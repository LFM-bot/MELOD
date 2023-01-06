import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import dataset
from src.dataset.dataset import load_specified_dataset
from src.dataset.data_processor import DataProcessor
from src.evaluation.estimator import Estimator
from src.utils.recorder import Recorder
import src.model as model
from src.train.config import experiment_hyper_load, config_override


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model_name = config.model
        self._config_override(self.model_name, config)

        # pretraining args
        self.pretraining_model = None
        self.do_pretraining = self.config.do_pretraining
        self.pretraining_task = self.config.pretraining_task
        self.pretraining_epoch = self.config.pretraining_epoch
        self.pretraining_batch = self.config.pretraining_batch
        self.pretraining_lr = self.config.pretraining_lr
        self.pretraining_l2 = self.config.pretraining_l2

        # training args
        self.training_model = None
        self.user_target_len = self.config.use_tar_len
        self.num_worker = self.config.num_worker
        self.train_batch = self.config.train_batch
        self.eval_batch = self.config.eval_batch
        self.lr = self.config.learning_rate
        self.l2 = self.config.l2
        self.epoch_num = self.config.epoch_num
        self.dev = torch.device(self.config.device)
        self.split_type = self._set_split_mode(self.config.split_type)
        self.do_test = self.split_type == 'valid_and_test'
        self.num_items = 0

        # main components
        self.data_processor = DataProcessor(self.config)
        self.estimator = Estimator(self.config)
        self.recorder = Recorder(self.config)

        # load data
        self.data_pair_dict = self.data_processor.prepare_data()
        self.estimator.load_item_popularity(self.data_processor.popularity)
        self._set_num_items()

    def start_training(self):
        if self.do_pretraining:
            self.pretraining()
        self.training()

    def pretraining(self):

        if self.pretraining_task in ['MISP', 'MIM', 'PID']:
            pretrain_dataset = getattr(dataset, f'{self.pretraining_task}PretrainDataset')
            pretrain_dataset = pretrain_dataset(self.num_items, self.config, self.data_pair_dict['train'])
        else:
            raise NotImplementedError(f'No such pretraining task: {self.pretraining_task}, '
                                      f'choosing from [MIP, MIM, PID]')
        train_loader = DataLoader(pretrain_dataset, batch_size=self.train_batch,
                                  shuffle=True, num_workers=0, drop_last=False)

        pretrain_model = self._load_model()

        opt = torch.optim.Adam(filter(lambda x: x.requires_grad, pretrain_model.parameters()),
                               self.pretraining_lr, weight_decay=self.pretraining_l2)

        self.experiment_setting_verbose(pretrain_model, training=False)

        logging.info('Start pretraining...')
        for epoch in range(self.pretraining_epoch):
            pretrain_model.train()
            self.recorder.epoch_restart()
            self.recorder.tik_start()
            train_iter = tqdm(enumerate(train_loader), total=len(train_loader))
            train_iter.set_description(f'pretraining  epoch: {epoch}')
            for i, batch in train_iter:
                batch = (t.to(self.dev) for t in batch)
                loss = getattr(pretrain_model, f'{self.pretraining_task}_pretrain_forward')(*batch)
                opt.zero_grad()
                loss.backward()
                opt.step()

                self.recorder.save_batch_loss(loss.item())
            self.recorder.tik_end()
            self.recorder.train_log_verbose(len(train_loader))

        self.pretraining_model = pretrain_model
        logging.info('Pre-training is over, prepare for training...')

    def training(self):

        SpecifiedDataSet = load_specified_dataset(self.model_name, self.config)
        train_dataset = SpecifiedDataSet(self.num_items, self.config, self.data_pair_dict['train'])
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch,
                                  shuffle=True, num_workers=0, drop_last=False)

        eval_dataset = SpecifiedDataSet(self.num_items, self.config, self.data_pair_dict['eval'], train=False)
        eval_loader = DataLoader(eval_dataset, batch_size=self.eval_batch,
                                 shuffle=False, num_workers=self.num_worker, drop_last=False)

        self.training_model = self._load_model()

        opt = torch.optim.Adam(filter(lambda x: x.requires_grad, self.training_model.parameters()), self.lr,
                               weight_decay=self.l2)
        self.recorder.reset()
        self.experiment_setting_verbose(self.training_model)

        logging.info('Start training...')
        for epoch in range(self.epoch_num):
            self.training_model.train()
            self.recorder.epoch_restart()
            self.recorder.tik_start()
            train_iter = tqdm(enumerate(train_loader), total=len(train_loader))
            train_iter.set_description('training  ')
            for i, batch in train_iter:
                # training forward
                batch = (t.to(self.dev) for t in batch)
                loss = self.training_model.train_forward(*batch, epoch=i)
                # backward propagation
                opt.zero_grad()
                loss.backward()
                opt.step()

                self.recorder.save_batch_loss(loss.item())
            self.recorder.tik_end()
            self.recorder.train_log_verbose(len(train_loader))

            # evaluation
            self.recorder.tik_start()
            eval_metric_result, eval_loss = self.estimator.evaluate(eval_loader, self.training_model)
            self.recorder.tik_end(mode='eval')
            self.recorder.eval_log_verbose(eval_metric_result, eval_loss, self.training_model)

            if self.recorder.early_stop:
                break

        self.recorder.report_best_res()

        if self.do_test:
            test_metric_res = self.test_model(self.data_pair_dict['test'])
            self.recorder.report_test_result(test_metric_res)

    def _set_split_mode(self, split_mode):
        assert split_mode in ['valid_and_test', 'valid_only'], f'Invalid split mode: {split_mode} !'
        return split_mode

    def _load_model(self):
        if self.do_pretraining and self.pretraining_model is not None:  # return pretraining model
            return self.pretraining_model

        if self.config.model_type.lower() == 'knowledge':
            return self._load_kg_model()
        elif self.config.model_type.lower() == 'sequential':
            return self._load_sequential_model()
        elif self.config.model_type.lower() == 'graph':
            return self._load_graph_model()
        else:
            pass

    def _load_sequential_model(self):
        Model = getattr(model, self.model_name)
        seq_model = Model(self.num_items, self.config).to(self.dev)
        return seq_model

    def _load_graph_model(self):
        # graph = self.data_processor.prepare_graph().to(self.dev)
        graph = self.data_processor.prepare_graph()
        Model = getattr(model, self.model_name)
        graph_model = Model(self.num_items, self.config, graph).to(self.dev)
        return graph_model

    def _load_kg_model(self):
        kg_map = self.data_processor.prepare_kg_map().float().to(self.dev)
        Model = getattr(model, self.model_name)
        kg_model = Model(self.num_items, self.config, kg_map).to(self.dev)
        return kg_model

    def _config_override(self, model_name, cmd_config):
        self.model_config = getattr(model, f'{model_name}_config')()
        self.config = config_override(cmd_config, self.model_config)

    def _set_num_items(self):
        self.num_items = self.data_processor.num_items

    def experiment_setting_verbose(self, model, training=True):
        if self.do_pretraining and training:
            return
        # model config
        logging.info('[1] Model Hyper-Parameter '.ljust(47, '-'))
        for arg in self.model_config.keys():
            logging.info(f'{arg}: {getattr(self.model_config, arg)}')
        # experiment config
        logging.info('[2] Experiment Hyper-Parameter '.ljust(47, '-'))
        # verbose_order = ['Data', 'Training', 'Evaluation', 'Save']
        hyper_types, exp_setting = experiment_hyper_load(self.config)
        for i, hyper_type in enumerate(hyper_types):
            hyper_start_log = (f'[2-{i + 1}] ' + hyper_type.lower() + ' hyper-parameter ').ljust(47, '-')
            logging.info(hyper_start_log)
            for hyper, value in exp_setting[hyper_type].items():
                logging.info(f'{hyper}: {value}')
        # data statistic
        self.data_processor.data_log_verbose(3)
        # model architecture
        self.report_model_info(model)

    def report_model_info(self, model):
        # model architecture
        logging.info('[1] Model Architecture '.ljust(47, '-'))
        logging.info(f'total parameters: {model.calc_total_params()}')
        logging.info(model)

    def test_model(self, test_data_pair=None):
        SpecifiedDataSet = load_specified_dataset(self.training_model, self.config)
        test_dataset = SpecifiedDataSet(self.num_items, self.config, test_data_pair, train=False)
        test_loader = DataLoader(test_dataset, batch_size=self.eval_batch, num_workers=self.num_worker,
                                 drop_last=False, shuffle=False)
        # load the best model
        self.recorder.load_best_model(self.training_model)
        self.training_model.eval()
        test_metric_result = self.estimator.test(test_loader, self.training_model)

        return test_metric_result


if __name__ == '__main__':
    pass
