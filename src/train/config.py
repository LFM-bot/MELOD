import argparse
from easydict import EasyDict
import torch.cuda

EXP_HYPER_LIST = {'Data': {'dataset': None, 'data_aug': None, 'seq_filter_len': None, 'if_filter_target': None,
                           'use_tar_len': None, 'target_len': None, 'max_len': None},
                  'Pretraining': {'do_pretraining': None, 'pretraining_task': None, 'pretraining_epoch': None,
                                  'pretraining_batch': None, 'pretraining_lr': None, 'pretraining_l2': None},
                  'Training': {'epoch_num': None, 'train_batch': None,
                               'learning_rate': None, 'l2': None, 'patience': None,
                               'device': None, 'num_worker': None},
                  'Evaluation': {'split_type': None, 'split_mode': None, 'eval_mode': None, 'metric': None, 'k': None,
                                 'valid_metric': None, 'eval_batch': None},
                  'Save': {'log_save': None, 'model_save': None}}


def experiment_hyper_load(exp_config):
    hyper_types = EXP_HYPER_LIST.keys()
    for hyper_dict in EXP_HYPER_LIST.values():
        for hyper in hyper_dict.keys():
            hyper_dict[hyper] = getattr(exp_config, hyper)
    return list(hyper_types), EXP_HYPER_LIST


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_default_config():
    default_config = EasyDict({})

    # Model
    default_config.model = 'MELOD'  # default model name
    default_config.model_type = 'Knowledge'  # choices=['Knowledge', 'Sequential', 'RL']
    default_config.loss_type = 'CUSTOM'  # choices=['CE', 'BPR', 'BCE', 'CUSTOM']

    # Data
    default_config.dataset = 'beauty'  # choices=['beauty', 'cellphone', 'cloth', ...]
    default_config.data_aug = True  # if do sequence data augmentation
    default_config.seq_filter_len = 3  # threshold to filter short sequences
    default_config.if_filter_target = True  # if filter target item appearing in previous item sequence'
    default_config.use_tar_len = False  # if use multi-step target sequence
    default_config.target_len = 3  # length for mult-step target sequence
    default_config.separator = ' '  # separator to split item sequence from data file, choices=[' ', ',']
    default_config.max_len = 50  # max item sequence length
    default_config.graph_type = None  # choices=['None', 'BIPARTITE', 'TRANSITION']

    # Pretraining
    default_config.do_pretraining = False  # if do pretraining
    default_config.pretraining_task = 'MISP'  # choices=[None, 'MISP', 'MIM', 'PID']
    # MISP: Mask Item Prediction and Mask Segment Prediction
    # MIM: Mutual Information Maximization
    # PID: Pseudo Item Discrimination
    default_config.pretraining_epoch = 10
    default_config.pretraining_batch = 512
    default_config.pretraining_lr = 1e-3
    default_config.pretraining_l2 = 0.

    # Training
    default_config.epoch_num = 100
    default_config.train_batch = 512
    default_config.learning_rate = 1e-3
    default_config.l2 = 0.
    default_config.patience = 5  # early stop patience
    default_config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # choices=['cuda:0', 'cpu']
    default_config.num_worker = 0  # num_workers for dataloader

    # Evaluation
    default_config.eval_batch = 512 
    default_config.split_type = 'valid_and_test'  # choices=['valid_only', 'valid_and_test']
    default_config.split_mode = 'LS_R@0.2'  # how to split sequence data, choices=['LS', 'LS_R@0.x', 'PS']
    # LS: Leave-one-out splitting, last item for test, second item for eval, rest for training.
    # LS_R@0.2: use LS to split train and test, then split 0.x data from test set for eval if use valid_and_test.
    # PS: Pre-Splitting, prepare xx.train and xx.eval, and xx.test is needed if use valid_and_test
    default_config.eval_mode = 'uni100'  # choices=[unixx: random neg sample, popxx: pop-based neg sample, full: all]
    default_config.metric = ['hit', 'ndcg']  # choices='[hit, ndcg, mrr, recall]'
    default_config.k = [5, 10]  # top k for each metric
    default_config.valid_metric = 'hit@10'  # metric to apply early stop

    # save
    default_config.log_save = 'log'  # log saving path
    default_config.model_save = 'save'  # model saving path

    return default_config


def config_override(cmd_config, model_config=None):
    default_config = get_default_config()
    command_args = set([arg for arg in vars(cmd_config)])

    # overwrite model config by cmd config
    for arg in model_config.keys():
        if arg in command_args:
            setattr(model_config, arg, getattr(cmd_config, arg))

    # overwrite default config by cmd config
    for arg in default_config.keys():
        if arg in command_args:
            setattr(default_config, arg, getattr(cmd_config, arg))

    # overwrite default config by model config
    for arg in model_config:
        setattr(default_config, arg, getattr(model_config, arg))

    return default_config


if __name__ == '__main__':
    config = get_default_config()
    print(config.use_tar_len)
    arr = [1, 2, 3]
    print(arr[2: 5])
