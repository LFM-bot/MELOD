import argparse
from easydict import EasyDict
import torch.cuda
from src.utils.utils import HyperParamDict

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
    parser = HyperParamDict("Default hyper-parameters for training.")
    # Model
    parser.add_argument('--model', default='MELOD')
    parser.add_argument('--model_type', default='Sequential', choices=['Sequential', 'Knowledge'])
    # Data
    parser.add_argument('--dataset', default='toys', type=str,
                        choices=['beauty', 'cellphone', 'cloth', 'cd', 'grocery', 'yelp', 'toys'])
    parser.add_argument('--data_aug', action='store_false', help='data augmentation')
    parser.add_argument('--target_len', default=3, type=int, help='target length for target sequence')
    parser.add_argument('--use_tar_len', action='store_false', help='if use target sequence')
    parser.add_argument('--seq_filter_len', default=3, type=int, help='filter seq less than 3')
    parser.add_argument('--if_filter_target', action='store_false',
                        help='if filter target appearing in previous sequence')
    parser.add_argument('--separator', default=' ', type=str, help='separator to split item sequence')
    parser.add_argument('--graph_type', default='None', type=str, help='do not use graph',
                        choices=['None', 'BIPARTITE', 'TRANSITION'])
    parser.add_argument('--max_len', default=50, type=int, help='max sequence length')
    # Pretraining
    parser.add_argument('--do_pretraining', default=False, action='store_true')
    parser.add_argument('--pretraining_task', default='MISP', type=str, choices=['MISP', 'MIM', 'PID'],
                        help='pretraining task:' \
                             'MISP: Mask Item Prediction and Mask Segment Prediction' \
                             'MIM: Mutual Information Maximization' \
                             'PID: Pseudo Item Discrimination'
                        )
    parser.add_argument('--pretraining_epoch', default=10, type=int)
    parser.add_argument('--pretraining_batch', default=512, type=int)
    parser.add_argument('--pretraining_lr', default=1e-3, type=float)
    parser.add_argument('--pretraining_l2', default=0., type=float, help='l2 normalization')
    # Training
    parser.add_argument('--epoch_num', default=100, type=int)
    parser.add_argument('--train_batch', default=512, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--l2', default=0., type=float, help='l2 normalization')
    parser.add_argument('--patience', default=5, help='early stop patience')
    parser.add_argument('--device', default=get_device(), choices=['cuda:0', 'cpu'],
                        help='training on gpu or cpu, default gpu')
    parser.add_argument('--num_worker', default=0, type=int,
                        help='num_workers for dataloader, best: 6')

    # Evaluation
    parser.add_argument('--split_type', default='valid_and_test', choices=['valid_only', 'valid_and_test'])
    parser.add_argument('--split_mode', default='LS_R0.2', type=str,
                        help='LS: Leave-one-out splitting.'
                             'LS_R@0.2: use LS and a ratio 0.x of test data for validate if use valid_and_test.'
                             'PS: Pre-Splitting, prepare xx.train and xx.eval, also xx.test if use valid_and_test')
    parser.add_argument('--eval_mode', default='uni100', help='[uni100, pop100, full]')
    parser.add_argument('--metric', default=['hit', 'ndcg'], help='[hit, ndcg, mrr, recall]')
    parser.add_argument('--k', default=[5, 10], help='top k for each metric')
    parser.add_argument('--valid_metric', default='hit@10', help='specifies which indicator to apply early stop')
    parser.add_argument('--eval_batch', default=512, type=int)

    # save
    parser.add_argument('--log_save', default='log', type=str, help='log saving path')
    parser.add_argument('--model_save', default='save', type=str, help='model saving path')

    return parser


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
    print(config.data_aug)
