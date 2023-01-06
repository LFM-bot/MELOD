import pickle
import logging
import datetime
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time as t
import numpy as np


def load_pickle(file_path):
    return pickle.load(open(file_path, 'rb'), encoding='latin1')


def pkl_to_txt(dataset='beauty'):
    data_dir = '../../dataset'
    file_path = os.path.join(data_dir, f'{dataset}/{dataset}_seq.pkl')
    record = load_pickle(file_path)

    # write to xxx.txt
    target_file = os.path.join(data_dir, f'{dataset}/{dataset}_seq.txt')
    with open(target_file, 'w') as fw:
        for seq in record:
            seq = list(map(str, seq))
            seq_str = ' '.join(seq) + '\n'
            fw.write(seq_str)


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


def get_activate(act='relu'):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leaky_relu':
        return nn.LeakyReLU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise KeyError(f'Not support current activate function: {act}, please add by yourself.')


if __name__ == '__main__':
    datasets = ['ml-10m']
    for dataset in datasets:
        pkl_to_txt(dataset)