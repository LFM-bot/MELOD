import copy
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate


def load_specified_dataset(model_name, config):
    if model_name == 'MELOD':
        return KERLDataset
    else:
        return SequentialDataset


class BaseSequentialDataset(Dataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(BaseSequentialDataset, self).__init__()
        self.num_items = config.num_items
        self.SR_task = config.SR_task
        self.session_len = config.session_len
        self.config = config
        self.train = train
        self.dataset = config.dataset
        self.max_len = config.max_len
        self.item_seq = data_pair[0]
        self.label = data_pair[1]

    def get_SRtask_input(self, idx):
        item_seq = self.item_seq[idx]
        target = self.label[idx]

        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        if not self.train and self.SR_task == 'Next-Session':
            assert len(target) == self.session_len

        return (torch.tensor(item_seq, dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))

    def __getitem__(self, idx):
        return self.get_SRtask_input(idx)

    def __len__(self):
        return len(self.item_seq)

    def collate_fn(self, x):
        return self.basic_SR_collate_fn(x)

    def basic_SR_collate_fn(self, x):
        """
        x: [(seq_1, len_1, tar_1), ..., (seq_n, len_n, tar_n)]
        """
        item_seq, seq_len, target = default_collate(x)
        batch_dict = {}
        batch_dict['item_seq'] = item_seq
        batch_dict['seq_len'] = seq_len
        batch_dict['target'] = target
        return batch_dict


class SequentialDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(SequentialDataset, self).__init__(config, data_pair, additional_data_dict, train)


class KERLDataset(BaseSequentialDataset):
    def __init__(self, config, data_pair, additional_data_dict=None, train=True):
        super(KERLDataset, self).__init__(config, data_pair, additional_data_dict, train)
        self.tar_length = config.episode_len

    def __getitem__(self, idx):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(idx)

        # for training
        item_seq = self.item_seq[idx]
        target_seq = self.label[idx]
        if not isinstance(target_seq, list):
            target_seq = [target_seq]

        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        target_len = len(target_seq)

        # crop
        item_seq = item_seq[-self.max_len:]
        # padding
        item_seq = item_seq + (self.max_len - seq_len) * [0]
        target_seq = target_seq + (self.tar_length - target_len) * [0]

        assert len(item_seq) == self.max_len
        assert len(target_seq) == self.tar_length

        return (torch.tensor(item_seq, dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long),
                torch.tensor(target_len, dtype=torch.long))

    def collate_fn(self, x):
        if not self.train:
            return self.basic_SR_collate_fn(x)
        tensor_list = default_collate(x)
        item_seq, seq_len, target_seq, target_len = tensor_list

        batch_dict = {}
        batch_dict['item_seq'] = item_seq
        batch_dict['seq_len'] = seq_len
        batch_dict['target_seq'] = target_seq
        batch_dict['target_len'] = target_len

        return batch_dict

