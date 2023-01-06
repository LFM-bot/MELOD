import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.utils import neg_sample
from src.model.data_augmentation import Crop, Mask, Reorder
from src.model.data_augmentation import AUGMENTATIONS


def load_specified_dataset(model_name, config):
    if model_name in ['BERT4Rec']:
        return MaskItemPredictDataset
    elif model_name in ['KERL', 'MELOD', 'MELOD_DIIN', "MELOD_GRU"]:
        return SequentialDatasetWithTargetSeq
    elif model_name in ['CL4SRec'] and not config.do_pretraining:
        return RecWithContrastLearningDataset
    elif model_name in ['STGCR']:
        return ReversedSeqDataset
    else:
        return SequentialDataset


class BaseSequentialDataset(Dataset):
    def __init__(self, num_item, config, data_pair, train=True):
        super(BaseSequentialDataset, self).__init__()
        self.num_item = num_item
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

        return (torch.tensor(item_seq, dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))

    def __getitem__(self, idx):
        return self.get_SRtask_input(idx)

    def __len__(self):
        return len(self.item_seq)

    @staticmethod
    def collate_fn(x):
        return [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))], 0).long() for j in range(len(x[0]))]


class SequentialDataset(BaseSequentialDataset):
    def __init__(self, num_item, config, data_pair, train=True):
        super(SequentialDataset, self).__init__(num_item, config, data_pair, train)


class SequentialDatasetWithTargetSeq(BaseSequentialDataset):
    def __init__(self, num_items, config, data_pair, train=True):
        super(SequentialDatasetWithTargetSeq, self).__init__(num_items, config, data_pair, train)
        self.tar_length = config.target_len

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


class MaskItemPredictDataset(BaseSequentialDataset):
    """
    For bert training.
    """
    def __init__(self, num_items, config, data_pair, train=True):
        super(MaskItemPredictDataset, self).__init__(num_items, config, data_pair, train)
        self.mask_id = num_items
        self.num_items = num_items + 1
        self.max_len = config.max_len + 1  # add mask at last pos

        self.mask_ratio = config.mask_ratio
        self.one_mask_ratio = config.one_mask_ratio  # only mask last

    def __getitem__(self, index):
        sequence = self.item_seq[index]  # pos_items

        # eval and test phase
        if not self.train:
            item_sequence = sequence
            item_sequence.append(self.mask_id)
            seq_len = len(item_sequence) if len(item_sequence) < self.max_len else self.max_len
            target = self.label[index]

            item_sequence = item_sequence[-self.max_len:]
            item_sequence = item_sequence + (self.max_len - seq_len) * [0]

            return (torch.tensor(item_sequence, dtype=torch.long),
                    torch.tensor(seq_len, dtype=torch.long),
                    torch.tensor(target, dtype=torch.long))

        # for training: Masked Item Prediction
        masked_item_sequence = []
        pos_items = copy.deepcopy(sequence)
        if random.random() < self.one_mask_ratio:
            masked_item_sequence = copy.deepcopy(sequence)  # keep the same
        else:
            for item in sequence:
                prob = random.random()
                if prob < self.mask_ratio:
                    masked_item_sequence.append(self.mask_id)
                else:
                    masked_item_sequence.append(item)
        # add mask at the last position
        masked_item_sequence.append(self.mask_id)
        pos_items.append(self.label[index])

        assert len(masked_item_sequence) == len(sequence) + 1
        assert len(pos_items) == len(sequence) + 1

        # crop sequence
        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]

        # padding sequence
        pad_len = self.max_len - len(masked_item_sequence)
        masked_item_sequence = masked_item_sequence + [0] * pad_len
        pos_items = pos_items + [0] * pad_len

        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len

        cur_tensors = (torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long))
        return cur_tensors


class RecWithContrastLearningDataset(BaseSequentialDataset):
    def __init__(self, num_items, config, data_pair, train=True):
        super(RecWithContrastLearningDataset, self).__init__(num_items, config, data_pair, train)
        self.mask_id = num_items
        self.aug_types = config.aug_types
        self.n_views = 2
        self.augmentations = []

        self.load_augmentor()

    def load_augmentor(self):
        for aug in self.aug_types:
            if aug == 'mask':
                self.augmentations.append(Mask(gamma=self.config.mask_ratio, mask_id=self.mask_id))
            else:
                self.augmentations.append(AUGMENTATIONS[aug](getattr(self.config, f'{aug}_ratio')))

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # for training
        # contrast learning augmented views
        item_seq = self.item_seq[index]
        target = self.label[index]
        aug_type = np.random.choice([i for i in range(len(self.augmentations))],
                                    size=self.n_views, replace=False)
        aug_seq_1 = self.augmentations[aug_type[0]](item_seq)
        aug_seq_2 = self.augmentations[aug_type[1]](item_seq)

        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_2 = aug_seq_2[-self.max_len:]

        aug_len_1 = len(aug_seq_1)
        aug_len_2 = len(aug_seq_2)

        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))
        assert len(aug_seq_1) == self.max_len
        assert len(aug_seq_2) == self.max_len

        # recommendation sequences
        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        cur_tensors = (torch.tensor(item_seq, dtype=torch.long),
                       torch.tensor(seq_len, dtype=torch.long),
                       torch.tensor(target, dtype=torch.long),
                       torch.tensor(aug_seq_1, dtype=torch.long),
                       torch.tensor(aug_seq_2, dtype=torch.long),
                       torch.tensor(aug_len_1, dtype=torch.long),
                       torch.tensor(aug_len_2, dtype=torch.long))

        return cur_tensors


class ReversedSeqDataset(BaseSequentialDataset):
    def __init__(self, num_items, config, data_pair, train=True):
        super(ReversedSeqDataset, self).__init__(num_items, config, data_pair, train)

    def __getitem__(self, index):
        # for eval and test
        if not self.train:
            return self.get_SRtask_input(index)

        # for training
        item_seq = self.item_seq[index]
        reversed_seq = list(reversed(item_seq))
        target = self.label[index]

        seq_len = len(item_seq) if len(item_seq) < self.max_len else self.max_len
        item_seq = item_seq[-self.max_len:]
        item_seq = item_seq + (self.max_len - seq_len) * [0]

        reversed_seq = reversed_seq[-self.max_len:]
        reversed_seq = reversed_seq + (self.max_len - seq_len) * [0]

        assert len(item_seq) == self.max_len

        return (torch.tensor(reversed_seq, dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long),
                torch.tensor(target, dtype=torch.long))


class MISPPretrainDataset(Dataset):
    """
    Masked Item & Segment Prediction (MISP)
    """
    def __init__(self, num_items, config, data_pair):
        self.mask_id = num_items
        self.mask_ratio = config.mask_ratio
        self.num_items = num_items + 1
        self.config = config
        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.max_len = config.max_len
        self.long_sequence = []

        for seq in self.item_seq:
            self.long_sequence.extend(seq)

    def __len__(self):
        return len(self.item_seq)

    def __getitem__(self, index):
        sequence = self.item_seq[index]  # pos_items

        # Masked Item Prediction
        masked_item_sequence = []
        neg_items = []
        pos_items = sequence

        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.mask_ratio:
                masked_item_sequence.append(self.mask_id)
                neg_items.append(neg_sample(item_set, self.num_items))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)
        # add mask at the last position
        masked_item_sequence.append(self.mask_id)
        neg_items.append(neg_sample(item_set, self.num_items))

        assert len(masked_item_sequence) == len(sequence)
        assert len(pos_items) == len(sequence)
        assert len(neg_items) == len(sequence)

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id: start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
            masked_segment_sequence = sequence[:start_id] + [self.mask_id] * sample_length + sequence[
                                                                                      start_id + sample_length:]
            pos_segment = [self.mask_id] * start_id + pos_segment + [self.mask_id] * (
                        len(sequence) - (start_id + sample_length))
            neg_segment = [self.mask_id] * start_id + neg_segment + [self.mask_id] * (
                        len(sequence) - (start_id + sample_length))

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # crop sequence
        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]
        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = masked_item_sequence + [0] * pad_len
        pos_items = pos_items + [0] * pad_len
        neg_items = neg_items + [0] * pad_len
        masked_segment_sequence = masked_segment_sequence + [0] * pad_len
        pos_segment = pos_segment + [0] * pad_len
        neg_segment = neg_segment + [0] * pad_len

        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        cur_tensors = (torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long),
                       torch.tensor(neg_items, dtype=torch.long),
                       torch.tensor(masked_segment_sequence, dtype=torch.long),
                       torch.tensor(pos_segment, dtype=torch.long),
                       torch.tensor(neg_segment, dtype=torch.long))
        return cur_tensors


class MIMPretrainDataset(Dataset):
    def __init__(self, num_items, config, data_pair):
        self.mask_id = num_items
        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.config = config
        self.max_len = config.max_len
        self.n_views = 2
        self.augmentations = [Crop(tao=config.crop_ratio),
                              Mask(gamma=config.mask_ratio, mask_id=self.mask_id),
                              Reorder(beta=config.reorder_ratio)]

    def __getitem__(self, index):
        aug_type = np.random.choice([i for i in range(len(self.augmentations))],
                                    size=self.n_views, replace=False)
        item_seq = self.item_seq[index]
        aug_seq_1 = self.augmentations[aug_type[0]](item_seq)
        aug_seq_2 = self.augmentations[aug_type[1]](item_seq)

        aug_seq_1 = aug_seq_1[-self.max_len:]
        aug_seq_2 = aug_seq_2[-self.max_len:]

        aug_len_1 = len(aug_seq_1)
        aug_len_2 = len(aug_seq_2)

        aug_seq_1 = aug_seq_1 + [0] * (self.max_len - len(aug_seq_1))
        aug_seq_2 = aug_seq_2 + [0] * (self.max_len - len(aug_seq_2))
        assert len(aug_seq_1) == self.max_len
        assert len(aug_seq_2) == self.max_len

        aug_seq_tensors = (torch.tensor(aug_seq_1, dtype=torch.long),
                           torch.tensor(aug_seq_2, dtype=torch.long),
                           torch.tensor(aug_len_1, dtype=torch.long),
                           torch.tensor(aug_len_2, dtype=torch.long))

        return aug_seq_tensors

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.item_seq)


class PIDPretrainDataset(Dataset):
    def __init__(self, num_items, config, data_pair):
        self.num_items = num_items
        self.item_seq = data_pair[0]
        self.label = data_pair[1]
        self.config = config
        self.max_len = config.max_len
        self.pseudo_ratio = config.pseudo_ratio

    def __getitem__(self, index):
        item_seq = self.item_seq[index]
        pseudo_seq = []
        target = []

        for item in item_seq:
            if random.random() < self.pseudo_ratio:
                pseudo_item = neg_sample(item_seq, self.num_items)
                pseudo_seq.append(pseudo_item)
                target.append(0)
            else:
                pseudo_seq.append(item)
                target.append(1)

        pseudo_seq = pseudo_seq[-self.max_len:]
        target = target[-self.max_len:]

        pseudo_seq = pseudo_seq + [0] * (self.max_len - len(pseudo_seq))
        target = target + [0] * (self.max_len - len(target))
        assert len(pseudo_seq) == self.max_len
        assert len(target) == self.max_len
        pseudo_seq_tensors = (torch.tensor(pseudo_seq, dtype=torch.long),
                          torch.tensor(target, dtype=torch.float))

        return pseudo_seq_tensors

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.item_seq)


