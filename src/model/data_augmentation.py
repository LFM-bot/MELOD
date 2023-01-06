import copy
import math
import random
import numpy as np
import torch


class AbstractDataAugmentor:
    def __init__(self, intensity):
        self.intensity = intensity

    def transform(self, item_seq, seq_len):
        """
        :param item_seq: torch.LongTensor, [batch, max_len]
        :param seq_len: torch.LongTensor, [batch]
        :return: aug_seq: torch.LongTensor, [batch, max_len]
        """
        raise NotImplementedError


class CropDataAugmentor(AbstractDataAugmentor):
    def __init__(self, intensity):
        super(CropDataAugmentor, self).__init__(intensity)

    def transform(self, item_seq, seq_len):
        """
        :param item_seq: torch.LongTensor, [batch, max_len]
        :param seq_len: torch.LongTensor, [batch]
        :return: aug_seq: torch.LongTensor, [batch, max_len]
        """
        max_len = item_seq.size(-1)
        aug_seq_len = torch.ceil(seq_len * self.intensity).long()
        # get start index
        index = torch.arange(max_len, device=seq_len.device)
        index = index.expand_as(item_seq)
        up_bound = (seq_len - aug_seq_len).unsqueeze(-1)
        prob = torch.zeros_like(item_seq, device=seq_len.device).float()
        prob[index <= up_bound] = 1.
        start_index = torch.multinomial(prob, 1)
        # item indices in subsequence
        gather_index = torch.arange(max_len, device=seq_len.device)
        gather_index = gather_index.expand_as(item_seq)
        gather_index = gather_index + start_index
        max_seq_len = aug_seq_len.unsqueeze(-1)
        gather_index[index >= max_seq_len] = 0
        # augmented subsequence
        aug_seq = torch.gather(item_seq, -1, gather_index).long()
        aug_seq[index >= max_seq_len] = 0

        return aug_seq, aug_seq_len


class MaskDataAugmentor(AbstractDataAugmentor):
    def __init__(self, intensity, mask_id=0):
        super(MaskDataAugmentor, self).__init__(intensity)
        self.mask_id = mask_id

    def transform(self, item_seq, seq_len):
        """
        :param item_seq: torch.LongTensor, [batch, max_len]
        :param seq_len: torch.LongTensor, [batch]
        :return: aug_seq: torch.LongTensor, [batch, max_len]
        """
        max_len = item_seq.size(-1)
        aug_seq = item_seq.clone()
        aug_seq_len = seq_len.clone()
        # get mask item id
        mask_item_size = math.ceil(max_len * self.intensity)
        prob = torch.ones_like(item_seq, device=seq_len.device).float()
        masked_item_id = torch.multinomial(prob, mask_item_size)
        # mask
        aug_seq = aug_seq.scatter(-1, masked_item_id, self.mask_id)

        return aug_seq, aug_seq_len


class ReorderDataAugmentor(AbstractDataAugmentor):
    def __init__(self, intensity):
        super(ReorderDataAugmentor, self).__init__(intensity)

    def transform(self, item_seq, seq_len):
        """
        :param item_seq: torch.LongTensor, [batch, max_len]
        :param seq_len: torch.LongTensor, [batch]
        :return: aug_seq: torch.LongTensor, [batch, max_len]
        """
        copied_item_seq = item_seq.cpu().numpy()
        copied_seq_len = seq_len.cpu().numpy()
        sub_seq_length = np.ceil(copied_seq_len * self.intensity).astype(int)
        up_bound = copied_seq_len - sub_seq_length - 1
        up_bound[up_bound < 0] = 0

        for i in range(len(copied_item_seq)):
            one_item_seq = copied_item_seq[i]
            start_index = random.randint(0, up_bound[i])
            sub_seq = one_item_seq[start_index:start_index + sub_seq_length[i]]
            random.shuffle(sub_seq)

            left_seq = one_item_seq[:start_index]
            right_seq = one_item_seq[start_index + sub_seq_length[i]:]
            reordered_seq = np.concatenate([left_seq, sub_seq, right_seq], axis=0)

            copied_item_seq[i] = reordered_seq

        aug_item_seq = torch.from_numpy(copied_item_seq).to(seq_len.device)
        aug_seq_len = seq_len

        return aug_item_seq, aug_seq_len


class Crop(object):
    """Randomly crop a subseq from the original sequence"""
    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.tao*len(copied_sequence))
        #randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence)-sub_seq_length-1)
        if sub_seq_length<1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index:start_index+sub_seq_length]
            return cropped_seq


class Mask(object):
    """Randomly mask k items given a sequence"""
    def __init__(self, gamma=0.7, mask_id=0):
        self.gamma = gamma
        self.mask_id = mask_id

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.gamma*len(copied_sequence))
        mask = [self.mask_id for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence


class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""
    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.beta*len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence)-sub_seq_length-1)
        sub_seq = copied_sequence[start_index:start_index+sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + \
                        copied_sequence[start_index+sub_seq_length:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq


AUGMENTATIONS = {'crop': Crop, 'mask': Mask, 'reorder': Reorder}

if __name__ == '__main__':
   seq = np.arange(10).tolist()
   mask_aug = Mask(0.2)
   print(mask_aug(seq))
   print(Crop(0.5)(seq))
   print(Reorder(0.5)(seq))
