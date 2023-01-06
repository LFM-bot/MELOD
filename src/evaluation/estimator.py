import re
import torch
from tqdm import tqdm
from src.evaluation.metrics import Metric


class Estimator:
    def __init__(self, config):
        self.popularity = None
        self.config = config
        self.metrics = config.metric
        self.k_list = config.k
        self.dev = config.device
        self.metric_res_dict = {}
        self.eval_loss = 0.
        self.max_k = max(self.k_list)
        self.split_type = config.split_type
        self.eval_mode = config.eval_mode
        self.neg_size = 0
        if self.eval_mode != 'full':
            self.neg_size = int(re.findall(r'\d+', self.eval_mode)[0])
            self.eval_mode = self.eval_mode[:3]
        self._reset_metric_rec()
    
    def _reset_metric_rec(self):
        for metric in self.metrics:
            for k in self.k_list:
                self.metric_res_dict[f'{metric}@{k}'] = 0.
        self.eval_loss = 0.
    
    def load_item_popularity(self, pop):
        self.popularity = torch.tensor(pop, dtype=torch.float, device=self.dev)
    
    @torch.no_grad()
    def evaluate(self, eval_loader, model):
        model.eval()
        self._reset_metric_rec()

        eval_iter = tqdm(enumerate(eval_loader), total=len(eval_loader))
        eval_iter.set_description('evaluating')
        for _, batch in eval_iter:
            batch = (t.to(self.dev) for t in batch)
            batch_input, batch_length, batch_target = batch
            logits = model(batch_input, batch_length)
            model_loss = model.get_loss(batch_input, logits, batch_target)
            logits = self.neg_item_select(logits, batch_input, batch_target)

            self.calc_metrics(logits, batch_target)
            self.eval_loss += model_loss.item()

        for metric in self.metrics:
            for k in self.k_list:
                self.metric_res_dict[f'{metric}@{k}'] /= len(eval_loader)

        eval_loss = self.eval_loss / len(eval_loader)

        return self.metric_res_dict, eval_loss

    @torch.no_grad()
    def test(self, test_loader, model):
        self._reset_metric_rec()

        for i, batch in enumerate(test_loader):
            batch = (t.to(self.dev) for t in batch)
            batch_input, batch_length, batch_target = batch
            prediction = model(batch_input, batch_length)
            prediction = self.neg_item_select(prediction, batch_input, batch_target)
            self.calc_metrics(prediction, batch_target)

        for metric in self.metrics:
            for k in self.k_list:
                self.metric_res_dict[f'{metric}@{k}'] /= len(test_loader)

        return self.metric_res_dict

    def calc_metrics(self, prediction, target):
        _, topk_index = torch.topk(prediction, self.max_k, -1)  # [batch, max_k]
        topk_socre = torch.gather(prediction, index=topk_index, dim=-1)
        idx_sorted = torch.argsort(topk_socre, dim=-1, descending=True)
        max_k_item_sorted = torch.gather(topk_index, index=idx_sorted, dim=-1)

        for metric in self.metrics:
            for k in self.k_list:
                score = getattr(Metric, f'calc_{metric.upper()}')(max_k_item_sorted, target, k)
                self.metric_res_dict[f'{metric}@{k}'] += score

    def calc_metrics_(self, prediction, target):
        _, topk_index = torch.topk(prediction, self.max_k, -1)  # [batch, max_k]
        topk_socre = torch.gather(prediction, index=topk_index, dim=-1)
        idx_sorted = torch.argsort(topk_socre, dim=-1, descending=True)
        max_k_item_sorted = torch.gather(topk_index, index=idx_sorted, dim=-1)

        metric_res_dict = {}
        for metric in self.metrics:
            for k in self.k_list:
                score = getattr(Metric, f'calc_{metric.upper()}')(max_k_item_sorted, target, k)
                metric_res_dict[f'{metric}@{k}'] += score

        return metric_res_dict

    def neg_item_select(self, prediction, input, target):
        """
        Leave scores for one pos items and n neg items
        :param prediction: [batch_size, num_items]
        :param input: [batch_size, max_len]
        :param target: [batch_size]
        """
        if self.eval_mode == 'full':
            return prediction
        # sample negative items
        target = target.unsqueeze(-1)
        mask_item = torch.cat([input, target], dim=-1)  # [batch, max_len + 1]

        if self.eval_mode == 'uni':
            sample_prob = torch.ones_like(prediction, device=self.dev) / prediction.size(-1)
        elif self.eval_mode == 'pop':
            sample_prob = self.popularity.unsqueeze(0).repeat(prediction.size(0), 1)
        else:
            raise NotImplementedError('Choose eval_model from [full, popxxx, unixxx]')
        sample_prob = sample_prob.scatter(dim=-1, index=mask_item, value=0.)
        neg_item = torch.multinomial(sample_prob, self.neg_size)  # [batch, neg_size]
        # mask non-rank items
        rank_item = torch.cat([neg_item, target], dim=-1)  # [batch, neg_size + 1]
        mask = torch.ones_like(prediction, device=self.dev).bool()
        mask = mask.scatter(dim=-1, index=rank_item, value=False)
        masked_pred = torch.masked_fill(prediction, mask, 0.)

        return masked_pred


if __name__ == '__main__':
    a = torch.arange(0, 9).view(3, -1)
    mask = torch.ones_like(a).bool()
    mask[-1][-1] = False
    print(~mask)
    res = torch.masked_fill(a, mask, False)
    print(a)
    print(res)
