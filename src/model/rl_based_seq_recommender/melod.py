import argparse
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch.nn.init import xavier_normal_
from src.model.abstract_recommeder import AbstractRLRecommender
from src.utils.utils import HyperParamDict


class MELOD(AbstractRLRecommender):
    def __init__(self, num_items, config, kg_map):
        super(MELOD, self).__init__(num_items, config)
        # hyperparams
        self.alpha = config.alpha
        self.lamda = config.lamda
        self.sas_prob = self.set_sas_prob(config.sas_prob)
        self.DIIN_loss_type = config.DIIN_loss_type.upper()
        self.episode_num = config.episode_num
        self.episode_len = config.episode_len
        self.embed_size = config.embed_size
        self.sample_size = config.sample_size if config.sample_size > 0 else num_items
        self.prob_sharpen = config.prob_sharpen
        self.hit_range = config.hit_range
        self.hit_r = 1.
        self.layer_norm_eps = 1e-12
        self.device = config.device

        # modules
        self.indu_loss_func = torch.nn.MarginRankingLoss(margin=config.alpha) if self.DIIN_loss_type == 'MR' else nn.BCELoss()
        self.kg_embedding = kg_map[: self.num_items]
        self.kg_embedding.requires_grad = not config.freeze_kg
        self.item_embedding = nn.Embedding(self.num_items, self.embed_size)

        self.seq_encoder = Transformer(embed_size=config.embed_size,
                                       num_heads=config.num_heads,
                                       num_blocks=config.num_blocks,
                                       attn_dropout=config.attn_dropout,
                                       ffn_hidden=config.ffn_hidden,
                                       ffn_dropout=config.ffn_dropout,
                                       max_len=self.max_len,
                                       layer_norm_eps=self.layer_norm_eps)

        self.W1 = nn.Linear(self.embed_size * 2, self.embed_size * 2)
        self.W2 = nn.Linear(self.embed_size * 2, 2)
        # self.W2 = nn.Sequential(nn.Linear(self.embed_size * 2, self.embed_size),
        #                         nn.GELU(),
        #                         nn.Linear(self.embed_size, 2))
        self.emb_drop = nn.Dropout(p=config.ffn_dropout)
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            xavier_normal_(module.weight)

    def forward(self, item_seq, seq_len):
        """
        Parameters
        ----------
        item_seq: torch.LongTensor, [batch_size,max_len]
        seq_len: torch.LongTensor, [batch_size]
        """
        # sequential state
        seq_embedding = self.item_embedding(item_seq)
        seq_embedding = self.emb_drop(seq_embedding)
        seq_states = self.seq_encoder(seq_embedding)
        seq_state = self.gather_index(seq_states, seq_len - 1)
        # kg state
        kg_state = self.kg_encoder_avg(item_seq, seq_len, drop=True)
        final_state = self.W1(torch.cat([seq_state, kg_state], dim=1))

        logits = self.DIIN(final_state)

        return logits

    def train_forward(self, item_seq, seq_len, target, *args, **kwargs):
        """
        Parameters
        ----------
        item_seq: [batch, max_len]
        seq_len: [batch]
        target: [batch]
        """
        tar_len = args[0]  # [batch, ep_len]
        sas_prob = self.get_sas_prob(target, tar_len)
        prediction, action, reward, indu_weight = self.teach_and_explore(item_seq, target, seq_len, tar_len, sas_prob)
        return self.get_final_loss(prediction, action, reward, indu_weight)

    def teach_and_explore(self, item_seq, targets, train_len, tar_len, sas_prob):
        """
        RL training by Teach and Explore components.

        Parameters
        ----------
        item_seq : torch.LongTensor, [batch_size,L]
            each training sequence in batch, max sequence length L : 50
        targets : [batch_size,3] , max targets size T : 3
        sas_prob : extra added probability , [batch_size,num_items]
        train_len : true lengths of training sequences , [batch_size]
        tar_len : true lengths of tragets , [batch_size]

        Returns
        -------
        origin_prob : original probability , [batch_size,item_nums]
        first_sample : items sampled for the first time , [batch_size]
        reward : rewards for this sampling episode , [batch_size]
        """
        # sequential state
        seq_embeddings = self.item_embedding(item_seq)
        seq_embeddings = self.emb_drop(seq_embeddings)
        seq_states = self.seq_encoder(seq_embeddings)
        seq_state = self.gather_index(seq_states, train_len - 1)
        # kg state
        kg_state = self.kg_encoder_avg(item_seq, train_len, drop=True)

        final_state = self.W1(torch.cat([seq_state, kg_state], dim=1))

        logits, indu_factor = self.DIIN(final_state, target=targets[:, 0])
        logits = torch.scatter(logits, -1, item_seq, -sys.maxsize)  # mask existing items
        prob = F.softmax(logits, dim=1)  # [batch_size, num_items]
        prob_double = prob.repeat(2, 1)  # [2 * batch_size, num_items]

        exploration_sample = self.sample_from_topk(prob, k=self.sample_size)
        teach_sample = self.sample_from_topk(sas_prob, sharpen=False)
        total_sample = torch.cat([exploration_sample, teach_sample], dim=0)  # [batch_size * 2]

        # data duplication
        item_seq = item_seq.repeat(2, 1)
        targets = targets.repeat(2, 1)
        train_len = train_len.repeat(2)
        tar_len = tar_len.repeat(2)

        reward = self.generateReward(prob_double, item_seq, total_sample, self.episode_num,
                                     self.episode_len, targets, train_len, tar_len)

        return prob_double, total_sample, reward, indu_factor

    def kg_encoder_avg(self, item_seq, seq_len, drop=True):
        """
        Get encoded knowledge state.
        Parameters
        ----------
        item_seq: [batch_size, max_len]
        seq_len: [batch_size]

        Returns
        -------
        batch_kg: [batch_size,dims]
        """
        seq_len = seq_len.unsqueeze(-1)  # [batch_size,1]
        batch_kg = self.kg_embedding[item_seq]  # [batch_size,seq_len,dims]
        batch_kg_drop = self.emb_drop(batch_kg) if drop else batch_kg
        # mask invalid item
        max_len = batch_kg_drop.size(1)
        batch_size = batch_kg_drop.size(0)
        indices = torch.arange(max_len).repeat(batch_size, 1).to(self.device)  # [batch_size,seq_len]
        mask = indices.lt(seq_len).unsqueeze(-1)  # [batch, max_len, 1]
        batch_kg_drop.masked_fill_(mask == 0, value=0.)

        encoded_kg_state = batch_kg_drop.sum(dim=1) / seq_len  # [batch_size,dims]

        return encoded_kg_state

    def generateReward(self, initial_prob, pre_item_seq, action_taken, episode_num,
                       episode_len, ground_item_seq, seq_len, tar_len):
        """
        Calculate cumulative discounted rewards.

        Parameters
        ----------
        initial_prob: [batch_size, num_items]
        pre_item_seq: [batch_size , max_len]
        action_taken: first sampled items , [batch_size]
        episode_len: the number of items to be sampled , here is T = 3
        episode_num: the number of total sample episodes , here is 3
        ground_item_seq: targets for train data in this batch , [batch_size, 3]
        seq_len : torch.LongTensor, [batch_size]
            True lengths of training sequences.
        tar_len: true lengths of targets , [batch_size]

        Returns
        -------
        Rewards: final reward , [batch_size]
        """

        seq_reward = []
        kg_reward = []
        ground_kg = self.kg_encoder_avg(ground_item_seq, tar_len, drop=False)  # [batch_size,dims]

        for paths in range(episode_num):
            cur_item_seq, cur_seq_len = self.state_transfer(pre_item_seq, action_taken, seq_len)
            episode_items = action_taken.unsqueeze(-1)  # [batch_size, 1]
            episode_hit_reward = [self.hit_ratio_reward(initial_prob, ground_item_seq, i=0)]
            for i in range(episode_len - 1):
                out_logits = self.forward(cur_item_seq, cur_seq_len)
                out_logits = torch.scatter(out_logits, -1, cur_item_seq, -sys.maxsize)
                out_prob = torch.softmax(out_logits, dim=-1)
                cur_sample = self.sample_from_topk(out_prob, k=self.sample_size)
                episode_items = torch.cat([episode_items, cur_sample.unsqueeze(-1)], dim=-1)
                # hit reward
                episode_hit_reward.append(self.hit_ratio_reward(out_prob, ground_item_seq, i + 1))
                # transfer to next state
                cur_item_seq, cur_seq_len = self.state_transfer(cur_item_seq, cur_sample, cur_seq_len)
            # calculate kg reward
            episode_item_len = 3 * torch.ones(len(episode_items)).to(self.device)
            episode_kg = self.kg_encoder_avg(episode_items, episode_item_len, drop=False)  # [batch_size, dims]
            kg_reward.append(self.normalized_cos(ground_kg, episode_kg))  # [batch_size]
            # calculate seq reward
            episode_hit_reward = torch.stack(episode_hit_reward, dim=0)  # [episode_len, batch_size]
            # discounted_ep_reward = self.get_discount_factor().unsqueeze(-1) * episode_hit_reward
            seq_reward.append(torch.mean(episode_hit_reward, dim=0))

        seq_reward = torch.stack(seq_reward, dim=0)
        seq_reward = torch.mean(seq_reward, dim=0)  # [batch_size]
        kg_reward = torch.stack(kg_reward, dim=0)  # [episode_num, batch_size]
        kg_reward = torch.mean(kg_reward, dim=0)  # [batch_size]

        final_reward = (2 * seq_reward * kg_reward) / (seq_reward + kg_reward)  # harmonic mean

        return final_reward

    def DIIN(self, final_state, target=None):
        """
        generate probabilities of next item by DIIN.

        Parameters
        ---------
        final_state: [batch_size, embed_size * 2]
        target: [batch_size]
        """
        final_seq_state, final_kg_state = final_state[:, : self.embed_size], final_state[:, self.embed_size:]
        candidate_item_emb = self.item_embedding.weight  # [batch_size, embed_size]
        candidate_kg_emb = self.kg_embedding

        induction_weights = torch.softmax(self.W2(final_state), dim=-1)  # [batch_size, 2]
        seq_weight, kg_weight = induction_weights[:, 0].unsqueeze(-1), induction_weights[:, 1].unsqueeze(-1)

        # ground-truth weight
        sorted_pred_weight = None
        if target is not None:
            ground_seq_emb = self.item_embedding(target)  # [batch_size, embed_size]
            ground_kg_emb = self.kg_embedding[target]
            ground_kg_coef = torch.mul(final_seq_state, ground_seq_emb).sum(dim=-1)  # [batch_size]
            ground_seq_coef = torch.mul(final_kg_state, ground_kg_emb).sum(dim=-1)
            ground_coef = torch.stack([ground_seq_coef, ground_kg_coef], dim=-1)  # [batch_size, 2]
            sorted_idx = torch.argsort(ground_coef, dim=-1, descending=True)
            sorted_pred_weight = torch.gather(induction_weights, dim=-1, index=sorted_idx)

        seq_logits = torch.matmul(final_seq_state, candidate_item_emb.t())  # [batch_size, num_items]
        kg_logits = torch.matmul(final_kg_state, candidate_kg_emb.t())
        out = seq_weight * seq_logits + kg_weight * kg_logits

        return (out, sorted_pred_weight) if target is not None else out

    def sample_from_topk(self, sample_prob, k=None, masked_item=None, sharpen=True):
        """
        sample_prob: [batch_size, num_items]
        k: int, default:None
        returns:
        sample: [batch_size]
        masked_item: [batch_size, m]
        """
        masked_prob = sample_prob
        if masked_item is not None:
            masked_prob = torch.scatter(sample_prob, -1, masked_item, 0.)
        if k is not None:
            assert k <= masked_prob.size(-1), 'top k must less than prob size !'
            _, topk_idx = torch.topk(masked_prob, k, -1)
            mask = torch.ones_like(masked_prob).bool()
            mask.scatter_(-1, topk_idx, False)
            masked_prob = torch.masked_fill(masked_prob, mask, 0.)
        # sharpen
        if sharpen:
            masked_prob = masked_prob ** self.prob_sharpen
        sampler = torch.distributions.categorical.Categorical(masked_prob)
        item_sampled = sampler.sample()
        return item_sampled

    def normalized_cos(self, vec1, vec2):
        return (self.cos(vec1, vec2) + 1) / 2

    def hit_ratio_reward(self, prob, targets, i=0):
        """
        prob: [batch_size, num_items]
        target: [batch_size, T]

        returns:
        hit_reward: [batch_size]
        """
        rank = prob.argsort().argsort() + 1  # [batch_size, num_items]
        rank_reward = torch.zeros(targets.size(0), device=self.device)

        for i in range(targets.size(-1)):
            target = targets[:, i]
            target_rank = torch.gather(rank, -1, target.unsqueeze(-1)).squeeze()  # [batch_size]
            target_rank = torch.where(target_rank > self.hit_range, 0., self.hit_r)
            rank_reward += target_rank
        rank_reward[rank_reward > 1.] = 1.
        return rank_reward

    def get_loss(self, item_input, logits, target):
        return torch.zeros((1,))

    def get_final_loss(self, prediction, action, reward, induction_factor):
        # DIIN loss
        DIIN_Loss = self.get_DIIN_loss_func(self.DIIN_loss_type, self.indu_loss_func, induction_factor)
        # RL loss
        prob = torch.index_select(prediction, 1, action)  # [batch_size * 2,batch_size]
        prob = torch.diagonal(prob, 0)  # prob for each batch, [batch_size * 2]
        RLloss = -torch.mean(torch.mul(reward, torch.log(prob)))

        loss = RLloss + self.lamda * DIIN_Loss
        return loss

    def set_sas_prob(self, teach_prob):
        prob_dict = {
            1: [0.4, 0.3, 0.3],
            2: [0.5, 0.3, 0.2],
            3: [0.8, 0.1, 0.1],
            4: [1., 0., 0.]
        }
        return prob_dict[teach_prob]

    def get_DIIN_loss_func(self, indu_loss_type, indu_loss, inductoin_factor):
        indu_target = torch.ones(inductoin_factor.size(0)).to(self.device)  # [batch_size]
        if indu_loss_type == 'MR':  # margin rank loss
            max_factor = inductoin_factor[:, 0]  # [batch_size]
            min_factor = inductoin_factor[:, 1]  # [batch_size]
            loss = indu_loss(max_factor, min_factor, indu_target)  # scalar
        elif indu_loss_type == 'BCE':
            max_factor = inductoin_factor[:, 0]  # [batch_size]
            loss = indu_loss(max_factor, indu_target)
        return loss

    def get_sas_prob(self, target, tar_len):
        teach_prob = np.zeros((len(tar_len), self.num_items))  # [batch_size,num_items]
        for i, tar in enumerate(target.detach().cpu().numpy()):
            sample_prob = np.where(tar != 0, self.sas_prob, 0.)
            teach_prob[i][tar] = sample_prob
        teach_prob = torch.from_numpy(teach_prob).float().to(self.device)
        return teach_prob

    def position_embedding(self, seq_embedding):
        seq_embedding *= self.embed_size ** 0.5  # divide dim^2
        positions = np.tile(np.array(range(seq_embedding.shape[1])), [seq_embedding.shape[0], 1])
        seq_embedding += self.pos_emb(torch.LongTensor(positions).to(seq_embedding.device))
        return seq_embedding


class Transformer(torch.nn.Module):
    def __init__(self, embed_size, num_heads, num_blocks, attn_dropout, ffn_hidden, ffn_dropout, max_len, layer_norm_eps):
        super(Transformer, self).__init__()
        self.embed_size = embed_size
        self.ffn_hidden = ffn_hidden
        self.pos_emb = torch.nn.Embedding(max_len, embed_size)
        self.emb_dropout = nn.Dropout(ffn_dropout)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(embed_size, eps=layer_norm_eps)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(embed_size,
                                                         num_heads,
                                                         attn_dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(embed_size, eps=layer_norm_eps)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(embed_size, ffn_hidden, ffn_dropout)
            self.forward_layers.append(new_fwd_layer)

    def forward(self, seq_embedding):
        """
        :param log_seqs: [batch_size, max_seq_len]
        :param seq_embedding: torch.FloatTensor, [batch_size, max_seq_len, dim]
        :return:
        """
        seq_embedding *= self.embed_size ** 0.5
        positions = np.tile(np.array(range(seq_embedding.shape[1])), [seq_embedding.shape[0], 1])
        seq_embedding += self.pos_emb(torch.LongTensor(positions).to(seq_embedding.device))
        seqs = self.emb_dropout(seq_embedding)

        seq_len = seqs.size(1)
        attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=seqs.device))  # [max_len, max_len]

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)  # [max_len, batch_size, embed_size]
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        return seqs


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, embed_size, hidden_size, ffn_dropout):
        super(PointWiseFeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(embed_size, hidden_size)
        self.dropout1 = torch.nn.Dropout(p=ffn_dropout)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, embed_size)
        self.dropout2 = torch.nn.Dropout(p=ffn_dropout)

    def forward(self, inputs):
        outputs = self.dropout2(self.fc2(self.act(self.dropout1(self.fc1(inputs)))))
        outputs += inputs
        return outputs


def MELOD_config():
    parser = HyperParamDict('MELOD default hyper-parameters')
    parser.add_argument('--model', default='MELOD', type=str)
    parser.add_argument('--embed_size', default=128, type=int, help='embedding size')
    parser.add_argument('--DIIN_loss_type', default='MR', type=str, choices=['MR', 'BCE'],
                        help='loss type for DIIN, mr: margin rank loss, bec: binary cross entropy loss')
    parser.add_argument('--alpha', default=0.6, type=float, help='margin for DIIN loss')
    parser.add_argument('--lamda', default=0.5, type=float, help='weight for induction loss')
    parser.add_argument('--sas_prob', default=2, type=int, choices=[1, 2, 3, 4],
                        help='teach sample strategy, 1:[0.4, 0.3, 0.3], 2:[0.5, 0.3, 0.2], 3:[0.8, 0.1, 0.1], '
                             '4:[1., 0., 0.]')
    parser.add_argument('--freeze_kg', action='store_true', help='if freeze kg embedding')
    parser.add_argument('--episode_num', default=1, type=int, help='episode number')
    parser.add_argument('--episode_len', default=3, type=int, help='episode length')
    parser.add_argument('--hit_range', default=100, type=int, help='valid hit range for seq reward')
    parser.add_argument('--sample_size', default=100, help='sample from top k item, [100, 200, -1:all]')
    parser.add_argument('--prob_sharpen', default=1., help='sharpen sample probability')
    parser.add_argument('--hit_r', default=1., type=float, help='reward for each hit')

    # transformer
    parser.add_argument('--num_blocks', default=2, type=int, help='number of transformer block')
    parser.add_argument('--num_heads', default=2, type=int, help='number of transformer head')
    parser.add_argument('--ffn_hidden', default=256, type=int, help='transformer ffn hidden size')
    parser.add_argument('--attn_dropout', default=0.5, type=float, help='attention score dropout')
    parser.add_argument('--ffn_dropout', default=0.5, type=float, help='embedding dropout probability')
    # other
    parser.add_argument('--model_type', default='Knowledge', choices=['Knowledge', 'Sequential'])
    parser.add_argument('--loss_type', default='CUSTOM', type=str, choices=['CE', 'BPR', 'BCE','CUSTOM'])
    return parser


if __name__ == '__main__':
    config = MELOD_config()
    config.max_len = 50
    config.device = torch.device('cuda:0')
    num_items = 100
    kg_map = torch.randn(100, 128)
    model = MELOD(num_items, config, kg_map)
    print(model)