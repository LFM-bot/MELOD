import torch.nn as nn
import torch
import numpy as np
import copy
import math
import torch.nn.functional as F


class Transformer(torch.nn.Module):
    def __init__(self, embed_size, num_heads, num_blocks, attn_dropout, ffn_hidden, ffn_dropout, max_len,
                 layer_norm_eps):
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
        attention_mask = ~torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=seqs.device))  # [max_len, max_len]

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
