# PoetryModel
# 2026 Spring Deep Learning Lab3
# Embedding + 多层LSTM + LayerNorm + weight tying 输出

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoetryModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=300,
                 num_layers=2, dropout=0.5, padding_idx=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocab_size,                # 词表大小
            embedding_dim,             # 字向量维度
            padding_idx=padding_idx    # padding token 不参与 embedding 更新
        )
        self.emb_dropout = nn.Dropout(dropout)   # 对输入字向量做 dropout
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))  # 输出层偏置

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        batch_size, seq_len = x.size()
        embeds = self.embedding(x)       # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        embeds = self.emb_dropout(embeds)
        if hidden is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)   # 初始 hidden state
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)   # 初始 cell state
            hidden = (h_0, c_0)

        output, hidden = self.lstm(embeds, hidden)
        output = self.norm(output)
        output = F.linear(output, self.embedding.weight, self.output_bias)  # 共享 embedding 权重作为输出权重
        output = output.reshape(batch_size * seq_len, -1)
        return output, hidden
