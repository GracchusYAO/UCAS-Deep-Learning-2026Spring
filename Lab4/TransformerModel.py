import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):	# 正弦位置编码
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(1, max_len, embed_dim)	# (1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):	# x: (batch, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):	# token embedding 与尺度缩放
    def __init__(self, vocab_size, embed_dim, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.embed_dim)


class Seq2SeqTransformer(nn.Module):	# Encoder-Decoder Transformer
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        embed_dim=256,
        num_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        max_len=256,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.src_embedding = TokenEmbedding(src_vocab_size, embed_dim, src_pad_idx)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, embed_dim, tgt_pad_idx)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_len)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer.encoder.enable_nested_tensor = False	# 关闭实验性 nested tensor warning
        self.transformer.encoder.use_nested_tensor = False
        self.generator = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src, tgt_input):	# 训练时的前向传播
        src_key_padding_mask, tgt_key_padding_mask, tgt_mask = create_masks(
            src, tgt_input, self.src_pad_idx, self.tgt_pad_idx
        )

        src_emb = self.positional_encoding(self.src_embedding(src))	# 源语言 embedding
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt_input))

        memory = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask
        )
        output = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.generator(output)

    def encode(self, src):	# 推理时编码源句
        src_key_padding_mask = src == self.src_pad_idx
        src_emb = self.positional_encoding(self.src_embedding(src))
        return self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask
        )

    def decode(self, tgt, memory, src_key_padding_mask):	# 推理时逐步解码
        tgt_key_padding_mask = tgt == self.tgt_pad_idx
        tgt_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device),
            diagonal=1
        )
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))
        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )


def create_masks(src, tgt_input, src_pad_idx, tgt_pad_idx):	# 构造 padding mask 和因果 mask
    src_key_padding_mask = src == src_pad_idx
    tgt_key_padding_mask = tgt_input == tgt_pad_idx
    tgt_mask = torch.triu(	# 禁止 decoder 看到未来 token
        torch.ones(tgt_input.size(1), tgt_input.size(1), dtype=torch.bool, device=tgt_input.device),
        diagonal=1
    )
    return src_key_padding_mask, tgt_key_padding_mask, tgt_mask


def count_parameters(model):	# 统计可训练参数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
