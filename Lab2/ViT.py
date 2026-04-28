import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
        # TODO


class Embedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        # 对 CIFAR-10，image_size 通常就是 32
        self.num_patches = (image_size // patch_size) ** 2
        self.embedding = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = embed_dim,
                kernel_size = patch_size,
                stride = patch_size,
                bias = False
            ),  # 用卷积切 patch 并映射到 embed_dim
            nn.Flatten(2)  # 展成 [B, embed_dim, num_patches]
        )
        nn.init.xavier_uniform_(self.embedding[0].weight)   # 使用xavier初始化

    def forward(self, x):
        output = self.embedding(x)
        output = output.transpose(1, 2) # 转成 [B, num_patches, embed_dim]
        return output

class Attention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8,  dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x):
        x = self.norm(x)
        output = self.attention(x, x, x)[0]
        return output
    
class Mlp(nn.Module):
    def __init__(self, embed_dim=256, mlp_hidden_dim=512, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = self.norm(x)
        output = self.mlp(x)
        return output

class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        mlp_hidden_dim=512,
        depth=1,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        drop_path_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(embed_dim, num_heads, dropout),
                Mlp(embed_dim, mlp_hidden_dim, dropout),
                DropPath(drop_path_rates[layer_idx]),
                DropPath(drop_path_rates[layer_idx]),
            ])
            for layer_idx in range(depth)
        ])

    def forward(self, x):
        for attn, mlp, attn_drop_path, mlp_drop_path in self.layers:
            x = x + attn_drop_path(attn(x))
            x = x + mlp_drop_path(mlp(x))
        return x

class ViT(nn.Module):
    """
    patch embedding + transformer块 + 分类头
    """
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_heads=8,
        mlp_hidden_dim=512,
        depth = 1,
        dropout=0.0,
        emb_dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.patch_embed = Embedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        # position embedding
        self.norm  = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(emb_dropout)
        # transformer块
        self.transformer = Transformer(
            embed_dim,
            num_heads,
            mlp_hidden_dim,
            depth,
            dropout,
            drop_path_rate,
        )
        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        embedx = self.patch_embed(x)    # 嵌入成[B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(embedx.size(0), -1, -1)
        embedx = torch.cat((cls_tokens, embedx), dim=1)
        embedx = embedx + self.pos_embedding[:, :embedx.size(1), :]
        embedx = self.pos_drop(embedx)
        embedx = self.transformer(embedx)
        embedx = self.norm(embedx)
        pooled = embedx[:, 1:, :].mean(dim=1)  # 用 patch token 的均值做分类
        output = self.head(pooled)
        return output
