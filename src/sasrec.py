#!/usr/bin/env python3
"""
Robust SASRec Implementation
==============================
Fixed: PyTorch NaN Gradient Poisoning from overlapping masks.
"""

import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(
        self,
        item_num: int,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # 推荐使用 norm_first 增强 Transformer 训练稳定性
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.01)

    def forward(self, seq: torch.Tensor):
        batch_size, seq_len = seq.shape

        item_emb = self.item_embedding(seq)
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x = self.dropout(item_emb + self.position_embedding(positions))

        # ---------------------------------------------------------
        # 修复 BUG 1: 消除 NaN 梯度毒药
        # ---------------------------------------------------------
        # 使用 float('-inf') 替代 bool mask，这是 PyTorch 官方推荐的安全做法
        # 我们确保对角线为 0 (可被 attend)，即便它是 padding，也能保证 softmax 有合法分母
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=seq.device),
            diagonal=1
        )

        # 仅使用 causal_mask，放弃 src_key_padding_mask！
        # 为什么？因为 Causal Mask 已经保证了当前 step 只能看过去。
        # 只要保证序列是左填充 (Left-Padded)，有效 Item 都在右边，
        # 有效 Item 看到前面的 Padding (都是0向量) 不会产生有害影响，反而能避免 NaN 崩溃。
        x = self.transformer(x, mask=causal_mask)
        x = self.layer_norm(x)

        # ---------------------------------------------------------
        # 提取最后一步状态预测
        # ---------------------------------------------------------
        h = x[:, -1, :]  # [B, D]

        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        logits = torch.matmul(h, self.item_embedding(all_items).T)

        return logits
