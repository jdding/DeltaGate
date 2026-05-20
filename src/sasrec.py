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
            norm_first=True  # Improves Transformer training stability.
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

        # Use a float causal mask with a valid diagonal to avoid all-masked rows.
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=seq.device),
            diagonal=1
        )

        # With left padding, valid items are on the right and the final state is at -1.
        # We rely on the causal mask here to avoid NaNs from fully masked rows.
        x = self.transformer(x, mask=causal_mask)
        x = self.layer_norm(x)

        # Extract the final position for next-item prediction.
        h = x[:, -1, :]  # [B, D]

        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        logits = torch.matmul(h, self.item_embedding(all_items).T)

        return logits
