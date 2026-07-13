#!/usr/bin/env python3
"""GRU4Rec backbone used in the experiments."""

import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    """GRU-based sequential recommender."""

    def __init__(
        self,
        item_num: int,
        embedding_dim: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.item_embedding = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

    def forward(self, seq: torch.Tensor):
        x = self.item_embedding(seq)
        x = self.dropout(x)

        h, _ = self.gru(x)
        h = h[:, -1, :]

        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        logits = torch.matmul(h, self.item_embedding(all_items).T)

        return logits


if __name__ == '__main__':
    model = GRU4Rec(item_num=1000, embedding_dim=64, hidden_size=64)
    seq = torch.randint(1, 1001, (4, 50))
    logits = model(seq)
    print(f"Input: {seq.shape}, Output: {logits.shape}")
    print("GRU4Rec model test passed!")
