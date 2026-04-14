#!/usr/bin/env python3
"""
GRU4Rec Implementation
======================

Session-based Recommendations with RNNs
"""

import torch
import torch.nn as nn


class GRU4Rec(nn.Module):
    """
    GRU4Rec: Session-based Recommendations with RNNs.

    Architecture:
    - Item embedding with padding
    - GRU encoder
    - Output projection
    """

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

        # Item embedding (0 is padding)
        self.item_embedding = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # GRU encoder
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
        """Initialize embeddings."""
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)

    def forward(self, seq: torch.Tensor):
        """
        Forward pass.

        Args:
            seq: [batch_size, seq_len] item sequence indices

        Returns:
            logits: [batch_size, item_num] scores over all items
        """
        # Embedding
        x = self.item_embedding(seq)  # [B, L, D]
        x = self.dropout(x)

        # GRU
        h, _ = self.gru(x)  # [B, L, H]
        h = h[:, -1, :]  # [B, H]

        # Compute scores
        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        logits = torch.matmul(h, self.item_embedding(all_items).T)  # [B, item_num]

        return logits


if __name__ == '__main__':
    # Quick test
    model = GRU4Rec(item_num=1000, embedding_dim=64, hidden_size=64)
    seq = torch.randint(1, 1001, (4, 50))
    logits = model(seq)
    print(f"Input: {seq.shape}, Output: {logits.shape}")
    print("GRU4Rec model test passed!")
