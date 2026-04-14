#!/usr/bin/env python3
"""
BERT4Rec Implementation
=======================

BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations
"""

import torch
import torch.nn as nn


class BERT4Rec(nn.Module):
    """
    BERT4Rec: Sequential Recommendation with Bidirectional Encoder.

    Architecture:
    - Item embedding with padding and mask token
    - Bidirectional Transformer encoder
    - Masked item prediction
    """

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
        self.pad_token = 0
        self.mask_token = item_num + 1

        # Item embedding (0=padding, item_num+1=mask)
        self.item_embedding = nn.Embedding(
            item_num + 2,
            embedding_dim,
            padding_idx=0
        )

        # Position embedding
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Transformer encoder (bidirectional - no causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Output bias
        self.output_bias = nn.Parameter(torch.zeros(item_num + 1))

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings."""
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.01)

    def forward(self, seq: torch.Tensor, return_all: bool = False):
        """
        Forward pass.

        Args:
            seq: [batch_size, seq_len] item sequence with possible masks
            return_all: if True, return logits for all positions

        Returns:
            logits: [batch_size, item_num] or [batch_size, seq_len, item_num]
        """
        B, L = seq.shape

        # Embedding
        x = self.item_embedding(seq)  # [B, L, D]
        x = x + self.position_embedding(torch.arange(L, device=seq.device).unsqueeze(0))
        x = self.dropout(x)

        # Padding mask
        mask = (seq != self.pad_token).float().unsqueeze(-1)

        # Transformer (bidirectional)
        x = self.transformer(x)
        x = x * mask
        x = self.layer_norm(x)

        # Output projection
        weights = self.item_embedding.weight[1:self.item_num + 1]

        if return_all:
            return torch.matmul(x, weights.T) + self.output_bias[1:]

        return torch.matmul(x[:, -1, :], weights.T) + self.output_bias[1:]


if __name__ == '__main__':
    # Quick test
    model = BERT4Rec(item_num=1000, embedding_dim=64)
    seq = torch.randint(1, 1001, (4, 50))
    logits = model(seq)
    print(f"Input: {seq.shape}, Output: {logits.shape}")
    print("BERT4Rec model test passed!")
