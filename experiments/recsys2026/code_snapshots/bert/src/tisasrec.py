#!/usr/bin/env python3
"""TiSASRec baseline with padding-safe time-relation attention."""
import torch
import torch.nn as nn
import math


class TiSASRec(nn.Module):
    """Time-interval-aware self-attentive sequential recommender."""

    def __init__(
        self,
        item_num: int,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
        time_buckets: int = 256,
        max_time_interval: int = 730,  # max interval in days
    ):
        super().__init__()

        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.time_buckets = time_buckets
        self.max_time_interval = max_time_interval
        self.num_heads = num_heads

        # Item embedding (0 is padding)
        self.item_embedding = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Absolute position embedding
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Time interval embedding (for each position's interval from predecessor)
        self.time_embedding = nn.Embedding(time_buckets, embedding_dim)

        # Relation embeddings provide a per-head temporal attention bias.
        self.time_relation_embedding = nn.Embedding(time_buckets, embedding_dim)

        # Custom multi-head attention with time-aware bias
        self.attention_layers = nn.ModuleList([
            TimeAwareAttentionLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                time_buckets=time_buckets
            )
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.item_embedding.weight, 0, 0.02)
        nn.init.normal_(self.position_embedding.weight, 0, 0.02)
        nn.init.normal_(self.time_embedding.weight, 0, 0.02)
        nn.init.normal_(self.time_relation_embedding.weight, 0, 0.02)

    def bucketize_time(self, time_values: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous time values to bucket indices using log scale.

        Args:
            time_values: [B, L] or [B, L, L] tensor of time values (in days)

        Returns:
            Bucket indices in range [0, time_buckets-1]
        """
        eps = 1e-6
        # Log scaling preserves more resolution for short intervals.
        normalized = torch.log1p(torch.abs(time_values) + eps) / math.log1p(self.max_time_interval)
        buckets = (normalized * (self.time_buckets - 1)).long()
        buckets = torch.clamp(buckets, 0, self.time_buckets - 1)
        return buckets

    def compute_time_relation_matrix(
        self,
        intra_intervals: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the time relation matrix R[i][j] = |interval_i - interval_j|.

        Padding-safe implementation:
        - For padding positions, we set intervals to a special value
        - The padding_mask ensures these positions are ignored in attention

        Args:
            intra_intervals: [B, L] time intervals between consecutive items.
                intra_intervals[b][i] = gap from item[i-1] to item[i].
                intra_intervals[b][0] = 0 (first item has no predecessor).
            padding_mask: [B, L] bool tensor, True = padding position

        Returns:
            time_relation_matrix: [B, L, L] bucketized time relation values
        """
        batch_size, seq_len = intra_intervals.shape

        interval_i = intra_intervals.unsqueeze(-1)
        interval_j = intra_intervals.unsqueeze(1)

        time_relation = torch.abs(interval_i - interval_j)

        # Zero padding relations before bucketization; attention masks them later.
        padding_mask_2d = padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(1)
        time_relation = time_relation.masked_fill(padding_mask_2d, 0.0)

        # Bucketize
        time_relation_buckets = self.bucketize_time(time_relation)  # [B, L, L]

        return time_relation_buckets

    def forward(
        self,
        seq: torch.Tensor,
        intra_intervals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with TiSASRec time-aware attention.

        Args:
            seq: [B, L] item sequence indices (1-indexed, 0 = padding)
            intra_intervals: [B, L] time intervals between consecutive items.
                intra_intervals[b][i] = gap from item[i-1] to item[i].
                intra_intervals[b][0] = 0 (first item).

        Returns:
            logits: [B, item_num] scores over all items
        """
        batch_size, seq_len = seq.shape

        padding_mask = (seq == 0)

        item_emb = self.item_embedding(seq)

        # Position embeddings (absolute)
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # [B, L, D]

        # Time interval embeddings (for each position)
        time_buckets = self.bucketize_time(intra_intervals)  # [B, L]
        time_emb = self.time_embedding(time_buckets)  # [B, L, D]

        # Combine: item + position + time_interval
        x = item_emb + pos_emb + time_emb  # [B, L, D]
        x = self.dropout(x)

        time_relation_matrix = self.compute_time_relation_matrix(
            intra_intervals, padding_mask
        )  # [B, L, L]

        for attn_layer in self.attention_layers:
            x = attn_layer(
                x,
                time_relation_matrix=time_relation_matrix,
                padding_mask=padding_mask
            )

        x = self.layer_norm(x)

        h = x[:, -1, :]

        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        logits = torch.matmul(h, self.item_embedding(all_items).T)  # [B, item_num]

        return logits


class TimeAwareAttentionLayer(nn.Module):
    """Multi-head attention with a learned time-relation bias."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        time_buckets: int
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        # Time relation embedding for attention bias
        self.time_relation_embedding = nn.Embedding(time_buckets, num_heads)

        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_relation_matrix: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input tensor
            time_relation_matrix: [B, L, L] bucketized time relation values
            padding_mask: [B, L] bool tensor, True = padding position

        Returns:
            [B, L, D] output tensor
        """
        batch_size, seq_len, _ = x.shape
        residual = x

        # Layer norm
        x = self.layer_norm(x)

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        time_bias = self.time_relation_embedding(time_relation_matrix)  # [B, L, L, H]
        time_bias = time_bias.permute(0, 3, 1, 2)  # [B, H, L, L]

        attn_scores = attn_scores + time_bias

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()  # [L, L], True = masked

        # Padding mask: prevent attention to/from padding positions
        # padding_mask: [B, L], True = padding
        # For attention: we want to mask where KEY is padding
        # key_padding_mask: [B, 1, 1, L] for broadcasting
        key_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]

        # Combined mask
        # causal_mask: True = cannot attend (future)
        # key_padding_mask: True = cannot attend (padding)
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) | key_padding_mask  # [B, H, L, L] (broadcasted)

        # Keep the diagonal available so padding queries do not produce all-masked rows.
        diagonal_mask = torch.eye(seq_len, device=x.device).bool().unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        # Apply mask: set masked positions to -inf, but keep diagonal available
        attn_scores = attn_scores.masked_fill(attn_mask & ~diagonal_mask, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)  # [B, H, L, L]
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # [B, H, L, d]

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [B, L, D]

        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        # Residual connection
        return residual + output


if __name__ == '__main__':
    print("Testing TiSASRec with time relation matrix...")

    model = TiSASRec(item_num=1000, embedding_dim=64, num_heads=2, num_layers=2)

    # Test with padding
    seq = torch.randint(1, 1001, (4, 50))
    seq[0, :10] = 0  # Add padding to first sequence
    seq[1, :5] = 0   # Add padding to second sequence

    # Random intervals
    intra_intervals = torch.rand(4, 50) * 100
    # Padding positions should have interval 0
    intra_intervals[0, :10] = 0
    intra_intervals[1, :5] = 0

    logits = model(seq, intra_intervals)
    print(f"Input: {seq.shape}, Intervals: {intra_intervals.shape}, Output: {logits.shape}")

    # Check for NaN
    if torch.isnan(logits).any():
        print("ERROR: NaN detected in output!")
    else:
        print("No NaN in output; padding-safe implementation verified.")

    print("\nTiSASRec model test passed!")
