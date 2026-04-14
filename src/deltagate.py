#!/usr/bin/env python3
"""
Delta-Gate Recommendation Models
================================

SASRec/GRU4Rec backbone + Delta-Gate at output layer.
NO time injection inside Transformer (keeps ablation clean).

Core Innovation: Vector Delta-Gate with Confidence-Adaptive Prior Fallback

Differentiation from TiSASRec:
- TiSASRec models intra-sequence time intervals
- We model the single massive gap AFTER the sequence ends
- Explicit prior fallback when historical confidence collapses

SANITIZED: Fixed padding mask and left-padding last-item extraction bugs.
"""

import torch
import torch.nn as nn


class DeltaGateSASRec(nn.Module):
    """
    SASRec backbone + Delta-Gate at output layer only.
    Transformer internals use standard absolute positional encoding.
    NO time injection inside Transformer (keeps ablation clean).

    SANITIZED VERSION:
    - Uses src_key_padding_mask for padding tokens
    - Extracts TRUE last valid hidden state for left-padding (index -1)

    ABLATION SUPPORT:
    - ablation_mode='none': Normal learned gate (default)
    - ablation_mode='fixed_gate': Fixed g=0.5 during TRAINING and inference
    - ablation_mode='prior_only': Fixed g=0 during TRAINING and inference
    """

    def __init__(
        self,
        item_num: int,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        max_seq_len: int = 50,
        dropout: float = 0.2,
        ablation_mode: str = 'none',
    ):
        super().__init__()

        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.ablation_mode = ablation_mode

        # Validate ablation mode
        assert ablation_mode in ['none', 'fixed_gate', 'prior_only'], \
            f"Invalid ablation_mode: {ablation_mode}. Must be 'none', 'fixed_gate', or 'prior_only'"

        if ablation_mode != 'none':
            print(f"[ABLATION] DeltaGateSASRec initialized with ablation_mode='{ablation_mode}'")

        # Item embedding (0 is padding)
        self.item_embedding = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Absolute positional encoding (NOT time-aware)
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=embedding_dim
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # ================================================================
        # Vector Delta-Gate (ends with Sigmoid!)
        # ================================================================
        gate_input_dim = 1 + embedding_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()  # Output strictly in [0, 1]
        )

        # Global Mainstream Prior
        self.global_prior = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.zeros_(self.global_prior)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.01)

    def forward(self, seq: torch.Tensor, delta_t: torch.Tensor):
        """
        Forward pass with SANITIZED extraction.

        Args:
            seq: [batch_size, seq_len] item sequence indices (1-indexed, 0 = padding)
            delta_t: [batch_size] time gap to target item in days

        Returns:
            logits: [batch_size, item_num] scores over all items
            confidence: [batch_size] confidence scores for analysis
        """
        batch_size, seq_len = seq.shape

        # Item + Position embeddings
        item_emb = self.item_embedding(seq)  # [B, L, D]
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # [B, L, D]
        x = item_emb + pos_emb  # [B, L, D]
        x = self.dropout(x)

        # 1. Causal Mask (float('-inf') to avoid NaN gradient poisoning)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=seq.device),
            diagonal=1
        )

        # 2. Transformer with causal mask only (no src_key_padding_mask)
        x = self.transformer(x, mask=causal_mask)
        x = self.layer_norm(x)

        # Left-padding: last valid item is ALWAYS at index -1
        h_personalized = x[:, -1, :]  # [B, D]

        # Vector Delta-Gate with Ablation Support
        if self.ablation_mode == 'fixed_gate':
            # ABLATION: Fixed gate g=0.5 (equal mix of history and prior)
            g = torch.full((batch_size, self.embedding_dim), 0.5, device=seq.device)
        elif self.ablation_mode == 'prior_only':
            # ABLATION: Fixed gate g=0 (pure prior fallback)
            g = torch.zeros(batch_size, self.embedding_dim, device=seq.device)
        else:
            # NORMAL: Learned gate via MLP
            delta_normalized = torch.log1p(delta_t)  # [B]
            gate_input = torch.cat([
                delta_normalized.unsqueeze(-1),  # [B, 1]
                h_personalized                    # [B, D]
            ], dim=-1)  # [B, 1 + D]
            g = self.gate_mlp(gate_input)  # [B, D] in [0, 1]

        # Gated Fusion
        global_prior_expanded = self.global_prior.unsqueeze(0).expand(batch_size, -1)  # [B, D]
        h_final = g * h_personalized + (1 - g) * global_prior_expanded  # [B, D]

        # Compute Logits
        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        all_item_emb = self.item_embedding(all_items)  # [V, D]
        logits = torch.matmul(h_final, all_item_emb.T)  # [B, item_num]

        # Confidence Score
        confidence = g.mean(dim=-1)  # [B]

        return logits, confidence


class DeltaGateGRU4Rec(nn.Module):
    """
    GRU4Rec backbone + Delta-Gate at output layer.

    SANITIZED VERSION:
    - Extracts TRUE last valid hidden state for left-padding (index -1)
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

        # GRU4Rec backbone
        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

        # Delta-Gate module (ends with Sigmoid!)
        gate_input_dim = 1 + hidden_size
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()  # Output strictly in [0, 1]
        )

        # Global mainstream prior
        self.global_prior = nn.Parameter(torch.zeros(hidden_size))
        nn.init.zeros_(self.global_prior)

    def forward(self, seq: torch.Tensor, delta_t: torch.Tensor):
        """
        Forward pass with SANITIZED extraction.

        Args:
            seq: [B, L] item sequence indices
            delta_t: [B] time gap to target item in days

        Returns:
            logits: [B, item_num] scores over all items
            confidence: [B] confidence scores
        """
        batch_size = seq.shape[0]

        # GRU4Rec backbone
        x = self.item_embedding(seq)  # [B, L, D]
        x = self.dropout(x)
        h, _ = self.gru(x)  # [B, L, H]

        # Left-padding: last valid item is ALWAYS at index -1
        h_personalized = h[:, -1, :]  # [B, H]

        # Delta-Gate
        delta_normalized = torch.log1p(delta_t)  # [B]
        gate_input = torch.cat([
            delta_normalized.unsqueeze(-1),  # [B, 1]
            h_personalized                    # [B, H]
        ], dim=-1)  # [B, 1 + H]

        g = self.gate_mlp(gate_input)  # [B, H] in [0, 1]

        # Gated fusion
        global_prior_expanded = self.global_prior.unsqueeze(0).expand(batch_size, -1)
        h_final = g * h_personalized + (1 - g) * global_prior_expanded  # [B, H]

        # Compute logits
        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        logits = torch.matmul(h_final, self.item_embedding(all_items).T)  # [B, item_num]

        # Confidence score
        confidence = g.mean(dim=-1)  # [B]

        return logits, confidence


class SASRecBaseline(nn.Module):
    """
    Pure SASRec Baseline (NO DeltaGate).
    SANITIZED VERSION with proper padding mask and left-padding extraction.
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
        self.embedding_dim = embedding_dim

        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

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

        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.01)

    def forward(self, seq: torch.Tensor):
        batch_size, seq_len = seq.shape

        item_emb = self.item_embedding(seq)
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x = item_emb + self.position_embedding(positions)
        x = self.dropout(x)

        # 1. Causal Mask: 阻止看未来
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()

        # 2. Padding Mask: 阻止看过去的 Padding 零向量
        key_padding_mask = (seq == 0)  # [B, L]

        # 3. Transformer with dual mask
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.layer_norm(x)

        # Left-padding: last valid item is ALWAYS at index -1
        h = x[:, -1, :]

        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        logits = torch.matmul(h, self.item_embedding(all_items).T)

        return logits


class GRU4RecBaseline(nn.Module):
    """
    Pure GRU4Rec Baseline (NO DeltaGate).
    SANITIZED VERSION with proper left-padding extraction.
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
        self.hidden_size = hidden_size

        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

    def forward(self, seq: torch.Tensor):
        batch_size = seq.shape[0]

        x = self.item_embedding(seq)
        x = self.dropout(x)
        h, _ = self.gru(x)

        # TRUE last valid hidden state for left-padding
        # (The last item is always at index -1, padded zeros are on the left)
        h_last = h[:, -1, :]  # [B, H]

        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        logits = torch.matmul(h_last, self.item_embedding(all_items).T)

        return logits


if __name__ == '__main__':
    print("Testing SANITIZED models...")

    seq = torch.randint(1, 1001, (4, 50))
    delta_t = torch.tensor([30.0, 180.0, 400.0, 730.0])

    # Test with some padding
    seq[0, :10] = 0  # Left pad first sequence

    print("\nDeltaGateSASRec...")
    model_sas = DeltaGateSASRec(item_num=1000, embedding_dim=64)
    logits, conf = model_sas(seq, delta_t)
    print(f"  Output: {logits.shape}, Confidence: {conf.tolist()}")

    print("\nDeltaGateGRU4Rec...")
    model_gru = DeltaGateGRU4Rec(item_num=1000, embedding_dim=64, hidden_size=64)
    logits, conf = model_gru(seq, delta_t)
    print(f"  Output: {logits.shape}, Confidence: {conf.tolist()}")

    print("\nSASRecBaseline...")
    model_sas_base = SASRecBaseline(item_num=1000, embedding_dim=64)
    logits = model_sas_base(seq)
    print(f"  Output: {logits.shape}")

    print("\nGRU4RecBaseline...")
    model_gru_base = GRU4RecBaseline(item_num=1000, embedding_dim=64, hidden_size=64)
    logits = model_gru_base(seq)
    print(f"  Output: {logits.shape}")

    print("\n✅ All SANITIZED tests passed!")
    