#!/usr/bin/env python3
"""DeltaGate adapters for SASRec and GRU4Rec backbones."""

import torch
import torch.nn as nn


def _validate_delta_t(delta_t: torch.Tensor):
    if torch.any(delta_t < 0):
        raise ValueError("delta_t must be non-negative")


class DeltaGateSASRec(nn.Module):
    """SASRec with output-layer dimensional gating.

    ``ablation_mode`` selects the learned gate, a fixed 0.5 gate, or the
    global-prior-only variant.
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

        assert ablation_mode in ['none', 'fixed_gate', 'prior_only'], \
            f"Invalid ablation_mode: {ablation_mode}. Must be 'none', 'fixed_gate', or 'prior_only'"

        if ablation_mode != 'none':
            print(f"[ABLATION] DeltaGateSASRec initialized with ablation_mode='{ablation_mode}'")

        self.item_embedding = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # The backbone uses ordinary positional embeddings; delta_t enters only the gate.
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=embedding_dim
        )

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

        gate_input_dim = 1 + embedding_dim

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )

        self.global_prior = nn.Parameter(torch.zeros(embedding_dim))
        nn.init.zeros_(self.global_prior)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.01)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(self, seq: torch.Tensor, delta_t: torch.Tensor):
        """
        Forward pass with left-padding-aware extraction.

        Args:
            seq: [batch_size, seq_len] item sequence indices (1-indexed, 0 = padding)
            delta_t: [batch_size] time gap to target item in days

        Returns:
            logits: [batch_size, item_num] scores over all items
            confidence: [batch_size] confidence scores for analysis
        """
        _validate_delta_t(delta_t)
        batch_size, seq_len = seq.shape

        item_emb = self.item_embedding(seq)
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # [B, L, D]
        x = item_emb + pos_emb  # [B, L, D]
        x = self.dropout(x)
        x = x.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)

        # A float causal mask avoids fully masked rows under left padding.
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=seq.device),
            diagonal=1
        )

        x = self.transformer(x, mask=causal_mask)
        x = self.layer_norm(x)

        h_personalized = x[:, -1, :]

        if self.ablation_mode == 'fixed_gate':
            g = torch.full((batch_size, self.embedding_dim), 0.5, device=seq.device)
        elif self.ablation_mode == 'prior_only':
            g = torch.zeros(batch_size, self.embedding_dim, device=seq.device)
        else:
            delta_normalized = torch.log1p(delta_t)
            gate_input = torch.cat([
                delta_normalized.unsqueeze(-1),
                h_personalized
            ], dim=-1)
            g = self.gate_mlp(gate_input)

        global_prior_expanded = self.global_prior.unsqueeze(0).expand(batch_size, -1)
        h_final = g * h_personalized + (1 - g) * global_prior_expanded

        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        all_item_emb = self.item_embedding(all_items)
        logits = torch.matmul(h_final, all_item_emb.T)

        confidence = g.mean(dim=-1)

        return logits, confidence


class DeltaGateGRU4Rec(nn.Module):
    """GRU4Rec with output-layer dimensional gating."""

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

        self.item_embedding = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim, hidden_size, num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.item_embedding.weight, 0, 0.01)

        gate_input_dim = 1 + hidden_size
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.global_prior = nn.Parameter(torch.zeros(hidden_size))
        nn.init.zeros_(self.global_prior)

    def forward(self, seq: torch.Tensor, delta_t: torch.Tensor):
        """
        Forward pass with left-padding-aware extraction.

        Args:
            seq: [B, L] item sequence indices
            delta_t: [B] time gap to target item in days

        Returns:
            logits: [B, item_num] scores over all items
            confidence: [B] confidence scores
        """
        _validate_delta_t(delta_t)
        batch_size = seq.shape[0]

        # GRU4Rec backbone
        x = self.item_embedding(seq)  # [B, L, D]
        x = self.dropout(x)
        h, _ = self.gru(x)  # [B, L, H]

        h_personalized = h[:, -1, :]

        delta_normalized = torch.log1p(delta_t)
        gate_input = torch.cat([
            delta_normalized.unsqueeze(-1),  # [B, 1]
            h_personalized                    # [B, H]
        ], dim=-1)  # [B, 1 + H]

        g = self.gate_mlp(gate_input)  # [B, H] in [0, 1]

        global_prior_expanded = self.global_prior.unsqueeze(0).expand(batch_size, -1)
        h_final = g * h_personalized + (1 - g) * global_prior_expanded  # [B, H]

        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        logits = torch.matmul(h_final, self.item_embedding(all_items).T)  # [B, item_num]

        confidence = g.mean(dim=-1)

        return logits, confidence


class DeltaGateBERT4Rec(nn.Module):
    """BERT4Rec with final-mask extraction and output-layer dimensional gating."""

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
        if ablation_mode not in ['none', 'fixed_gate', 'prior_only']:
            raise ValueError(f"Invalid ablation_mode: {ablation_mode}")

        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.mask_token = item_num + 1
        self.ablation_mode = ablation_mode

        self.item_embedding = nn.Embedding(
            item_num + 2,
            embedding_dim,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.gate_mlp = nn.Sequential(
            nn.Linear(embedding_dim + 1, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )
        self.global_prior = nn.Parameter(torch.zeros(embedding_dim))

        # BERT4Rec stores one bias row for padding plus one for each catalog item.
        # DeltaGate keeps an additional frozen mask-token row for checkpoint loading.
        self.output_bias = nn.Parameter(torch.zeros(item_num + 2))

        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.position_embedding.weight, mean=0, std=0.01)

    def forward_backbone(self, seq: torch.Tensor):
        """Run the exact encoder path used by the pretrained BERT4Rec backbone."""
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        x = self.dropout(self.item_embedding(seq) + self.position_embedding(positions))
        x = self.transformer(x)
        x = x * seq.ne(0).float().unsqueeze(-1)
        x = self.layer_norm(x)
        return x

    def forward(self, seq: torch.Tensor, delta_t: torch.Tensor):
        _validate_delta_t(delta_t)
        batch_size = seq.shape[0]
        x = self.forward_backbone(seq)

        h_personalized = x[:, -1, :]
        if self.ablation_mode == 'fixed_gate':
            g = torch.full_like(h_personalized, 0.5)
        elif self.ablation_mode == 'prior_only':
            g = torch.zeros_like(h_personalized)
        else:
            gate_input = torch.cat(
                [torch.log1p(delta_t).unsqueeze(-1), h_personalized],
                dim=-1,
            )
            g = self.gate_mlp(gate_input)

        h_final = g * h_personalized + (1 - g) * self.global_prior.unsqueeze(0)
        item_embeddings = self.item_embedding.weight[1:self.item_num + 1]
        logits = torch.matmul(h_final, item_embeddings.T)
        logits = logits + self.output_bias[1:self.item_num + 1]
        return logits, g.mean(dim=-1)


class SASRecBaseline(nn.Module):
    """SASRec baseline without DeltaGate."""

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

        # 1. Causal mask: prevent attending to future positions.
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()

        # 2. Padding mask: prevent attending to padded positions.
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
    """GRU4Rec baseline without DeltaGate."""

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

        h_last = h[:, -1, :]

        all_items = torch.arange(1, self.item_num + 1, device=seq.device)
        logits = torch.matmul(h_last, self.item_embedding(all_items).T)

        return logits


if __name__ == '__main__':
    print("Testing DeltaGate models...")

    seq = torch.randint(1, 1001, (4, 50))
    delta_t = torch.tensor([30.0, 180.0, 400.0, 730.0])

    seq[0, :10] = 0

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

    print("\nAll model smoke tests passed.")
