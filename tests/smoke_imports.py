#!/usr/bin/env python3
"""Minimal import and forward-pass smoke test for the public artifact."""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402,F401
import evaluate_offline  # noqa: E402,F401
from src import DeltaGateGRU4Rec, DeltaGateSASRec  # noqa: E402


def main():
    seq = torch.randint(1, 101, (2, 10))
    seq[0, :3] = 0
    delta_t = torch.tensor([45.0, 420.0])

    sas = DeltaGateSASRec(item_num=100, embedding_dim=32, num_heads=2, num_layers=1, max_seq_len=10)
    gru = DeltaGateGRU4Rec(item_num=100, embedding_dim=32, hidden_size=32)

    sas_logits, sas_conf = sas(seq, delta_t)
    gru_logits, gru_conf = gru(seq, delta_t)

    assert sas_logits.shape == (2, 100)
    assert gru_logits.shape == (2, 100)
    assert sas_conf.shape == (2,)
    assert gru_conf.shape == (2,)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
