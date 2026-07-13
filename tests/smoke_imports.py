#!/usr/bin/env python3
"""Minimal import and forward-pass smoke test for the public artifact."""

import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402,F401
import evaluate_offline  # noqa: E402,F401
from src import (  # noqa: E402
    BERT4Rec,
    DeltaGateBERT4Rec,
    DeltaGateGRU4Rec,
    DeltaGateSASRec,
    GRU4Rec,
    SASRec,
)


EXPECTED_PLUGIN_PARAMETERS = {
    'gate_mlp.0.weight',
    'gate_mlp.0.bias',
    'gate_mlp.2.weight',
    'gate_mlp.2.bias',
    'global_prior',
}


def check_frozen_plugin(backbone, plugin):
    with tempfile.NamedTemporaryFile(suffix='.pt') as checkpoint:
        torch.save(backbone.state_dict(), checkpoint.name)
        train.load_and_freeze_deltagate_backbone(plugin, checkpoint.name)

    trainable = {
        name for name, parameter in plugin.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == EXPECTED_PLUGIN_PARAMETERS


def main():
    seq = torch.randint(1, 101, (2, 10))
    seq[0, :3] = 0
    delta_t = torch.tensor([45.0, 420.0])

    sas = DeltaGateSASRec(item_num=100, embedding_dim=32, num_heads=2, num_layers=1, max_seq_len=10)
    gru = DeltaGateGRU4Rec(item_num=100, embedding_dim=32, hidden_size=32)
    bert = DeltaGateBERT4Rec(
        item_num=100,
        embedding_dim=32,
        num_heads=2,
        num_layers=1,
        max_seq_len=10,
    )
    bert_seq = seq.clone()
    bert_seq[:, -1] = bert.mask_token

    sas_logits, sas_conf = sas(seq, delta_t)
    gru_logits, gru_conf = gru(seq, delta_t)
    bert_logits, bert_conf = bert(bert_seq, delta_t)

    assert sas_logits.shape == (2, 100)
    assert gru_logits.shape == (2, 100)
    assert sas_conf.shape == (2,)
    assert gru_conf.shape == (2,)
    assert bert_logits.shape == (2, 100)
    assert bert_conf.shape == (2,)

    check_frozen_plugin(
        SASRec(item_num=100, embedding_dim=32, num_heads=2, num_layers=1, max_seq_len=10),
        DeltaGateSASRec(item_num=100, embedding_dim=32, num_heads=2, num_layers=1, max_seq_len=10),
    )
    check_frozen_plugin(
        GRU4Rec(item_num=100, embedding_dim=32, hidden_size=32),
        DeltaGateGRU4Rec(item_num=100, embedding_dim=32, hidden_size=32),
    )
    check_frozen_plugin(
        BERT4Rec(item_num=100, embedding_dim=32, num_heads=2, num_layers=1, max_seq_len=10),
        DeltaGateBERT4Rec(item_num=100, embedding_dim=32, num_heads=2, num_layers=1, max_seq_len=10),
    )
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
