#!/usr/bin/env python3
"""Validate a DG-BERT4Rec checkpoint against the public model schema."""

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import DeltaGateBERT4Rec  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-heads', type=int, default=2)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    item_rows, embedding_dim = state['item_embedding.weight'].shape
    item_num = item_rows - 2
    max_seq_len = state['position_embedding.weight'].shape[0]
    layer_indices = {
        int(key.split('.')[2])
        for key in state
        if key.startswith('transformer.layers.')
    }
    num_layers = max(layer_indices) + 1

    model = DeltaGateBERT4Rec(
        item_num=item_num,
        embedding_dim=embedding_dim,
        num_heads=args.num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    seq = torch.zeros((2, max_seq_len), dtype=torch.long)
    seq[0, -4:] = torch.tensor([1, 2, 3, model.mask_token])
    seq[1, -3:] = torch.tensor([4, 5, model.mask_token])
    delta_t = torch.tensor([45.0, 420.0])
    with torch.no_grad():
        logits, confidence = model(seq, delta_t)

    print(json.dumps({
        'strict_load': True,
        'state_tensors': len(state),
        'item_num': item_num,
        'embedding_dim': embedding_dim,
        'num_layers': num_layers,
        'num_heads': args.num_heads,
        'max_seq_len': max_seq_len,
        'logits_shape': list(logits.shape),
        'confidence_shape': list(confidence.shape),
    }, indent=2))


if __name__ == '__main__':
    main()
