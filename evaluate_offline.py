#!/usr/bin/env python3
"""
Offline Evaluation Script for SIGIR
Calculates: Hit@10, NDCG@10, MRR@10, Catalog Coverage
Extracts: Gate Vectors for Heatmap Visualization
"""

import argparse
import numpy as np
import torch
import os
import pickle
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import SASRec, GRU4Rec, BERT4Rec, TiSASRec, DeltaGateSASRec, DeltaGateGRU4Rec
from train import make_batch_standard, make_batch_tisasrec, make_batch_deltagate, make_batch_bert4rec

BUCKET_NAMES = ['30-90d', '90-180d', '180-365d', '>365d']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--max-len', type=int, default=50)
    parser.add_argument('--ablation_mode', type=str, default='none')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating model: {args.model_path} on {device}")

    # 1. Load Data
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    test_samples = data.get('test_samples', data.get('test_data', []))
    item_vocab_size = data.get('item_vocab_size', data.get('vocab_size'))
    print(f"Test samples: {len(test_samples)}, Vocab: {item_vocab_size}")

    # 2. Init Model
    if args.model == 'sasrec':
        model = SASRec(item_num=item_vocab_size, embedding_dim=args.embed_dim, max_seq_len=args.max_len)
    elif args.model == 'gru4rec':
        model = GRU4Rec(item_num=item_vocab_size, embedding_dim=args.embed_dim, hidden_size=args.embed_dim)
    elif args.model == 'bert4rec':
        model = BERT4Rec(item_num=item_vocab_size, embedding_dim=args.embed_dim, max_seq_len=args.max_len)
    elif args.model == 'deltagate-sasrec':
        model = DeltaGateSASRec(item_num=item_vocab_size, embedding_dim=args.embed_dim, max_seq_len=args.max_len, ablation_mode=args.ablation_mode)
    else:
        raise ValueError("Model not supported for offline eval yet.")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Setup Forward Hook for Gate Extraction (Only for DeltaGate)
    gate_vectors = []
    bucket_labels = []
    
    if 'deltagate' in args.model:
        def gate_hook(module, input, output):
            gate_vectors.append(output.detach().cpu().numpy())
        model.gate_mlp.register_forward_hook(gate_hook)

    # 4. Evaluation Metrics Setup
    hits = {b: 0 for b in BUCKET_NAMES + ['short']}
    ndcgs = {b: 0 for b in BUCKET_NAMES + ['short']}
    mrrs = {b: 0 for b in BUCKET_NAMES + ['short']}
    counts = {b: 0 for b in BUCKET_NAMES + ['short']}
    recommended_items = set()

    batch_size = 512
    with torch.no_grad():
        for i in tqdm(range(0, len(test_samples), batch_size), desc="Evaluating"):
            batch = test_samples[i:i+batch_size]

            if args.model == 'deltagate-sasrec':
                seq, delta, target, buckets = make_batch_deltagate(batch, args.max_len)
                seq, delta = seq.to(device), delta.to(device)
                logits, _ = model(seq, delta)
                bucket_labels.extend(buckets)
            elif args.model == 'bert4rec':
                seq, target, buckets = make_batch_bert4rec(batch, args.max_len, mask_token=model.mask_token, training=False)
                seq = seq.to(device)
                logits = model(seq)
            else:
                seq, target, buckets = make_batch_standard(batch, args.max_len)
                seq = seq.to(device)
                logits = model(seq)

            _, top10 = torch.topk(logits, 10, dim=-1)
            top10_items = top10 + 1 
            
            recommended_items.update(top10_items.cpu().numpy().flatten().tolist())

            for j in range(len(target)):
                b = buckets[j]
                tgt = target[j].item()
                top = top10_items[j].cpu().tolist()

                if tgt in top:
                    hits[b] += 1
                    rank = top.index(tgt) + 1
                    ndcgs[b] += 1.0 / np.log2(rank + 1)
                    mrrs[b] += 1.0 / rank
                counts[b] += 1

    # 5. Print & Save Results
    print("\n" + "="*50)
    print(f"OFFLINE EVALUATION RESULTS: {args.model}")
    print("="*50)
    print(f"{'Bucket':<12} | {'Hit@10':<8} | {'NDCG@10':<8} | {'MRR@10':<8} | {'Count':<8}")
    print("-" * 55)
    
    for b in BUCKET_NAMES + ['short']:
        if counts[b] > 0:
            h = hits[b] / counts[b]
            n = ndcgs[b] / counts[b]
            m = mrrs[b] / counts[b]
            print(f"{b:<12} | {h:.4f}   | {n:.4f}   | {m:.4f}   | {counts[b]}")

    catalog_coverage = len(recommended_items) / item_vocab_size
    print("-" * 55)
    print(f"Catalog Coverage: {catalog_coverage:.4%} ({len(recommended_items)} / {item_vocab_size})")

    if 'deltagate' in args.model and len(gate_vectors) > 0:
        all_gates = np.concatenate(gate_vectors, axis=0)
        save_dict = {'gates': all_gates, 'buckets': np.array(bucket_labels)}
        out_name = args.model_path.replace('.pt', '_gate_analysis.pkl')
        with open(out_name, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"\n[+] Saved {all_gates.shape[0]} gate vectors to {out_name}.")

if __name__ == '__main__':
    main()
