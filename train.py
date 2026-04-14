#!/usr/bin/env python3
"""
Training Script for Gap-Aware Sequential Recommendation
========================================================

Supports multiple model architectures:
- sasrec: Standard SASRec
- gru4rec: GRU4Rec
- bert4rec: BERT4Rec
- tisasrec: Time Interval-aware SASRec
- deltagate-sasrec: SASRec + Delta-Gate
- deltagate-gru4rec: GRU4Rec + Delta-Gate

SANITIZED: 
- Fixed GPU starvation by migrating to PyTorch DataLoader for streaming.
- Added Linear LR Warmup to prevent Transformer attention collapse.
- Kept Full-Ranking evaluation to maintain academic integrity.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SASRec, GRU4Rec, BERT4Rec, TiSASRec, DeltaGateSASRec, DeltaGateGRU4Rec

BUCKET_NAMES = ['30-90d', '90-180d', '180-365d', '>365d']


def make_batch_standard(samples, max_len=50):
    """Batch for standard models (SASRec, GRU4Rec)."""
    seqs, targets, buckets = [], [], []
    for s in samples:
        seq = s['seq_idx'][-max_len:]
        seq = [0] * (max_len - len(seq)) + seq
        seqs.append(seq)
        targets.append(s['target_idx'])
        buckets.append(s.get('bucket', 'short'))
    return torch.LongTensor(seqs), torch.LongTensor(targets), buckets


def make_batch_tisasrec(samples, max_len=50):
    """Batch for TiSASRec - uses ONLY intra_intervals."""
    seqs, intra_intervals, targets, buckets = [], [], [], []
    for s in samples:
        seq = s['seq_idx'][-max_len:]
        seq = [0] * (max_len - len(seq)) + seq
        seqs.append(seq)

        intervals = s.get('intervals', [])[-(max_len-1):]
        intervals = [0] * (max_len - len(intervals) - 1) + [0] + intervals
        intervals = intervals[-max_len:]
        intra_intervals.append(intervals)

        targets.append(s['target_idx'])
        buckets.append(s.get('bucket', 'short'))

    return (
        torch.LongTensor(seqs),
        torch.FloatTensor(intra_intervals),
        torch.LongTensor(targets),
        buckets
    )


def make_batch_deltagate(samples, max_len=50):
    """Batch for DeltaGate models - uses gap_days."""
    seqs, deltas, targets, buckets = [], [], [], []
    for s in samples:
        seq = s['seq_idx'][-max_len:]
        seq = [0] * (max_len - len(seq)) + seq
        seqs.append(seq)
        deltas.append(s.get('gap_days', 0.0))
        targets.append(s['target_idx'])
        buckets.append(s.get('bucket', 'short'))

    return (
        torch.LongTensor(seqs),
        torch.FloatTensor(deltas),
        torch.LongTensor(targets),
        buckets
    )


def make_batch_bert4rec(samples, max_len=50, mask_prob=0.2, mask_token=None, vocab_size=None, training=True):
    """Batch for BERT4Rec with masking."""
    if training:
        seqs, labels = [], []
        for s in samples:
            seq = s['seq_idx'][-(max_len-1):] + [s['target_idx']]
            seq = [0] * (max_len - len(seq)) + seq
            valid = [i for i, x in enumerate(seq) if x != 0]
            n_mask = max(1, int(len(valid) * mask_prob))
            n_mask = min(n_mask, len(valid))
            mask_pos = np.random.choice(valid, n_mask, replace=False)
            masked = seq.copy()
            lbl = [-1] * max_len
            for p in mask_pos:
                orig = seq[p]
                r = np.random.random()
                if r < 0.8:
                    masked[p] = mask_token
                elif r < 0.9:
                    masked[p] = np.random.randint(1, vocab_size + 1)
                lbl[p] = orig - 1
            seqs.append(masked)
            labels.append(lbl)
        return torch.LongTensor(seqs), torch.LongTensor(labels)
    else:
        seqs, targets, buckets = [], [], []
        for s in samples:
            hist = s['seq_idx'][-(max_len-1):]
            seq = [0] * (max_len - len(hist) - 1) + hist + [mask_token]
            seqs.append(seq)
            targets.append(s['target_idx'])
            buckets.append(s.get('bucket', 'short'))
        return torch.LongTensor(seqs), torch.LongTensor(targets), buckets


def evaluate(model, test_samples, model_type, max_len, device, mask_token=None):
    """Evaluate model on test samples using Full-Ranking."""
    model.eval()
    hits = {b: 0 for b in BUCKET_NAMES + ['short']}
    ndcgs = {b: 0 for b in BUCKET_NAMES + ['short']}
    counts = {b: 0 for b in BUCKET_NAMES + ['short']}

    with torch.no_grad():
        batch_size = 256
        for i in range(0, len(test_samples), batch_size):
            batch = test_samples[i:i+batch_size]

            if model_type == 'bert4rec':
                seq, target, buckets = make_batch_bert4rec(batch, max_len, mask_token=mask_token, training=False)
                seq, target = seq.to(device), target.to(device)
                logits = model(seq)
                
                # BERT4Rec specifically returns [B, V], we just rank it
                _, top10 = torch.topk(logits, 10, dim=-1)
                top10_items = top10 + 1
            else:
                if model_type == 'tisasrec':
                    seq, intra, target, buckets = make_batch_tisasrec(batch, max_len)
                    seq, intra, target = seq.to(device), intra.to(device), target.to(device)
                    logits = model(seq, intra)
                elif model_type in ['deltagate-sasrec', 'deltagate-gru4rec']:
                    seq, delta, target, buckets = make_batch_deltagate(batch, max_len)
                    seq, delta, target = seq.to(device), delta.to(device), target.to(device)
                    logits, _ = model(seq, delta)
                else:
                    seq, target, buckets = make_batch_standard(batch, max_len)
                    seq, target = seq.to(device), target.to(device)
                    logits = model(seq)

                _, top10 = torch.topk(logits, 10, dim=-1)
                top10_items = top10 + 1

            for j in range(len(target)):
                b = buckets[j]
                tgt = target[j].item()
                top = top10_items[j].cpu().tolist()

                if tgt in top:
                    hits[b] += 1
                    rank = top.index(tgt) + 1
                    ndcgs[b] += 1.0 / np.log2(rank + 1)
                counts[b] += 1

    metrics = {}
    for b in BUCKET_NAMES + ['short']:
        if counts[b] > 0:
            metrics[f'Hit@10_{b}'] = hits[b] / counts[b]
            metrics[f'NDCG@10_{b}'] = ndcgs[b] / counts[b]
        else:
            metrics[f'Hit@10_{b}'] = 0.0
            metrics[f'NDCG@10_{b}'] = 0.0

    return metrics


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear Warmup Scheduler to prevent early attention collapse in Transformers."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    """Main training function."""
    print("=" * 70)
    print(f"Training: {args.model}")
    print("=" * 70)

    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"\n[1] Loading data from {args.data}...")
    from data_adapter import load_data
    data = load_data(args.data, seed=args.seed)

    is_streaming = hasattr(data, '__iter__') and not isinstance(data, dict)
    
    if is_streaming:
        train_dataset = data
        test_samples = data.test_samples
        item_vocab_size = data.item_vocab_size
        train_samples_count = len(data)
    else:
        train_dataset = data['train_samples']
        test_samples = data['test_samples']
        item_vocab_size = data['item_vocab_size']
        train_samples_count = len(train_dataset)
    
    steps_per_epoch = train_samples_count // args.batch_size

    print(f"  Train: {train_samples_count:,}")
    print(f"  Test: {len(test_samples):,}")
    print(f"  Vocab: {item_vocab_size:,}")
    print(f"  Steps/Epoch: {steps_per_epoch:,}")

    print(f"\n[2] Creating model...")
    if args.model == 'sasrec':
        model = SASRec(
            item_num=item_vocab_size, embedding_dim=args.embed_dim,
            num_heads=args.num_heads, num_layers=args.num_layers,
            max_seq_len=args.max_len, dropout=args.dropout,
        )
        model_type = 'sasrec'
    elif args.model == 'gru4rec':
        model = GRU4Rec(
            item_num=item_vocab_size, embedding_dim=args.embed_dim,
            hidden_size=args.embed_dim, dropout=args.dropout,
        )
        model_type = 'gru4rec'
    elif args.model == 'bert4rec':
        model = BERT4Rec(
            item_num=item_vocab_size, embedding_dim=args.embed_dim,
            num_heads=args.num_heads, num_layers=args.num_layers,
            max_seq_len=args.max_len, dropout=args.dropout,
        )
        model_type = 'bert4rec'
    elif args.model == 'tisasrec':
        model = TiSASRec(
            item_num=item_vocab_size, embedding_dim=args.embed_dim,
            num_heads=args.num_heads, num_layers=args.num_layers,
            max_seq_len=args.max_len, dropout=args.dropout,
            time_buckets=args.time_buckets, max_time_interval=args.max_time_interval,
        )
        model_type = 'tisasrec'
    elif args.model == 'deltagate-sasrec':
        model = DeltaGateSASRec(
            item_num=item_vocab_size, embedding_dim=args.embed_dim,
            num_heads=args.num_heads, num_layers=args.num_layers,
            max_seq_len=args.max_len, dropout=args.dropout,
            ablation_mode=args.ablation_mode,
        )
        model_type = 'deltagate-sasrec'
    elif args.model == 'deltagate-gru4rec':
        model = DeltaGateGRU4Rec(
            item_num=item_vocab_size, embedding_dim=args.embed_dim,
            hidden_size=args.embed_dim, dropout=args.dropout,
        )
        model_type = 'deltagate-gru4rec'
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1) if model_type == 'bert4rec' else nn.CrossEntropyLoss()

    # Configure DataLoader and Collate Fn
    if model_type == 'tisasrec':
        collate_fn = partial(make_batch_tisasrec, max_len=args.max_len)
    elif model_type in ['deltagate-sasrec', 'deltagate-gru4rec']:
        collate_fn = partial(make_batch_deltagate, max_len=args.max_len)
    elif model_type == 'bert4rec':
        collate_fn = partial(make_batch_bert4rec, max_len=args.max_len, mask_prob=args.mask_prob, mask_token=model.mask_token, vocab_size=item_vocab_size, training=True)
    else:
        collate_fn = partial(make_batch_standard, max_len=args.max_len)

    if is_streaming:
        # Single-thread loading to prevent OOM from multi-process shard deserialization spikes
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn, 
            pin_memory=True
        )

    # Configure Learning Rate Warmup
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(5000, int(0.05 * total_steps))  # Warmup for first 5% or 5000 steps
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n[3] Training for {args.epochs} epochs (Warmup steps: {warmup_steps})...")
    best_avg_hit = 0
    patience_counter = 0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        
        for step, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Unpack dynamically based on model type
            if model_type == 'tisasrec':
                seq, intra, target, _ = batch_data
                seq, intra, target = seq.to(device), intra.to(device), target.to(device)
                logits = model(seq, intra)
                loss = criterion(logits, target - 1)
            elif model_type in ['deltagate-sasrec', 'deltagate-gru4rec']:
                seq, delta, target, _ = batch_data
                seq, delta, target = seq.to(device), delta.to(device), target.to(device)
                logits, _ = model(seq, delta)
                loss = criterion(logits, target - 1)
            elif model_type == 'bert4rec':
                seq, labels = batch_data
                seq, labels = seq.to(device), labels.to(device)
                logits = model(seq, return_all=True)
                B, L, V = logits.shape
                loss = criterion(logits.view(-1, V), labels.view(-1))
            else:
                seq, target, _ = batch_data
                seq, target = seq.to(device), target.to(device)
                logits = model(seq)
                loss = criterion(logits, target - 1)

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Print status every 1000 steps
            if (step + 1) % 1000 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"    Step {step+1}/{steps_per_epoch} | Loss: {total_loss/(step+1):.4f} | LR: {current_lr:.2e}")

        avg_loss = total_loss / steps_per_epoch

        # Evaluate at the end of epoch
        mask_token = model.mask_token if model_type == 'bert4rec' else None
        metrics = evaluate(model, test_samples, model_type, args.max_len, device, mask_token)
        
        # Calculate Average Hit@10 across predefined buckets
        valid_buckets = [b for b in BUCKET_NAMES if f'Hit@10_{b}' in metrics and metrics[f'Hit@10_{b}'] > 0]
        if valid_buckets:
            avg_hit = np.mean([metrics[f'Hit@10_{b}'] for b in valid_buckets])
        else:
            avg_hit = 0.0

        print(f"  Epoch {epoch:2d}/{args.epochs} | Loss: {avg_loss:.4f} | Avg Hit@10: {avg_hit:.4f}")

        if avg_hit > best_avg_hit:
            best_avg_hit = avg_hit
            best_metrics = metrics.copy()
            patience_counter = 0

            # Save best model
            os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
            torch.save(model.state_dict(), args.output)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n{'Bucket':<12} {'Hit@10':<12} {'NDCG@10':<12}")
    print("-" * 36)
    for b in BUCKET_NAMES:
        hit_val = best_metrics.get(f'Hit@10_{b}', 0.0)
        ndcg_val = best_metrics.get(f'NDCG@10_{b}', 0.0)
        print(f"{b:<12} {hit_val:.4f}       {ndcg_val:.4f}")

    print(f"\nModel saved to: {args.output}")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description='Train gap-aware recommendation models')

    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to processed data pickle')
    parser.add_argument('--output', type=str, default='best_model.pt', help='Path to save model')

    # Model
    parser.add_argument('--model', type=str, required=True,
                        choices=['sasrec', 'gru4rec', 'bert4rec', 'tisasrec',
                                 'deltagate-sasrec', 'deltagate-gru4rec'],
                        help='Model architecture')

    # Hyperparameters
    parser.add_argument('--embed-dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--max-len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--time-buckets', type=int, default=256, help='Number of time buckets (TiSASRec)')
    parser.add_argument('--max-time-interval', type=int, default=730, help='Max time interval in days (TiSASRec)')
    parser.add_argument('--mask-prob', type=float, default=0.2, help='Mask probability (BERT4Rec)')

    # Training
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda/mps)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--ablation_mode', type=str, default='none',
                        choices=['none', 'fixed_gate', 'prior_only'],
                        help='Ablation mode for DeltaGate models')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Auto-detect device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            args.device = 'mps'
        elif torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'

    train(args)


if __name__ == '__main__':
    main()