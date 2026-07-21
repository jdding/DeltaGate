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
- deltagate-bert4rec: BERT4Rec + Delta-Gate
"""

import argparse
import copy
import json
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import (
    SASRec,
    GRU4Rec,
    BERT4Rec,
    TiSASRec,
    DeltaGateSASRec,
    DeltaGateGRU4Rec,
    DeltaGateBERT4Rec,
    TimeConcatSASRec,
    ScalarGateSASRec,
    LinearConcatSASRec,
)
from protocol import (
    PROTOCOL_VERSION,
    file_digest,
    git_state,
    parameter_digest,
    reactivation_selection_score,
)

BUCKET_NAMES = ['30-90d', '90-180d', '180-365d', '>365d']
DELTAGATE_MODELS = {
    'deltagate-sasrec',
    'deltagate-gru4rec',
    'deltagate-bert4rec',
}
ADAPTER_MODELS = DELTAGATE_MODELS | {
    'scalar-gate-sasrec',
    'linear-concat-sasrec',
}
GAP_MODELS = ADAPTER_MODELS | {'timeconcat-sasrec'}
BACKBONE_AUGMENTED_MODELS = GAP_MODELS


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
    """Build a TiSASRec batch from intra-sequence intervals."""
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


def make_batch_deltagate_bert4rec(samples, max_len=50, mask_token=None):
    """Build histories ending in the BERT4Rec extraction mask."""
    seqs, deltas, targets, buckets = [], [], [], []
    for sample in samples:
        history = sample['seq_idx'][-(max_len - 1):]
        seq = [0] * (max_len - len(history) - 1) + history + [mask_token]
        seqs.append(seq)
        deltas.append(sample.get('gap_days', 0.0))
        targets.append(sample['target_idx'])
        buckets.append(sample.get('bucket', 'short'))

    return (
        torch.LongTensor(seqs),
        torch.FloatTensor(deltas),
        torch.LongTensor(targets),
        buckets,
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


def evaluate(
    model,
    samples,
    model_type,
    max_len,
    device,
    mask_token=None,
    batch_size=256,
):
    """Evaluate samples with full ranking and per-bucket metrics."""
    model.eval()
    hits = {b: 0 for b in BUCKET_NAMES + ['short']}
    ndcgs = {b: 0 for b in BUCKET_NAMES + ['short']}
    counts = {b: 0 for b in BUCKET_NAMES + ['short']}

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]

            if model_type == 'bert4rec':
                seq, target, buckets = make_batch_bert4rec(batch, max_len, mask_token=mask_token, training=False)
                seq, target = seq.to(device), target.to(device)
                logits = model(seq)
                
                _, top10 = torch.topk(logits, 10, dim=-1)
                top10_items = top10 + 1
            else:
                if model_type == 'tisasrec':
                    seq, intra, target, buckets = make_batch_tisasrec(batch, max_len)
                    seq, intra, target = seq.to(device), intra.to(device), target.to(device)
                    logits = model(seq, intra)
                elif model_type == 'deltagate-bert4rec':
                    seq, delta, target, buckets = make_batch_deltagate_bert4rec(
                        batch,
                        max_len,
                        mask_token,
                    )
                    seq, delta, target = seq.to(device), delta.to(device), target.to(device)
                    logits, _ = model(seq, delta)
                elif model_type in ['deltagate-sasrec', 'deltagate-gru4rec']:
                    seq, delta, target, buckets = make_batch_deltagate(batch, max_len)
                    seq, delta, target = seq.to(device), delta.to(device), target.to(device)
                    logits, _ = model(seq, delta)
                elif model_type == 'scalar-gate-sasrec':
                    seq, delta, target, buckets = make_batch_deltagate(batch, max_len)
                    seq, delta, target = seq.to(device), delta.to(device), target.to(device)
                    logits, _ = model(seq, delta)
                elif model_type in ['linear-concat-sasrec', 'timeconcat-sasrec']:
                    seq, delta, target, buckets = make_batch_deltagate(batch, max_len)
                    seq, delta, target = seq.to(device), delta.to(device), target.to(device)
                    logits = model(seq, delta)
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
        metrics[f'Count_{b}'] = counts[b]

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


def load_backbone_state(
    model,
    checkpoint_path,
    added_prefixes,
):
    """Load shared backbone tensors while allowing declared new modules."""
    if not checkpoint_path:
        raise ValueError("--backbone is required for backbone-based adaptation")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    expected_state = model.state_dict()
    if (
        'output_bias' in state
        and 'output_bias' in expected_state
        and state['output_bias'].numel() + 1 == expected_state['output_bias'].numel()
    ):
        padded_bias = expected_state['output_bias'].clone()
        padded_bias[:state['output_bias'].numel()] = state['output_bias']
        state = dict(state)
        state['output_bias'] = padded_bias

    missing, unexpected = model.load_state_dict(state, strict=False)
    critical_missing = [
        key for key in missing
        if not any(
            key == prefix or key.startswith(f'{prefix}.')
            for prefix in added_prefixes
        )
    ]
    if critical_missing or unexpected:
        raise RuntimeError(
            "Backbone checkpoint is incompatible: "
            f"missing={critical_missing}, unexpected={unexpected}"
        )


def load_and_freeze_adapter_backbone(
    model,
    checkpoint_path,
    trainable_prefixes,
):
    """Load a pretrained backbone and leave only adapter parameters trainable."""
    load_backbone_state(model, checkpoint_path, trainable_prefixes)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            any(
                name == prefix or name.startswith(f'{prefix}.')
                for prefix in trainable_prefixes
            )
        )

    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def load_and_freeze_deltagate_backbone(model, checkpoint_path):
    """Compatibility wrapper for the DeltaGate plugin contract."""
    return load_and_freeze_adapter_backbone(
        model,
        checkpoint_path,
        ('gate_mlp', 'global_prior'),
    )


def keep_frozen_modules_in_eval(model):
    """Keep frozen backbone modules deterministic during plugin training."""
    model.eval()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_training_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    model_type,
    device,
    steps_per_epoch,
    plugin_training,
):
    model.train()
    if plugin_training:
        keep_frozen_modules_in_eval(model)
    total_loss = 0.0
    num_batches = 0

    for step, batch_data in enumerate(train_loader):
        optimizer.zero_grad()

        if model_type == 'tisasrec':
            seq, intra, target, _ = batch_data
            seq, intra, target = seq.to(device), intra.to(device), target.to(device)
            logits = model(seq, intra)
            loss = criterion(logits, target - 1)
        elif model_type in DELTAGATE_MODELS or model_type == 'scalar-gate-sasrec':
            seq, delta, target, _ = batch_data
            seq, delta, target = seq.to(device), delta.to(device), target.to(device)
            logits, _ = model(seq, delta)
            loss = criterion(logits, target - 1)
        elif model_type in ['linear-concat-sasrec', 'timeconcat-sasrec']:
            seq, delta, target, _ = batch_data
            seq, delta, target = seq.to(device), delta.to(device), target.to(device)
            logits = model(seq, delta)
            loss = criterion(logits, target - 1)
        elif model_type == 'bert4rec':
            seq, labels = batch_data
            seq, labels = seq.to(device), labels.to(device)
            logits = model(seq, return_all=True)
            _, _, vocab_size = logits.shape
            loss = criterion(
                logits.view(-1, vocab_size),
                labels.view(-1),
            )
        else:
            seq, target, _ = batch_data
            seq, target = seq.to(device), target.to(device)
            logits = model(seq)
            loss = criterion(logits, target - 1)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % 1000 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"    Step {step+1}/{steps_per_epoch} | "
                f"Loss: {total_loss/(step+1):.4f} | LR: {current_lr:.2e}"
            )

    return total_loss / max(1, num_batches)


def make_train_loader(dataset, batch_size, collate_fn, is_streaming):
    if is_streaming:
        # Avoid simultaneous shard deserialization in multiple workers.
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def train(args):
    """Main training function."""
    set_seed(args.seed)
    print("=" * 70)
    print(f"Training: {args.model}")
    print("=" * 70)

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"\n[1] Loading data from {args.data}...")
    from data_adapter import load_data
    data = load_data(
        args.data,
        seed=args.seed,
        derive_validation=True,
        strict_split=args.strict_split,
    )

    is_streaming = hasattr(data, '__iter__') and not isinstance(data, dict)
    
    if is_streaming:
        raise ValueError(
            "protocol v2 full-development refit requires monolithic processed "
            "data so validation examples can be restored without ambiguity"
        )
    else:
        train_dataset = data['train_samples']
        development_train_dataset = data['development_train_samples']
        validation_samples = data['validation_samples']
        test_samples = data['test_samples']
        item_vocab_size = data['item_vocab_size']
        selection_train_samples_count = len(train_dataset)
        refit_train_samples_count = len(development_train_dataset)
        split_metadata = data['split_metadata']
    
    selection_steps_per_epoch = math.ceil(
        selection_train_samples_count / args.batch_size
    )
    refit_steps_per_epoch = math.ceil(
        refit_train_samples_count / args.batch_size
    )

    print(f"  Selection train: {selection_train_samples_count:,}")
    print(f"  Refit train: {refit_train_samples_count:,}")
    print(f"  Validation: {len(validation_samples):,}")
    print(f"  Test: {len(test_samples):,}")
    print(f"  Vocab: {item_vocab_size:,}")
    print(f"  Selection Steps/Epoch: {selection_steps_per_epoch:,}")
    print(f"  Refit Steps/Epoch: {refit_steps_per_epoch:,}")

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
    elif args.model == 'deltagate-bert4rec':
        model = DeltaGateBERT4Rec(
            item_num=item_vocab_size, embedding_dim=args.embed_dim,
            num_heads=args.num_heads, num_layers=args.num_layers,
            max_seq_len=args.max_len, dropout=args.dropout,
            ablation_mode=args.ablation_mode,
        )
        model_type = 'deltagate-bert4rec'
    elif args.model == 'timeconcat-sasrec':
        model = TimeConcatSASRec(
            item_num=item_vocab_size, embedding_dim=args.embed_dim,
            num_heads=args.num_heads, num_layers=args.num_layers,
            max_seq_len=args.max_len, dropout=args.dropout,
        )
        model_type = 'timeconcat-sasrec'
    elif args.model == 'scalar-gate-sasrec':
        model = ScalarGateSASRec(
            item_num=item_vocab_size, embedding_dim=args.embed_dim,
            num_heads=args.num_heads, num_layers=args.num_layers,
            max_seq_len=args.max_len, dropout=args.dropout,
        )
        model_type = 'scalar-gate-sasrec'
    elif args.model == 'linear-concat-sasrec':
        model = LinearConcatSASRec(
            item_num=item_vocab_size, embedding_dim=args.embed_dim,
            num_heads=args.num_heads, num_layers=args.num_layers,
            max_seq_len=args.max_len, dropout=args.dropout,
        )
        model_type = 'linear-concat-sasrec'
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if model_type in ADAPTER_MODELS and args.training_mode == 'plugin':
        if model_type in DELTAGATE_MODELS or model_type == 'scalar-gate-sasrec':
            prefixes = ('gate_mlp', 'global_prior')
        else:
            prefixes = ('fusion_linear',)
        trainable_parameters = load_and_freeze_adapter_backbone(
            model,
            args.backbone,
            prefixes,
        )
    elif args.backbone and model_type in BACKBONE_AUGMENTED_MODELS:
        if model_type in DELTAGATE_MODELS or model_type == 'scalar-gate-sasrec':
            prefixes = ('gate_mlp', 'global_prior')
        elif model_type == 'linear-concat-sasrec':
            prefixes = ('fusion_linear',)
        else:
            prefixes = ('time_projection',)
        load_backbone_state(model, args.backbone, prefixes)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        trainable_parameters = list(model.parameters())
    else:
        trainable_parameters = list(model.parameters())

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_names = sorted(
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    print(f"  Trainable parameters: {total_params:,}")
    frozen_digest_before = (
        parameter_digest(model, trainable=False)
        if model_type in ADAPTER_MODELS and args.training_mode == 'plugin'
        else None
    )
    initial_state = copy.deepcopy(model.state_dict())

    optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1) if model_type == 'bert4rec' else nn.CrossEntropyLoss()

    if model_type == 'tisasrec':
        collate_fn = partial(make_batch_tisasrec, max_len=args.max_len)
    elif model_type == 'deltagate-bert4rec':
        collate_fn = partial(
            make_batch_deltagate_bert4rec,
            max_len=args.max_len,
            mask_token=model.mask_token,
        )
    elif model_type in GAP_MODELS:
        collate_fn = partial(make_batch_deltagate, max_len=args.max_len)
    elif model_type == 'bert4rec':
        collate_fn = partial(make_batch_bert4rec, max_len=args.max_len, mask_prob=args.mask_prob, mask_token=model.mask_token, vocab_size=item_vocab_size, training=True)
    else:
        collate_fn = partial(make_batch_standard, max_len=args.max_len)

    train_loader = make_train_loader(
        train_dataset,
        args.batch_size,
        collate_fn,
        is_streaming=False,
    )

    selection_total_steps = selection_steps_per_epoch * args.epochs
    selection_warmup_steps = min(
        5000,
        int(0.05 * selection_total_steps),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        selection_warmup_steps,
        selection_total_steps,
    )

    print(
        f"\n[3] Selecting epoch on validation for up to {args.epochs} epochs "
        f"(Warmup steps: {selection_warmup_steps})..."
    )
    best_avg_hit = float('-inf')
    patience_counter = 0
    best_validation_metrics = None
    best_epoch = None
    validation_history = []
    plugin_training = (
        model_type in ADAPTER_MODELS and args.training_mode == 'plugin'
    )

    for epoch in range(1, args.epochs + 1):
        avg_loss = run_training_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            model_type,
            device,
            selection_steps_per_epoch,
            plugin_training,
        )

        mask_token = model.mask_token if model_type in ['bert4rec', 'deltagate-bert4rec'] else None
        metrics = evaluate(
            model,
            validation_samples,
            model_type,
            args.max_len,
            device,
            mask_token,
            batch_size=args.eval_batch_size,
        )
        avg_hit = reactivation_selection_score(metrics)
        validation_history.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'selection_score': avg_hit,
            'metrics': metrics,
        })

        print(
            f"  Epoch {epoch:2d}/{args.epochs} | Loss: {avg_loss:.4f} | "
            f"Validation Reactivation Hit@10: {avg_hit:.4f}"
        )

        if avg_hit > best_avg_hit:
            best_avg_hit = avg_hit
            best_validation_metrics = metrics.copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_epoch is None:
        raise RuntimeError("validation selection did not produce a best epoch")

    print(
        f"\n[4] Refitting from the identical initialization on "
        f"{refit_train_samples_count:,} development samples for "
        f"{best_epoch} epoch(s)..."
    )
    set_seed(args.seed)
    model.load_state_dict(initial_state, strict=True)
    refit_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(refit_parameters, lr=args.lr)
    refit_total_steps = refit_steps_per_epoch * args.epochs
    refit_warmup_steps = min(5000, int(0.05 * refit_total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        refit_warmup_steps,
        refit_total_steps,
    )
    refit_loader = make_train_loader(
        development_train_dataset,
        args.batch_size,
        collate_fn,
        is_streaming=False,
    )
    refit_history = []
    for epoch in range(1, best_epoch + 1):
        avg_loss = run_training_epoch(
            model,
            refit_loader,
            optimizer,
            scheduler,
            criterion,
            model_type,
            device,
            refit_steps_per_epoch,
            plugin_training,
        )
        refit_history.append({'epoch': epoch, 'train_loss': avg_loss})
        print(
            f"  Refit epoch {epoch:2d}/{best_epoch} | Loss: {avg_loss:.4f}"
        )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), args.output)

    frozen_digest_after = (
        parameter_digest(model, trainable=False)
        if model_type in ADAPTER_MODELS and args.training_mode == 'plugin'
        else None
    )
    if frozen_digest_before != frozen_digest_after:
        raise RuntimeError("frozen backbone drift detected")

    mask_token = model.mask_token if model_type in ['bert4rec', 'deltagate-bert4rec'] else None
    test_metrics = evaluate(
        model,
        test_samples,
        model_type,
        args.max_len,
        device,
        mask_token,
        batch_size=args.eval_batch_size,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"\n{'Bucket':<12} {'Hit@10':<12} {'NDCG@10':<12}")
    print("-" * 36)
    for b in BUCKET_NAMES:
        hit_val = test_metrics.get(f'Hit@10_{b}', 0.0)
        ndcg_val = test_metrics.get(f'NDCG@10_{b}', 0.0)
        print(f"{b:<12} {hit_val:.4f}       {ndcg_val:.4f}")

    print(f"\nModel saved to: {args.output}")
    result = {
        'protocol_version': PROTOCOL_VERSION,
        'selection_split': 'validation',
        'selection_metric': 'mean Hit@10 over 30-90d, 90-180d, 180-365d, >365d',
        'final_fit_split': 'development_train',
        'refit_performed': True,
        'refit_epochs': best_epoch,
        'test_evaluations': 1,
        'model': args.model,
        'training_mode': (
            args.training_mode if args.model in ADAPTER_MODELS else 'e2e'
        ),
        'seed': args.seed,
        'data_path': os.path.abspath(args.data),
        'data_sha256': args.data_sha256 or file_digest(args.data),
        'backbone_path': os.path.abspath(args.backbone) if args.backbone else None,
        'checkpoint_path': os.path.abspath(args.output),
        'checkpoint_sha256': file_digest(args.output),
        'backbone_sha256': file_digest(args.backbone) if args.backbone else None,
        'code': git_state(os.path.dirname(os.path.abspath(__file__))),
        'train_samples': refit_train_samples_count,
        'selection_train_samples': selection_train_samples_count,
        'refit_train_samples': refit_train_samples_count,
        'validation_samples': len(validation_samples),
        'test_samples': len(test_samples),
        'split_metadata': split_metadata,
        'best_epoch': best_epoch,
        'best_validation_score': best_avg_hit,
        'best_validation_metrics': best_validation_metrics,
        'test_metrics': test_metrics,
        'trainable_parameters': total_params,
        'trainable_names': trainable_names,
        'frozen_digest_before': frozen_digest_before,
        'frozen_digest_after': frozen_digest_after,
        'frozen_unchanged': frozen_digest_before == frozen_digest_after,
        'validation_history': validation_history,
        'refit_history': refit_history,
        'selection_steps_per_epoch': selection_steps_per_epoch,
        'refit_steps_per_epoch': refit_steps_per_epoch,
        'selection_warmup_steps': selection_warmup_steps,
        'refit_warmup_steps': refit_warmup_steps,
        'command_args': vars(args),
    }
    result_path = args.result_json or f"{args.output}.result.json"
    result_dir = os.path.dirname(result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(f"Result metadata saved to: {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Train gap-aware recommendation models')

    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to processed data pickle')
    parser.add_argument('--output', type=str, default='best_model.pt', help='Path to save model')
    parser.add_argument('--result-json', type=str, default='',
                        help='Path for protocol and metric metadata')
    parser.add_argument(
        '--data-sha256',
        type=str,
        default='',
        help='Prevalidated input SHA256 supplied by the matrix runner',
    )
    parser.add_argument('--backbone', type=str, default='',
                        help='Pretrained backbone checkpoint required by DeltaGate plugin models')

    # Model
    parser.add_argument('--model', type=str, required=True,
                        choices=['sasrec', 'gru4rec', 'bert4rec', 'tisasrec',
                                 'deltagate-sasrec', 'deltagate-gru4rec',
                                 'deltagate-bert4rec', 'timeconcat-sasrec',
                                 'scalar-gate-sasrec', 'linear-concat-sasrec'],
                        help='Model architecture')
    parser.add_argument(
        '--training-mode',
        choices=['plugin', 'e2e'],
        default='plugin',
        help='Freeze a pretrained backbone or train the full adapter model',
    )

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
    parser.add_argument('--eval-batch-size', type=int, default=256,
                        help='Validation and test batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cpu', help='Device (auto/cpu/cuda/mps)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--ablation_mode', type=str, default='none',
                        choices=['none', 'fixed_gate', 'prior_only'],
                        help='Ablation mode for DeltaGate models')
    parser.add_argument(
        '--strict-split',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Require exact validation-target to test-history continuity',
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

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
