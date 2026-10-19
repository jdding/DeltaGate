#!/usr/bin/env python3
"""Audited frozen-backbone DG-BERT4Rec training for the RecSys 2026 snapshot."""

import argparse
import copy
import hashlib
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


BUCKETS = ["30-90d", "90-180d", "180-365d", ">365d"]
ALLOWLIST = {
    "gate_mlp.0.weight",
    "gate_mlp.0.bias",
    "gate_mlp.2.weight",
    "gate_mlp.2.bias",
    "global_prior",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--run-kind", choices=["preflight", "formal"], default="formal")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(model, include_trainable):
    digest = hashlib.sha256()
    params = dict(model.named_parameters())
    for name, tensor in sorted(model.state_dict().items()):
        parameter = params.get(name)
        is_trainable = parameter is not None and parameter.requires_grad
        if is_trainable != include_trainable:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def expected_inputs(manifest_path, dataset, seed):
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    data_entries = [
        item for item in manifest["inputs"]
        if item["kind"] == "data" and item["dataset"] == dataset
    ]
    checkpoint_entries = [
        item for item in manifest["inputs"]
        if item["kind"] == "backbone_checkpoint"
        and item["dataset"] == dataset
        and item["seed"] == seed
    ]
    if len(data_entries) != 1 or len(checkpoint_entries) != 1:
        raise RuntimeError(f"Manifest identity mismatch for {dataset}/seed-{seed}")
    return data_entries[0], checkpoint_entries[0]


def normalize_samples(samples):
    for sample in samples:
        sample["seq_idx"] = sample.get("pre_items", sample.get("seq_idx", []))
        sample["target_idx"] = sample.get(
            "target_item", sample.get("target_idx", sample.get("target"))
        )
    return samples


def make_batch(samples, max_len, mask_token):
    seqs, targets, deltas, buckets = [], [], [], []
    for sample in samples:
        history = sample["seq_idx"][-(max_len - 1):]
        seqs.append([0] * (max_len - len(history) - 1) + history + [mask_token])
        targets.append(sample["target_idx"] - 1)
        deltas.append(sample.get("gap_days", 0.0))
        buckets.append(sample.get("bucket", "short"))
    return (
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(deltas, dtype=torch.float32),
        buckets,
    )


def evaluate(model, samples, max_len, batch_size, device):
    model.eval()
    hits = Counter()
    ndcgs = Counter()
    counts = Counter()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            seq, targets_zero, delta_t, buckets = make_batch(
                samples[start:start + batch_size], max_len, model.mask_token
            )
            logits, _ = model(seq.to(device), delta_t.to(device))
            top10 = torch.topk(logits, 10, dim=-1).indices.cpu()
            for row, target, bucket in zip(top10, targets_zero, buckets):
                counts[bucket] += 1
                matches = torch.nonzero(row.eq(target), as_tuple=False)
                if len(matches):
                    rank = int(matches[0, 0]) + 1
                    hits[bucket] += 1
                    ndcgs[bucket] += 1.0 / np.log2(rank + 1)
    metrics = {}
    for bucket in BUCKETS + ["short"]:
        count = counts[bucket]
        metrics[f"Hit@10_{bucket}"] = hits[bucket] / count if count else 0.0
        metrics[f"NDCG@10_{bucket}"] = ndcgs[bucket] / count if count else 0.0
        metrics[f"count_{bucket}"] = count
    return metrics


def resolve_code_root(repo_root):
    """Return the directory that contains ``src/`` and ``train.py``.

    The public snapshot is laid out at ``code_snapshots/bert``. The historical
    runner looked for an ``anonymous_repo`` subdirectory of ``--repo-root``.
    """
    repo_root = Path(repo_root)
    candidates = (repo_root, repo_root / "anonymous_repo")
    for candidate in candidates:
        if (candidate / "src" / "bert4rec.py").is_file() and (candidate / "train.py").is_file():
            return candidate
    raise FileNotFoundError(
        "DG-BERT4Rec sources were not found under "
        + " or ".join(str(candidate / "src") for candidate in candidates)
    )


def load_models(repo_root, item_num, backbone_path, device):
    code_root = resolve_code_root(repo_root)
    sys.path.insert(0, str(code_root))
    from src import BERT4Rec, DeltaGateBERT4Rec
    from train import load_and_freeze_deltagate_backbone

    kwargs = dict(
        item_num=item_num,
        embedding_dim=128,
        num_heads=2,
        num_layers=2,
        max_seq_len=50,
        dropout=0.2,
    )
    baseline = BERT4Rec(**kwargs)
    state = torch.load(backbone_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    baseline.load_state_dict(state, strict=True)

    model = DeltaGateBERT4Rec(**kwargs)
    trainable = load_and_freeze_deltagate_backbone(model, str(backbone_path))
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    count = sum(parameter.numel() for parameter in trainable)
    if names != ALLOWLIST or count != 66304:
        raise RuntimeError(f"Trainable surface mismatch: names={sorted(names)}, count={count}")

    baseline.eval().to(device)
    model.eval().to(device)
    probe = torch.zeros((3, 50), dtype=torch.long, device=device)
    probe[0, -4:] = torch.tensor([1, 2, 3, model.mask_token], device=device)
    probe[1, -3:] = torch.tensor([4, 5, model.mask_token], device=device)
    probe[2, -2:] = torch.tensor([6, model.mask_token], device=device)
    with torch.no_grad():
        baseline_hidden = baseline.forward_backbone(probe)
        plugin_hidden = model.forward_backbone(probe)
    max_abs_diff = float((baseline_hidden - plugin_hidden).abs().max().cpu())
    if max_abs_diff != 0.0:
        raise RuntimeError(f"Frozen backbone representation mismatch: {max_abs_diff}")
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model, names, count, max_abs_diff


def main():
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.run_kind == "formal" and (args.max_train_batches or args.max_eval_samples):
        raise ValueError("Formal runs cannot cap training batches or evaluation samples")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expected_data, expected_backbone = expected_inputs(
        args.input_manifest, args.dataset, args.seed
    )
    actual_data_sha = sha256_file(args.data)
    actual_backbone_sha = sha256_file(args.backbone)
    if actual_data_sha != expected_data["sha256"]:
        raise RuntimeError("Data SHA-256 does not match INPUT_MANIFEST.json")
    if actual_backbone_sha != expected_backbone["sha256"]:
        raise RuntimeError("Backbone SHA-256 does not match INPUT_MANIFEST.json")

    data = load_pickle(args.data)
    train_samples = normalize_samples(data.get("train_samples", data.get("train_data", [])))
    test_samples = normalize_samples(data.get("test_samples", data.get("test_data", [])))
    item_num = data.get("item_vocab_size", data.get("vocab_size"))
    if (
        len(train_samples) != expected_data["train_samples"]
        or len(test_samples) != expected_data["test_samples"]
        or item_num != expected_data["item_vocab_size"]
    ):
        raise RuntimeError("Data counts or vocabulary do not match INPUT_MANIFEST.json")
    if args.max_eval_samples:
        per_bucket = max(1, args.max_eval_samples // len(BUCKETS))
        test_samples = [
            sample
            for bucket in BUCKETS
            for sample in [s for s in test_samples if s.get("bucket") == bucket][:per_bucket]
        ]

    model, names, trainable_count, representation_diff = load_models(
        args.repo_root, item_num, args.backbone, device
    )
    frozen_digest_before = state_digest(model, include_trainable=False)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()
    best_metric = float("-inf")
    best_metrics = None
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = np.random.permutation(len(train_samples))
        loss_sum = 0.0
        steps = 0
        for start in range(0, len(permutation), args.batch_size):
            if args.max_train_batches and steps >= args.max_train_batches:
                break
            indices = permutation[start:start + args.batch_size]
            batch = [train_samples[index] for index in indices]
            seq, targets, delta_t, _ = make_batch(batch, 50, model.mask_token)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(seq.to(device), delta_t.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 5.0
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            steps += 1

        metrics = evaluate(model, test_samples, 50, args.eval_batch_size, device)
        selection = float(np.mean([metrics[f"Hit@10_{bucket}"] for bucket in BUCKETS]))
        history.append({
            "epoch": epoch,
            "steps": steps,
            "mean_loss": loss_sum / max(steps, 1),
            "selection_metric": selection,
            "metrics": metrics,
        })
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        if selection > best_metric:
            best_metric = selection
            best_metrics = copy.deepcopy(metrics)
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
        if args.run_kind == "preflight":
            break

    frozen_digest_after = state_digest(model, include_trainable=False)
    if frozen_digest_before != frozen_digest_after:
        raise RuntimeError("Frozen backbone drift detected")
    checkpoint_path = args.output_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    result = {
        "status": "complete",
        "run_kind": args.run_kind,
        "dataset": args.dataset,
        "seed": args.seed,
        "data_path": str(args.data),
        "data_sha256": actual_data_sha,
        "backbone_path": str(args.backbone),
        "backbone_sha256": actual_backbone_sha,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "item_vocab_size": item_num,
        "trainable_names": sorted(names),
        "trainable_parameters": trainable_count,
        "backbone_representation_max_abs_diff": representation_diff,
        "frozen_digest_before": frozen_digest_before,
        "frozen_digest_after": frozen_digest_after,
        "best_selection_metric": best_metric,
        "best_metrics": best_metrics,
        "history": history,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
        "torch_version": torch.__version__,
        "runner_sha256": sha256_file(Path(__file__)),
        "bert4rec_source_sha256": sha256_file(
            resolve_code_root(args.repo_root) / "src/bert4rec.py"
        ),
        "deltagate_source_sha256": sha256_file(
            resolve_code_root(args.repo_root) / "src/deltagate.py"
        ),
        "command_args": vars(args),
    }
    result["command_args"] = {key: str(value) if isinstance(value, Path) else value
                              for key, value in result["command_args"].items()}
    with open(args.output_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
