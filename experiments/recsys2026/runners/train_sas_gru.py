#!/usr/bin/env python3
"""Audited frozen DG-SASRec/DG-GRU4Rec training for RecSys 2026."""

import argparse
import copy
import hashlib
import importlib.util
import json
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
SOURCE_FILES = [
    "src/__init__.py",
    "src/sasrec.py",
    "src/gru4rec.py",
    "src/deltagate.py",
    "train.py",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["deltagate-sasrec", "deltagate-gru4rec"],
        required=True,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--lr", type=float)
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


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def expected_inputs(manifest_path, model, dataset, seed):
    manifest = load_json(manifest_path)
    data_entries = [
        item
        for item in manifest["inputs"]
        if item["kind"] == "data" and item["dataset"] == dataset
    ]
    checkpoint_entries = [
        item
        for item in manifest["inputs"]
        if item["kind"] == "backbone_checkpoint"
        and item["model"] == model
        and item["dataset"] == dataset
        and item["seed"] == seed
    ]
    source_entries = {
        item["relative_path"]: item
        for item in manifest["inputs"]
        if item["kind"] == "source"
    }
    if (
        len(data_entries) != 1
        or len(checkpoint_entries) != 1
        or set(source_entries) != set(SOURCE_FILES)
    ):
        raise RuntimeError(f"Manifest identity mismatch for {model}/{dataset}/seed-{seed}")
    return data_entries[0], checkpoint_entries[0], source_entries


def normalize_samples(samples):
    for sample in samples:
        sample["seq_idx"] = sample.get("pre_items", sample.get("seq_idx", []))
        sample["target_idx"] = sample.get(
            "target_item", sample.get("target_idx", sample.get("target"))
        )
    return samples


def make_batch(samples, max_len):
    seqs, targets, deltas, buckets = [], [], [], []
    for sample in samples:
        history = sample["seq_idx"][-max_len:]
        seqs.append([0] * (max_len - len(history)) + history)
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
                samples[start:start + batch_size], max_len
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


def sasrec_hidden(model, seq):
    batch_size, seq_len = seq.shape
    positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(
        batch_size, -1
    )
    x = model.dropout(model.item_embedding(seq) + model.position_embedding(positions))
    x = x.masked_fill(seq.eq(0).unsqueeze(-1), 0.0)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=seq.device),
        diagonal=1,
    )
    x = model.transformer(x, mask=causal_mask)
    return model.layer_norm(x)[:, -1, :]


def gru4rec_hidden(model, seq):
    x = model.dropout(model.item_embedding(seq))
    x, _ = model.gru(x)
    return x[:, -1, :]


def load_and_freeze_deltagate_backbone(model, checkpoint_path):
    """Load a backbone checkpoint and expose only the declared gate parameters."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = (
        raw["model_state_dict"]
        if isinstance(raw, dict) and "model_state_dict" in raw
        else raw
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    critical_missing = [
        name
        for name in missing
        if not (
            name == "global_prior"
            or name.startswith("gate_mlp.")
        )
    ]
    if critical_missing or unexpected:
        raise RuntimeError(
            "Backbone checkpoint is incompatible: "
            f"missing={critical_missing}, unexpected={unexpected}"
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name == "global_prior" or name.startswith("gate_mlp.")
        )
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def load_snapshot_classes(source_root):
    """Load the hashed snapshot modules by file path.

    The retained GRU snapshot's ``src/__init__.py`` also imports ``bert4rec``,
    ``tisasrec``, and ``variants``, which are not in this public tree. The
    runner only needs the modules recorded in the input manifest.
    """
    source_root = Path(source_root)
    modules = {}
    classes = {}
    for filename, class_names in (
        ("sasrec.py", ("SASRec",)),
        ("gru4rec.py", ("GRU4Rec",)),
        ("deltagate.py", ("DeltaGateSASRec", "DeltaGateGRU4Rec")),
    ):
        path = source_root / "src" / filename
        module_name = f"deltagate_snapshot_{source_root.name}_{filename[:-3]}"
        module = modules.get(filename)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load snapshot module: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            modules[filename] = module
        for class_name in class_names:
            classes[class_name] = getattr(module, class_name)
    return classes


def load_models(source_root, model_name, item_num, backbone_path, hp, device):
    classes = load_snapshot_classes(source_root)
    DeltaGateGRU4Rec = classes["DeltaGateGRU4Rec"]
    DeltaGateSASRec = classes["DeltaGateSASRec"]
    GRU4Rec = classes["GRU4Rec"]
    SASRec = classes["SASRec"]

    if model_name == "deltagate-sasrec":
        kwargs = dict(
            item_num=item_num,
            embedding_dim=hp["embedding_dim"],
            num_heads=hp["num_heads"],
            num_layers=hp["sasrec_num_layers"],
            max_seq_len=hp["max_len"],
            dropout=hp["dropout"],
        )
        baseline = SASRec(**kwargs)
        model = DeltaGateSASRec(**kwargs)
        hidden_fn = sasrec_hidden
    elif model_name == "deltagate-gru4rec":
        kwargs = dict(
            item_num=item_num,
            embedding_dim=hp["embedding_dim"],
            hidden_size=hp["embedding_dim"],
            num_layers=hp["gru4rec_num_layers"],
            dropout=hp["dropout"],
        )
        baseline = GRU4Rec(**kwargs)
        model = DeltaGateGRU4Rec(**kwargs)
        hidden_fn = gru4rec_hidden
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    state = torch.load(backbone_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    baseline.load_state_dict(state, strict=True)
    trainable = load_and_freeze_deltagate_backbone(model, str(backbone_path))
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    count = sum(parameter.numel() for parameter in trainable)
    if names != ALLOWLIST or count != 66304:
        raise RuntimeError(f"Trainable surface mismatch: names={sorted(names)}, count={count}")

    baseline.eval().to(device)
    model.eval().to(device)
    probe = torch.zeros((3, hp["max_len"]), dtype=torch.long, device=device)
    probe[0, -4:] = torch.tensor([1, 2, 3, 4], device=device)
    probe[1, -3:] = torch.tensor([4, 5, 6], device=device)
    probe[2, -2:] = torch.tensor([6, 7], device=device)
    with torch.no_grad():
        baseline_hidden = hidden_fn(baseline, probe)
        plugin_hidden = hidden_fn(model, probe)
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
    config = load_json(args.config)
    hp = config["hyperparameters"]
    for name, key in [
        ("batch_size", "train_batch_size"),
        ("eval_batch_size", "eval_batch_size"),
        ("epochs", "epochs"),
        ("patience", "patience"),
        ("lr", "learning_rate"),
    ]:
        if getattr(args, name) is None:
            setattr(args, name, hp[key])

    if args.model not in config["models"] or args.dataset not in config["datasets"]:
        raise ValueError("Model or dataset is not authorized by the experiment config")
    if args.seed not in config["seeds"]:
        raise ValueError("Seed is not authorized by the experiment config")
    if args.run_kind == "formal" and (args.max_train_batches or args.max_eval_samples):
        raise ValueError("Formal runs cannot cap training batches or evaluation samples")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_spec = config["datasets"][args.dataset]
    model_spec = config["models"][args.model]
    data_path = args.root / dataset_spec["data"]
    backbone_path = (
        args.root
        / dataset_spec["checkpoint_dir"]
        / model_spec["checkpoint_pattern"].format(seed=args.seed)
    )
    source_root = args.root / config["code_snapshot"]

    expected_data, expected_backbone, expected_sources = expected_inputs(
        args.input_manifest, args.model, args.dataset, args.seed
    )
    actual_data_sha = sha256_file(data_path)
    actual_backbone_sha = sha256_file(backbone_path)
    if actual_data_sha != expected_data["sha256"]:
        raise RuntimeError("Data SHA-256 does not match INPUT_MANIFEST.json")
    if actual_backbone_sha != expected_backbone["sha256"]:
        raise RuntimeError("Backbone SHA-256 does not match INPUT_MANIFEST.json")
    source_hashes = {}
    for relative, expected in expected_sources.items():
        actual = sha256_file(source_root / relative)
        if actual != expected["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {relative}")
        source_hashes[relative] = actual

    data = load_pickle(data_path)
    train_samples = normalize_samples(data.get("train_samples", data.get("train_data", [])))
    test_samples = normalize_samples(data.get("test_samples", data.get("test_data", [])))
    item_num = data.get("item_vocab_size", data.get("vocab_size"))
    if (
        len(train_samples) != expected_data["train_samples"]
        or len(test_samples) != expected_data["test_samples"]
        or item_num != expected_data["item_vocab_size"]
    ):
        raise RuntimeError("Data counts or vocabulary do not match INPUT_MANIFEST.json")
    formal_train_samples = len(train_samples)
    formal_test_samples = len(test_samples)
    if args.max_eval_samples:
        per_bucket = max(1, args.max_eval_samples // len(BUCKETS))
        test_samples = [
            sample
            for bucket in BUCKETS
            for sample in [s for s in test_samples if s.get("bucket") == bucket][:per_bucket]
        ]

    model, names, trainable_count, representation_diff = load_models(
        source_root, args.model, item_num, backbone_path, hp, device
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
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Gradients remain enabled for gate/prior in eval mode. Keeping the
        # frozen backbone deterministic prevents dropout from changing the
        # pretrained representation during plugin fitting.
        model.eval()
        permutation = np.random.permutation(len(train_samples))
        loss_sum = 0.0
        steps = 0
        for start in range(0, len(permutation), args.batch_size):
            if args.max_train_batches and steps >= args.max_train_batches:
                break
            indices = permutation[start:start + args.batch_size]
            batch = [train_samples[index] for index in indices]
            seq, targets, delta_t, _ = make_batch(batch, hp["max_len"])
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(seq.to(device), delta_t.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                hp["gradient_clip_norm"],
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            steps += 1

        metrics = evaluate(
            model, test_samples, hp["max_len"], args.eval_batch_size, device
        )
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
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break
        if args.run_kind == "preflight":
            break

    frozen_digest_after = state_digest(model, include_trainable=False)
    if frozen_digest_before != frozen_digest_after:
        raise RuntimeError("Frozen backbone drift detected")
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")

    checkpoint_path = args.output_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    result = {
        "status": "complete",
        "run_kind": args.run_kind,
        "experiment_id": config["experiment_id"],
        "code_revision": config["code_revision"],
        "model": args.model,
        "backbone_model": model_spec["backbone_model"],
        "dataset": args.dataset,
        "seed": args.seed,
        "selection_split": "historical_held_out_test",
        "unbiased_final_test": False,
        "data_path": str(data_path),
        "data_sha256": actual_data_sha,
        "backbone_path": str(backbone_path),
        "backbone_sha256": actual_backbone_sha,
        "source_hashes": source_hashes,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "formal_train_samples": formal_train_samples,
        "formal_test_samples": formal_test_samples,
        "evaluated_test_samples": len(test_samples),
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
        "command_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with open(args.output_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
