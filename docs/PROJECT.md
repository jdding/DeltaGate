# DeltaGate Project Documentation

## Project Scope

DeltaGate studies user reactivation in sequential recommendation when a user
returns after a long inactivity gap. The core method is a lightweight
gap-conditioned plugin that operates on top of a sequential recommendation
backbone. The plugin uses the post-sequence gap length, `gap_days`, to route the
final user representation between the historical sequence representation and a
global prior.

This public artifact is intended to support inspection of:

- model implementations for SASRec, GRU4Rec, BERT4Rec, TiSASRec, and DeltaGate
- the training and full-ranking evaluation entrypoints
- the data format expected by the code
- the aggregated results reported in the paper
- a minimal smoke test that verifies imports and model forward passes

## Artifact Contents

| Path | Purpose |
|---|---|
| `README.md` | Main setup, data, training, evaluation, and result pointers |
| `train.py` | Training entrypoint for standard backbones and DeltaGate variants |
| `evaluate_offline.py` | Full-ranking offline evaluation entrypoint |
| `data_adapter.py` | Loader for monolithic or sharded processed data |
| `src/` | Model implementations |
| `data/README.md` | Processed-data schema expected by the scripts |
| `results/` | Aggregated result summaries and machine-readable result files |
| `tests/smoke_imports.py` | Minimal import and forward-pass smoke test |

## Method Components

DeltaGate is implemented as a post-processing module over a backbone sequence
encoder. Given a history representation `h_personalized` and a time gap
`gap_days`, the model computes a vector gate:

```text
g = sigmoid(MLP([log1p(gap_days); h_personalized]))
h_final = g * h_personalized + (1 - g) * h_global
```

The public implementation includes DeltaGate variants for SASRec, GRU4Rec,
and BERT4Rec. The bidirectional variant appends the extraction mask at the
chronological end and applies the same output-layer gate.

## Data Contract

The repository does not include raw Amazon review data or processed pickle
files. To run training or evaluation, place processed files under `data/`.
Each processed dataset should provide:

- `train_samples`: training examples, or sharded training metadata
- `test_samples`: full-ranking test examples
- `item_vocab_size`: item vocabulary size
- `max_len`: maximum sequence length

Each sample should contain:

- `seq_idx`: historical item ids
- `target_idx`: held-out target item id
- `bucket`: inactivity-gap bucket label
- `gap_days`: post-sequence inactivity gap in days for DeltaGate models
- `intervals`: intra-sequence time intervals for TiSASRec

See `data/README.md` for the detailed schema.

## Reproduction Levels

This artifact supports three levels of inspection.

### Level 1: Code and Smoke Test

This level does not require the Amazon datasets:

```bash
python tests/smoke_imports.py
```

Expected result:

```text
Smoke test passed.
```

### Level 2: Training or Evaluation with Processed Data

After placing processed data under `data/`, run training:

```bash
python train.py \
    --data data/amazon_movies_tv_processed.pkl \
    --model deltagate-sasrec \
    --backbone checkpoints/movies_tv_sasrec_seed42.pt \
    --embed-dim 128 \
    --max-len 50 \
    --lr 0.001 \
    --epochs 200 \
    --batch-size 128 \
    --output checkpoints/movies_tv_deltagate_sasrec_seed42.pt \
    --seed 42 \
    --device auto
```

For all three DeltaGate plugin runs, `--backbone` is mandatory. The checkpoint is loaded
before training, after which only `gate_mlp` and `global_prior` remain
trainable; the sequence encoder and item embeddings are frozen.

Evaluate a trained checkpoint:

```bash
python evaluate_offline.py \
    --data data/amazon_movies_tv_processed.pkl \
    --model_path checkpoints/movies_tv_deltagate_sasrec_seed42.pt \
    --model deltagate-sasrec \
    --embed-dim 128 \
    --max-len 50
```

### Level 3: Result Inspection

Aggregated paper results are available in:

- `results/PAPER_RESULTS.md`
- `results/results.json`
- `results/results_all.json`
- `results/results_e2e.json`

The result summaries include macro-gap performance by dataset and model family.

## Lightweight Checks

Recommended lightweight checks:

```bash
python tests/smoke_imports.py
python train.py --help
python evaluate_offline.py --help
```

Optional static checks:

```bash
python -m py_compile train.py evaluate_offline.py data_adapter.py src/*.py tests/smoke_imports.py
```

## Known Scope Boundaries

- Raw Amazon review data is not included.
- Processed dataset pickle files and checkpoints are not included.
- Full experiment reproduction requires the processed data files described in
  `data/README.md`.
- The exported DG-BERT4Rec class is schema- and forward-compatible with the
  recovered RecSys checkpoint. The DG-BERT4Rec summaries under `results/` were
  regenerated through the frozen-plugin path over three datasets and five
  matched seeds; the recovered joint-training checkpoint is retained only as a
  compatibility artifact.
- The repository is designed as a paper-facing code artifact, not as a
  packaged Python library.

## Primary Result Pointer

For the main paper table and macro-gap summary, start with
`results/PAPER_RESULTS.md`.
