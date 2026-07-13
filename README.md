# Zero-Observation User Reactivation with Gap-Driven Dimensional Gating

This repository contains the public code artifact for DeltaGate, a lightweight
gap-conditioned plugin for sequential recommendation under long user inactivity
gaps. The implementation includes standard sequential backbones, DeltaGate
variants, evaluation utilities, and result tables used by the paper.

## Repository Layout

```text
.
|-- train.py                 # Training entrypoint
|-- evaluate_offline.py      # Full-ranking evaluation entrypoint
|-- data_adapter.py          # Loader for processed data or sharded training data
|-- src/                     # Model implementations
|-- results/                 # Aggregated paper results
|-- data/README.md           # Expected processed-data format
`-- tests/smoke_imports.py   # Import and forward-pass smoke test
```

## Dependencies

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The code was written for PyTorch 2.x and standard scientific Python packages.
GPU acceleration is recommended for full training runs.

## Data

Processed data files should be placed under `data/`:

```text
data/
|-- amazon_cds_vinyl_processed.pkl
|-- amazon_movies_tv_processed.pkl
`-- amazon_video_games_processed.pkl
```

Each processed file should provide:

- `train_samples`: training examples, or metadata for sharded training data
- `test_samples`: full-ranking test examples
- `item_vocab_size`: item vocabulary size
- `max_len`: maximum sequence length

Each sample is expected to contain `seq_idx`, `target_idx`, `bucket`, and, for
DeltaGate models, `gap_days`. TiSASRec additionally expects `intervals`.

## Quick Smoke Test

Run this command to verify imports and model forward passes:

```bash
python tests/smoke_imports.py
```

This smoke test does not require the Amazon datasets.

## Training

Example DeltaGate-SASRec training command:

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

Supported training models:

- `sasrec`
- `gru4rec`
- `bert4rec`
- `tisasrec`
- `deltagate-sasrec`
- `deltagate-gru4rec`
- `deltagate-bert4rec`

DeltaGate models also support `--ablation_mode fixed_gate` and
`--ablation_mode prior_only`. DeltaGate plugin training requires a compatible
pretrained backbone checkpoint. The loader freezes all backbone and item
embedding parameters and optimizes only `gate_mlp` and `global_prior`.
For `deltagate-bert4rec`, batching appends `[MASK]` after the chronological
history and extracts that final hidden state, matching Algorithm 1.

## Evaluation

Example full-ranking evaluation command:

```bash
python evaluate_offline.py \
    --data data/amazon_movies_tv_processed.pkl \
    --model_path checkpoints/movies_tv_deltagate_sasrec_seed42.pt \
    --model deltagate-sasrec \
    --embed-dim 128 \
    --max-len 50
```

The offline evaluation script supports `sasrec`, `gru4rec`, `bert4rec`, and
all three DeltaGate variants.

## Results

The `results/` directory contains aggregated Hit@10 and NDCG@10 results for:

- Amazon Video Games
- Amazon CDs & Vinyl
- Amazon Movies & TV

The main result summary is in `results/PAPER_RESULTS.md`; machine-readable
tables are provided in `results/results.json`, `results/results_all.json`, and
`results/results_e2e.json`.

## Project Documentation

For a paper-artifact description of the project scope, artifact contents,
reproduction levels, and expected checks, see `docs/PROJECT.md`.
