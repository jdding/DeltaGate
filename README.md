# Zero-Observation User Reactivation with Gap-Driven Dimensional Gating

[![arXiv](https://img.shields.io/badge/arXiv-2607.19802-b31b1b.svg)](https://arxiv.org/abs/2607.19802)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3773078.3831771-blue)](https://doi.org/10.1145/3773078.3831771)
[![RecSys 2026](https://img.shields.io/badge/RecSys-2026-orange)](https://dl.acm.org/doi/10.1145/3773078.3831771)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

DeltaGate is a frozen-backbone plugin for sequential recommendation: after a long gap with no observations, it uses that gap to route each dimension of a returning user's representation between their pre-gap history and a learned global prior, without retraining the backbone.

- Paper (arXiv): https://arxiv.org/abs/2607.19802
- ACM Digital Library: https://dl.acm.org/doi/10.1145/3773078.3831771
- Project page: https://jdding.github.io/deltagate

This repository is the public code artifact for the RecSys 2026 main-track paper. It is not a packaged library.

## Results

Five-seed mean Hit@10 in the **>365-day** bucket, full ranking over the item catalog. The metric columns are rounded to three decimals, as in Table 3 of the paper. The relative change is rounded to one decimal from the unrounded corrected-SASRec and DG-SASRec means in `results/formal_claims.json` (the same percentages are listed in `results/PAPER_RESULTS.md`).

| Dataset | SASRec | DG-SASRec | Relative change | BERT4Rec | DG-BERT4Rec |
|---|---:|---:|---:|---:|---:|
| Video Games | 0.031 | 0.047 | +52.2% | 0.025 | 0.046 |
| CDs & Vinyl | 0.022 | 0.040 | +85.9% | 0.026 | 0.041 |
| Movies & TV | 0.021 | 0.034 | +62.6% | 0.024 | 0.035 |

BERT4Rec cells are the three-decimal rounding of the means in `results/results_all.json` and match the paper. At embedding size 128 the plugin trains **66,304** parameters (`gate_mlp` and `global_prior`); the audited runners require that count. On Video Games, the TimeConcat end-to-end comparator reaches **0.080** Hit@10 in this same bucket (`results/formal_claims.json` mean 0.07976594941487354), which is higher than the frozen plugin and retrains the backbone.

These runs selected checkpoints by mean Hit@10 over the four non-active buckets of the same held-out leave-one-out set that is reported. The numbers are descriptive under that historical protocol. They are not unbiased final-test estimates. Bucket-level tables for every model are in `results/PAPER_RESULTS.md`. The canonical machine-readable claim file is `results/formal_claims.json`.

## Install and smoke test

Python 3.9+ and PyTorch 2.x. GPU is useful for full training. The smoke test runs on CPU and does not need the Amazon data.

```bash
python3 -m pip install -r requirements.txt
python3 tests/smoke_imports.py
```

Expected output:

```text
Smoke test passed.
```

The default PyTorch wheel can also install CUDA libraries. Those libraries are optional for the smoke test.

Inspect the command-line interfaces without data:

```bash
python3 train.py --help
python3 evaluate_offline.py --help
python3 experiments/recsys2026/runners/train_sas_gru.py --help
python3 experiments/recsys2026/runners/train_bert.py --help
```

## What this repository does not include

- Raw Amazon review files.
- Processed `.pkl` files and training shards.
- Backbone or DeltaGate checkpoints.
- A script that downloads the reviews or builds the processed files.

The paper uses Amazon Product Reviews (2018), categories Video Games, CDs & Vinyl, and Movies & TV, after iterative 5-core filtering. Its Gap-Synthesize protocol is user-level chronological leave-one-out, the natural gap before the target, pre-gap histories of at least five items, and the buckets `<30d`, `30-90d`, `90-180d`, `180-365d`, and `>365d`. To train or evaluate, prepare files that match `data/README.md`.

The processed sizes recorded for the paper claims, and repeated in `experiments/recsys2026/config_sas_public.json`, are:

| Dataset | Items | Train samples | Test samples |
|---|---:|---:|---:|
| Video Games | 17,408 | 387,125 | 55,220 |
| CDs & Vinyl | 73,713 | 1,218,826 | 112,319 |
| Movies & TV | 60,175 | 2,814,911 | 297,498 |

These counts match Table 2 of the paper. `experiments/recsys2026/provenance/` stores SHA-256 hashes of the processed files and backbone checkpoints used by the claim-bearing runs. The runners stop if a local file differs. This repository does not distribute those files and does not contain a URL for them.

`train.py` is the entrypoint for newly prepared data and does not check those hashes. Exact protocol reruns start at `experiments/recsys2026/README.md`. Do not run `experiments/recsys2026/code_snapshots/*/train.py` directly: those copies import a `protocol` module that is not published, and the retained GRU snapshot's package init also imports `bert4rec`, `tisasrec`, and `variants`, which are not in this tree. The runners load the hashed model files themselves.

## Overview

The repository includes standard sequential backbones, DeltaGate plugin variants, training and full-ranking evaluation entrypoints, lightweight checks, and aggregated paper-result summaries.

![DeltaGate framework](assets/figure3_deltagate_framework.png)

Figure 3 from the paper summarizes the setup: a pre-gap user history is encoded by one frozen backbone, DeltaGate receives the exposed user representation and the post-sequence gap, and the final representation is produced by dimension-wise fusion.

## Repository layout

| Path | Purpose |
|---|---|
| `src/` | SASRec, GRU4Rec, BERT4Rec, TiSASRec, and DeltaGate model implementations |
| `train.py` | Training entrypoint for backbone and DeltaGate variants |
| `evaluate_offline.py` | Full-ranking offline evaluation entrypoint |
| `data_adapter.py` | Loader for processed monolithic or sharded data |
| `data/README.md` | Expected processed-data schema |
| `results/` | Aggregated paper-result summaries and machine-readable tables |
| `experiments/recsys2026/` | Audited runners, code snapshots, configs, and input manifests |
| `tests/` | Smoke and checkpoint-compatibility checks |
| `docs/PROJECT.md` | Additional artifact scope and reproduction notes |
| `CITATION.cff` | Citation metadata for the paper |

## Data

Place processed files under `data/`:

```text
data/
|-- amazon_cds_vinyl_processed.pkl
|-- amazon_movies_tv_processed.pkl
`-- amazon_video_games_processed.pkl
```

Each processed dataset should provide:

- `train_samples`: training examples, or sharded training metadata
- `test_samples`: full-ranking test examples
- `item_vocab_size`: item vocabulary size
- `max_len`: maximum sequence length

Each sample should contain:

- `seq_idx`: historical item ids, left-padded by the batching code
- `target_idx`: held-out target item id
- `bucket`: inactivity-gap bucket label, such as `30-90d` or `>365d`
- `gap_days`: inactivity gap in days, required by DeltaGate models
- `intervals`: intra-sequence intervals, required by TiSASRec

See `data/README.md` for the detailed data contract.

## Training

For exact paper-protocol reproduction, start with `experiments/recsys2026/README.md`. It preserves the eval-mode frozen-backbone fit, fixed Adam learning rate, gradient clipping, historical checkpoint selection rule, and fail-closed input hashes used by the claim-bearing runs. The root entrypoint below is for ordinary use on newly prepared data.

Train a backbone first:

```bash
python3 train.py \
    --data data/amazon_movies_tv_processed.pkl \
    --model sasrec \
    --embed-dim 128 \
    --max-len 50 \
    --lr 0.001 \
    --epochs 200 \
    --batch-size 128 \
    --output checkpoints/movies_tv_sasrec_seed42.pt \
    --seed 42 \
    --device auto
```

Then train the corresponding DeltaGate plugin from the pretrained backbone:

```bash
python3 train.py \
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

Supported models:

- `sasrec`
- `gru4rec`
- `bert4rec`
- `tisasrec`
- `deltagate-sasrec`
- `deltagate-gru4rec`
- `deltagate-bert4rec`

DeltaGate plugin training requires a compatible pretrained backbone checkpoint. The loader freezes the backbone and item embeddings, then optimizes only `gate_mlp` and `global_prior`. DeltaGate models also support `--ablation_mode fixed_gate` and `--ablation_mode prior_only`.

## Evaluation

Evaluate a trained checkpoint with full-ranking evaluation:

```bash
python3 evaluate_offline.py \
    --data data/amazon_movies_tv_processed.pkl \
    --model_path checkpoints/movies_tv_deltagate_sasrec_seed42.pt \
    --model deltagate-sasrec \
    --embed-dim 128 \
    --max-len 50
```

## BERT4Rec and DG-BERT4Rec

For the bidirectional backbone, the public implementation appends `[MASK]` at the chronological end of the pre-gap sequence and uses the final mask hidden state as the personalized representation. DG-BERT4Rec then applies the same output-layer DeltaGate fusion used by the other backbone variants.

## Citation

Please cite the paper. Machine-readable metadata is in `CITATION.cff`.

```bibtex
@inproceedings{ding2026deltagate,
  title = {Zero-Observation User Reactivation with Gap-Driven Dimensional Gating},
  author = {Ding, Jiandong and Liu, Tianying and Liu, Fuyuan and Qin, Huijie and Wu, Tiandeng},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems},
  year = {2026},
  doi = {10.1145/3773078.3831771}
}
```

## License

This repository is released under the MIT license in `LICENSE`.
