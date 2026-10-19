# RecSys 2026 Audited Protocol

This directory preserves the exact training semantics and provenance records
behind the paper's final claim-bearing DeltaGate rows. It is separate from the
root `train.py`, which is a convenient general-purpose entrypoint.

## Evidence Tracks

| Track | Runs | Code snapshot | Immutable input record |
|---|---:|---|---|
| Corrected DG-SASRec | 3 datasets x 5 seeds | `code_snapshots/sas_corrected/` | `provenance/sas_corrected_input_manifest.json` |
| Retained DG-GRU4Rec | 3 datasets x 5 seeds | `code_snapshots/gru_a98c/` | `provenance/gru_retained_input_manifest.json` |
| Frozen DG-BERT4Rec | 3 datasets x 5 seeds | `code_snapshots/bert/` | `provenance/bert_input_manifest.json` |

The JSON files in `provenance/` are immutable records of the actual data,
backbone checkpoints, source files, seeds, and hyperparameters. The repository
does not redistribute the processed Amazon files or pretrained checkpoints, so
their hashes can be verified only after those inputs are prepared locally.
This repository also does not include a script that builds those files.

Run the runners in this directory, not `code_snapshots/*/train.py`. Those
snapshot scripts import a `protocol` module that is not published. The retained
GRU snapshot's `src/__init__.py` also imports `bert4rec`, `tisasrec`, and
`variants`, which are not in this tree. `runners/train_sas_gru.py` loads the
hashed `sasrec.py`, `gru4rec.py`, and `deltagate.py` files directly.
`runners/train_bert.py` accepts either `code_snapshots/bert` or a historical
checkout whose sources live in an `anonymous_repo` subdirectory.

## Required Layout

Use the processed-data schema in `../../data/README.md` and place inputs under:

```text
data/
|-- amazon_video_games_processed.pkl
|-- amazon_cds_vinyl_processed.pkl
`-- amazon_movies_tv_processed.pkl

checkpoints/
|-- video_games/{sasrec,gru4rec}_seedSEED_abl-none.pt
|-- cds_and_vinyl/{sasrec,gru4rec}_seedSEED_abl-none.pt
`-- movies_and_tv/{sasrec,gru4rec}_seedSEED_abl-none.pt
```

## Audited SASRec/GRU4Rec Runner

From the repository root, a DG-SASRec run has this form:

```bash
python3 experiments/recsys2026/runners/train_sas_gru.py \
  --root . \
  --config experiments/recsys2026/config_sas_public.json \
  --input-manifest experiments/recsys2026/provenance/sas_corrected_input_manifest.json \
  --model deltagate-sasrec \
  --dataset video_games \
  --seed 42 \
  --output-dir outputs/formal/deltagate-sasrec/video_games/seed_42 \
  --device cuda
```

For DG-GRU4Rec, use `config_gru_public.json`, the retained GRU manifest, and
`--model deltagate-gru4rec`.

The runner fails closed if data, checkpoint, or source hashes differ from the
manifest. For newly prepared inputs, create a new manifest rather than editing
the archived provenance record.

## Audited BERT4Rec Runner

The DG-BERT4Rec runner accepts the data and backbone paths directly:

```bash
python3 experiments/recsys2026/runners/train_bert.py \
  --repo-root experiments/recsys2026/code_snapshots/bert \
  --data data/amazon_video_games_processed.pkl \
  --backbone checkpoints/video_games/bert4rec_seed42_abl-none.pt \
  --input-manifest experiments/recsys2026/provenance/bert_input_manifest.json \
  --dataset video_games \
  --seed 42 \
  --output-dir outputs/formal/deltagate-bert4rec/video_games/seed_42 \
  --device cuda --batch-size 128 --epochs 30
```

## Historical Selection Protocol

The accepted-paper pipeline did not release a separate validation split.
Checkpoint selection used mean Hit@10 over the four non-active buckets on the
held-out leave-one-out set, and that same set was reported. The public result
files therefore label these results as descriptive under the historical
protocol, not as unbiased final-test estimates.

The canonical machine-readable result surface is
`../../results/formal_claims.json`.
