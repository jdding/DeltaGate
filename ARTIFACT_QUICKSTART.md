# Quickstart

Install the dependencies and run the import-and-forward smoke test:

```bash
python3 -m pip install -r requirements.txt
python3 tests/smoke_imports.py
```

To inspect the training and evaluation interfaces without downloading data:

```bash
python3 train.py --help
python3 evaluate_offline.py --help
```

For the exact paper protocol and its immutable input hashes, see:

```bash
python3 experiments/recsys2026/runners/train_sas_gru.py --help
python3 experiments/recsys2026/runners/train_bert.py --help
```

Then follow `experiments/recsys2026/README.md`.

An available historical DG-BERT4Rec checkpoint can be checked without data:

```bash
python3 tests/check_dg_bert4rec_checkpoint.py --checkpoint PATH_TO_CHECKPOINT
```

Processed-data schemas and full training commands are documented in
`data/README.md` and `README.md`.
