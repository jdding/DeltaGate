# Zero-Observation User Reactivation with Gap-Driven Dimensional Gating

Sequential recommendation model incorporating time-decayed item semantics into transformer architectures.

## Overview

This repository contains the model implementation and evaluation pipeline for DeltaGate, a sequential recommendation approach that integrates temporal decay patterns with semantic item representations.

## Dependencies

```
torch>=2.0
numpy
scipy
pandas
scikit-learn
tqdm
transformers
```

Install with: `pip install -r requirements.txt`

## Data Preparation

Place processed data files in the `data/` directory. Expected format:

```
data/
├── amazon_cds_vinyl_processed.pkl    # CDs & Vinyl dataset
├── amazon_movies_tv_processed.pkl    # Movies & TV dataset
└── amazon_video_games_processed.pkl   # Video Games dataset
```

Each pickle file should contain:
- `train_samples`: training sequences
- `test_samples`: test sequences
- `item_vocab_size`: vocabulary size
- `max_len`: maximum sequence length

## Running Instructions

### Training

```bash
python train.py \
    --dataset movies_tv \
    --model deltagate-sasrec \
    --embed_dim 128 \
    --max_len 50 \
    --lr 0.001 \
    --num_epochs 200 \
    --batch_size 128 \
    --output_dir ./checkpoints/
```

### Evaluation

```bash
python evaluate_offline.py \
    --data data/amazon_movies_tv_processed.pkl \
    --model_path checkpoints/movies_tv_s42/model.pt \
    --model deltagate-sasrec \
    --embed_dim 128 \
    --max_len 50
```

### Models

- `SASRec` — SASRec baseline
- `GRU4Rec` — GRU4Rec baseline
- `BERT4Rec` — BERT4Rec baseline
- `TiSASRec` — Time-informed SASRec
- `DeltaGateSASRec` — DeltaGate with SASRec backbone
- `DeltaGateGRU4Rec` — DeltaGate with GRU4Rec backbone

## Results

See `results/` for detailed experimental results across three Amazon categories:
- CDs & Vinyl
- Movies & TV
- Video Games

Results include Hit@10, NDCG@10, MRR@10 by dormancy bucket (30-90d, 90-180d, 180-365d, >365d).
