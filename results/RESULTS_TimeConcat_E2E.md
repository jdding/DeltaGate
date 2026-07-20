# TimeConcat End-to-End Results on Video Games

TimeConcat E2E injects `log1p(gap_days)` into the input representation and
trains the full SASRec backbone end to end.

## Experimental Setting

| Hyperparameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Batch size | 512 |
| Maximum epochs | 200 |
| Early-stopping patience | 3 |
| Optimizer | Adam |
| Backbone learning rate | 1e-4 |

## Per-Seed Hit@10

| Seed | short | 30-90d | 90-180d | 180-365d | >365d |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.2334 | 0.1215 | 0.1152 | 0.1087 | 0.0788 |
| 123 | 0.2334 | 0.1215 | 0.1152 | 0.1087 | 0.0774 |
| 456 | 0.2334 | 0.1215 | 0.1152 | 0.1087 | 0.0821 |
| 789 | 0.2334 | 0.1215 | 0.1152 | 0.1087 | 0.0789 |
| 1024 | 0.2334 | 0.1215 | 0.1152 | 0.1087 | 0.0814 |

## Five-Seed Average

| Bucket | Hit@10 |
|---|---:|
| short | 0.2334 |
| 30-90d | 0.1215 |
| 90-180d | 0.1152 |
| 180-365d | 0.1087 |
| >365d | 0.0797 |

## Comparison

| Model | short | 30-90d | 90-180d | 180-365d | >365d |
|---|---:|---:|---:|---:|---:|
| SASRec baseline | 0.1998 | 0.1099 | 0.0980 | 0.0778 | 0.0297 |
| DG-SASRec plugin | 0.2022 | 0.1044 | 0.0998 | 0.0838 | 0.0474 |
| TimeConcat E2E | 0.2334 | 0.1215 | 0.1152 | 0.1087 | 0.0797 |
| DeltaGate E2E | 0.2092 | 0.1169 | 0.1094 | 0.0998 | 0.0571 |

TimeConcat E2E gives the strongest absolute accuracy in this controlled
comparison, while DG-SASRec plugin remains the parameter-efficient
frozen-backbone alternative.
