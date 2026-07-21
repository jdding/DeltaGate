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
| 42 | 0.2330 | 0.1224 | 0.1179 | 0.1089 | 0.0790 |
| 123 | 0.2302 | 0.1158 | 0.1143 | 0.1059 | 0.0774 |
| 456 | 0.2348 | 0.1228 | 0.1118 | 0.1068 | 0.0822 |
| 789 | 0.2344 | 0.1208 | 0.1159 | 0.1111 | 0.0789 |
| 1024 | 0.2347 | 0.1255 | 0.1163 | 0.1112 | 0.0814 |

## Five-Seed Average

| Bucket | Hit@10 |
|---|---:|
| short | 0.2334 |
| 30-90d | 0.1215 |
| 90-180d | 0.1152 |
| 180-365d | 0.1088 |
| >365d | 0.0798 |

## Comparison

| Model | short | 30-90d | 90-180d | 180-365d | >365d |
|---|---:|---:|---:|---:|---:|
| SASRec baseline | 0.1848 | 0.0909 | 0.0837 | 0.0661 | 0.0311 |
| DG-SASRec plugin | 0.2022 | 0.1044 | 0.0998 | 0.0838 | 0.0474 |
| TimeConcat E2E | 0.2334 | 0.1215 | 0.1152 | 0.1088 | 0.0798 |
| DeltaGate E2E | 0.2092 | 0.1169 | 0.1094 | 0.0998 | 0.0571 |

TimeConcat E2E gives the strongest absolute accuracy in this controlled
comparison, while DG-SASRec plugin remains the parameter-efficient
frozen-backbone alternative.
