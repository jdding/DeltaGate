# DeltaGate End-to-End Results on Video Games

This experiment trains the DeltaGate architecture end to end on Amazon Video
Games and compares it with TimeConcat E2E and the frozen-backbone DG-SASRec
plugin setting.

## Experimental Setting

| Hyperparameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Batch size | 512 |
| Maximum epochs | 200 |
| Early-stopping patience | 3 |
| Optimizer | Adam |
| Backbone learning rate | 1e-4 |
| Gate learning rate | 1e-3 |

## Per-Seed Hit@10

| Seed | Epochs | short | 30-90d | 90-180d | 180-365d | >365d |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 25 | 0.2078 | 0.1156 | 0.1089 | 0.0999 | 0.0544 |
| 123 | 28 | 0.2124 | 0.1164 | 0.1116 | 0.1026 | 0.0554 |
| 456 | 25 | 0.2084 | 0.1177 | 0.1069 | 0.0931 | 0.0548 |
| 789 | 27 | 0.2118 | 0.1191 | 0.1110 | 0.1045 | 0.0632 |
| 1024 | 23 | 0.2058 | 0.1158 | 0.1085 | 0.0988 | 0.0578 |

## Five-Seed Average

| Bucket | Hit@10 | Std |
|---|---:|---:|
| short | 0.2092 | 0.0025 |
| 30-90d | 0.1169 | 0.0015 |
| 90-180d | 0.1094 | 0.0017 |
| 180-365d | 0.0998 | 0.0040 |
| >365d | 0.0571 | 0.0034 |

## Comparison

| Model | short | 30-90d | 90-180d | 180-365d | >365d |
|---|---:|---:|---:|---:|---:|
| SASRec baseline | 0.1848 | 0.0909 | 0.0837 | 0.0661 | 0.0311 |
| DG-SASRec plugin | 0.2022 | 0.1044 | 0.0998 | 0.0838 | 0.0474 |
| TimeConcat E2E | 0.2334 | 0.1215 | 0.1152 | 0.1088 | 0.0798 |
| DeltaGate E2E | 0.2092 | 0.1169 | 0.1094 | 0.0998 | 0.0571 |

DeltaGate E2E improves over the SASRec baseline but does not match the
TimeConcat E2E accuracy on this dataset. The frozen-backbone plugin setting
achieves lower macro-gap accuracy than DeltaGate E2E while avoiding backbone
updates.
