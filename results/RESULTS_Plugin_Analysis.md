# Plugin and End-to-End Adaptation Analysis

This note compares three adaptation regimes on Amazon Video Games:

- SASRec baseline without a macro-gap signal
- TimeConcat E2E, which injects `log1p(gap_days)` into the input stream and
  retrains the backbone
- DG-SASRec plugin, which freezes the backbone and trains only the DeltaGate
  module

## Summary

| Model | short Hit@10 | >365d Hit@10 | >365d NDCG@10 | Trainable Params | Gain per 1M Params |
|---|---:|---:|---:|---:|---:|
| SASRec baseline | 0.1998 | 0.0297 | 0.0067 | 0 | N/A |
| TimeConcat E2E | 0.2334 | 0.0797 | 0.0195 | 2,648,320 | 0.0189 |
| DG-SASRec plugin | 0.2166 | 0.0573 | 0.0138 | 66,048 | 0.8676 |

TimeConcat E2E obtains the highest absolute accuracy. DG-SASRec plugin obtains
lower absolute accuracy but uses about 1/40 of the trainable parameters and
keeps the backbone fixed.

## Hit@10 by Gap Bucket

| Bucket | TimeConcat E2E | DG-SASRec plugin | Difference |
|---|---:|---:|---:|
| short | 0.2334 | 0.2166 | +7.8% |
| 30-90d | 0.1215 | 0.1173 | +3.6% |
| 90-180d | 0.1152 | 0.1099 | +4.8% |
| 180-365d | 0.1087 | 0.0995 | +9.2% |
| >365d | 0.0797 | 0.0573 | +39.1% |

## NDCG@10 by Gap Bucket

| Model | short | 30-90d | 90-180d | 180-365d | >365d |
|---|---:|---:|---:|---:|---:|
| SASRec baseline | 0.0583 | 0.0289 | 0.0251 | 0.0198 | 0.0067 |
| DG-SASRec plugin | 0.0647 | 0.0312 | 0.0301 | 0.0278 | 0.0138 |
| TimeConcat E2E | 0.0712 | 0.0358 | 0.0331 | 0.0301 | 0.0195 |

## Parameter Efficiency

| Model | Trainable Params | >365d Hit@10 Gain vs SASRec | Gain per 1M Params |
|---|---:|---:|---:|
| TimeConcat E2E | 2,648,320 | +0.0500 | 0.0189 |
| DG-SASRec plugin | 66,048 | +0.0276 | 0.8676 |

## Active-User Stability

| Model | short Hit@10 | Gain vs SASRec |
|---|---:|---:|
| SASRec baseline | 0.1998 | 0.0% |
| TimeConcat E2E | 0.2334 | +16.8% |
| DG-SASRec plugin | 0.2166 | +8.4% |

Both adaptation methods improve the short-gap bucket in this experiment.

## Backbone Drift

| Model | Backbone L2 Drift | Relative to Original Norm |
|---|---:|---:|
| TimeConcat E2E | 108.63 | 123.6% |
| DeltaGate E2E | 89.71 | 102.1% |
| DG-SASRec plugin | 0.00 | 0.0% |

The plugin configuration leaves the pretrained backbone unchanged by design.

## Interpretation

TimeConcat E2E is the best option when full retraining is allowed and absolute
accuracy is the only objective. DG-SASRec plugin is the more constrained
adaptation setting: it preserves the backbone and introduces only a small
gap-conditioned module. The two settings therefore answer different deployment
questions rather than forming a single strict accuracy ranking.
