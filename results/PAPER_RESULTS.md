# Paper Result Summary

This file summarizes the aggregated Hit@10 results used by the paper. Full
per-seed values are available in `results_all.json`.

## Main Results

### Movies & TV

| Model | 30-90d | 90-180d | 180-365d | >365d |
|---|---:|---:|---:|---:|
| SASRec | 0.1001 +/- 0.0012 | 0.0761 +/- 0.0007 | 0.0532 +/- 0.0008 | 0.0229 +/- 0.0004 |
| GRU4Rec | 0.0970 +/- 0.0003 | 0.0718 +/- 0.0010 | 0.0486 +/- 0.0003 | 0.0218 +/- 0.0006 |
| TiSASRec | 0.1018 +/- 0.0006 | 0.0769 +/- 0.0005 | 0.0548 +/- 0.0004 | 0.0248 +/- 0.0003 |
| BERT4Rec | 0.0915 +/- 0.0026 | 0.0693 +/- 0.0023 | 0.0510 +/- 0.0008 | 0.0239 +/- 0.0008 |
| DG-SASRec | 0.0920 +/- 0.0009 | 0.0734 +/- 0.0013 | 0.0599 +/- 0.0008 | 0.0340 +/- 0.0019 |
| DG-GRU4Rec | 0.0971 +/- 0.0005 | 0.0775 +/- 0.0011 | 0.0624 +/- 0.0008 | 0.0344 +/- 0.0028 |
| DG-BERT4Rec | 0.0939 +/- 0.0023 | 0.0756 +/- 0.0024 | 0.0631 +/- 0.0016 | 0.0348 +/- 0.0016 |
| Ablation-Fixed | 0.1007 +/- 0.0009 | 0.0755 +/- 0.0004 | 0.0531 +/- 0.0004 | 0.0224 +/- 0.0002 |
| Ablation-Prior | 0.0071 +/- 0.0001 | 0.0089 +/- 0.0001 | 0.0088 +/- 0.0004 | 0.0072 +/- 0.0005 |

### CDs & Vinyl

| Model | 30-90d | 90-180d | 180-365d | >365d |
|---|---:|---:|---:|---:|
| SASRec | 0.0816 +/- 0.0014 | 0.0735 +/- 0.0020 | 0.0649 +/- 0.0012 | 0.0332 +/- 0.0008 |
| GRU4Rec | 0.0710 +/- 0.0014 | 0.0649 +/- 0.0007 | 0.0572 +/- 0.0004 | 0.0294 +/- 0.0002 |
| TiSASRec | 0.0856 +/- 0.0017 | 0.0741 +/- 0.0009 | 0.0657 +/- 0.0008 | 0.0353 +/- 0.0008 |
| BERT4Rec | 0.0705 +/- 0.0074 | 0.0569 +/- 0.0076 | 0.0479 +/- 0.0071 | 0.0261 +/- 0.0045 |
| DG-SASRec | 0.0755 +/- 0.0043 | 0.0680 +/- 0.0039 | 0.0627 +/- 0.0039 | 0.0403 +/- 0.0035 |
| DG-GRU4Rec | 0.0711 +/- 0.0014 | 0.0651 +/- 0.0007 | 0.0572 +/- 0.0004 | 0.0295 +/- 0.0003 |
| DG-BERT4Rec | 0.0757 +/- 0.0046 | 0.0646 +/- 0.0063 | 0.0609 +/- 0.0063 | 0.0407 +/- 0.0052 |
| Ablation-Fixed | 0.0799 +/- 0.0016 | 0.0718 +/- 0.0018 | 0.0637 +/- 0.0006 | 0.0321 +/- 0.0005 |
| Ablation-Prior | 0.0073 +/- 0.0002 | 0.0051 +/- 0.0002 | 0.0057 +/- 0.0001 | 0.0050 +/- 0.0001 |

### Video Games

| Model | 30-90d | 90-180d | 180-365d | >365d |
|---|---:|---:|---:|---:|
| SASRec | 0.1099 +/- 0.0011 | 0.0980 +/- 0.0019 | 0.0778 +/- 0.0015 | 0.0297 +/- 0.0005 |
| GRU4Rec | 0.1004 +/- 0.0012 | 0.0931 +/- 0.0019 | 0.0700 +/- 0.0015 | 0.0279 +/- 0.0012 |
| TiSASRec | 0.1104 +/- 0.0018 | 0.0986 +/- 0.0026 | 0.0789 +/- 0.0029 | 0.0311 +/- 0.0006 |
| BERT4Rec | 0.0981 +/- 0.0024 | 0.0818 +/- 0.0037 | 0.0638 +/- 0.0033 | 0.0254 +/- 0.0015 |
| DG-SASRec | 0.1044 +/- 0.0029 | 0.0998 +/- 0.0026 | 0.0838 +/- 0.0016 | 0.0474 +/- 0.0018 |
| DG-GRU4Rec | 0.1014 +/- 0.0029 | 0.0944 +/- 0.0032 | 0.0726 +/- 0.0052 | 0.0315 +/- 0.0063 |
| DG-BERT4Rec | 0.1108 +/- 0.0026 | 0.0975 +/- 0.0013 | 0.0869 +/- 0.0048 | 0.0464 +/- 0.0021 |
| Ablation-Fixed | 0.1093 +/- 0.0033 | 0.0952 +/- 0.0013 | 0.0756 +/- 0.0032 | 0.0288 +/- 0.0012 |
| Ablation-Prior | 0.0090 +/- 0.0014 | 0.0109 +/- 0.0021 | 0.0088 +/- 0.0006 | 0.0075 +/- 0.0009 |

## Key Macro-Gap Results

| Dataset | SASRec >365d | DG-SASRec >365d | Relative Gain |
|---|---:|---:|---:|
| Movies & TV | 0.0229 | 0.0340 | +48.7% |
| CDs & Vinyl | 0.0332 | 0.0403 | +21.4% |
| Video Games | 0.0297 | 0.0474 | +59.7% |

## Ablations

The fixed-gate ablation performs close to the unmodified SASRec baseline, while
the prior-only ablation collapses to a low-popularity global prior. These two
controls indicate that both personalized history and adaptive dimensional
routing are necessary for the reported DeltaGate gains.

## End-to-End Comparison on Video Games

| Model | short | 30-90d | 90-180d | 180-365d | >365d |
|---|---:|---:|---:|---:|---:|
| SASRec baseline | 0.1998 | 0.1099 | 0.0980 | 0.0778 | 0.0297 |
| DG-SASRec plugin | 0.2022 | 0.1044 | 0.0998 | 0.0838 | 0.0474 |
| TimeConcat E2E | 0.2334 | 0.1215 | 0.1152 | 0.1087 | 0.0797 |
| DeltaGate E2E | 0.2092 | 0.1169 | 0.1094 | 0.0998 | 0.0571 |

TimeConcat E2E reaches the highest absolute accuracy when full backbone
retraining is allowed. DeltaGate plugin preserves a frozen backbone and uses a
small trainable gate, making it a parameter-efficient adaptation option for
settings where backbone retraining or embedding drift is undesirable.

## Evidence Boundary

DG-SASRec and DG-GRU4Rec rows are synchronized to the corrected 30-run formal
archive from the camera-ready repair pass. Baseline, DG-BERT4Rec, ablation, and
end-to-end rows remain the original camera-ready evidence unless separately
listed above.
