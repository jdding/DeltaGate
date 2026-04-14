# DeltaGate E2E 实验结果 (Video Games)

> 实验时间: 2026-04-12
> 目的: 对比 DeltaGate (gating at OUTPUT) vs TimeConcat (additive time concat at INPUT)，在相同 E2E 训练超参下

---

## 实验设计

**控制变量**: 除 mechanism 外，所有超参完全一致

| 超参 | TimeConcat E2E | DeltaGate E2E |
|------|----------------|---------------|
| LR | 1e-3 | 1e-3 |
| Backbone LR | 1e-4 | 1e-4 |
| Gate LR | N/A | 1e-3 |
| Batch Size | 512 | 512 |
| Epochs | 200 | 200 |
| Patience | 3 | 3 |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| embed_dim | 128 | 128 |

**DeltaGate E2E Architecture**:
```
h_final = g * h_personalized + (1 - g) * h_global
g = sigmoid(MLP([log1p(Δt); h_personalized]))
h_global: zero-initialized Parameter
```

---

## DeltaGate E2E 结果 (5 seeds × Video Games)

### Per-Seed 结果

| Seed | Epochs | short | 30-90d | 90-180d | 180-365d | >365d |
|------|--------|-------|--------|---------|---------|-------|
| 42 | 25 | 0.2078 | 0.1156 | 0.1089 | 0.0999 | 0.0544 |
| 123 | 28 | 0.2124 | 0.1164 | 0.1116 | 0.1026 | 0.0554 |
| 456 | 25 | 0.2084 | 0.1177 | 0.1069 | 0.0931 | 0.0548 |
| 789 | 27 | 0.2118 | 0.1191 | 0.1110 | 0.1045 | 0.0632 |
| 1024 | 23 | 0.2058 | 0.1158 | 0.1085 | 0.0988 | 0.0578 |

### 5-Seed 平均

| Bucket | Hit@10 | Std |
|--------|--------|-----|
| short | 0.2092 | ±0.0025 |
| 30-90d | 0.1169 | ±0.0015 |
| 90-180d | 0.1094 | ±0.0017 |
| 180-365d | 0.0998 | ±0.0040 |
| **>365d** | **0.0571** | **±0.0034** |

---

## 完整对比表

| Model | short | 30-90d | 90-180d | 180-365d | >365d |
|-------|-------|--------|---------|---------|-------|
| SASRec baseline (checkpoint) | 0.1998 | 0.1099 | 0.0980 | 0.0778 | 0.0297 |
| DG-SASRec (checkpoint 加载) | 0.2166 | 0.1173 | 0.1099 | 0.0995 | 0.0573 |
| **TimeConcat E2E** | **0.2334** | **0.1215** | **0.1152** | **0.1087** | **0.0797** |
| **DeltaGate E2E** | 0.2092 | 0.1169 | 0.1094 | 0.0998 | 0.0571 |

**DeltaGate E2E vs TimeConcat E2E**: >365d 仅达 **71.6%**

---

## 三大硬核分析

### 1. Backbone Drift (L2 distance from original SASRec)

| Model | L2 Drift | Relative to Original |
|-------|----------|---------------------|
| TimeConcat E2E | 108.6 | 122% of orig norm |
| DeltaGate E2E | 89.7 | 104% of orig norm |

→ TimeConcat 漂移更大但性能更高，"牺牲基座通用性"说法不成立。

### 2. Active User Stability (short bucket)

| Model | short Hit@10 |
|-------|-------------|
| TimeConcat E2E | **0.2334** |
| DG-SASRec checkpoint | 0.2166 |
| DeltaGate E2E | 0.2092 |
| SASRec baseline | 0.1998 |

→ TimeConcat 在活跃用户上最强，未发现"牺牲基座"问题。

### 3. Gate Entropy (DeltaGate E2E, seed=789, 10K samples)

| Bucket | Mean(g) | 预期 | 实际 |
|--------|---------|------|------|
| short (<30d) | 0.800 | high | high ✓ |
| 30-90d | 0.821 | ↓ | ↑ |
| 90-180d | 0.826 | ↓ | ↑ |
| 180-365d | 0.830 | ↓ | ↑ |
| >365d | **0.836** | low | **high ✗** |

⚠️ **意外发现**: gate 值随 Δt 增加而非减少。global_prior 未被有效利用。

---

## 结论

DeltaGate E2E 在所有 bucket 均不如 TimeConcat E2E，gating mechanism 在 E2E 训练范式下：

1. 无性能优势（71.6% of TimeConcat on >365d）
2. 无 backbone 保护优势
3. gate 路由方向与预期相反
4. E2E 训练下 gating 不如简单 concat

**DeltaGate 的价值存在于"预训练 checkpoint + frozen backbone + gating head"范式，而非 E2E 训练。**

---

## 溯源

- 训练日志: `./logs_deltagate_e2e/deltagate_e2e_s{seed}.log`
- 模型 checkpoint: `./outputs_deltagate_e2e/video_games_s{seed}/best_model.pt`
- 训练脚本: `scripts/deltagate_e2e_baseline/train.py`
- 分析脚本: `scripts/deltagate_e2e_baseline/analyze_backbone_drift.py`

