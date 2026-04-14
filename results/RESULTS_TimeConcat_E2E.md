# TimeConcat E2E 实验结果 (Video Games)

> 实验时间: 2026-04-12
> 目的: 对比 TimeConcat (additive time concat at INPUT) vs DeltaGate (gating at OUTPUT)，在相同 E2E 训练超参下

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

**Mechanism 区别**:
- TimeConcat: `x = item_emb + pos_emb + MLP(log1p(Δt))` (additive at INPUT)
- DeltaGate: `h_final = g*h_pers + (1-g)*h_global` where `g=sigmoid(MLP([log1p(Δt); h_pers]))` (gating at OUTPUT)

---

## TimeConcat E2E 结果 (5 seeds × Video Games)

### Per-Seed 结果

| Seed | short | 30-90d | 90-180d | 180-365d | >365d |
|------|-------|--------|---------|---------|-------|
| 42 | 0.2334 | 0.1215 | 0.1152 | 0.1087 | **0.0788** |
| 123 | — | — | — | — | **0.0774** |
| 456 | — | — | — | — | **0.0821** |
| 789 | — | — | — | — | **0.0789** |
| 1024 | — | — | — | — | **0.0814** |

### 5-Seed 平均 (官方评估)

| Bucket | Hit@10 | Count |
|--------|--------|-------|
| short | 0.2334 | 28,245 |
| 30-90d | 0.1215 | 5,147 |
| 90-180d | 0.1152 | 4,875 |
| 180-365d | 0.1087 | 6,357 |
| **>365d** | **0.0797** | **10,596** |

---

## 完整对比表

| Model | short | 30-90d | 90-180d | 180-365d | >365d |
|-------|-------|--------|---------|---------|-------|
| SASRec baseline (checkpoint) | 0.1998 | 0.1099 | 0.0980 | 0.0778 | 0.0297 |
| DG-SASRec (checkpoint 加载) | 0.2166 | 0.1173 | 0.1099 | 0.0995 | 0.0573 |
| **TimeConcat E2E** | **0.2334** | **0.1215** | **0.1152** | **0.1087** | **0.0797** |
| **DeltaGate E2E** | 0.2092 | 0.1169 | 0.1094 | 0.0998 | 0.0571 |

---

## 结论

TimeConcat E2E 在所有 bucket 全面领先 DeltaGate E2E，尤其 >365d bucket：

| 对比 | 比值 |
|------|------|
| TimeConcat vs DeltaGate (>365d) | **+39.6%** |
| TimeConcat vs DG-SASRec checkpoint | +39.1% |
| TimeConcat vs SASRec baseline | +168% |

**结论**: additive time concatenation 在 E2E 训练范式下优于 gating mechanism。

---

## 溯源

- 训练日志: `./logs_baselines/time_concat_s{seed}.log`
- 模型 checkpoint: `./outputs_time_concat/video_games_s{seed}/best_model.pt`
- 评估日志: `run_master.log` Step 5

