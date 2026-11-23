# CVF-0 Nine-Critic Checkpoint Evolution

**Script**: [da04_evolution_and_error_decomp.py](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da04_evolution_and_error_decomp.py)  
**Artifacts**: [da04_evolution_and_decomp.json](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/da04_evolution_and_decomp.json), [checkpoint_evolution.csv](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/checkpoint_evolution.csv)

## 1. Common-Policy Evaluation Boundary

All 9 Critics are evaluated on the **same fixed evaluation set** generated under `pi_old@2100` (`online_B0`). Thus:
- `b0@1000 -> b0@1500 -> b0@2000 -> B0@2100_pre -> B0@2100_post -> B0@2300` represents a **single baseline lineage** (with `B0@2100_post` being matched-policy).
- `B1@2300`, `B2@2300`, `B3@2300` are **parallel experimental branches** forking from `b0@2000` (common-policy evaluation only; matched-policy Simulator rollouts for `B1/B2/B3` were not collected).

## 2. Longitudinal Metrics Table

| Critic | `val_r` (Sim128) | `pose_c` Sign64 | `vel_c` Sign64 | `vel_r` Sign64 (Sp) | O2 `Q4` Sign | O2 `TrueTot` Sign (Sp) |
|---|---:|---:|---:|---:|---:|---:|
| `b0@1000` | 0.856 | 0.69 | 0.62 | 0.50 (+0.05) | 0.56 | 0.54 (+0.15) |
| `b0@1500` | 0.963 | 0.81 | 0.62 | 0.50 (-0.03) | 0.58 | 0.48 (+0.09) |
| `b0@2000` | 0.963 | 0.81 | 0.75 | 0.44 (+0.00) | 0.56 | 0.50 (+0.03) |
| `B0@2100_pre` | 0.986 | 0.81 | 0.75 | 0.44 (-0.02) | 0.62 | 0.54 (+0.08) |
| `B0@2100_post` | 0.987 | 0.81 | 0.75 | 0.38 (-0.01) | 0.58 | 0.56 (+0.08) |
| `B0@2300` | 0.990 | 0.81 | 0.75 | 0.44 (+0.05) | 0.65 | 0.50 (+0.09) |
| `B1@2300` | 0.988 | 0.81 | 0.75 | 0.38 (-0.03) | 0.65 | 0.52 (+0.07) |
| `B2@2300` | 0.976 | 0.81 | 0.75 | 0.56 (+0.12) | 0.60 | 0.54 (+0.10) |
| `B3@2300` | 0.982 | 0.81 | 0.75 | 0.62 (+0.03) | 0.60 | 0.52 (+0.06) |

## 3. Key Longitudinal Findings

1. **Scalar value accuracy improves monotonically** (`r = 0.856 -> 0.990` on baseline).
2. **`pose_safe_critic` and `vel_critic` improve early and plateau** by iter 1500–2000 (`0.81` and `0.75`).
3. **`vel_reachable` never learns** on the baseline lineage (`0.50 -> 0.50 -> 0.44 -> 0.38 -> 0.44`, Spearman $\approx 0$ throughout). The mismatch is already present at iter 1000 and does not resolve with more scalar TD($\lambda$) training.
4. **Oracle 2 `TrueTot` remains near chance (`0.48–0.56`, Spearman `+0.03` to `+0.15`)** across all 9 Critics and all 4 branches (`B0/B1/B2/B3`).

![Checkpoint Evolution](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/figures/fig07_checkpoint_evolution.png)
