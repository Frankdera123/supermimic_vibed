# CVF-0 Value Calibration: Global Scalar vs Within-Start Local Response

**Script**: [da02_scalar_and_local_response.py](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da02_scalar_and_local_response.py)  
**Artifacts**: [da02_scalar_and_local.json](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/da02_scalar_and_local.json), [scalar_calibration_per_start.csv](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/scalar_calibration_per_start.csv)

## 1. Global Scalar Value Calibration (`B0@2100_post`, 16 Starts)

| Reference | Mean Pred | Mean Ref | Pearson $r$ | Spearman $\rho$ | RMSE | Bias | Cal Slope | Cal Intercept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `V_sim(128, soft, unboot)` | 7.686 | 5.583 | **0.987** | 0.956 | 2.125 | +2.104 | 0.782 | -0.431 |
| `V_wm(128, boot_post)` | 7.686 | 7.543 | **0.977** | 0.962 | **0.339** | +0.144 | **1.089** | -0.827 |
| `V_wm(32, boot_post)` | 7.686 | 7.599 | **0.984** | — | **0.263** | +0.087 | — | — |

**Conclusion on Scalar Calibration**: Once the $\gamma^T c_T V(S_T)$ truncation tail is accounted for, `B0@2100_post` is **both highly correlated ($r=0.977\text{–}0.987$) and well-calibrated** (bias `+0.09` to `+0.14`, RMSE `0.26–0.34`, calibration slope `1.09`). It is NOT merely correlated with an arbitrary offset.

## 2. Within-Start Local Response ($\Delta V_c$ vs $\Delta G_s$)

Across the 5 Oracle 1 directions within each of the 16 starts (`eps=0.015`):
- **Critic FD vs Autograd**: Max diff `0.0033`, RMSE `0.00057` (Critic is locally linear).
- **Within-start 5-direction Spearman** (`B0@2100_post` vs `Sim64`): mean **`0.531`** (median `0.650`); vs `Sim128`: mean **`0.325`**. (WM64 vs Sim64: mean `0.663`).
- **Demeaned 80-point Pearson** (removing start-level mean, `eps=0.015`):
  - `Critic` vs `Sim64`: **`0.559`**; vs `Sim128`: **`0.379`**.
  - `WM64` vs `Sim64`: **`0.743`**.
- **Pooled 80-point Sign Agreement** (`eps=0.015`): `Critic` vs `Sim64` = **`0.613`**, vs `Sim128` = **`0.650`** (vs `WM64` vs `Sim64` = `0.763`).

**Key Insight**: Global scalar accuracy ($r=0.987$) reflects cross-clip/cross-state value differences across the 16 starts. Within a fixed start, directional sensitivity drops sharply (`r = 0.38–0.56`, sign `0.61–0.65`), with severe failure concentrated in `vel_reachable` (`sign = 0.31` at $T=32$, `0.38` at $T=64$, `p = 0.454` vs chance).

![Value Prediction vs Sim](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/figures/fig01_value_pred_vs_sim.png)
![Global vs Local Accuracy](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/figures/fig02_global_vs_local_accuracy.png)
