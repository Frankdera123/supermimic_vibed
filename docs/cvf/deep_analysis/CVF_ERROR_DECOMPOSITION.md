# CVF-0 Error Decomposition: $e_A = D_c - D_m$ vs $e_B = D_m - D_s$

**Script**: [da04_evolution_and_error_decomp.py](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da04_evolution_and_error_decomp.py)  
**Artifact**: [da04_evolution_and_decomp.json](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/da04_evolution_and_decomp.json)

## 1. Exact MSE Identity (ALL16, $\epsilon=0.015$, `B0@2100_post`)

Using $e_{\mathrm{tot}} = e_A + e_B$ where $e_A = D_c - D_m$ and $e_B = D_m - D_s$, we verify $\mathbb{E}[e_{\mathrm{tot}}^2] = \mathbb{E}[e_A^2] + \mathbb{E}[e_B^2] + 2\mathbb{E}[e_A e_B]$ (identity residual $< 10^{-7}$):

| Horizon | Direction | $\mathrm{RMSE}(e_A)$ | $\mathrm{RMSE}(e_B)$ | Share $e_A^2 / (e_A^2+e_B^2)$ | Interaction $\frac{2\mathbb{E}[e_A e_B]}{\mathbb{E}[e_A^2+e_B^2]}$ | $r(e_A, e_B)$ |
|---|---|---:|---:|---:|---:|---:|
| $T=64$ | `pose_safe_critic` | **0.665** | 0.246 | **0.88** | -0.18 | -0.69 |
| $T=64$ | `pose_safe_random` | 0.244 | 0.252 | 0.48 | -0.08 | -0.08 |
| $T=64$ | `vel_critic` | 0.321 | 0.343 | 0.47 | +0.20 | +0.24 |
| $T=64$ | `vel_reachable` | **0.043** | **0.244** | **0.03** | -0.04 | -0.12 |
| $T=64$ | `vel_random` | **0.065** | **0.352** | **0.03** | +0.06 | +0.11 |
| $T=128$ | `pose_safe_critic` | 0.644 | 0.665 | 0.48 | +0.09 | +0.21 |
| $T=128$ | `vel_critic` | 0.509 | 1.142 | 0.17 | -0.29 | -0.32 |
| $T=128$ | `vel_reachable` | **0.101** | **0.684** | **0.02** | -0.14 | -0.41 |
| $T=128$ | `vel_random` | **0.142** | **0.683** | **0.04** | -0.06 | -0.16 |

## 2. Critical Nuance: Directional Failure vs Magnitude-MSE Dominance

This decomposition reveals a crucial nuance that `CVF0_FINAL_REPORT.md` missed by focusing solely on sign agreement:
1. **In MSE / RMSE terms**, `e_B` (WM-to-Sim gap) dominates `vel_reachable` (`97–98%` of MSE!) and `vel_random` (`96–97%` of MSE), because `D_sim` has larger magnitude and variance while both `D_crit` and `D_wm` underestimate velocity slopes (`ratio = 0.08–0.35x`).
2. **In directional terms (sign / rank)**, `WM` still preserves the sign of `D_sim` (`0.69–0.81` in `vel_reachable`) while `Critic` does not (`0.31–0.38`).
3. **Horizon confounding**: Because `D_m` omits the terminal bootstrap $\gamma^T c_T V(S_T)$, `e_A` mixes Critic fitting error with finite-horizon truncation error. Adding the bootstrap tail slightly improves `C vs WM` in `vel_reachable` (`0.44 -> 0.50`) but does not rescue it.
4. **Error compensation**: `r(e_A, e_B)` is negative in some regimes (e.g., `-0.69` in `pose_safe_critic` T64, `-0.41` in `vel_reachable` T128) and positive in others (`+0.24` in `vel_critic` T64).

![Error A vs Error B](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/figures/fig08_error_A_vs_B.png)
![Clip Fidelity Heatmap](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/figures/fig09_clip_fidelity_heatmap.png)
