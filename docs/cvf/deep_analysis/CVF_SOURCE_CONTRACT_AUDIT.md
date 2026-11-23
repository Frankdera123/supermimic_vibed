# CVF-0 Mathematical Contract & Source Audit

**Script**: [da01_inventory_and_contract.py](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da01_inventory_and_contract.py)  
**Artifact**: [da01_inventory_contract.json](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/da01_inventory_contract.json)

## 1. Exact Mathematical Definitions in CVF-0

1. **70-D Tangent Perturbation** (`cvf_core.perturb_state`):
   $$S \oplus \delta = (p + H(R)\delta p,\; R\exp(\delta\theta),\; v + R\delta v,\; w + R\delta w,\; q + \delta q,\; \dot q + \delta\dot q)$$
2. **Critic Response ($D_c$)**:
   - **Autograd at center**: `critic_tangent_grad` computes $\nabla_\delta V_\psi(S \oplus \delta)|_{\delta=0}^\top u$.
   - **Oracle 1 vs Oracle 2 clamp discrepancy**:
     - Oracle 1 (`cvf_analyze_all.py:196`) uses `grad_raw` (unclamped `V_raw`).
     - Oracle 2 (`cvf_analyze_all.py:263, 295`) uses `.clamp(max=10.0)` (`V_eff`).
     - **Observed impact**: On all 16 Oracle 1 `S_star` states and all 256 Oracle 2 `S_32` states, `V_raw < 10.0` (`max = 9.69` for `B0@2100_post`, `0/16` and `0/256` clamped). Thus the discrepancy has **zero numerical impact** on the primary Critic.
3. **World Model & Simulator Continuation ($D_m, D_s$)**:
   - **Oracle 1**: Finite-horizon closed-loop `pi_old` rollout of length $T \in \{32, 64, 96, 128\}$ from $S^* \oplus (\pm \epsilon u)$ with `last_only` observation history (`oh_prev9` unchanged, only current slot updated). **No terminal critic bootstrap**.
   - **Oracle 2**: Open-loop action sequence $A_{0:31} \pm \epsilon d_A$ for $H=32$, followed by closed-loop `pi_old` continuation for $T_{\text{cont}}=96$ from $S_{32}$.

## 2. Survival Indexing, Batching, and Independent Cross-Check

- **Survival indexing**: Verified in `supermimic_opcl_proto/supermimic/train.py:1136-1144` that training uses $c_0 = 1$, $c_t = \prod_{k=0}^{t-1}(1 - d_k)$, $J = \sum_t \gamma^t c_t r_t + \gamma^H c_H V(S_H)$. Both `cvf_core.py` and `cvf_analyze_all.py` match this **100%** (did NOT inherit the $c_{t+1} r_t$ bug from CRGF/PADA).
- **Independent WM rollout cross-check**: Re-implementing `rollout_wm_continuation` independently yields `J_max_abs_diff = 0.0` and `grad_max_abs_diff = 0.0` across `T = 32, 64, 96, 128`.
- **Batching consistency ($B=1$ vs $B=16$)**: Max diff is `1.43e-6` for Critic value, `7.69e-6` for Critic gradient, `2.15e-6` for WM $J_{64}$, `1.32e-4` for WM $\nabla J_{64}$ (fp32 precision only; zero cross-batch reduction bugs).

## 3. Protocol Deviations & Confounding Gaps

1. **Finite-horizon vs Infinite-horizon mismatch**: `D_crit` estimates $\nabla_s V^\pi_\infty(S^*)$ (trained with TD($\lambda=0.95$) and $\gamma=0.99$), whereas Oracle 1 `D_sim(T)` and `D_wm(T)` omit the terminal bootstrap $\gamma^T c_T V(S_T)$. At $T=32$, the omitted tail is `5.39` out of `7.60` (71%); even at $T=128$, the omitted tail is `1.95` out of `7.54` (26%).
2. **Protocol vs Report Pooling**: `CVF0_FINAL_REPORT.md` Table 2 reports `ALL16` at `eps=0.015` while `gated` uses pooled `(0.015, 0.030)`. Table 3 reports `ALL_POLICY (n=48)` by pooling `delta_2100`, `delta_2300`, `ascent` without gate filtering. Both are reproducible from `oracle1_sim.pt` and `oracle2_sim.pt`.
3. **Extended state**: Critic receives `(S, f, xi_n, wm_goal)` but **not** `o_hist` (10-step noisy observation history) or `a_prev`. Thus Critic lacks history/action-rate state variables that govern `pi_old` and `r_rate`.
