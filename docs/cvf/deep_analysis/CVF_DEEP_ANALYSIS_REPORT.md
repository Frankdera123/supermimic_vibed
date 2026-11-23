# CVF-0 Deep Mechanism Analysis — Independent Master Report

**Date**: 2026-11-19 | **Mode**: CPU-Only Offline Independent Audit | **Codebase**: `supermimic_opcl_proto` (`H=32, gamma=0.99, lam=0.95, V_MAX=10.0`)

## Executive Summary

An independent, CPU-only audit of all CVF-0 raw trajectories, 9 Critic checkpoints, and source code confirms several key empirical phenomena from `CVF0_FINAL_REPORT.md` **while overturning two central conclusions**:

1. **Counterfactual B (`WM_cont96`) overturns the "WM continuation is a reliable 1st-order teacher" claim**: On the reachable manifold (Oracle 2, `H=32 + T_cont=96`, $n=48$), replacing the learned Critic with 96-step closed-loop WM continuation yields `sign = 0.583`, `Spearman = +0.124` (vs `0.562` / `+0.078` for the learned Critic) against true Simulator total return. Meanwhile, replacing it with Simulator Oracle continuation jumps to **`sign = 0.958`, `Spearman = +0.954`, `r = +0.995`**.
2. **MSE error decomposition overturns "Error A dominates everywhere"**: While Critic fails directionally in `vel_reachable` (`sign = 0.31–0.38`, `sp ~ 0`), **in MSE/RMSE terms `e_B = D_m - D_s` accounts for 97–98% of the squared error** in `vel_reachable` (`rmseB = 0.244–0.684` vs `rmseA = 0.043–0.101`).
3. **Simulator bifurcation & Critic flatness**: In `vel_reachable`, the Critic is smooth and locally linear (`1.1%` asymmetry, `0/16` clamped), whereas the **Simulator itself exhibits 94.9% plus/minus asymmetry** at $\epsilon=0.015$.

---

## Direct Answers to the 10 Research Questions

1. **Is the CVF-0 implementation and evaluation contract reliable?**  
   **PARTIALLY / YES WITH BOUNDARIES.** Survival indexing ($c_t r_t$) matches training 100%, `repeat_range = 0.0`, and $B=1$ vs $B=16$ batching is exact (`<1.3e-4`). Minor protocol deviations exist: Oracle 1 uses `grad_raw` while Oracle 2 uses `clamp(max=10.0)` (zero numerical impact here since `0/16` and `0/256` states reach `10.0`), and Oracle 1 compares an infinite-horizon Critic against unbootstrapped finite-horizon $T$-step returns (omitting `26%–71%` of value tail).
2. **Is the Critic's scalar value accurate, well-calibrated, or merely correlated?**  
   **Accurate AND well-calibrated** once the $\gamma^T c_T V(S_T)$ tail is included: vs `V_wm(128, bootstrapped)`, `B0@2100_post` achieves $r = 0.977$, `RMSE = 0.339`, `bias = +0.144`, `slope = 1.089` (vs `V_wm(32, boot)`: $r = 0.984$, `RMSE = 0.263`, `bias = +0.087`).
3. **Does the Critic have a systematic local value-gradient mismatch?**  
   **YES, within-start directional sensitivity collapses relative to global value accuracy.** Global $r = 0.987$ drops to within-start 5-direction `Spearman = 0.531` ($T=64$) / `0.325` ($T=128$) and demeaned `Pearson = 0.559` ($T=64$) / `0.379` ($T=128$).
4. **Is the mismatch concentrated in velocity and policy-reachable directions?**  
   **YES.** `pose_safe_critic` achieves `sign = 0.81–1.00` ($p=0.021$), `vel_critic` achieves `0.69–0.88`, while `vel_reachable` drops to `0.31–0.38` ($T=32, 64$, `p=0.454`, `Sp ~ 0`). Furthermore, `>90%` of the Euclidean energy of $dS_{32} = J_{\mathrm{WM},A} dA$ lies in velocity coordinates (`joint_vel` + `root_angvel` + `root_linvel`).
5. **How does the mismatch evolve across training (9 Critics)?**  
   On the baseline lineage (`1000 -> 2300`), scalar value $r$ rises (`0.856 -> 0.990`), `pose_safe_critic` and `vel_critic` plateau by iter `1500–2000` (`0.81` and `0.75`), while **`vel_reachable` (`0.38–0.50`, `Sp ~ 0`) and Oracle 2 `TrueTot` (`0.48–0.56`) never improve**—the mismatch exists from iter 1000.
6. **What are the most likely structural causes, and what remains indistinguishable?**  
   Observed structural contributors: (i) **Feature asymmetry** (`mdp.critic_features` includes `fk_body_pos_err`, `q_err`, `goal_rel_pos` but **zero** velocity tracking error features), (ii) **Normalizer + MLP suppression** (`joint_vel` has `std=1.19` vs `0.10–0.19` for pose/height AND 10× smaller MLP-space gradient RMS `0.013` vs `0.128–0.167`), and (iii) **Simulator reference instability** (`94.9%` plus/minus asymmetry in `vel_reachable`). Indistinguishable offline: exact causal share of zeroth-order TD loss (H1) vs feature/normalizer design (H2/H3) vs imagined data coverage (H5).
7. **Is World Model continuation reliable enough to serve as a 1st-order teacher for the Critic?**  
   **NO (CONTRADICTED on Oracle 2 reachable manifold).** While WM continuation preserves sign in Oracle 1 `vel_reachable` (`0.69–0.81`), in Oracle 2 (`H=32 + T_cont=96`) `WM_cont96` vs `Sim_cont96` achieves only `sign = 0.583` (`Sp = +0.164`), and Counterfactual B (`0.583`) fails to improve over the learned Critic (`0.562`).
8. **Does Critic error genuinely harm the optimization direction relative to the same physical objective?**  
   **YES.** Against true Simulator total return (`n=48`), `A_learned_critic` achieves `sign = 0.562` (`p=0.471`), `Sp = +0.078`, worse than `C_prefix_only` (`0.604`, `Sp = +0.133`) and far below `D_sim_oracle_cont` (`0.958`, `Sp = +0.954`).
9. **If the Critic's directional response is replaced, does the theoretical control gradient improve?**  
   **YES if replaced with an accurate continuation oracle (D: `0.562 -> 0.958`, `Sp: +0.078 -> +0.954`), NO if replaced with 96-step WM continuation (B: `0.562 -> 0.583`, `Sp: +0.078 -> +0.124`).**
10. **What is the single most worthwhile mechanism to verify next, and why?**  
    **Simulator-Anchored Critic Supervision (Route B)** (optionally paired in a 2×2 ablation with velocity tracking-error features, Route D), because Counterfactual D proves the 32-step WM prefix is already sufficient (`95.8%` sign) when paired with Sim-accurate continuation gradients, whereas WM continuation alone (Route A) drifts too much over long horizons (`128` steps).

---

## Independent Verdict Table

| Research Proposition | Evidence Grade | Primary Evidence | Key Limitations | Next-Stage Recommendation |
|---|---|---|---|---|
| **CVF-0 Contract & Determinism** | **SUPPORTED** | `repeat_range=0.0`; `B=1` vs `B=16` diff `<1.3e-4`; survival $c_t r_t$ matches `train.py`. | Finite-horizon $T$ omits 26–71% tail in O1; `B1/B2/B3` common-policy only. | Standardize bootstrapped vs unbootstrapped comparison. |
| **Critic Scalar Value Accuracy & Calibration** | **SUPPORTED** | `B0@2100_post` vs `V_wm128_boot`: $r=0.977$, `RMSE=0.339`, `bias=+0.144`, `slope=1.089`. | 16 starts, nominal $\xi$, `opcl_proto` only. | No scalar calibration fix needed. |
| **Systematic Local Gradient Mismatch (Velocity / Reachable)** | **SUPPORTED** | Demeaned `r=0.38–0.56`; `vel_reachable` `sign=0.31–0.38` (`p=0.454`); `Q4=0.583`. | `Sim64` has `94.9%` asym in `vel_reachable`; `n_gated=3–4/16`. | Address both Critic sensitivity & Sim reference noise. |
| **H-CV2: In-Model Sobolev Error (A) Solely Dominates** | **PARTIALLY SUPPORTED / REFINED** | `C vs Sim` fails directionally (`0.38`) while `WM vs Sim` holds (`0.69–0.81`); **HOWEVER** in MSE, `e_B` is **97–98%** of `vel_reachable` MSE. | Horizon truncation confounds `e_A` in O1. | Treat as **H-CV4 (composite)**: Critic direction + WM long-horizon drift. |
| **WM Continuation Reliable as 1st-Order Teacher (Route A)** | **CONTRADICTED** (on O2) | Counterfactual B (`WM_cont96`): `sign=0.583`, `Sp=+0.124` (vs Critic `0.562`); `WM_cont96 vs Sim_cont96` `sign=0.583`. | Tested at `H=32 + T=96`; shorter $T$ unprobed for distillation. | Do **not** pursue pure WM-to-Critic distillation. |
| **Accurate Continuation Rescues Control Gradient** | **SUPPORTED** | Counterfactual D (`Sim_cont96` oracle): **`sign=0.958`, `Sp=+0.954`, `r=+0.995`**. | Oracle upper bound, not a trainable Critic. | Pursue **Route B (Simulator-Anchored Critic)**. |

---

## Sub-Reports & Artifacts Index

- [CVF_DATA_INTEGRITY.md](file:///home/sustech/Desktop/supermimic_vibed/docs/cvf/deep_analysis/CVF_DATA_INTEGRITY.md) | [CVF_SOURCE_CONTRACT_AUDIT.md](file:///home/sustech/Desktop/supermimic_vibed/docs/cvf/deep_analysis/CVF_SOURCE_CONTRACT_AUDIT.md)
- [CVF_VALUE_CALIBRATION.md](file:///home/sustech/Desktop/supermimic_vibed/docs/cvf/deep_analysis/CVF_VALUE_CALIBRATION.md) | [CVF_LOCAL_GEOMETRY.md](file:///home/sustech/Desktop/supermimic_vibed/docs/cvf/deep_analysis/CVF_LOCAL_GEOMETRY.md)
- [CVF_FEATURE_ATTRIBUTION.md](file:///home/sustech/Desktop/supermimic_vibed/docs/cvf/deep_analysis/CVF_FEATURE_ATTRIBUTION.md) | [CVF_CHECKPOINT_EVOLUTION.md](file:///home/sustech/Desktop/supermimic_vibed/docs/cvf/deep_analysis/CVF_CHECKPOINT_EVOLUTION.md)
- [CVF_ERROR_DECOMPOSITION.md](file:///home/sustech/Desktop/supermimic_vibed/docs/cvf/deep_analysis/CVF_ERROR_DECOMPOSITION.md) | [CVF_POLICY_GRADIENT_PROPAGATION.md](file:///home/sustech/Desktop/supermimic_vibed/docs/cvf/deep_analysis/CVF_POLICY_GRADIENT_PROPAGATION.md)
- [CVF_COUNTERFACTUAL_ANALYSIS.md](file:///home/sustech/Desktop/supermimic_vibed/docs/cvf/deep_analysis/CVF_COUNTERFACTUAL_ANALYSIS.md) | [CVF_HYPOTHESIS_MATRIX.md](file:///home/sustech/Desktop/supermimic_vibed/docs/cvf/deep_analysis/CVF_HYPOTHESIS_MATRIX.md)
- [CVF_NEXT_RESEARCH_DECISION.md](file:///home/sustech/Desktop/supermimic_vibed/docs/cvf/deep_analysis/CVF_NEXT_RESEARCH_DECISION.md)
- Scripts: [da01](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da01_inventory_and_contract.py), [da02](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da02_scalar_and_local_response.py), [da03](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da03_geometry_and_attribution.py), [da04](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da04_evolution_and_error_decomp.py), [da05](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da05_propagation_and_counterfactual.py), [da06](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da06_clip_stats_and_plots.py)
- Figures (`results/cvf/deep_analysis/figures/`): `fig01`–`fig10` (all generated from real data).
