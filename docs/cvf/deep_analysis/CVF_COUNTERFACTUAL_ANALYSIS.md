# CVF-0 Counterfactual Critic Replacement Analysis

**Script**: [da05_propagation_and_counterfactual.py](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da05_propagation_and_counterfactual.py)  
**Artifacts**: [da05_prop_and_cf.json](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/da05_prop_and_cf.json), [counterfactual_per_start.csv](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/counterfactual_per_start.csv)

## 1. Unified Evaluation Against True Simulator Total Return

On the exact same Oracle 2 samples (`ALL_POLICY`, $n=48$: 16 starts × `{delta_2100, delta_2300, ascent}`), using the **same** Simulator Total-Return reference $D_{\text{sim,tot}}^{\text{true}} = D_{\text{sim,state}} + D_{\text{sim,direct}} + D_{\text{sim,cont}}(T_{\text{cont}}=96)$:

| Counterfactual | Definition | Sign Agree [Clip CI95] | Exact $p$ | Spearman | Pearson $r$ | Med Ratio | RMSE |
|---|---|---|---:|---:|---:|---:|---:|
| **A. Learned Critic** | $D_{\text{WM,pref}} + D_{\text{WM,critic}}$ | 0.562 [0.396, 0.708] | 0.471 | +0.078 | +0.138 | 0.48 | 0.46 |
| **B. WM Continuation (96)** | $D_{\text{WM,pref}} + D_{\text{WM,cont96}}$ | **0.583** [0.437, 0.729] | 0.312 | **+0.124** | +0.097 | 0.54 | **0.50** |
| **C. No Critic (Prefix)** | $D_{\text{WM,pref}}$ | 0.604 [0.396, 0.812] | 0.193 | +0.133 | +0.336 | 0.08 | 0.45 |
| **D. Sim Oracle Cont (96)** | $D_{\text{WM,pref}} + D_{\text{sim,cont96}}$ | **0.958** [0.896, 1.000] | <0.0001 | **+0.954** | **+0.995** | **0.98** | **0.05** |

### Continuation-Only Comparison ($D_{\text{cont}}$ vs $D_{\text{sim,cont96}}$)
- `WM_cont96` vs `Sim_cont96`: `sign = 0.583`, `Spearman = +0.164`.
- `Learned_Crit` vs `Sim_cont96`: `sign = 0.562`, `Spearman = +0.154`.

## 2. Core Mechanism Verdict from Counterfactuals

1. **If Critic is replaced by an accurate continuation oracle (D)**, the existing 32-step WM prefix produces **`95.8%` sign agreement and `0.954` Spearman** with the true Simulator total-return gradient. Thus the WM prefix ($H=32$) is **not** the primary bottleneck when paired with an accurate continuation value gradient.
2. **HOWEVER, replacing Critic with 96-step WM continuation (B) FAILS to rescue the optimization direction**: sign agreement moves only from `0.562` to `0.583` (`p=0.312`, indistinguishable from chance), Spearman moves from `+0.078` to `+0.124`, and RMSE worsens (`0.46 -> 0.50`).
3. **Why?** Because after 32 open-loop steps + 96 closed-loop WM steps (= 128 total WM steps), the WM itself has drifted off the true Simulator manifold (`WM_cont96` vs `Sim_cont96` has only `0.583` sign agreement and `+0.164` Spearman!).
4. **Implication for Sobolev / Gradient Distillation**: Simply distilling 96-step WM continuation gradients into the Critic (Route A) **cannot** be expected to fix the Oracle 2 policy-gradient bottleneck, because WM continuation at $H=32 + T=96$ is nearly as misaligned with `Sim_cont96` as the learned Critic is.

![Counterfactual Replacement](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/figures/fig10_counterfactual_replacement.png)
