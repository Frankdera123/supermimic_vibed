# CVF-0 Policy Gradient Propagation through World Model

**Script**: [da05_propagation_and_counterfactual.py](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da05_propagation_and_counterfactual.py)  
**Artifact**: [da05_prop_and_cf.json](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/da05_prop_and_cf.json)

## 1. Controllable Terminal State Subspace $dS_{32} = J_{\mathrm{WM}, A} \, dA$

Using exact JVP of the 32-step WM open-loop rollout along the 4 Oracle 2 action directions (`delta_2100`, `delta_2300`, `ascent`, `random`), the Euclidean energy distribution of $dS_{32}$ in 70-D tangent space is:

| Action Direction | $\|dS_{32}\|$ Mean | `root_pos` | `root_ori` | `root_linvel` | `root_angvel` | `joint_pos` | `joint_vel` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `delta_2100` | 0.74 | 0.002 | 0.011 | 0.029 | 0.089 | 0.049 | **0.820** |
| `delta_2300` | 1.09 | 0.001 | 0.010 | 0.022 | 0.075 | 0.060 | **0.832** |
| `ascent` | 3.86 | 0.001 | 0.005 | 0.012 | 0.071 | 0.011 | **0.900** |
| `random` | 0.88 | 0.000 | 0.000 | 0.001 | 0.063 | 0.002 | **0.934** |

- **Over 90% of the Euclidean energy** of the WM-propagated terminal state perturbation $dS_{32}$ lies in **velocity coordinates** (`joint_vel` + `root_angvel` + `root_linvel`: `93.8%–99.8%`).
- Note that in Bryson-normalized units (`joint_vel` scale `3.14` vs `joint_pos` `0.1`), `joint_pos` carries more physical weight than its raw `rad` norm suggests, but velocity still dominates the controllable terminal state tangent—precisely the subspace where the Critic's directional fidelity is weakest.

## 2. Prefix vs Continuation Weighting & Conflict (`ALL_POLICY`, $n=48$)

- In `D_sim_total_true = D_sim_prefix + D_sim_cont96`:
  - `D_wm_prefix` alone has magnitude ratio **`0.08×`** of `D_sim_total_true` (`rmse = 0.45`), meaning **~92% of the total return gradient magnitude comes from the post-$H=32$ continuation**!
  - `D_wm_prefix` agrees in sign with `D_sim_state` (`0.71`, Spearman `+0.51`), and with `D_sim_total_true` (`0.604`, Spearman `+0.133`).
  - `D_wm_prefix` vs `D_wm_critic` sign agreement is `0.50–0.65`; adding the learned Critic (`A`) **reduces** total-return sign agreement from `0.604` (`C: prefix-only`) to `0.562` (`A: learned Critic`), and reduces Spearman from `+0.133` to `+0.078`.
- **Adam Optimizer Caveat**: `delta_2100` and `delta_2300` are actual policy-update action shifts (which include Adam preconditioning), whereas `ascent` is raw WM+Critic gradient ascent. Full parameter-space Adam `m_t / sqrt(v_t)` cannot be reconstructed from action directions alone.
