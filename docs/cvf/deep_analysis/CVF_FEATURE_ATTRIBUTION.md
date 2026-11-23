# CVF-0 Critic Feature Attribution & Normalizer Diagnostics

**Script**: [da03_geometry_and_attribution.py](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da03_geometry_and_attribution.py)  
**Artifacts**: [da03_geometry_attribution.json](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/da03_geometry_attribution.json), [feature_attribution.csv](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/feature_attribution.csv)

## 1. Exact 938-D Feature Index Map

Verified directly from `supermimic_opcl_proto/supermimic/mdp.py:159-174` and `motion.py:111-131`:
- `0`: `height` ($p_z$, 1-D)
- `1..6`: `tilt6d` (6-D)
- `7..9`: `root_linvel_b` ($R^\top v$, 3-D)
- `10..12`: `root_angvel_b` ($R^\top w$, 3-D)
- `13..41`: `joint_pos` ($q$, 29-D)
- `42..70`: `joint_vel` ($\dot q$, 29-D)
- `71..99`: `joint_pos_err` ($q_{\text{ref}} - q$, 29-D)
- `100..141`: `fk_body_pos_err` (14×3 = 42-D)
- `142..144`: `root_pos_err` (3-D)
- `145..177`: `xi_n` (33-D)
- `178..907`: 10 goal frames × 73-D (`goal_ref_only` 67-D per frame + `goal_rel_ori` 6-D per frame)
- `908..937`: `goal_rel_pos` (10×3 = 30-D)

## 2. Feature-Group Gradient Attribution $D_g(u) = \nabla_{x_g} V_\psi^\top J_{\phi_g} u$

Exact JVP decomposition verified with $\max |D_c(u) - \sum_g D_g(u)| \le 5.96 \times 10^{-7}$:

| Direction | Net/Sum Ratio | Top Contributing Feature Groups (Mean $|D_g|$) |
|---|---:|---|
| `pose_safe_critic` | 0.917 | `goal_rel_pos` (0.65), `joint_pos` (0.13), `goal_rel_ori` (0.08), `root_pos_err` (0.07) |
| `pose_safe_random` | **0.353** | `goal_rel_pos` (0.15), `joint_pos` (0.08), `root_linvel_b` (0.05), `goal_rel_ori` (0.05) — strong cross-group cancellation |
| `vel_critic` | 1.000 | `root_linvel_b` (0.36), `joint_vel` (0.01), `root_angvel_b` (0.01) |
| `vel_reachable` | 0.743 | `root_linvel_b` (0.02), `root_angvel_b` (0.01), `joint_vel` (0.01) |
| `vel_random` | 0.814 | `root_linvel_b` (0.05), `joint_vel` (0.01), `root_angvel_b` (0.01) |

- **Velocity-direction structure**: Note that `mdp.critic_features` includes **no** body velocity error (`fk_body_vel_err`) and **no** joint velocity tracking error (`qd_ref - qd`). The only velocity inputs are `root_linvel_b`, `root_angvel_b`, and raw `joint_vel`.
- **Pose-direction dominance**: `goal_rel_pos` (30-D) + `fk_body_pos_err` + `root_pos_err` + `joint_pos` provide strong immediate tracking-error signals for pose perturbations.

## 3. Normalizer & MLP-Input Gradient Diagnostics (`B0@2100_post`)

| Feature Group | `norm.std` Mean | Max $|z|$ | MLP Grad RMS ($|\nabla_{\tilde x} V|$) | Raw Grad RMS ($|\nabla_x V|$) |
|---|---:|---:|---:|---:|
| `height` | 0.191 | 0.58 | **0.1671** | **0.8768** |
| `tilt6d` | 0.198 | 1.10 | **0.1182** | **0.6243** |
| `root_linvel_b` | 0.448 | 2.32 | **0.1276** | 0.2400 |
| `fk_body_pos_err` | 0.100 | 2.48 | 0.0354 | **0.5566** |
| `root_pos_err` | 0.720 | 0.70 | 0.0765 | 0.1060 |
| `goal_rel_pos` | 0.811 | 1.15 | 0.0535 | 0.0590 |
| `root_angvel_b` | **1.076** | 1.60 | **0.0284** | **0.0277** |
| `joint_vel` | **1.194** | 4.48 | **0.0131** | **0.0122** |

- **Dual Bottleneck in `joint_vel` and `root_angvel_b`**:
  1. **MLP sensitivity**: Even in *normalized* $\tilde x$ space, the Critic MLP places 9–13× smaller gradient RMS on `joint_vel` (`0.0131`) than on `height` (`0.1671`) or `root_linvel_b` (`0.1276`).
  2. **Normalizer scaling**: Because `joint_vel` has larger variance (`std = 1.194` vs `0.100–0.198` for pose/height), dividing by `std` reduces the physical-space gradient RMS by another ~6–10× (`0.0122` vs `0.557–0.877`).
- No extreme OOD explosion is observed on `S_star` (`z_gt3_frac` is small; `goal_ref_only` has `z_max = 7.94`, which has zero derivative w.r.t. robot state `S`).

![Feature Attribution](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/figures/fig05_feature_attribution.png)
![Normalizer & Physical Grad](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/figures/fig06_normalizer_and_physical_grad.png)
