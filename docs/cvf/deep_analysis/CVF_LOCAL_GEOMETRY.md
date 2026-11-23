# CVF-0 Critic Local Value-Function Geometry

**Script**: [da03_geometry_and_attribution.py](file:///home/sustech/Desktop/supermimic_vibed/cvf_proto/deep_analysis/da03_geometry_and_attribution.py)  
**Artifact**: [da03_geometry_attribution.json](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/da03_geometry_attribution.json)

## 1. First-Order Geometry Across 6 Physical Tangent Blocks (`B0@2100_post`)

Comparing $\nabla_s V_\psi$ against $\nabla_s V_{\mathrm{WM}}(T=64)$ and $\nabla_s V_{\mathrm{WM}}(T=128)$ in both raw physical units and Bryson-normalized units (`NOMINAL_ERROR` from `networks.py`):

| Tangent Block | Dim | Bryson Scale | Raw Norm (Crit / WM64) | Bryson Norm (Crit / WM64 / WM128) | Cos(Crit, WM64) Mean / Med | Cos(Crit, WM128) Mean / Med |
|---|---:|---:|---:|---:|---:|---:|
| `root_pos` | 3 | 0.01 | 1.491 / 1.159 | 0.015 / 0.012 / 0.019 | +0.216 / +0.166 | +0.157 / +0.289 |
| `root_ori` | 3 | 0.02 | 0.773 / 1.367 | 0.015 / 0.027 / 0.042 | +0.211 / +0.061 | +0.321 / +0.562 |
| `root_linvel` | 3 | 0.10 | 0.373 / 0.554 | 0.037 / 0.055 / 0.089 | **+0.550 / +0.703** | **+0.539 / +0.696** |
| `root_angvel` | 3 | 0.30 | 0.046 / 0.074 | 0.014 / 0.022 / 0.035 | +0.242 / +0.506 | +0.325 / +0.433 |
| `joint_pos` | 29 | 0.10 | 0.656 / 0.973 | 0.066 / 0.097 / 0.148 | **+0.069 / -0.020** | **+0.080 / +0.107** |
| `joint_vel` | 29 | 3.14 | 0.062 / 0.077 | **0.195 / 0.241 / 0.375** | +0.310 / +0.318 | +0.331 / +0.303 |

- Notice two critical facts:
  1. In Bryson-normalized units, `joint_vel` actually has the **largest** gradient norm (`0.195`) of any block in the Critic, comparable to `WM64` (`0.241`), so the velocity mismatch is **not** simply "zero velocity gradient everywhere".
  2. However, the **directional alignment** is weak (`cos ~ 0.31` in `joint_vel`, `0.07` in `joint_pos`), and in `vel_reachable` specifically, `d1_abs_median` is tiny (`0.0213` vs `0.925` in `pose_safe_critic`).

## 2. Second-Order Geometry (Directional Curvature & Asymmetry)

Using exact directional Hessian $\kappa(u) = u^\top \nabla_s^2 V_\psi(s) u$:

| Direction | $|D_1|$ Med | $|\kappa|$ Med | $\frac{0.015|\kappa|}{2|D_1|}$ Med | Critic Asym ($\epsilon=0.015$) | Sim64 Asym ($\epsilon=0.015$) | Critic Taylor-1 Rel Err ($\epsilon=0.015$) |
|---|---:|---:|---:|---:|---:|---:|
| `pose_safe_critic` | 0.925 | 4.713 | 3.8% | 3.8% | 26.7% | 4.0% |
| `vel_reachable` | **0.021** | **0.0065** | **0.48%** | **1.1%** | **94.9%** | **1.0%** |

- **Key Finding**: The Critic itself is **extremely smooth and locally linear** along `vel_reachable` (`1.1%` asymmetry, `1.0%` Taylor-1 error across $\epsilon \in [0.005, 0.060]$). The mismatch in `vel_reachable` is **not** caused by excessive Critic curvature or value clamping (`0/16` clamped).
- Conversely, **the Simulator (`Sim64`) exhibits 94.9% plus/minus asymmetry** in `vel_reachable` at $\epsilon=0.015$, indicating that small velocity perturbations in `vel_reachable` trigger strong nonlinearities/contact-gait bifurcations in the real Simulator while the Critic remains nearly flat (`|D_1| = 0.021`).

![O1 Direction Scatter](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/figures/fig03_o1_direction_scatter.png)
![Epsilon Stability](file:///home/sustech/Desktop/supermimic_vibed/results/cvf/deep_analysis/figures/fig04_eps_stability.png)
