# CVF-0 Next Research Mechanism Decision

## 1. Evaluation of Candidate Routes (A–F)

- **Route A: World Model-to-Critic Gradient Distillation (Sobolev Learning using WM Continuation)** — **NOT RECOMMENDED as standalone next step.**
  - *Evidence*: Counterfactual B proves that replacing the learned Critic with 96-step closed-loop WM continuation (`B_wm_continuation`) yields `sign = 0.583`, `Spearman = +0.124` (vs `0.562` / `+0.078` for the learned Critic) against true Simulator total return. At $H=32 + T=96$, WM continuation itself has drifted (`WM_cont96 vs Sim_cont96`: `sign = 0.583, sp = +0.164`). Distilling an inaccurate teacher will not solve Oracle 2's bottleneck.
- **Route B: Simulator-Anchored Critic (Real-Trajectory / Multi-Step Sim-Anchored Value & Gradient Supervision)** — **RECOMMENDED (Primary Candidate), subject to single-variable verification.**
  - *Evidence*: Counterfactual D (`D_sim_oracle_cont`) achieves **`sign = 0.958`, `Spearman = +0.954`, `r = +0.995`** using the exact same 32-step WM prefix! This proves that if the Critic's continuation value gradient matches real Simulator continuation, the existing WM prefix already produces a near-ideal policy optimization direction.
  - *Why it avoids OPCL/PADA pitfalls*: It directly anchors the Critic's value/gradient target to real Simulator rollouts (which are already collected in the buffer) rather than trusting long-horizon imagined WM rollouts.
- **Route C: History-Conditioned Critic** — **Low Priority.**
  - `o_hist` and `a_prev` are missing from `critic_features`, but scalar $V$ RMSE is already `0.26–0.34` and Oracle 1 `vel_reachable` fails even with fixed `oh_prev9`.
- **Route D: Critic Feature / Normalization Redesign** — **Secondary / Complementary.**
  - Adding `qd_err` / `body_vel_err` and addressing `joint_vel` normalizer scaling (`std=1.19` vs `0.10–0.19`) is a clean, low-cost structural improvement that can be tested independently or alongside Route B.
- **Route E: Adaptive Bootstrap** — **Insufficient.**
  - Prefix-only (`C_prefix_only`, `no_critic`) gets `sign = 0.604`, `ratio = 0.08` on `H=32`; dropping the Critic without lengthening $H$ (e.g., to $H=64$) loses 92% of the total return gradient magnitude.
- **Route F: Keep Original Critic Unchanged** — **Rejected.**
  - The learned Critic degrades total-return gradient Spearman from `+0.51` (prefix) / `+0.954` (Oracle D) down to `+0.078` (`A`).

## 2. Recommended Single Next Mechanism to Verify

**Simulator-Anchored Value/Gradient Critic Supervision (Route B)**, paired in a clean 2×2 single-variable ablation with **Velocity Tracking-Error Feature Augmentation (Route D)**:
1. Use real on-policy Simulator trajectories already in the replay buffer to compute multi-step discounted return targets (and/or finite-difference / short-horizon trajectory-pair value differences) so Critic is anchored to real Sim dynamics rather than 32-step WM rollouts alone.
2. Keep `H=32`, `gamma=0.99`, `pi_old/critic` architecture fixed for clean single-variable comparison against `online_B0`.
