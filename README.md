# SuperMimic reproduction on Isaac Lab 3 (kit-less, MuJoCo-Warp)

**Unofficial** reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
Model* (arXiv 2610.28480) — no official code is released, so everything here is implemented from
the paper. Target experiment: the controlled LAFAN1 comparison (Fig. 4a): 40 G1-retargeted LAFAN1
clips (2.45 h), one GPU; the paper reaches evaluation normalized return 0.82 after 1.2 h on an H100.

Author: **Chong Zhang** (ETH Zurich). License: [MIT](LICENSE).

> **Disclaimer:** this repository was speed-vibecoded in 1 hour with an AI coding
> assistant. It
> deliberately does not live under any lab's organization, because the code is too much "slop"
> for that: fast and unreviewed. Use it at
> your own risk.

> **One G1 motion tracker for all 40 LAFAN1 clips, trained in only ~1.5 GPU-hours**
> (1.47 h on a single RTX PRO 6000): evaluation return **0.829** with **0/40 failed clips**.
> For scale, the paper's PPO-in-simulator baseline does not reach return 0.82 within 18 h on up to
> 8 GPUs.

![SuperMimic G1 tracking LAFAN1 motions](results/supermimic_lafan1_highlight.gif)

*Solid robot: the trained policy simulated in Isaac Lab with MuJoCo-Warp. Blue ghost: the reference
motion. Clips: dance, jumps, fall-and-get-up, sprint.*

Progress and timestamps: [MILESTONES.md](MILESTONES.md).

## Layout

| Path | What |
|---|---|
| `IsaacLab/` | Isaac Lab `main` @ 46d6cc3 (VERSION 3.0.0), unmodified — cloned by `tools/setup.sh`, not tracked |
| `docker/` | `Dockerfile`: `supermimic:latest` = `isaac-lab-kitless:main-46d6cc3` (built from `IsaacLab/docker/Dockerfile.kitless`) + tmux, ffmpeg, gcc (for `torch.compile`); `run_container.sh` |
| `data/` | not tracked; `tools/download_data.sh` fetches LAFAN1 G1 retargets (HF `lvhaidong/LAFAN1_Retargeting_Dataset`) and the G1 URDF/meshes, then builds `g1_29dof.xml` (MJCF) and the USD |
| `supermimic/` | implementation (see below) |
| `tools/` | setup, asset conversion, tests, benchmark, plotting, rendering, failure analysis |
| `logs/<run>/` | not tracked: `train_log.csv`, `eval.csv`, checkpoints, stdout |
| `results/` | success curve and highlight GIF (full videos are rendered locally with `tools/render.py`) |

## Setup

Neither Isaac Lab nor any data is included in this repository; the steps below fetch them.

**Requirements:** Linux, an NVIDIA GPU (tested: RTX PRO 6000 Blackwell, driver 580 / CUDA 13),
Docker with the NVIDIA Container Toolkit (`docker run --gpus` must work), ~20 GB of disk for the
images and ~0.2 GB for the data.

Physics: Newton with the MuJoCo-Warp solver (`NewtonCfg(MJWarpSolverCfg(...))`), no Kit, no PhysX;
50 Hz control, 4 substeps of 5 ms, implicitfast integrator, MuJoCo collision pipeline.

### 1. Isaac Lab (kit-less Docker image)

The code runs against Isaac Lab 3 `main` at commit
[`46d6cc3`](https://github.com/isaac-sim/IsaacLab/commit/46d6cc3d7b115906fad12efd372fbf69f4e3d3ed)
(VERSION 3.0.0), unmodified, inside Isaac Lab's **kit-less** image (no Isaac Sim / Kit, no PhysX).
Newer `main` commits may work but are untested.

```bash
tools/setup.sh
```

which is equivalent to:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout 46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
(cd IsaacLab && docker build -f docker/Dockerfile.kitless -t isaac-lab-kitless:main-46d6cc3 .)
docker build -t supermimic:latest docker/     # + tmux, ffmpeg, gcc, NVIDIA EGL vendor file
```

Then start the long-lived container (repository mounted at `/workspace/supermimic`):

```bash
docker/run_container.sh 0 128g     # GPU index 0, 128 GB memory cap
```

### 2. Data: LAFAN1 motions and the G1 model

| What | Source | License | Goes to |
|---|---|---|---|
| 40 LAFAN1 clips retargeted to the 29-DoF G1 (30 fps CSV: root pos, root quat xyzw, 29 joint angles) | HF [`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), folder `g1/` | CC BY-NC-ND 4.0 (Ubisoft La Forge) | `data/lafan1_g1/*.csv` |
| G1 description `g1_29dof_rev_1_0.urdf` + meshes (the model the clips were retargeted to) | same dataset, folder `robot_description/g1/` | BSD-3-Clause (Unitree) | `data/g1_description/` |

With the container from step 1 running:

```bash
tools/download_data.sh
```

This also caches Isaac Lab's official ground plane and textures in `data/ground/`.
For existing datasets, run `bash tools/download_ground.sh` once. This avoids a startup
stall in OmniClient's S3 download. No Docker rebuild is needed.

It downloads both, then converts the robot inside the container:
`tools/urdf_to_mjcf.py` compiles the URDF into `data/g1_description/g1_29dof.xml` (free-floating
pelvis), and `tools/mjcf_to_usd.py` converts that MJCF to `data/g1_usd/g1_29dof/g1_29dof.usda`,
which Isaac Lab spawns. The conversion is offline because the kit-less importers cannot run inside
an Isaac Lab process (see Deviations). Expected result:

```
data/
├── lafan1_g1/                  40 CSVs (dance, fallAndGetUp, fight, jumps, run, sprint, walk)
├── g1_description/
│   ├── g1_29dof_rev_1_0.urdf
│   ├── g1_29dof.xml            generated MJCF
│   └── meshes/*.STL            38 meshes
└── g1_usd/g1_29dof/            generated USD (+ payloads/)
```

Check: `ls data/lafan1_g1 | wc -l` prints `40`, and the clips total 264,705 frames at 30 fps
(2.45 h). `tools/test_env.py` checks the converted robot against the reference motions (FK vs.
simulator should agree to ~1e-6):

```bash
docker exec supermimic python tools/test_env.py
```

## Usage

```bash
CPUS=0-23 ./run_train.sh lafan1_main          # training in tmux session 'lafan1_main' (optional CPU pinning)
docker exec -it supermimic tmux attach -t lafan1_main
docker exec supermimic tmux new -d -s eval_main "cd /workspace/supermimic && python supermimic/evaluate.py --run_dir logs/lafan1_main --watch"
docker exec supermimic python tools/plot.py logs/lafan1_main --out results/success_curve.png
docker exec supermimic python supermimic/evaluate.py --run_dir logs/lafan1_main --ckpt logs/lafan1_main/ckpt_014000.pt --record traj.npz
docker exec supermimic python tools/render.py traj.npz out.mp4 --clips dance2_subject1 --seconds 10
```

### Optional dispersion-gated policy updates

The default training path is unchanged. To enable the experimental state-dependent
world-model gate, pass the flags through `run_train.sh` or invoke `train.py` directly:

```bash
./run_train.sh lafan1_gate01 --dispersion_gate 0.1 --wm_ema_beta 0.995
```

The trainer maintains an EMA target world model and compares its full-horizon imagined
return with the online model. Starts whose relative disagreement exceeds the threshold
are excluded from the imagined policy gradient; the target model is never updated by
policy gradients. `gate_fraction` is recorded in `train_log.csv` and printed with each
training log line. This is an experimental Tier-0-inspired extension and has not yet
been benchmarked on the full Isaac Lab LAFAN1 task.

If the EMA disagreement is too large during early model fitting, delay activation with
`--dispersion_gate_warmup 1000`; the default `0` enables the gate immediately.
To replace the binary decision with a smooth trust weight, set for example
`--dispersion_gate_temperature 0.2`; the default `0` keeps the hard gate.
For a fixed per-batch trust budget, set `--dispersion_gate_fraction 0.5`; this
trusts the least-disagreeing half of starts and overrides the fixed threshold.
Add `--dispersion_gate_normalize` when the gate should preserve the mean policy
gradient scale after filtering; without it, filtering also reduces the update magnitude.

For the separate model-error replay ablation, add `--wm_prioritized --wm_priority_alpha 0.6`.
This changes only world-model window sampling; policy start sampling remains uniform.

To test update ordering inside an iteration, add `--policy_before_wm`. This runs the
policy step using the previous world model and fits the world model afterward; the
default remains the paper order, world-model step followed by policy step.

The experimental `--wm_sensitivity_weight λ` weights real-transition model errors by
the local tracking-reward gradient with respect to the next state. It is intended for
policy-sensitive fitting studies and is disabled by default.

`--wm_rollout_sensitivity_weight λ` is a slower prototype that computes reward and
termination sensitivity along each k-step predicted window. It is for profiling and
method development only; the current autograd implementation is intentionally disabled
by default because it substantially reduces throughput.

`--wm_td_weight λ` is a value-aware prototype that weights replay model errors by the
critic's detached absolute TD-error. It requires a critic and should be treated as an
unstable research ablation until the value estimate is calibrated.
When enabled, `--wm_td_target_beta` controls the EMA target critic used for that
diagnostic; the target critic is saved in checkpoints.

For diagnosis without changing the objective, `--wm_disagreement_diagnostic`
maintains an EMA world model and logs `wm_dis_corr` and `wm_diag_err1`, measuring
whether online/EMA one-step disagreement predicts real replay error. This mode does
not enable policy gating unless `--dispersion_gate` is also set.

`--dispersion_gate_corr_min` optionally enables the gate only while the diagnostic
correlation exceeds a running EMA threshold; this is an experimental calibration
ablation and is disabled by default.

`--dispersion_return_penalty λ` keeps every start but penalizes online/EMA imagined
return disagreement. It is another experimental, disabled-by-default robustness
objective; disagreement is not assumed to be a calibrated risk estimate.

`--dispersion_truncate_temperature T` applies a per-step exponential attenuation
to imagined returns based on online/EMA state disagreement. It is a finer-grained
alternative to dropping an entire start, and is disabled by default.

`--policy_updates N` repeats the policy/critic update N times per iteration. The raw
trainer keeps a backward-compatible default of `1`, while the current validated
`tools/run_concurrent_ramp.sh` recipe uses `2`; comparisons should be compute-matched
before interpreting it as a concurrent-training method.

`--policy_updates_warmup_iters N` keeps the policy at one update for N iterations
after `--wm_only_iters`, then switches to `--policy_updates`. It is disabled by
default; the current audit found no return gain from delaying the second update.

`--policy_second_update_scale s` multiplies the policy optimizer learning rate only
for the second update in an iteration. The default `1` uses equal learning rates;
the current audit found that `0.5` weakens the validated recipe.

`--policy_reuse_batch` reuses one sampled policy batch for all updates in an
iteration. It is disabled by default; the current ablation found that independent
resampling is stronger for the two-update recipe.

For the reproduction script, set `POLICY_REUSE_BATCH=1` to enable this ablation;
the default `0` keeps independent sampling.

Set `INTERLEAVE_POLICY_WM=1` to test the experimental `WM → policy₁ → WM → policy₂`
ordering. It is disabled by default and does not change the validated recipe.

Set `POLICY_ACCUMULATE_UPDATES=1` to average two independently sampled policy gradients
into one optimizer step. This is an ablation for sequential adaptation and is disabled by
default.

`--policy_period N` runs policy updates every N iterations while keeping WM updates
at every iteration. The default is 1; larger values are schedule ablations and can
cause the policy and replay distribution to drift apart.

`--policy_bc_weight λ` adds a replay-action behavior-cloning anchor to the imagined
policy loss. It is disabled by default; raw replay actions include execution noise,
so a large value can constrain the policy toward a poor or noisy teacher.

`--policy_awbc_weight` and `--policy_awbc_beta` provide a reward-weighted variant of
that anchor. The current implementation uses only normalized one-step replay reward,
so it is an exploratory proxy rather than a true counterfactual advantage estimate.

`--policy_ramp_iters N` linearly ramps the policy optimizer learning rate from zero
to the current learning rate over N iterations after `--wm_only_iters`; it is
disabled by default and is intended to test whether the first post-warmup policy
update causes a transient distribution shift.

`--policy_ema_model` routes imagined policy rollouts through the optional EMA world
model while fitting the online model from replay. This is an experimental stability
ablation; it can fail early if the EMA model is not yet accurate enough.

`--imag_action_noise` optionally adds Gaussian noise to imagined actions. It is
disabled by default: matching the real-data action noise directly can destabilize
the deterministic policy gradient.

The `--horizon` flag controls policy imagination length independently of `--wm_window`.
Short-horizon experiments can reduce accumulated model error; comparisons should keep
the model window fixed when studying this coupling.

`--policy_bptt_truncation N` keeps the full imagined forward horizon but detaches the
state/history graph every N steps. It is a long-horizon gradient-stability ablation;
0 (the default) keeps full BPTT.

`--policy_bptt_state_only` is a diagnostic variant that detaches only the WM state
while retaining policy history/action graphs; it is not the recommended setting.

`--joint_wm_policy_weight λ` is a diagnostic concurrent-training ablation. It
allows a scaled copy of the imagined policy-return gradient to update the online
world model after the ordinary replay WM update. The default `0` keeps the WM
frozen during policy optimization. This naive coupling can exploit model error;
a principled joint objective should retain an explicit replay prediction term.

`--joint_wm_replay_weight λ` adds that replay prediction term to the same
policy/WM optimizer step. It is disabled by default and is intended to test a
supervised concurrent update; comparisons should report both this anchor and the
policy-gradient weight.

`--joint_replace_wm_step` replaces one standalone WM optimizer step with the
joint replay-anchored policy/WM step. Use it for compute-matched comparisons;
without this flag the joint mode adds an extra WM update per policy update.

`--joint_residual_hidden D` enables a small zero-initialized residual adapter.
With this option, the nominal WM branch remains replay-only during the joint
policy step; only the residual branch receives the policy gradient, while the
joint replay term anchors it to observed transitions. It requires
`--joint_wm_policy_weight > 0` and is disabled by default. Use
`--wm_norm_checkpoint PATH` to load and freeze shared input/output running
statistics across seeds; this is recommended for cross-environment residual
audits. Residual-only coupling has improved audit gaps in the mini testbed but
has not yet improved simulator return, so it remains an experimental mechanism.

`--joint_residual_occupancy_sigma σ` optionally gates that adapter by detached
compatibility between the current policy action and the replay action at the
imagination start. It is intended to suppress out-of-distribution residual
exploitation, not to provide an uncertainty guarantee. Current H=64 tests did
not improve control return, so keep it off unless explicitly comparing
occupancy-conditioned residuals.

`--dispersion_truncate_local` changes disagreement attenuation from a cumulative
trust product to a per-step weight. It is intended for long horizons where the
cumulative form can collapse all imagined return mass; it remains an experimental
uncertainty diagnostic, not a calibrated robust objective.

`--dispersion_truncate_normalize` normalizes those per-step trust weights by
their batch mean. It preserves the overall policy-gradient scale while retaining
relative disagreement weighting; use only with the local form for diagnostic
comparisons.

`--dispersion_softmin_temperature τ` optimizes a soft minimum of the online and
EMA imagined returns. Both WM parameter sets stay frozen during the actor step;
this is a two-model robust-return diagnostic and not a claim that EMA disagreement
is calibrated uncertainty.

`--term_soft_height` and `--term_soft_tilt` expose the sigmoid widths used by the
analytic imagined termination function. They are calibration ablations and default
to the original `.02` and `.04` values.

`--wm_aux_window K --wm_aux_weight λ` adds a second autoregressive WM loss at window
K while retaining the main `--wm_window`. This supports multi-scale loss experiments
without a hard curriculum switch and is disabled by default.

`--wm_occupancy_weight λ` enables a detached policy-action compatibility weight
for sampled WM windows. It reconstructs the start observation history, compares
the current policy action with the stored replay action, and keeps a uniform
floor via `--wm_occupancy_floor`. This is an exploratory nominal-fitting mode;
it is mutually exclusive with prioritized/sensitivity/TD WM weighting and does
not send gradients into the policy.

`--wm_loss_mean` normalizes the WM regression by the number of valid transitions;
the historical default sums across the window for compatibility with existing runs.

`--termination_diagnostic` logs the Brier score and empirical rate of this soft
termination on WM autoregressive replay predictions without changing training.

The gate and replay tests can be run without Isaac Lab; the residual branch
test imports the project networks and should be run inside the Isaac Lab
container:

```bash
python3 -m unittest tools.test_buffer tools.test_dispersion tools.test_residual -v
```

The first iterations are slow while `torch.compile` builds its kernels (cached in `.cache/`).
Checkpoints are written every 2000 iterations to `logs/<run>/`; the eval watcher scores each on all
40 clips and appends to `logs/<run>/eval.csv`.

### Community-trained checkpoint (Not Official)

[G1 policy weights on Hugging Face](https://huggingface.co/pantelismanouselis/g1-supermimic-lafan1),
published by Pantelis Manouselis, include evaluation results and motion videos.
This separate training run reached normalized return **0.8696**, with **0/40 failed training clips**.
The inference-only release uses Safetensors. Follow the
[compatible fork's loading instructions](https://github.com/Manouselis/supermimic_vibed_fork/blob/8bdf60a74c700720817669539e160f4f846b0cf4/docs/RELEASE.md).
The local `.pt` workflow above is unchanged.

## Result (LAFAN1, 40 clips, one RTX PRO 6000 Blackwell)

Evaluation return 0.824 at 1.24 h of training (paper: 0.82 at 1.2 h on an H100) and 0.829 with
**0/40 failed clips** at 1.47 h (iteration 14000).

![success curve](results/success_curve.png)

## Concurrent world-model / policy exploration

The experimental residual branch is enabled with `--joint_residual_hidden 128` and a
positive `--joint_wm_policy_weight`. It freezes the nominal replay-trained world model and
isolates policy-sensitive updates in a zero-initialized residual adapter. Optional controls
include `--joint_wm_replay_weight`, `--joint_wm_replay_window`,
`--joint_residual_audit_threshold`, `--onpolicy_refresh_steps`, and
`--onpolicy_separate_wm_update`.

Across H=64/truncation=4 runs, direct policy-to-WM gradients, adversarial residuals,
policy-action transition targets, on-policy refresh buffers, and calibrated residual gates did
not exceed the uniform truncated-BPTT baseline. The best use of the experimental code is as a
diagnostic framework: keep the nominal WM replay-trained, calibrate any residual gate on a
train environment split, validate it on held-out environments, and retain a nominal fallback.
Do not enable the experimental flags for the default training recipe without reproducing the
ablation evidence.

The full Chinese ablation log and decision summary are maintained outside this repository in
`SuperMimic_real_sim_mini_ablation.md` and `SuperMimic_exploration_decision_summary.md`.
The concise repository-level decision record is [docs/concurrent_exploration.md](docs/concurrent_exploration.md).
The offline candidate for a stricter residual gate is `tools/validity_classifier_audit.py`;
it must be calibrated on train environments and checked on held-out environments before
being connected to imagined policy rollouts.

## Implementation (paper section -> file)

* G1 29-DoF, BeyondMimic actuators (10 Hz natural frequency, damping ratio 2, motor armatures,
  action scale 0.25·τmax/Kp) — `supermimic/robot.py`.
* Reference library: 30 -> 50 Hz resampling, finite-difference velocities, FK of 14 tracked bodies,
  1 s bins — `supermimic/motion.py`, differentiable torch FK `supermimic/kinematics.py`
  (matches the simulator's link poses/velocities to 1e-6).
* MDP (App. A): observation (q−q0, q̇, IMU ω, projected gravity, a_{t−1}; 10-frame history; Table A.1
  noise), K=10 future reference frames as the goal, BeyondMimic tracking reward without the
  self-collision term and joint-limit weight −2, termination |e_z|>0.25 m or e_tilt>0.8, sim context
  ξ = (friction U[0.3,1.2], joint offsets U[±0.01], torso CoM offset) — `supermimic/mdp.py`.
* Simulator wrapper with return-based reset sampling (50% uniform / 40% softmax(−25 G_b/G_max) /
  10% past-softmax) — `supermimic/env.py`.
* World model f_ω: MLP(2048,2048,2048) on heading-invariant state features + action + ξ, predicting
  frame-local increments; fit on k=8-step autoregressive windows (Eq. 1); analytic reward,
  observation and termination heads — `supermimic/networks.py`, `supermimic/train.py`.
* Policy: motion encoder MLP(2048,1024,512,256)->z∈R^32 (tanh), shared decoder
  MLP(2048,1024,512,256); critic MLP(256,128). BPTT over H=32 imagined steps from replay-buffer
  states with a soft survival weight and critic bootstrap (Eq. 2); critic on GAE(λ=0.95) targets
  (Eq. 3), 5 epochs × 4 minibatches.
* Hyperparameters as Table A.5: 4096 envs, buffer 1000 steps/env, action noise N(0, 0.1²),
  UTD 0.4 (2.5 vectorized sim steps per iteration), 10 WM batches of 2048 windows per iteration,
  4096 imagined rollouts, Adam lr 2e-4 (policy β=(0.9,0.95)), cosine decay to 0.03× over 250k
  iterations, grad clip 1.0 (+ element-wise 0.1 for the policy), γ=0.99.
* Evaluation (App. F.1): each clip once from its first frame, friction 1.0, no DR/noise,
  deterministic policy; failure = training termination — `supermimic/evaluate.py`.

## Deviations / choices not specified by the paper

* **Self-collisions disabled** in the simulator (speed); robot–ground contacts use the URDF's
  convex mesh hulls and foot spheres.
* **bf16 autocast for MLP matmuls** (paper: TF32). On this RTX PRO 6000 it is 24 % faster; all
  state integration, FK and rewards stay fp32.
* Reward, observation and termination heads are **analytic** (allowed by §3.2); termination in
  imagination is a sigmoid-softened version of the hard thresholds.
* The policy observation additionally contains the reference root orientation relative to the
  robot root (6D), i.e. BeyondMimic's `motion_anchor_ori_b` — IMU-observable; no global position.
* 500 world-model-only warm-up iterations before policy updates start.
* Torso-CoM and friction are randomized once per environment at start-up (BeyondMimic's
  start-up events); joint offsets are resampled every episode. Episodes time out after 10 s or at
  the clip end (bootstrapped, not terminal).
* Do not run under a virtual-memory `ulimit -v`: Isaac Lab + Warp reserve ~430 GB of *virtual*
  address space (CUDA VA reservations; RSS ~3 GB) and Warp's CUDA-graph capture needs its mempool,
  which cannot initialize under such a limit. Cap memory with the container's cgroup
  (`--memory`) instead. Rootless docker ignores `--cpuset-cpus`; pin CPUs with `taskset`
  (`CPUS=...` in `run_train.sh`).
* Asset import: the kit-less URDF/MJCF importers fail inside a running Isaac Lab process
  (`PhysxJointAPI` schema not registered), so the USD is converted once, offline, by
  `tools/mjcf_to_usd.py` (registers the importer's schema plugin first) and spawned with
  `UsdFileCfg(variants={"Physics": "mujoco"})`.

## Attribution and licenses

* Method: Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath,
  *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*,
  CoRL 2026, arXiv:2610.28480. This code is an independent re-implementation, not affiliated with
  the authors.
* Isaac Lab (BSD-3-Clause), Newton and MuJoCo-Warp (Apache-2.0) are used unmodified.
* LAFAN1 motions (Ubisoft La Forge): CC BY-NC-ND 4.0; G1 retargets and description by Unitree
  (BSD-3-Clause). The motion data is not redistributed here; the demo GIF shows the
  reference motions for illustration only and is not covered by this repository's MIT license.

## License

The code in this repository is released under the [MIT License](LICENSE).

Copyright (c) 2026 ETH Zurich, Chong Zhang.

## Concurrent world-model/policy exploration

The current reproducible candidate uses a 4-step teacher-forced WM objective plus an
8-step free-running auxiliary objective whose weight ramps from zero to 0.05 over 300
iterations, with two policy updates after each WM update:

```bash
SEED=0 LOG_DIR=logs/concurrent_ramp_s0 tools/run_concurrent_ramp.sh
```

The two policy updates independently resample replay batches; `POLICY_REUSE_BATCH=1`
is reserved for the weaker same-batch ablation.

For the matched baseline or the update-count boundary check, override the policy update
count without changing the rest of the recipe:

```bash
POLICY_UPDATES=1 SEED=0 LOG_DIR=logs/concurrent_ramp_baseline_s0 tools/run_concurrent_ramp.sh
POLICY_UPDATES=3 SEED=0 LOG_DIR=logs/concurrent_ramp_updates3_s0 tools/run_concurrent_ramp.sh
```

The script also exposes `POLICY_UPDATES_WARMUP_ITERS` and
`POLICY_SECOND_UPDATE_SCALE` for schedule ablations; both default to `0` and `1.0`,
respectively, which are the validated settings.

The script defaults to 2000 iterations. Long-budget continuation was validated on both
held-out subject splits and three training seeds, with 2000-step checkpoints consistently
above their matched 1000-step checkpoints under perturbed starts. Set `ITERATIONS=1000`
for the shorter baseline budget. The one-update recipe remains the baseline for ablations;
use the script above for the current leading candidate.

### Adaptive policy-update ablation

For the concurrent-training experiments, `--policy_updates_until_iter K` keeps the configured number of policy updates through post-WM iteration `K`, then reduces to one update. This is an ablation for testing whether the second update is only useful early; it is not part of the default recipe.

`tools/policy_update_delta_audit.py` provides an offline diagnostic for the
marginal value of two virtual policy updates while keeping the world model
frozen. It loads the checkpoint policy optimizer state and reports held-out
policy-loss deltas; the diagnostic is not used as an online gate.

The next-stage matched-budget protocol is documented in
`docs/next_experiment_spec.md`.

The currently verified matched-budget results are indexed in
`docs/experiment_matrix.csv`.

Run `python tools/validate_experiment_matrix.py` after editing the matrix to
check all relative changes and matching-budget axes.

External implementation differences and their experiment status are tracked
in `docs/external_mechanism_gap_table.md`.

For deployment-aware shadow-policy probing, repeat `--steps` to require every
simulator horizon to pass independently. The optional first-failure threshold
guards against candidates that reach the same end-survival rate but fall sooner:

```bash
python tools/watch_probe_gate.py \
  --checkpoint-dir logs/shadow_run \
  --checkpoint-pattern 'shadow_ckpt_*.pt' --shadow-mode \
  --require-shadow-artifact --reference logs/shadow_run/ckpt_001000.pt \
  --motion-dir data/lafan1_g1 --seed 123 --seed 456 \
  --out-dir logs/shadow_run/probes --control-file logs/shadow_run/control \
  --steps 100 --steps 500 --max-first-failure-drop 5 \
  --promote-on-accept
```

The default single-horizon behavior remains unchanged; use the multi-horizon
form for candidates whose short-horizon gains may not persist.
For offline replay of already published checkpoints, add `--once` so the
watcher processes the current directory and exits instead of polling forever.

For WM-only per-component weighting experiments, pass
`--wm_component_weights` with 70 positive comma-separated values in the order
`position(3), rotation(3), linear_velocity(3), angular_velocity(3), q(29), qd(29)`.
The default remains uniform weighting; `--wm_root_height_weight` is applied on
top of this vector.

`--deterministic_collect_fraction f` mixes deterministic policy collection into
training: fraction `f` of environments receive no action noise while the rest
keep `--action_noise`. The default `0` preserves the original behavior.

`tools/derive_kinematic_component_weights.py` generates a conservative Jacobian
proxy (`docs/kinematic_component_weights.csv` plus metadata) from tracked-body
position and angular-velocity sensitivity. It is an ablation input, not a
validated default.

`tools/checkpoint_budget_audit.py` audits the actual checkpoint iteration, simulator
steps, estimated policy/WM optimizer steps, auxiliary-loss schedule, and wall-clock
for budget-matched comparisons. Do not infer these quantities from a log-directory
name.

`--ensemble_gate_nominal_prefix n` forces the first `n` imagined steps to use the
nominal world model before a calibrated bootstrap ensemble gate can select the
residual ensemble. It is an uncertainty-coupling diagnostic and is not a validated
default.

`--wm_recent_fraction f` restricts WM replay windows to the most recent `f` fraction
of the main buffer. It is an experimental data-coupling option; the default `0`
keeps uniform replay.
