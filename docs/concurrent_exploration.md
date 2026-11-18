# Concurrent world-model / policy exploration

This document records the reproducible conclusions from the experimental branches in
`supermimic/train.py`. The nominal recipe remains collect → replay WM fit → truncated-BPTT
policy update.

## Current evidence

| mechanism | outcome |
|---|---|
| Direct policy gradient into nominal WM | model exploitation; no stable control gain |
| Zero-initialized residual WM adapter | replay fidelity improves slightly; control remains below baseline |
| Replay anchor ratio/window scans | no scalar ratio or longer window gives a gain |
| Occupancy and disagreement gates | useful as diagnostics/fallbacks; no gain |
| Shared frozen normalization | improves audit consistency; no control gain |
| Action-aligned transition target | catastrophic mismatch because replay next state follows behavior action |
| On-policy refresh/shared or isolated buffer | lower control and/or higher WM error |
| Adversarial residual/min–max sign flip | strong control degradation |
| Reward-sensitivity weighting | no improvement |
| Train→audit calibrated gate | prevents catastrophic residual use, but no gain |

All comparisons should use the same seed, environment count, truncation interval, and iteration
budget. A residual result is only considered promising if it improves both held-out audit error
and real simulator return; either metric alone is insufficient.

## Current next-experiment boundary

The leading deployable recipe is now the nominal replay-trained WM with a small free-running
auxiliary term and two policy updates using independently sampled batches. Direct policy gradients
into the nominal WM and residual uncertainty gates remain diagnostic only: their long-run control
results are not stable enough to replace the nominal model. The next useful experiments should
therefore preserve the nominal WM objective and study policy batch diversity, action-conditioned
failure modes, and longer-horizon calibration separately, with held-out simulator rollouts as the
primary gate.

## Reproduction commands

Nominal truncated-BPTT baseline:

```bash
python -m supermimic.train --policy_bptt_truncation 4 --no_critic
```

Residual-only diagnostic with the shared frozen normalizer:

```bash
python -m supermimic.train --policy_bptt_truncation 4 \
  --joint_wm_policy_weight 0.005 --joint_wm_replay_weight 0.005 \
  --joint_residual_hidden 128 --wm_norm_checkpoint logs/shared_norm_avg.pt
```

The calibrated runtime gate uses a threshold obtained from a train environment split and must
be evaluated on a held-out split before being enabled:

```bash
python -m supermimic.train --policy_bptt_truncation 4 \
  --joint_wm_policy_weight 0.005 --joint_wm_replay_weight 0.005 \
  --joint_residual_hidden 128 --joint_residual_audit_threshold 0.0064 \
  --wm_norm_checkpoint logs/shared_norm_avg.pt
```

## Independent validity-classifier audit

`tools/validity_classifier_audit.py` fits a small classifier on deployment-time WM inputs
(normalized state/action/context features) to predict whether the residual lowers one-step
replay error. On the H=64/truncation=4 seed-0 residual checkpoint, a train-environment
calibrated top-10% classifier gate produced held-out residual-minus-nominal error delta
`-0.0082`, compared with `-0.0190` for the raw disagreement top-10% gate; the selected
held-out rate was 9.2%. Seed-1 at top-25% gave `-0.0063`, compared with `-0.0078` for
raw disagreement. Thus the classifier does not improve the simple disagreement gate in
these checks. Keep it as an offline audit until a simulator run shows a return gain; any
future rollout integration must reuse the train threshold and retain nominal fallback.

## Independent action-conditioned uncertainty audit

`tools/uncertainty_model_audit.py` freezes a nominal WM and fits an independent error
predictor from the WM input features. The one-step predictor had held-out log-error
correlation 0.73 (seed 0) and 0.61 (seed 1), so local uncertainty is predictable from
state/action/context. However, when the label is the true 8-step autoregressive error,
the held-out correlation fell to 0.44 on both seeds (train correlation stayed about
0.97), with calibration shift in the selected low-risk fraction. Therefore the signal is
useful for one-step diagnostics but not yet a reliable long-horizon policy gate. Any
future integration must train the target at the same rollout horizon as the policy and
must pass a held-out multi-seed audit first.

A horizon-matched rollout-feature variant was also tested. It summarizes the nominal
rollout's per-step WM inputs (initial, mean, absolute max, and final) before predicting
16-step error. This did not rescue transfer: held-out correlation was 0.096 (seed 0) and
0.394 (seed 1), versus 0.056 and 0.218 for the initial-input-only predictor. The feature
summary is therefore not a sufficient uncertainty representation; do not add it to the
policy path without a stronger sequence model and cross-environment validation.

A GRU over the full 16-step nominal input sequence was also tested. Held-out log-error
correlation was 0.050 (seed 0) and 0.223 (seed 1), essentially unchanged from the
initial-input predictor (0.056 and 0.218). Sequence capacity alone does not solve the
cross-environment risk shift; this branch should not be connected to policy optimization.

## Policy/replay support audit

`tools/policy_support_audit.py` compares the policy action with the replay behavior action
at sampled replay starts and correlates that distance with one-step replay WM error. For
seed 0, baseline and residual checkpoints had action-distance means `0.01264` and `0.01329`
(residual slightly larger), but distance/error correlation was `-0.078` and `-0.119`.
Thus simple action mismatch at replay starts does not explain the long-horizon risk shift;
the relevant support problem likely appears after imagined state rollouts, where this
static audit is insufficient.

## Imagined trajectory support drift

`tools/imagined_support_audit.py` follows the checkpoint policy and WM for 16 steps and
reports replay-normalized state/action support scores. For seed 0, nominal and residual
checkpoints were almost identical at step 16: state RMS z `0.950` vs `0.961`, action RMS z
`1.217` vs `1.181`, and WM-input RMS `1.192` vs `1.108`. Both action scores rise gradually
with horizon, but the residual branch does not create a large extra state-support shift.
Thus gross imagined-state OOD is not sufficient to explain the control gap; the failure is
more likely in subtle transition Jacobians/long-horizon error accumulation than in a simple
support-distance threshold.

## Local Jacobian audit

`tools/jacobian_audit.py` estimates one-step WM sensitivity by random-projection
Jacobians on the same replay sampling stream. On seed 0 with 1024 samples, the residual
checkpoint increased mean state-Jacobian norm from `2.638` to `2.702` (+2.4%) and mean
action-Jacobian norm from `1.365` to `1.469` (+7.6%). Neither Jacobian norm predicted
one-step error (correlations remained near -0.23 to -0.25). This supports a refined
hypothesis: residual updates may alter local policy-gradient geometry without producing
gross OOD or a direct fidelity-error signal.

## Horizon Jacobian audit

`tools/horizon_jacobian_audit.py` estimates the end-of-horizon invariant-state gradient
with respect to each independently perturbed policy action in the imagined sequence.
For seed 0, 16-step cumulative action sensitivity was `1.715` nominal versus `1.695`
residual (64 starts); at 4 steps it was `1.644` versus `1.692`. The residual branch
therefore does not show a consistent multi-step gradient explosion. Its local one-step
Jacobian increase is real but does not accumulate monotonically over this horizon, so the
current control gap cannot be explained by a simple norm-amplification mechanism alone.

## Horizon gradient alignment audit

`tools/gradient_alignment_audit.py` computes policy-task and replay-fidelity gradients on
the residual adapter at horizons 4/8/16. The task-gradient norm grows sharply with horizon
(seed 0: `0.086 → 0.284 → 0.746`; seed 1: `0.065 → 0.223 → 0.757`) while replay-gradient
norm stays roughly fixed. More importantly, alignment is not stable: seed 0 cosine is
`0.043/0.070/0.100`, whereas seed 1 is `-0.111/-0.104/-0.059`; a shared-normalized
seed-300 checkpoint changes from `-0.029` at H4 to `+0.060` at H16. A single scalar
policy coupling coefficient therefore cannot reliably preserve replay fidelity across
seeds and horizons. Conflict-aware gating/projection is the next principled experiment,
but it must be evaluated as a multi-seed policy run rather than inferred from one cosine.

The conflict-gate implementation is now available as `--joint_gradient_conflict_gate`.
It suppresses only the residual policy gradient when the current batch task/replay dot
product is negative and keeps the replay anchor update. A 30-iteration Isaac smoke with
shared normalization ran successfully; the logged cosine changed `+0.0243 → +0.0090 →
-0.0132`, confirming that the gate encounters both cooperative and conflicting batches.
This is an implementation smoke, not a control result; a matched multi-seed run is still
required before drawing a performance conclusion.

## Conflict gate control ablation

Matched H=64/trunc4, shared-normalized, coupling=.005, 300-iteration runs show the
conflict gate is harmful in both tested seeds:

| seed | ungated residual tail | conflict-gated tail |
|---:|---:|---:|
| 0 | `0.338` | `0.263` |
| 1 | `0.337` | `0.289` |

The gate is conservative in the intended replay-fidelity sense, but suppressing task
updates on negative-dot batches removes useful policy adaptation as well. Keep the flag
for diagnostics; it is not a candidate default or a control improvement.

## Gradient-norm matching ablation

`--joint_gradient_norm_match` rescales the residual policy gradient so its norm relative
to the replay gradient matches the requested coupling, compensating for the strong
horizon dependence measured by the gradient audit. Matched 300-iteration H=64/trunc4
runs with shared normalization gave:

| seed | nominal shared WM | ungated residual | norm-matched residual |
|---:|---:|---:|---:|
| 0 | `0.337` | `0.338` | **`0.310`** |
| 1 | `0.335` | `0.337` | **`0.323`** |

This is the first concurrent variant that improves over its matched nominal/residual
controls on both seeds, although it remains below the strongest default baseline trained
with the original normalization/configuration. Treat it as the leading candidate, not a
final result; it needs longer training and a fully matched default-baseline comparison.

## Long-run correction: gradient-norm matching is not stable

The apparent gain of `--joint_gradient_norm_match` at 300 iterations does not survive a
longer continuation. From the matched 300-step seed-0 checkpoint, the norm-matched
residual run reached episode return `0.212` at iteration 1000, while the matched nominal
shared-WM continuation ended at `0.339`. The norm-matched run improved transiently through
roughly 300--700 iterations, then continued to lose control performance. Therefore,
per-batch gradient-norm matching should remain a diagnostic experiment only; it is not a
concurrent-training candidate or default. The result also shows why short fixed-budget
ablation tails are insufficient for this problem: policy-task gradients can be rescaled
into a temporarily useful regime while still causing long-run model/policy co-adaptation
drift.

## Trust-region bug correction

The earlier cap=0.1 results were invalid: `sqrt(sum).clamp_min(eps)` produced NaN
backward gradients at the zero-initialized residual output, so the residual head stayed
frozen. The implementation now adds epsilon under the square root and has a regression
test for finite gradients. With the fix, a real seed-0 300-step norm-match + cap=0.1 run
ends at episode return `0.327`, below the unconstrained norm-match `0.310` and the
uniform truncation baseline around `0.380`; the cap is not currently a candidate.

## Residual parameter anchor scan

To test whether long-run drift was the dominant failure mode, `--joint_residual_l2`
adds an explicit penalty toward the zero-initialized residual parameters. On seed 0 with
H=64/trunc4, shared normalization, norm matching, and 300 iterations, L2=0.1 ended at
episode tail `0.327`, while a stronger L2=1.0 ended at `0.318`. Neither recovered the
uniform truncation baseline (~`0.380`); stronger anchoring suppresses useful adaptation.
Parameter shrinkage alone is therefore not a solution to the task/replay mismatch.

## Short-prefix policy-task gradient

`--joint_wm_policy_horizon` lets the full policy still optimize H=64 while the residual
adapter receives only a prefix task gradient. With norm matching and shared
normalization, seed-0 300-step episode tails were `0.326` for prefix 4 and `0.318` for
prefix 16, both below the full-task norm-matched result `0.310` and the uniform baseline
around `0.380`. Shortening the task gradient does not resolve the task/replay mismatch;
the prefix is retained as a diagnostic option only.

## Two-timescale residual task updates

`--joint_wm_policy_period` applies the residual policy-task gradient only every N policy
updates while replay fidelity is updated every iteration. With N=4, H=64/trunc4, shared
normalization, norm matching, and seed 0, the 300-step episode tail was `0.309`, with
worse early returns than the every-step variant. Reducing task-update frequency does not
resolve the concurrent mismatch; the flag remains a scheduling diagnostic only.

## Task-gradient EMA

`--joint_gradient_task_ema` smooths residual policy-task gradients across policy updates
before norm matching. With beta=0.9, H=64/trunc4, shared normalization, and seed 0, the
300-step episode tail fell to `0.269`, substantially below the unsmoothed norm-matched
run (`0.310`). Temporal smoothing therefore does not fix the mismatch and can erase
useful task directions; it remains diagnostic only.

## Literature calibration: what “concurrent” usually means

The Dreamer line trains the world model from replayed experience and trains actor/critic
from imagined trajectories in the same outer loop, but it keeps the actor/critic task
loss from updating the world-model parameters. This is a deliberate optimization-channel
separation, not a shared scalar loss. The rationale matches the present audits: when the
policy can change the dynamics model through its own imagined reward, it can exploit model
errors outside replay support. The literature therefore suggests the next research
question should be *how to couple the channels through data, targets, or uncertainty*
while preserving a replay-fidelity gradient boundary, rather than increasing direct
policy-to-WM coupling.

For SuperMimic, this yields a concrete design boundary:

1. Keep nominal WM parameters replay-only and treat policy imagination as a frozen-model
   operation for WM optimization.
2. Put coupling in a separately parameterized, audit-calibrated object (for example a
   bootstrap residual/uncertainty member), never in the nominal model's unrestricted
   parameters.
3. Require real-transition or held-out-audit evidence before a policy-sensitive member
   influences imagined return.
4. Evaluate the outer-loop concurrency by wall-clock/data reuse and policy improvement,
   not by whether the policy loss has a nonzero gradient into every WM parameter.

This is consistent with Dreamer’s replay/model split and with reports of model
exploitation when a behavior policy and learned model co-adapt outside environment
coverage.

## Residual capacity scan

Reducing the policy-sensitive residual adapter from hidden=128 to hidden=32 did not
prevent failure. Under H=64/trunc4, shared normalization, norm matching, and seed 0, the
300-step episode tail was `0.331`, below hidden=128 (`0.310`) and the uniform baseline
(about `0.380`). Adapter capacity is therefore not the sole source of model exploitation;
shrinking the residual is not a useful fix.

## Delayed residual task start

`--joint_wm_policy_start_iter` delays policy-task gradients while replay updates continue.
Starting the residual task gradient at iteration 100 caused a sharp late collapse: the
seed-0 H=64/trunc4 norm-matched run ended at episode tail `0.246` after 300 iterations,
versus `0.310` when enabled from the beginning. Stabilizing the nominal model first does
not solve co-adaptation; an abrupt late task update is worse.

## Dual replay constraint: implementation and validity correction

An adaptive replay multiplier was implemented as `--joint_residual_dual`. The first
fixed-reference version used the initial replay error, which was dominated by early WM
warm-up error; the multiplier therefore decayed to zero and produced an apparently strong
1000-step result that is not a valid constrained comparison. Checkpoints now save and
restore the dual state. A corrected version compares against the current nominal replay
loss, but its seed-0 300-step tail was only `0.339` and the multiplier still decayed to
`1.3e-4`. The dual mechanism is therefore unstable and is not a demonstrated improvement.

## Residual parameter-gradient clipping

`--joint_residual_grad_clip` clips only the residual parameter gradient after task/replay
composition, leaving the model function and task direction unchanged. A cap of `0.1`
with H=64/trunc4, shared normalization, norm matching, and seed 0 produced a 300-step
episode tail of `0.319`, below the unclipped norm-matched result `0.310`. Parameter-step
magnitude is not the main source of the concurrent mismatch.

## Bootstrap residual members: first calibrated audit signal

`tools/bootstrap_residual_train.py` now freezes a nominal checkpoint and trains an
independent zero-initialized residual head on a Bernoulli bootstrap replay mask. Two
members (hidden=128, 1000 one-step updates each) were trained from the shared-normalized
seed-0 nominal checkpoint. With residual rollouts, 8-step final disagreement/error
correlation was `0.620`. A threshold calibrated at the train 50th disagreement percentile
selected `42.5%` of held-out audit windows, with `error<=2` coverage `82.1%` (100-resample
CI `78.1–85.6%`). This is a useful uncertainty/audit signal, but it is not yet a policy
result; the residual members must be integrated with nominal fallback and evaluated in the
real simulator.

## Fixed-policy ensemble rollout audit

`tools/ensemble_policy_rollout_audit.py` evaluates the same nominal policy under a
nominal WM and bootstrap residual members on identical replay starts. At H=16, the
nominal imagined return mean was `0.02775`, while members were `0.02461` and `0.02438`.
The two residual members' final disagreement mean was `3.16` (p90 `5.19`). The members
therefore do not justify replacing nominal imagination; their current role is risk
screening with nominal fallback, not direct policy optimization.

## Nominal-fallback gate audit: replay thresholds do not transfer directly

`tools/ensemble_gate_audit.py` applies the mean of two residual members only when their
online disagreement is below a threshold; otherwise it uses nominal WM. Reusing the
replay-calibrated threshold `0.5623` opened the gate on `100%` of H=16 imagined steps and
reduced imagined return to `0.0245` versus nominal `0.0277`. A much tighter threshold
`0.01` opened only `0.18%` of steps and restored return to `0.0275`. The discrepancy shows
that replay-window disagreement calibration does not transfer directly to policy-induced
imagined states; any gate must be calibrated on rollout distributions and retain nominal
fallback.

## Rollout-distribution gate calibration

Calibrating the gate on nominal-policy imagined trajectories gives a different scale from
replay calibration. Rollout disagreement quantiles were p10=`0.0189`, p25=`0.0243`, and
p50=`0.0325`. With the fixed policy and H=16:

| threshold | gate-open fraction | imagined return |
|---:|---:|---:|
| nominal fallback | 0% | `0.02775` |
| 0.0189 (rollout p10) | 6.7% | `0.02819` |
| 0.0243 (rollout p25) | 19.9% | `0.02808` |
| 0.0325 (rollout p50) | 43.7% | `0.02662` |

The low-quantile rollout gates are a plausible next candidate, but this is only imagined
return. Real simulator evaluation is required before treating the small gain as useful;
larger gate fractions already reduce imagined return.

## Real-simulator training with rollout-calibrated ensemble gate

The frozen-member gate was integrated into policy imagination via
`--ensemble_gate_member0/1` and `--ensemble_gate_threshold`; nominal WM remained
replay-only and residual members were frozen. With threshold `0.0189` (rollout p10),
H=64/trunc4, shared normalization, and matched `wm_batches=2`:

| seed | nominal | ensemble gate |
|---:|---:|---:|
| 0 | `0.351` | `0.361` |
| 1 | `0.363` | `0.363` |

The gate did not harm either seed and gave a small +0.010 gain on seed 0, but this is not
yet a stable improvement: the members were trained offline from one nominal checkpoint,
and the budget is only 300 iterations. Longer/more-seed evaluation is required before
claiming control benefit.

## Long-run correction for ensemble gate

The seed-0 gate was extended to 1000 iterations with a matched nominal control. The
300-step apparent gain did not persist: nominal ended at episode tail `0.396`, while the
rollout-p10 ensemble gate ended at `0.393`. The gate remains non-catastrophic and useful
as a safety/fallback mechanism, but it is not a confirmed control improvement; the short
`+0.010` at 300 steps was transient variance.

## Rollout p25 gate long-run check

A wider rollout-calibrated threshold (`0.0243`, p25) was also trained for 1000 iterations
on seed 0. Its final episode tail was `0.396`, exactly matching the nominal control
(`0.396`), while the p10 gate ended at `0.393`. Increasing the gate fraction therefore
restored parity but did not produce a reproducible gain. The ensemble gate is best treated
as a non-harmful fallback/safety mechanism, not a performance claim.

### 18. Ensemble disagreement as a policy regularizer (2026-11-05)

A new `--ensemble_uncertainty_weight` option adds the mean one-step disagreement
between two frozen bootstrap residual members to the policy loss while keeping the
nominal WM as the rollout dynamics. This tests whether uncertainty is more useful as a
continuous risk penalty than as a hard residual fallback.

The matched real-simulator runs used H=64, BPTT truncation 4, shared frozen
normalization, 64 environments, 2 WM batches, no critic, seed 0, and 300 iterations:

| method | episode tail | 8-step WM error | imagined survival |
|---|---:|---:|---:|
| nominal | ~0.380 | — | — |
| ensemble gate p10 | 0.361 | 2.873 | 0.134 |
| disagreement penalty 0.01 | 0.359 | 3.134 | 0.116 |
| disagreement penalty 0.001 | 0.358 | 3.058 | 0.054 |

Both penalty weights underperform the nominal/gated controls and reduce imagined
survival. The result does not support direct disagreement minimization as a policy
objective; disagreement remains better treated as a detached fallback or audit signal.

### 19. Detached ensemble start gate (2026-11-05)

Added `--ensemble_policy_gate_fraction`, which keeps nominal WM dynamics but applies a
detached low-disagreement quantile gate to policy-gradient starts. At fraction 0.5, the
matched seed-0, H=64/trunc4, shared-normalization, 300-iteration run reached episode
tail `0.345` (gate fraction exactly `0.5`), below nominal (`~0.380`) and the hard
ensemble fallback (`0.361`). Filtering half of the imagined starts therefore removes
useful learning signal in this setting; uncertainty should not be used as a generic
policy-start filter without a stronger real-error target.

### 20. Robust-return audit: ensemble disagreement is not monotonic with real quality

Re-running the fixed-policy rollout audit at H=16 shows that bootstrap disagreement can
increase as the real policy improves. For the nominal seed-0 checkpoint at 1000 steps,
real episode tail is `0.339`, while the nominal/member imagined returns are
`0.0281/0.0251/0.0249` and final disagreement mean is `11.71`. The p25 ensemble-gate
checkpoint has real episode tail `0.396`, imagined returns `0.0297/0.0249/0.0257`, and
final disagreement mean `14.02`. Thus a disagreement threshold calibrated on early
rollouts would reject a later, better policy. This rules out treating raw ensemble
spread as a stationary policy-quality score; calibration must be conditioned on policy
and training stage, or used only as a local fallback.

### 21. Stage-conditioned calibration tool and three-stage audit

Added `tools/stage_conditioned_ensemble_audit.py`. It uses the same policy checkpoint
and replay-start distribution at each stage, reports disagreement quantiles, nominal and
member imagined returns, and reads the real `ep_ret_norm` tail from each stage log.
Using fixed nominal/member checkpoints and H=16:

| policy stage | real tail | disagreement mean | disagreement p50 |
|---|---:|---:|---:|
| nominal 300 | 0.337 | 4.97 | 3.62 |
| nominal 1000 | 0.339 | 4.32 | 2.73 |
| ensemble gate p25 1000 | 0.396 | 5.84 | 4.85 |

The best real policy has the largest spread, confirming stage/policy-conditioned drift
with a three-stage audit. The tool is now reusable for future calibration rather than
relying on a single threshold.

### 22. Dynamic per-batch p25 gate remains harmful

A milder stage-adaptive variant (`--ensemble_policy_gate_fraction=0.25`) retained only
one quarter of low-disagreement policy starts per batch while leaving nominal dynamics
unchanged. The matched seed-0, H=64/trunc4, 300-step run reached episode tail `0.347`
(`gate_fraction=0.25`), below the nominal control (`~0.380`) and the fixed p25 fallback
long run (`0.396`). Dynamic quantile selection therefore does not solve the calibration
drift; it discards useful policy gradients before disagreement has a validated causal
relationship to control error.

### 23. Clip-conditioned ensemble calibration

Added `tools/clip_conditioned_ensemble_audit.py` to separate motion-clip difficulty from
within-clip uncertainty. On the two bootstrap residual members, H=8, 6,572 valid
windows gave raw disagreement/error correlation `0.609`, clip-mean correlation `0.767`,
and within-clip correlation `0.555`. Thus motion clip explains a substantial part of
the signal, but disagreement still carries within-clip information. A useful calibration
should normalize or stratify by clip difficulty before applying any risk threshold; a
single global threshold mixes easy walking clips with difficult sprint/fight clips.

### 24. Per-clip median calibration is not automatically safer

The clip audit now compares equal-coverage median filters. On the same 6,5k-window
bootstrap audit, global median selection retained 50.1% of windows with mean error
`1.56` and `error<=2` coverage `76.3%`; per-clip medians retained 50.7% but mean error
rose to `1.86` and coverage fell to `66.5%`. Clip difficulty is a useful part of the
risk signal, so removing it by rank-normalizing within each clip can make selection
worse. Calibration should condition on clip difficulty while retaining a global
absolute-error scale, rather than blindly equalizing every clip.

### 25. Global threshold is strongly difficulty-selective

With one global median disagreement threshold, clip-difficulty quartiles have sharply
different selection rates. The easiest quartile contributes 31.9% selected windows with
mean error `1.17` and `87.96%` error<=2 coverage; the hardest quartile contributes only
2.96% selected windows with mean error `2.56` and `43.81%` coverage. Overall 50% coverage
therefore mostly means “select easy clips,” not uniform safety across the dataset. A
useful covariate-aware calibration should expose this tradeoff explicitly and choose
whether to preserve easy-clip throughput or allocate a controlled risk budget to hard
clips.

### 26. Held-out risk-budget calibration does not transfer yet

The clip audit now splits windows deterministically in half. Thresholds are fit on one
half and evaluated on the other. A global median retained `49.1%` of held-out windows,
with mean error `1.52` and `77.9%` error<=2 coverage. Equal per-difficulty median
thresholds retained `49.0%` but worsened mean error to `1.75` and coverage to `69.8%`.
A diagnostic threshold chosen at the 80th percentile of fit-window safe disagreement
(error<=2) retained `65.1%` but had mean error `1.87` and only `67.0%` coverage. This
shows that simple clip-wise risk budgets are not stable enough to drive policy updates;
they need more independent data or a calibrated predictor rather than threshold tuning.

### 27. Lightweight held-out calibration predictor does not beat raw disagreement

The audit now fits a linear predictor of `log(1+error)` from disagreement and fit-half
clip mean difficulty, then evaluates on the held-out half. At 50% predicted-risk
selection it achieved mean error `1.59` and `76.8%` error<=2 coverage, with predicted
error correlation `0.609`. The held-out global disagreement median was slightly better
(`1.52`, `77.9%`). This simple predictor is not a candidate for policy coupling; the
remaining gap likely needs richer temporal/state features and independent calibration
trajectories, not another scalar threshold.

### 28. Temporal calibration features are the first positive audit signal

The clip audit now gives the held-out predictor the final, mean, and max disagreement
across the full H=8 window plus fit-half clip difficulty. Two independent 32-batch
re-samplings gave 50% predicted-risk selection with mean error `1.38`/`1.36`,
error<=2 coverage `80.5%`/`81.2%`, and predicted-error correlation `0.650`/`0.684`.
The corresponding global-median filters were roughly `1.54`/`1.52` mean error and
`76.5%`/`77.9%` coverage. This is a promising offline calibration signal, but it is
not yet a policy result: the predictor must be trained only on held-out WM error and
validated on independent trajectories before it can gate or regularize policy updates.

### 29. Independent-buffer validation fails under cross-seed shift

The temporal calibration predictor was evaluated with the same bootstrap members but an
independent seed-1 replay buffer (`h64_nominal_sharednorm_s1_300`). On 32 batches, raw
sample disagreement/error correlation became `-0.231` and within-clip correlation
`-0.244`; the temporal predictor correlation was still `0.579`, but its 50% selected
windows had mean error `9.27` and zero error<=2 coverage. The global median was also
unsafe (mean error `11.48`, zero coverage). This is a real domain-shift failure: a
predictor calibrated on seed-0 residual members does not transfer to seed-1 trajectories.
Independent bootstrap members/calibration data are therefore mandatory before any policy
coupling claim.

### 30. Independent bootstrap members restore calibration on their own trajectories

Trained `bootstrap_member0_s1.pt` and `bootstrap_member1_s1.pt` from the seed-1 nominal
checkpoint with independent masks. On seed-1 trajectories, the temporal predictor at
50% held-out selection achieved mean error `1.04`, error<=2 coverage `94.0%`, and
correlation `0.729`; raw disagreement was already stronger than seed-0's audit
(`within-clip corr=.649`). The same seed-1 members evaluated on seed-0 trajectories
collapsed to mean error `5.85` and `0.23%` error<=2 coverage. This isolates the failure:
calibration works when members and trajectories share the data-generating distribution,
but is not transferable across independently collected replay distributions.

### 31. Same-distribution gate: short gain disappears at 1000 steps

Using seed-1-specific bootstrap members and a seed-1 rollout p10 threshold (`0.0150`),
the real simulator gate run reached `0.361` at 300 steps versus the matched nominal
`0.335`. After extending both configurations to 1000 steps, nominal reached `0.393`
while the gate reached `0.382`. Same-distribution calibration restores offline fidelity
but still does not produce a persistent control gain; the 300-step improvement is a
transient training-phase effect.

### 32. Ensemble-prioritized WM replay improves fidelity but not control

Added `--wm_ensemble_disagreement_weight`: frozen matched members provide a detached
weight for real replay WM loss, while policy imagination remains nominal and receives no
uncertainty gradient. On seed 1, weight=1.0, H=64/trunc4/no-critic:

| 1000-step run | episode tail | WM err8 |
|---|---:|---:|
| nominal | 0.393 | 1.987 |
| direct ensemble gate | 0.382 | 2.266 |
| ensemble-prioritized WM replay | 0.389 | **1.660** |

This is a useful separation result: uncertainty-prioritized WM fitting substantially
improves held-out WM error and is less harmful than direct policy gating, but the control
return still does not exceed nominal. Better model fidelity alone is not sufficient when
the policy objective and data distribution remain unchanged.

### 33. Always-on ensemble-average imagination is harmful

Using a huge gate threshold (`1e6`) forces policy imagination to use the average of the
two matched residual members at every step, with no nominal fallback. Seed 1, H=64/trunc4,
300 steps reached episode tail `0.308`, versus nominal `0.335` and calibrated fallback
`0.361`; imagined survival was `0.240` but real performance collapsed. Averaging
independent residual members is therefore not a safe dynamics estimate: disagreement
can be systematic bias, and nominal fallback is essential.

### 34. Replay-priority weight scan: short gain is not weight-sensitive

A matched seed-1 run with `--wm_ensemble_disagreement_weight=0.25` reached episode tail
`0.348` at 300 steps, essentially identical to weight=1.0 (`0.348`). Both are only
slightly above the nominal 300-step `0.335`, so the short improvement is not evidence
of a tuned priority coefficient; the long-run weight=1 result remained below nominal
(`0.389` vs `0.393`).

### 35. Short-horizon check: WM priority remains neutral

With H=32/trunc4, seed 1, 300 steps, ensemble-prioritized WM (weight=1.0) reached
`0.354` episode tail versus a matched nominal `0.355`; WM err8 was `3.272` versus
`2.965`. The H=64 result therefore does not hide a short-horizon control gain: replay
priority is essentially neutral or slightly worse for control while sometimes improving
model fidelity at longer training.

### 36. Seed-0 replication: priority improves WM error, control is a tie

A fully matched seed-0 H=64/trunc4/300-step nominal run reached episode tail `0.351`
and WM err8 `3.425`; ensemble-prioritized replay reached `0.352` and WM err8 `3.120`.
This replicates the pattern from seed 1: a modest WM fidelity improvement with no
meaningful control improvement.

### 37. Priority + on-policy refresh is unstable

Combining ensemble-prioritized WM loss with `policy_before_wm`, 8 fresh simulator
steps/iteration, and 50% on-policy replay produced very large WM losses (`wm err8`
~917 at 300 steps) and no control improvement (episode tail `0.356` vs matched nominal
`0.351`). Fresh-policy transitions and uncertainty weighting interact poorly in this
small buffer; this combination is rejected until replay normalization/weight clipping is
redesigned.

### 38. Excluding fresh data from uncertainty weighting does not fix instability

The priority+on-policy experiment was repeated with ensemble weights forced to one for
fresh transitions (priority applied only to historical replay). WM err8 still exploded
to `861.6` at 300 steps and episode tail was `0.348`. Therefore the instability is not
just an over-weighted fresh sample; the `policy_before_wm`/small fresh-buffer ordering
itself is incompatible with this configuration.

### 39. Enlarging the fresh buffer does not stabilize the ordering

Increasing `onpolicy_buffer_steps` from 64 to 512 while keeping the same priority,
policy-before-WM, refresh, and 50% mixing configuration still produced `wm_err8=845.6`
at 300 steps and episode tail `0.346`. Buffer capacity alone is not the cause; the
policy-before-WM/fresh-data update schedule is structurally unstable here.

### 40. Isolated fresh WM update still fails

Using `onpolicy_separate_wm_update` and disabling stale ensemble weighting inside the
fresh-only update reduced the explosion but did not stabilize it: WM loss remained
`68.6`, err8 `154.5`, and episode tail `0.355` at 300 steps. The fresh update schedule
is therefore not rescued by optimizer isolation alone; the current on-policy replay
path likely needs separate normalization/targets before it can be combined with this
WM objective.

### 41. One-step fresh supervision stabilizes the update, but control remains neutral

Changing `wm_window` to 1 while keeping policy-before-WM, separated on-policy replay,
and ensemble priority made training numerically stable: WM loss/err fell to `0.308`
and no multi-step explosion occurred. Episode tail was `0.353` versus the matched
H=64 nominal `0.351`. The apparent WM metric is one-step (not comparable to the
8-step err8 in other runs), and the control difference is negligible. One-step fresh
supervision is a viable stability repair, not yet a performance improvement.

### 42. Fresh-window scan: stability boundary lies between 1 and 4

With the same separated on-policy configuration, `wm_window=4` produced WM err8
`66.8` and episode tail `0.354`; `wm_window=1` stayed stable at one-step error
`0.308` and tail `0.353`. Thus reducing the window repairs numerical stability only at
one step; a four-step fresh autoregressive target already reintroduces large error,
without control benefit. The fresh path needs a different target construction (for
example, per-step teacher forcing or a separate normalizer), not merely a modest window
change.

### 43. Teacher forcing is the actual stabilizing factor

With `wm_window=4`, separated on-policy replay, and policy-before-WM, teacher-forced
WM fitting reached episode tail `0.373` without ensemble priority and `0.369` with
priority. Both beat the autoregressive fresh-window result (`0.354`) and the ordinary
nominal fresh baseline (`~0.351`), while WM error stayed below `0.2`. The priority term
is slightly harmful here; the positive effect comes from teacher forcing, which removes
error compounding in the fresh WM target.

### 44. Teacher forcing 在第二个 seed 上也保持优势，但收益幅度依赖 seed

将实验延长到 1000 iterations，并去掉 on-policy refresh，只保留 `wm_window=4` 的
teacher-forced WM loss：

| seed | teacher-forced episode tail | matched nominal | 差值 | teacher-forced WM err8* |
|---:|---:|---:|---:|---:|
| 0 | 0.3968 | 0.3388 | +0.0580 | 0.1427 |
| 1 | 0.4222 | 0.3933 | +0.0289 | 0.1325 |

两个 seed 都超过对应 nominal 基线，说明稳定性和短期 control 改善不是单一 seed
现象。由于 teacher forcing 的 `err8` 是逐步使用记录状态的监督指标，不能直接与
纯 autoregressive rollout 的多步 `err8` 比较；下一步应在固定 holdout 轨迹上统一评估
one-step、teacher-forced multi-step 和 free-running multi-step 误差，再判断这项改善
是否能迁移到更长 rollout 或真实环境。

### 45. 统一 holdout 审计：teacher-forced 误差下降，但 free-running 迁移依赖 seed

新增 `tools/teacher_forcing_audit.py`，在同一批 holdout 环境（env 48--63）上分别计算
teacher-forced 与 free-running 的 8-step 误差：

| seed / checkpoint | TF one-step | TF final | free final | free all steps |
|---|---:|---:|---:|---:|
| 0 / teacher-forced | 0.187 | 0.124 | **2.463** | **1.124** |
| 0 / nominal | 0.475 | 0.264 | 4.341 | 2.120 |
| 1 / teacher-forced | 0.161 | 0.110 | 2.166 | 0.998 |
| 1 / nominal | 0.252 | 0.120 | **1.474** | **0.810** |

结果说明：teacher forcing 对 one-step 和 teacher-forced multi-step supervision 都有
明确改善，seed0 也迁移为更好的 free-running rollout；但 seed1 的 free-running 误差
反而恶化。因而当前结论应限定为“有效的 WM 稳定化候选”，不能宣称已经得到稳定的
跨 seed 长时域预测或控制收益。下一步应固定数据生成与 holdout 轨迹，增加独立 seed，
并检查训练中 teacher-forced loss 与 free-running deployment objective 的差距。

### 46. Teacher forcing + free-running auxiliary loss：低权重混合未带来早期收益

为缩小 teacher-forced loss 与部署 free-running objective 的差距，测试了
`wm_window=4 --wm_teacher_forcing` 加 `wm_aux_window=8 --wm_aux_weight=0.25`。
seed1、300 iterations 的结果：

- 纯 teacher forcing：episode tail `0.356`，训练 WM err8 `0.161`；
- 加 8-step free-running auxiliary：episode tail `0.369`，训练 WM err8 `0.217`；
- 混合版本在 holdout 上的 free-running final error 为 `2.552`。

在当前预算下，辅助 free-running loss 同时增加了优化难度，没有改善 control 或 free
rollout，因此暂不延长该方向。后续若继续，应先做 loss-scale/梯度比例审计，确认 auxiliary
项是否压制了稳定的 teacher-forced 主梯度。

### 47. 梯度匹配后的 free-running auxiliary 仍未带来最终收益

梯度审计显示，在纯 teacher-forced checkpoint 上，8-step free-running auxiliary 的原始
梯度范数约为主 4-step teacher-forced 梯度的 `14.7--18.6` 倍，梯度余弦只有
`0.23--0.30`。因此将 `wm_aux_weight` 从 `0.25` 降到 `0.05`，使有效 auxiliary
梯度接近主梯度量级，并在 seed1 延长到 1000 iterations：

| 方法 | episode tail | holdout free final |
|---|---:|---:|
| teacher forcing | 0.4222 | 2.166 |
| teacher + aux(8), weight 0.05 | 0.4089 | 2.225 |

低权重版本比 `weight=0.25` 更稳定，但仍没有超过纯 teacher forcing。这排除了“仅仅
因为 auxiliary 梯度过大”这一解释；当前更可能是 free-running loss 与 policy/control
更新存在结构性目标错位。后续应优先研究 deployment-aware target 或 rollout horizon
curriculum，而不是继续微调 auxiliary 系数。

### 48. Horizon curriculum：改善 free-running WM，但未转化为 policy 收益

seed1 采用两阶段训练：前 300 iterations 使用 `wm_window=4` teacher forcing，随后从
同一 replay checkpoint 切换到 `wm_window=8` free-running，训练至 1000 iterations。
切换后的第一段 WM err8 瞬间达到 `66.8`，随后逐步恢复到 `1.94`。

| 方法 | episode tail | holdout free final |
|---|---:|---:|
| nominal | 0.3933 | 1.474 |
| 纯 teacher forcing | 0.4222 | 2.166 |
| 4-step TF → 8-step free curriculum | 0.3890 | 1.875 |

curriculum 确实比纯 teacher forcing 更接近部署时的 free-running 目标，但尚未超过
nominal control；这再次说明 WM 预测质量和 policy 收益之间存在脱钩。 abrupt horizon
切换还造成明显的短期误差尖峰，后续若继续应采用渐增 free-running 权重或 horizon
schedule，而不是一次性切换窗口。

### 49. 连续 auxiliary horizon ramp：目前最有希望的并行训练候选

新增 `--wm_aux_ramp_iters`，令 4-step teacher-forced 主损失保持稳定，同时将
8-step free-running auxiliary 权重从 0 线性 ramp 到 `0.05`（300 iterations）。在两个
seed、1000 iterations 下：

| seed | ramp episode tail | matched nominal | ramp free final | nominal free final |
|---:|---:|---:|---:|---:|
| 0 | **0.4292** | 0.3388 | **1.465** | 4.341 |
| 1 | **0.4131** | 0.3933 | **1.382** | 1.474 |

相比一次性 horizon 切换，ramp 没有出现 `66.8` 的 WM 误差尖峰；相比纯 teacher
forcing，它把部署式 free-running 误差显著拉低，并在两个 seed 上保留 control 提升。
seed0 的 control 增益尤其明显，seed1 则为温和增益。该配置目前是最值得继续做
独立 seed、不同 ramp 长度和真实环境验证的候选，但仍需更多重复实验确认统计稳定性。

### 50. 第三个 seed 的 paired baseline：ramp 改善在三 seed 上复现

补跑 seed2 的 matched nominal（`wm_window=8`）后，得到完整配对结果：

| seed | ramp tail | nominal tail | ramp free final 8 | nominal free final 8 | ramp free final 16 | nominal free final 16 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **0.4292** | 0.3388 | **1.465** | 4.341 | **3.675** | 10.260 |
| 1 | **0.4131** | 0.3933 | **1.382** | 1.474 | 3.445 | **3.342** |
| 2 | **0.4199** | 0.3730 | **1.681** | 2.179 | **3.720** | 4.718 |

seed1 的 16-step final 略逊于 nominal，但 all-step 误差略优；其余指标和 seed0/2
一致。三 seed 的 8-step free-running final 均改善，control tail 也均高于各自
nominal，说明 ramp 候选已经具备可重复性，但仍需更多随机种子和真实环境测试。

### 51. 长 horizon 审计揭示 8-step ramp 的边界

将三 seed 的 holdout 审计扩展到 32/64 steps 后，8-step ramp 在 32-step 仍有明显
改善，但 64-step 出现 seed 依赖：seed0 free final `11.62` 对 nominal `27.68`，
seed1 `7.51` 对 `23.95`，seed2 `23.18` 对 `7.35`。seed2 的 64-step 退化在 64
batches 重采样后仍存在，是真实长窗口分布问题，而不是一次抽样噪声。

### 52. 16-step auxiliary 能缓解但不能消除 64-step 退化

在 seed2 上使用 `wm_aux_window=16`、`wm_aux_weight=0.025`、同样 300-step ramp，
训练 1000 iterations：episode tail `0.4000`（nominal `0.3730`），free final 为
8-step `1.766`、16-step `3.613`、64-step `15.184`。相比 8-step ramp 的 64-step
`23.181` 有明显改善，但仍高于 nominal `7.352`。这表明 auxiliary horizon 越长，
free-running WM 越稳；但要覆盖 64-step deployment，可能需要多尺度 auxiliary loss
或逐步增加到 32/64 steps，而不是只把窗口从 8 改成 16。

### 53. 同时叠加 8+16-step auxiliary：简单多尺度并不奏效

新增第二个 auxiliary 窗口参数，测试 seed2 的
`aux8(weight=.05)+aux16(weight=.025)`、同一 300-step ramp。300 iterations 时：

- 8+16 多尺度：episode tail `0.379`，64-step free final `11.29`；
- 单独 aux16：episode tail `0.364`，64-step free final `7.09`。

简单把两个 auxiliary loss 相加反而恶化长时域误差，说明两个 horizon 的梯度可能
互相干扰。后续应测试分阶段 schedule（先 8，再逐步替换为 16/32），而不是同时
保持两个固定权重。

### 54. 分阶段 8→16 schedule：比同时叠加好，但仍不如单独 16-step

新增 `--wm_aux_switch_iters`：在前半段 ramp 8-step auxiliary，后半段衰减 8-step
并 ramp 16-step。seed2、300 iterations、switch=150：

- staged 8→16：episode tail `0.376`，64-step free final `10.38`；
- simultaneous 8+16：episode tail `0.379`，64-step free final `11.29`；
- 单独 16-step：episode tail `0.364`，64-step free final `7.09`。

分阶段替换确实减少了同时叠加的干扰，但在当前时长下仍没有超过单独 16-step。更
合理的后续方向是让 16-step 阶段占据更长训练比例，或进一步筛选 32-step auxiliary，
而不是继续增加同时激活的目标数。

### 55. 32-step auxiliary 短筛选：更长并非单调更好

修复 auxiliary 窗口大于 warmup buffer 的初始化 bug 后，测试 seed2 的
`wm_aux_window=32`、`weight=.0125`、300-step ramp：

- episode tail `0.353`；
- 32-step free all-step `4.79`；
- 64-step free all-step `5.11`。

结果不如单独 16-step（同类 64-step all-step 约 `3.30`），说明 auxiliary horizon
存在非单调 trade-off；不能简单地把窗口越拉越长。当前更合理的方向是围绕 8/16
做 schedule 和 loss weighting，而不是直接使用 32-step。

同时修复了启动时只按主窗口 warmup、导致大 auxiliary 窗口 buffer 不足的问题：初始
采集长度现在取主窗口和所有 auxiliary 窗口的最大值。

### 56. 16-step auxiliary 的 seed0 复现

seed0、16-step auxiliary（权重 `.025`、300-step ramp）训练 1000 iterations：episode
tail `0.4134`；free final 分别为 8-step `1.827`、16-step `4.050`、64-step `17.907`。
对应 nominal 的 64-step final 为 `27.679`，因此长时域确有改善，但仍不如 seed0 的
8-step ramp（64-step `11.618`）。目前 8-step 更偏 control，16-step 更偏长 horizon
稳定化；没有一个 horizon 在所有长度上都占优。

### 57. 三 seed paired 汇总

对 8-step ramp 与 matched nominal 的 1000-iteration 结果做简单 paired 汇总（仅三 seed，
不作为显著性检验）：

- ramp episode tail：均值 `0.4207`、样本标准差 `0.0081`；nominal `0.3684 ± 0.0275`；
  paired 差值 `+0.0524 ± 0.0356`；
- ramp holdout free final：`1.509 ± 0.154`；nominal `2.665 ± 1.494`；
  平均降低约 `43.4%`。

WM 指标的方差和 control 指标的 seed 差异都还不小，当前应把结果表述为“三 seed
可复现候选”，而不是最终统计结论；下一步需要固定评估轨迹、增加 seed，并在真实
环境或独立数据分布上验证。

### 58. 长窗口审计现在报告有效样本量

`tools/teacher_forcing_audit.py` 现在同时输出 one-step、final-step 和 all-step 的有效
mask 数量，避免 episode termination 让 final-step 指标被误读。seed2、64-step、64
batches 重跑中，8-step ramp 的 final-step 平均有效样本约 `81.0/256`，nominal 约
`48.2/256`；对应 free final 分别 `22.90` 和 `7.45`，all-step 分别 `8.07` 和
`3.74`。ramp 的长窗口退化不是 final-step 样本为零造成的，且 one-step 在该长窗口
子分布上也升到 `0.62`，说明需要针对长连续轨迹做额外数据/目标设计。

### 59. 交叉 replay 评估修正了长时域结论

给 `teacher_forcing_audit.py` 增加 `--buffer_checkpoint`，在相同 replay 上交叉评估
seed2 的 ramp 与 nominal WM（64-step、64 batches）：

| model | ramp replay all-step | nominal replay all-step |
|---|---:|---:|
| ramp WM | 7.98 | 33.15 |
| nominal WM | 50.84 | 3.80 |

在各自 replay 上，nominal 的数字看起来更好；但交叉后可见两个 WM 都严重依赖
policy 生成的数据分布。此前直接比较各自 checkpoint 的 free-running error 不能作为
纯模型泛化结论。更稳妥的结论是：ramp 促进了 policy 与 WM 的协同适配，是否改善
world-model 本身必须在共享 replay、固定轨迹或独立数据集上验证。

### 60. 8-step 共享 replay 也显示明显协同适配

seed2 的 8-step 交叉审计（64 batches）：

| model | ramp replay free all | nominal replay free all |
|---|---:|---:|
| ramp WM | **0.864** | 4.363 |
| nominal WM | 6.676 | **1.156** |

这种对角优势在 control 相关的 8-step horizon 上同样存在，说明 ramp 的主要收益是
WM-policy joint adaptation：WM 更适合当前 policy 访问的状态动作分布。后续比较应
以真实 episode return、固定初始状态 rollout，或共享 replay 上的 joint system 指标
为主，不能把孤立 WM 误差当成唯一目标。

### 61. 固定起点真实 simulator rollout：joint system 收益有趋势但仍依赖 seed

新增 `tools/fixed_policy_eval.py`，让多个 checkpoint 共用同一批 motion 起点和 100-step
真实 simulator batch。两组固定起点（seed 777/778）结果：

- seed0：ramp return_norm `0.297/0.305`，nominal `0.107/0.113`；
- seed1：ramp `0.315/0.295`，nominal `0.473/0.443`；
- seed2：ramp `0.371/0.350`，nominal `0.314/0.310`。

6 个 paired 条件中 ramp 胜出 4 个；平均 return_norm 为 ramp `0.322`、nominal
`0.293`。seed1 仍是明确反例，说明 joint system 的真实收益存在 seed 依赖。该
评估比训练日志更公平，但还需要更多固定起点、不同 episode 长度和更长 rollout。

### 62. 固定起点 evaluator 校正：ramp 在真实 rollout 上 6/6 胜出

修正 `fixed_policy_eval.py` 的起点采样：此前 `--seed` 没有参与 frame 选择，导致
不同 seed 重复相同起点。现在按 seed 随机选择合法 clip 和 offset，并重跑两组起点
（seed 777/778，每组 16 个起点、100 steps）：

- seed 777：ramp 三个训练 seed 的 return_norm 为 `0.202/0.211/0.202`，nominal
  为 `0.121/0.180/0.199`；
- seed 778：ramp 为 `0.259/0.277/0.281`，nominal 为 `0.159/0.229/0.215`。

合计 6 个 paired 条件中 ramp 全部胜出；总体均值 ramp `0.239`、nominal `0.184`，
相对提升约 `29.7%`。这比训练日志和 replay WM 误差更直接地支持 joint system 改善，
但仍需更多起点、不同 rollout 长度和更大 batch 复核。

### 63. 固定起点 rollout horizon 扩展

在正确随机起点采样下，seed 777、三个训练 seed 的固定真实 rollout：

| horizon | ramp mean return_norm | nominal mean | ramp 胜出 |
|---:|---:|---:|---:|
| 50 | 0.383 | 0.330 | 2/3 seeds |
| 100 | 0.205（此前结果） | 0.167 | 3/3 seeds |
| 200 | 0.110 | 0.091 | 2/3 seeds |

50/200-step 的收益比 100-step 小，seed2 在两个 horizon 都略逊于 nominal。因而
ramp 的 joint-system 优势能延伸到更长 rollout，但幅度和 seed 稳定性仍需更大起点
样本确认。

### 64. 三组独立固定起点的真实 rollout 汇总

用修正后的 evaluator，在 start seeds 777、778、779 上各取 16 个起点、100 steps，
三个训练 seed 共 9 个 paired 条件：ramp 在 **9/9** 条件胜出。所有条件合并后，
return_norm 均值为 ramp `0.2371`、nominal `0.1935`，相对提升约 `22.5%`。

这是目前最直接的 joint-system 证据；仍需扩大起点数和环境扰动，且应报告配对分布
而不只报告均值。

### 65. 大 batch 固定起点复核

在新的 start seed 780 上将 batch 扩大到 64、100-step，三个训练 seed 的 ramp/nominal
return_norm 分别为 `0.212/0.138`、`0.211/0.207`、`0.220/0.200`。ramp 在三个
paired 条件全部胜出，均值 `0.2145` 对 `0.1816`，相对提升约 `18.1%`。结合前面
三组 16 起点 batch，固定起点真实 rollout 的结果已具备较强一致性。

### 66. 四组固定起点 batch 的完整 paired 统计

汇总 start seeds 777、778、779（每组 16 起点）和 780（64 起点）的全部 12 个
训练-seed/batch 配对：

- ramp return_norm：均值 `0.2314`，样本 SD `0.0276`；
- nominal：均值 `0.1905`，样本 SD `0.0357`；
- paired 差值：`+0.0409 ± 0.0336`；
- ramp 胜出 `12/12`，均值相对提升约 `21.5%`。

该统计覆盖了相同起点、相同物理环境和相同 rollout 长度，当前是最强的 joint-system
证据；仍不等价于跨任务/跨环境泛化证明。

### 67. 共享物理扰动下的 joint rollout

`fixed_policy_eval.py --perturb` 让所有 policy 共享同一组随机 friction `[0.3,1.2]`
和 joint offsets，seed780、64 起点、100 steps：

- seed0：ramp/nominal `0.210/0.139`；
- seed1：`0.211/0.206`；
- seed2：`0.220/0.200`。

ramp 仍然 3/3 胜出，说明收益不依赖完全理想的 friction/context。当前扰动覆盖
friction 与 joint offsets，尚未覆盖 COM offset；后续可继续扩展扰动范围。

### 68. 完整共享 context 扰动：加入 COM offset 后仍保持优势

`--perturb` 现在同时施加相同的 friction、joint offsets 和 torso COM offset。seed780、
64 起点、100 steps：

- seed0：ramp/nominal `0.213/0.139`；
- seed1：`0.210/0.208`；
- seed2：`0.222/0.196`。

ramp 仍然 3/3 胜出，说明固定起点 joint-system 优势对当前 context 扰动具有一定鲁棒性。

### 69. 第二个大 batch 的完整 context 扰动复核

在新的 start seed 781、64 个固定起点、100 simulator steps 下，使用相同的 friction、joint
offsets 和 torso COM offset，三个训练 seed 的 ramp/nominal return_norm 为：

- seed0：`0.230/0.140`；
- seed1：`0.230/0.224`；
- seed2：`0.235/0.225`。

ramp 在 3/3 paired 条件胜出，且三组都提高了平均首次失败时间（分别约为
`47.7/35.5`、`48.7/47.9`、`49.2/49.6`；seed2 的生存时间略低但 return 仍更高）。
这为完整 context 扰动下的 joint-system 优势提供了第二个大 batch 复核，但仍属于单一
motion 数据集上的固定起点测试。

### 70. Subject-held-out 留出测试

为区分同分布 joint-system 收益与跨 motion 泛化，建立了 `subject1-4` 训练、`subject5`
测试的留出 split（训练 33 clips，测试 7 clips），并将 ramp 与 nominal 都训练到 1000
iterations。使用测试集相同的 64 个固定起点、100 simulator steps：

- 无扰动：ramp/nominal `0.1845/0.1725`，平均首次失败步数 `41.0/39.1`；
- 完整共享 context 扰动：`0.1858/0.1685`，平均首次失败步数 `43.1/40.7`。

两种条件下 ramp 均胜出。此前 500-iteration 检查中 ramp 略逊（`0.1697/0.1773`），
延长训练后优势反转，说明该 split 对训练长度敏感；目前只有一个留出 subject 和一个
训练 seed，不能据此宣称跨主体泛化已经充分验证。

### 71. 第二个训练 seed 的留出复核

在相同 subject1-4/subject5 split 上加入训练 seed1。subject5、64 起点、100 steps 的结果为：

- 无扰动：seed0 ramp/nominal `0.1845/0.1725`，seed1 `0.1732/0.1895`，两 seed 汇总为
  `0.1789/0.1810`，胜负 `1/2`；
- 完整共享 context 扰动：seed0 `0.1858/0.1685`，seed1 `0.1797/0.1725`，汇总为
  `0.1827/0.1705`，胜负 `2/2`。

因此，留出 subject 上的 ramp 优势依赖训练 seed和测试 context；当前证据支持“扰动条件下有希望”，
不支持无条件的跨主体泛化结论。

### 72. 第二个 subject-held-out split：subject4

为检查 subject5 结果是否偶然，使用 subject1-3、5 训练（33 clips），subject4 测试（7 clips），
seed0、1000 iterations。subject4 的 64 个固定起点、100-step rollout 结果：

- 无扰动：ramp/nominal return_norm `0.2211/0.2102`，平均首次失败 `44.5/46.0`，
  survival `0.0625/0.09375`；
- 完整共享 context 扰动：return_norm `0.2414/0.2224`，平均首次失败 `46.2/45.4`，
  survival `0.046875/0.03125`。

ramp 在 return 两种条件均胜出（约 +5.1% 和 +8.6%），但无扰动 survival 略低；因此收益
应描述为累计 tracking return 改善，而不是一致的寿命改善。结合 subject5 split，跨主体结果开始
显示可重复的 return 趋势，但仍需更多 subject/seed 才能定量确认。

### 73. subject4 split 的第二个训练 seed

subject4 留出 split 加入 seed1 后：

- 无扰动：ramp/nominal `0.2282/0.2277`，首次失败 `47.8/46.1`，survival `0.0781/0.0938`；
- 完整共享 context 扰动：`0.2279/0.2294`，首次失败 `47.6/46.6`，survival `0.0156/0.0625`。

与 seed0 合并，subject4 split 的平均 return 为无扰动 `0.2246/0.2190`、扰动
`0.2346/0.2259`；但单个 seed 的 paired 胜负并不一致，说明提升幅度温和且对 seed/context 敏感。

### 74. 辅助 free-running loss 权重扫描

在 subject4 留出 split、seed0、1000 iterations 下，固定同一批 64 个起点比较
`wm_aux_weight={0.02,0.05,0.10}` 与 nominal。无扰动 return_norm 依次为
`0.2134, 0.2328, 0.2200, 0.2092`；完整共享 context 扰动下依次为
`0.2276, 0.2537, 0.2397, 0.2315`。对应顺序均为 `0.05 > 0.10 > 0.02 > nominal`，
且四个模型在每组起点上的 survival 相同。

这说明 `0.05` 在当前设置下有局部最优迹象：辅助项太弱不足以塑造 free-running 行为，
太强则损害主 teacher-forced 拟合；仍需在更多 seed/split 上复核该结论。

### 75. Auxiliary horizon 对照：8-step 与 16-step

利用已训练 checkpoint，在 subject4 留出测试、seed0、64 起点、100 steps 上比较
8-step (`weight=.05`) 与 16-step (`weight=.025`) auxiliary。无扰动 return_norm
分别为 `0.2518` 与 `0.2472`，nominal 为 `0.1586`；完整 context 扰动下分别为
`0.2266`、`0.2189`，nominal 为 `0.1456`。三种模型每组的最终 survival 相同。

8-step 在两种条件均略优，说明更长 auxiliary horizon 并未自动带来更好的控制；当前
8-step 更像是 control return 与长时域 WM 约束之间的折中。

### 76. Auxiliary weight ramp 日程扫描

在 subject4 留出 split、seed0、1000 iterations、同一固定起点上比较 ramp 300 与 ramp 600：

- 无扰动：ramp300/ramp600/nominal 为 `0.2349/0.2321/0.2276`；
- 完整 context 扰动：`0.2668/0.2524/0.2317`。

ramp300 在两种条件均胜出；扰动下优势更明显。600-step 过慢的权重注入会减少 policy
在有限训练预算内适应 free-running 目标的时间，因此当前 300-step 日程比 600-step 更合适。

### 77. 并行更新顺序：policy-before-WM

在 subject4 留出 split、seed0、1000 iterations、同一固定起点上，将当前 ramp 配置改为
`--policy_before_wm`。标准顺序 / policy-before / nominal 的 return_norm 为：

- 无扰动：`0.2381/0.2292/0.2224`；
- 完整 context 扰动：`0.2215/0.2180/0.2028`。

policy-before 的 survival 略高（无扰动 `0.0469`、扰动 `0.0625`），但累计 tracking return
低于标准顺序。因此当前联合训练更适合在每轮先拟合 WM，再用更新后的 WM 更新 policy；这不是
简单的交换两个 optimizer step 就能改善。

### 78. WM-only warm-up 扫描

在 subject4 留出 split、seed0、1000 iterations、同一固定起点上比较默认
`wm_only_iters=5` 与 `wm_only_iters=50`：

- 无扰动：warm50 / default / nominal 为 `0.2535/0.2428/0.2246`；
- 完整 context 扰动：`0.2242/0.2346/0.2071`。

延长 WM-only 阶段只在无扰动起点上改善，加入 context 扰动后反而下降，说明它会减少
policy 对部署条件的适应时间；当前默认 5 更稳健。

### 79. Subject4 留出集的逐 clip 分解

新增 `fixed_policy_eval.py --clip_name`，在 subject4 的 7 个测试 clip 上以相同设置
（batch16、100 steps、seed921）逐一比较 ramp 与 nominal：

- dance2：`0.1982/0.1804`；
- fallAndGetUp1：`0.2160/0.1994`；
- fightAndSports1：`0.1904/0.1772`；
- run2：`0.0997/0.1011`；
- sprint1：`0.2274/0.2229`；
- walk2：`0.2659/0.2600`；
- walk3：`0.2794/0.2410`。

ramp 在 7 个 clip 中胜出 `6/7`，唯一反例是 run2。收益覆盖 dance、fall、fight、sprint
和 walk 多种动作，说明总体均值不是由单个 clip 拉动，但仍有动作类型依赖。

### 80. run2 反例的多起点复核

对唯一的逐 clip 初始反例 `run2_subject4` 增加 batch64、三个 start seeds（921/922/923）。
无扰动三组 ramp/nominal 分别为 `0.1135/0.1186`、`0.1431/0.1384`、`0.1140/0.1123`；
完整 context 扰动分别为 `0.1144/0.1191`、`0.1417/0.1353`、`0.1127/0.1116`。

汇总均值为：无扰动 `0.1235/0.1231`，扰动 `0.1230/0.1220`。因此 run2 的初始 batch16
反例不稳定，更准确的表述是该动作上的收益接近持平、seed 依赖，而不是稳定失败。

### 81. 逐 clip 的完整 context 扰动复核

对 subject4 的 7 个 clip 使用相同扰动生成器（batch16、100 steps、seed924）。ramp/nominal
return_norm 为：dance2 `0.2254/0.1850`，fallAndGetUp1 `0.2522/0.2612`，
fightAndSports1 `0.2004/0.1668`，run2 `0.1263/0.1400`，sprint1 `0.2509/0.1975`，
walk2 `0.2805/0.2644`，walk3 `0.2401/0.2577`。

ramp 胜出 `4/7`，逐 clip 平均 `0.2251/0.2103`（约 +7.0%）。因此 context 扰动下的
收益主要体现为总体平均鲁棒性，而不是每个动作都改善；fall、run2、walk3 是当前明确反例。

### 82. Termination 类型审计：反例主要是 root-height drift

`fixed_policy_eval.py` 现在额外报告 `height_failure_rate` 与 `tilt_failure_rate`。在 subject4
完整 context 扰动、batch64 的审计中，整体 ramp/nominal 的 return 为 `0.2486/0.2309`，
两者 failure rate 都是 `0.9531`，其中 height failure 都为 `0.9375`，tilt 仅 `0.0156`。

对动作级反例进一步检查：

- run2：ramp/nominal height failure `1.000/0.984`，tilt `0/0.0156`；
- walk3：`0.953/0.938`，tilt 均 `0.0156`；
- fallAndGetUp1：`0.688/0.641`，tilt 均 `0.0469`。

因此主要失败机制是 root-height drift，而不是姿态倾倒；ramp 在这些反例上并未降低 height
termination。后续若继续优化，应将 root-height drift 作为单独的 deployment-aware WM/policy
目标，而不是只调总 tracking return。

### 83. Root-z WM 误差重加权：负结果

新增可选 `--wm_root_height_weight`，只提高 WM state error 中 root-z 分量的权重。subject4
留出 split、seed0、1000 iterations、root-z 5 倍（flag=4）与默认 ramp 在同一批 64 起点上：

- 无扰动：root-z 加权 / 默认 ramp / nominal `0.2304/0.2358/0.2188`；
- 完整 context 扰动：`0.2323/0.2353/0.2191`。

root-z 加权没有降低 height failure（无扰动均 `0.9531`；扰动 `0.9688/0.9375`），虽然
减少了 tilt failure。说明 termination 审计指出的 root-height drift 不能靠简单提高单一 WM
分量权重修复，更可能需要改变 rollout target、状态表示或 policy 的高度约束。

### 84. Imagined policy 的结构化 root-height penalty：初步正结果

新增默认关闭的 `--policy_height_penalty`，在 imagined policy objective 中惩罚归一化的
root-z 轨迹误差；它不改变 nominal WM 的 replay loss。subject4 留出 split、seed0、
1000 iterations、penalty=0.05、batch64 同起点评估：

- 无扰动：height-penalty / 默认 ramp / nominal return `0.2425/0.2287/0.2164`；
  height failure `0.875/0.9219/0.9063`；
- 完整 context 扰动：`0.2440/0.2399/0.2175`；height failure `0.9375/0.9375/0.9219`。

该 penalty 在两种条件均提高累计 return；无扰动时明显降低 height failure，扰动时保持
相近 height failure 并提高 return。相比 root-z WM 重加权，这是更直接地针对 deployment
rollout 的 policy-side 修复，但还需要多 seed、多个 subject split 和 penalty 扫描。

### 85. Policy root-height penalty 权重扫描

在 subject4 留出 split、seed0、1000 iterations、五个 checkpoint 同起点评估：

- 无扰动（seed932）：penalty `0.02/0.05/0.10` return `0.2395/0.2326/0.2020`；
  height failure `0.8906/0.9063/0.6406`，tilt failure `0.0469/0.0469/0.3281`；
- 完整扰动（seed933）：return `0.2317/0.2347/0.2151`；height failure
  `0.9219/0.9375/0.7656`，tilt failure `0.0156/0/0.1875`。

`0.02–0.05` 是当前较合理的低权重区间；`0.10` 过强，虽然压低 height failure，却明显
增加 tilt failure 并损害 return。

### 86. Height penalty 的第二个 seed 与 tilt-aware 修复

在 subject4 split 的 seed1 上，height penalty=0.05 结果为：无扰动 return `0.2344`，
默认 ramp `0.2384`，nominal `0.2485`；height/tilt failure 为 `0.7813/0.1719`，
而默认 ramp 为 `0.875/0.0938`。完整扰动下 penalty/default/nominal return 为
`0.2492/0.2560/0.2582`，penalty 的 tilt failure `0.0938`。

为抑制 tilt 副作用，加入可选 `--policy_tilt_penalty`，测试 height=.05、tilt=.02：
无扰动 return `0.2274`，默认 ramp `0.2283`；扰动 `0.2324`，默认 ramp `0.2358`。
combined penalty 将 tilt failure 降至 0，但没有带来 return 提升。当前 policy-side penalty
仍是有希望的诊断分支，不能作为默认方案。

### 87. Height penalty ramp 日程：缓解但未消除 seed 依赖

新增 `--policy_height_penalty_ramp_iters`，在 WM-only warm-up 后渐增 height penalty。seed1、
subject4 留出、1000 iterations 下，height=.05/ramp=300 与固定 height=.05、默认 ramp、
nominal 比较：

- 无扰动：ramped `0.2071`，fixed `0.2093`，default `0.2119`，nominal `0.2095`；
- 完整扰动：ramped `0.2121`，fixed `0.1987`，default `0.2054`，nominal `0.2104`。

ramped schedule 在扰动条件下改善了固定 penalty 的 seed1 副作用，并将 height failure 降到
`0.875`，但无扰动仍低于默认 ramp。它是更稳健的候选调度，却不能消除跨 seed/context 差异。

## 88. Subject5 留出集上的 ramped height penalty

在 subject1-4 训练、subject5 测试的留出 split 上，加入 `policy_height_penalty=0.05` 并用 300 iterations ramp 的 checkpoint，在 batch64、100 steps、相同起点下，无扰动 return_norm 为 `0.1976`，高于默认 ramp 的 `0.1894` 和 nominal 的 `0.1727`；height failure 分别为 `0.9688/0.9531/0.9531`。完整 context 扰动后，ramped penalty 为 `0.1755`，默认 ramp 为 `0.1763`，nominal 为 `0.1620`；height failure 为 `0.9375/1.0000/1.0000`。因此该 penalty 在 subject5 上保持了相对 nominal 的收益，但没有超过默认 concurrent ramp 的扰动 return，结论仍是 split/context 条件性，不能替换当前默认配方。

## 89. Subject5 ramped height penalty 的第二个 seed：失败模式转移

subject5 留出集第二个训练 seed 重复 `policy_height_penalty=0.05`、ramp=300。无扰动下，ramped penalty/default ramp/nominal 的 return_norm 为 `0.1714/0.1850/0.1833`；完整 context 扰动下为 `0.1588/0.1840/0.1811`。两 seed 平均后，无扰动为 `0.1845/0.1872/0.1780`，扰动为 `0.1671/0.1801/0.1716`。

penalty 确实降低了 height failure：第二个 seed 的扰动 height failure 为 `0.7969`，default ramp 为 `0.9531`；但 tilt failure 升至 `0.2344`，default ramp 仅 `0.0313`。因此它主要把失败模式从 root-height 转移到姿态倾倒，并损失扰动 return。固定高度 penalty 不应进入默认 recipe，后续应改为基于 tilt/height margin 的自适应软约束。

## 90. Margin-gated height hinge：避免过度约束但收益不足

新增 `--policy_height_hinge_margin=0.5`，只在归一化 root-z 误差超过 termination margin 的一半后施加 height penalty。subject5、seed0、相同固定起点下，hinge/default ramp/fixed ramped penalty/nominal 的无扰动 return 为 `0.1747/0.1841/0.1812/0.1708`，完整 context 扰动为 `0.1842/0.1824/0.1859/0.1640`。hinge 没有造成明显 tilt 爆发，扰动下略高于默认 ramp，但无扰动明显落后，不能作为默认 coupling；它只说明 margin-gated constraint 比全程平方惩罚更安全，仍缺少稳定收益。

## 91. EMA disagreement return penalty：保守但未超越默认

启用 `--dispersion_return_penalty=0.1`，用冻结的 EMA WM 对 online imagined return 的偏差施加 detached penalty；nominal WM 仍只由 replay 更新。subject5、seed0、1000 iterations、相同固定起点下，penalty/default ramp/nominal 的无扰动 return 为 `0.1671/0.1679/0.1444`，完整 context 扰动为 `0.1903/0.1935/0.1752`。该项保持了相对 nominal 的收益，且没有明显增加 tilt failure，但两种 context 均略低于默认 ramp，暂不加入默认 recipe。

## 92. Checkpoint selection audit：收益不是末端偶然波动

对 subject4、seed0 的 ramp/nominal 中间 checkpoint 做同一起点评估。无扰动下 ramp 从 500→1000 iterations 的 return 为 `0.1900→0.2316`，nominal 为 `0.1719→0.2117`；完整 context 扰动下 ramp 为 `0.1863→0.2319`，nominal 为 `0.1716→0.2117`。ramp 的相对优势在 500 和 1000 均存在，且 1000 的绝对 return 更高，因此当前 1000-iteration 结果不是末端偶然 spike，也不支持用早停 checkpoint 替代默认训练预算。

## 93. Subject5 checkpoint audit：更长训练也改善未见主体

对 subject1-4 训练、subject5 测试的 seed0 checkpoint 做对照。ramp 从 500→1000 iterations 的 return 为：无扰动 `0.1404→0.1691`，完整 context 扰动 `0.1485→0.1744`；nominal 为无扰动 `0.1482→0.1528`、扰动 `0.1545→0.1576`。因此更长训练带来的 concurrent 增益也迁移到未见主体，且增幅大于 nominal；这进一步支持 1000 iterations 作为当前评估预算。

## 94. Policy learning-rate ramp：平滑接管但略损失 return

首次启用 `--policy_ramp_iters=100`，使 policy 在 WM-only warm-up 后逐步恢复到完整 LR。subject5、seed0、1000 iterations、相同固定起点下，policy-LR ramp/default ramp/nominal 的无扰动 return 为 `0.2017/0.2021/0.1804`，完整 context 扰动为 `0.2038/0.2107/0.1853`。平滑 LR 没有消除 termination，且在扰动下略低于默认；当前不加入默认 recipe。

## 95. Policy update period=2：降低更新频率会损失控制

启用 `--policy_period=2`，每两个 iteration 才更新一次 policy。subject5、seed0、1000 iterations、相同固定起点下，period=2/default period=1/nominal 的无扰动 return 为 `0.1611/0.1700/0.1626`，完整 context 扰动为 `0.1704/0.1894/0.1712`。height failure 几乎不变，period=2 只减少了 policy 适应速度并损失扰动 return；当前 1:1 WM/policy 更新频率更合适。

## 96. Policy updates=2：新的跨 split 正向候选

启用 `--policy_updates=2`，其余配置保持当前默认。subject5 留出集：seed0 的无扰动/扰动 return 为 `0.2426/0.2254`，默认 ramp 为 `0.2132/0.1910`；seed1 为 `0.2305/0.2455`，默认为 `0.2038/0.1959`。两 seed 平均为 `0.2366/0.2355`，相对默认 `0.2085/0.1935` 提升约 `13.5%/21.7%`。subject4 留出 seed0 也复现：updates=2 为 `0.2632/0.2530`，默认为 `0.2391/0.2270`，提升约 `10.1%/11.4%`。failure rate 基本不变，但 first-failure step 通常更晚，说明收益主要来自 policy 更充分地适应 WM，而不是简单减少 termination。

这条分支是目前最强的新候选：仍保持 nominal WM replay training，只增加同一 iteration 内的第二次 policy update；原 `policy_updates=1` 保留为严格基线。该结论已在 subject4/subject5 两个留出 split、四个训练 seed 上复核。

## 97. Policy updates=2 的第二个 subject4 seed

subject4 留出 seed1 的 updates=2/default ramp/nominal：无扰动 return `0.2791/0.2335/0.2328`，完整 context 扰动 `0.2891/0.2360/0.2434`。结合 seed0，subject4 两 seed 平均为 updates=2 `0.2712/0.2710`、default `0.2363/0.2315`；subject5 两 seed平均为 `0.2366/0.2355`、default `0.2085/0.1935`。因此 updates=2 已在两个留出 split、四个训练 seed 上一致优于 updates=1，当前可提升为推荐配置。failure rate 仍相近，收益主要体现为更高 tracking return 与更晚 first failure。

## 98. Policy updates=2 的计算公平性审计

updates=2 每 iteration 的 policy 更新次数翻倍，因此用 subject5 seed0 的 updates=2@500 checkpoint 与默认 updates=1@1000 比较。无扰动 return 为 `0.1861` vs `0.1911`，完整 context 扰动为 `0.1736` vs `0.1907`（updates=2@500 / default@1000）。这说明 updates=2 的优势不是简单以更少 WM iterations 换取同等 policy steps；它仍需要完整 1000 iterations 才超过默认，收益来自更充分的 WM-policy 共同适应。

## 99. Policy updates=2 不是简单增大 policy batch

subject5 seed0 做 `pi_batch=64, policy_updates=1` 对照。batch64/updates1 的无扰动/扰动 return 为 `0.1932/0.1978`；batch32/updates2 为 `0.2235/0.2314`；默认 batch32/updates1 为 `0.1937/0.1965`。因此 updates=2 的提升不能由一次性扩大 batch 解释，关键因素是同一 WM 更新后进行第二次 policy adaptation。

## 100. Policy updates=3：两次更新已接近甜点区

在 subject5 seed0 上测试 `policy_updates=3`。同一评估设置下，updates=3/updates=2/default 的无扰动 return 为 `0.2166/0.2119/0.1819`，完整 context 扰动为 `0.2398/0.2427/0.1998`。updates=3 在无扰动略高，但扰动略低于 updates=2，同时训练吞吐约进一步下降；没有证据支持超过两次更新。保留 `policy_updates=2` 为推荐值，updates=3 作为边界负对照。

## 101. Policy updates=2 与 critic 的组合：不提升扰动鲁棒性

subject5 seed0 上开启 critic、保持 `policy_updates=2`：critic/no-critic 的无扰动 return 为 `0.2209/0.2186`，完整 context 扰动为 `0.2030/0.2146`。critic 只带来轻微无扰动差异，却降低扰动 return，并增加训练成本；当前推荐候选继续使用 `--no_critic`。

## 102. Reproduction recipe synchronized

将当前验证后的主候选同步到 `tools/run_concurrent_ramp.sh` 与 README：默认现在包含 `--policy_updates 2`，同时保留 `--no_critic`。这样脚本复现的是 4-step teacher-forced WM、8-step auxiliary loss（weight 0.05、ramp 300）与双 policy update 的版本；`policy_updates=1` 仍作为严格 ablation baseline。

## 103. Training-log mechanism audit

字段完整的训练日志显示，updates=2 的收益不是来自更低的 replay WM loss。subject4 seed0 的最终 `wm_err8` 为 updates=2/default `0.1297/0.1214`，但 simulator reward 为 `0.4485/0.4233`；subject5 seed1 的 `wm_err8` 为 `0.1298/0.1135`，simulator reward 为 `0.4687/0.4400`。updates=2 的 WM 误差略高而真实控制更好，支持“nominal WM 保持 replay fidelity，第二次 policy update 改善对 WM 的适应”这一机制解释。旧实验日志存在不同 header 格式，未用于该数值比较。

## 104. Subject5 多起点扰动 robustness audit

对 subject5 两个训练 seed 的最终 checkpoint，使用新的扰动 start seeds 972/973/974：训练 seed0 的 updates=2 return 为 `0.2183/0.2179/0.2221`，默认为 `0.1881/0.1798/0.1882`，均值 `0.2194/0.1854`；训练 seed1 的 updates=2 为 `0.2217/0.2204/0.2273`，默认 `0.1845/0.1807/0.1853`，均值 `0.2231/0.1835`。六个 paired 条件全部胜出，平均提升约 18–22%，进一步排除了固定起点 batch 偶然性。

## 105. Reproduction script supports update-count ablations

`tools/run_concurrent_ramp.sh` 现在通过 `POLICY_UPDATES` 环境变量控制 policy update 次数，默认值为 2。使用 `POLICY_UPDATES=1` 可复现严格 baseline，`POLICY_UPDATES=3` 可复现边界扫描；其余 WM/policy 参数保持一致，减少手工命令造成的实验漂移。

## 106. Subject4 多起点扰动 robustness audit

使用新 start seeds 972/973/974 评估 subject4：训练 seed0 的 updates=2/default return 均值为 `0.2484/0.2181`，训练 seed1 为 `0.2485/0.2107`。六个 paired 条件全部由 updates=2 胜出。结合 subject5 的六个 paired 条件，两个留出 split、四个训练 seed 的 12 个新扰动比较均支持双 policy update。

## 107. Policy updates=2 的逐 clip 扰动分解

subject4 seed0、7 个测试 clip、batch16、完整 context 扰动下，updates=2/default return：dance2 `0.2490/0.2354`，fallAndGetUp1 `0.3198/0.2796`，fightAndSports1 `0.2506/0.2201`，run2 `0.1712/0.1379`，sprint1 `0.2294/0.2322`，walk2 `0.3337/0.3019`，walk3 `0.2883/0.2497`。updates=2 赢 6/7，逐 clip 平均 `0.2631/0.2367`（+11.2%）；收益不是由单一动作驱动，sprint1 是轻微动作级反例。

## 108. Policy updates=2 下增大 free-running auxiliary weight：0.05 仍是更稳配置

在 subject5、训练 seed0、`policy_updates=2` 保持不变的条件下，将 8-step free-running auxiliary WM loss 从当前 `0.05` 提高到 `0.10`，其余训练设置相同。固定评估起点下，aux=.10 / aux=.05 / nominal 的无扰动 return 分别为 `0.1940/0.1978/0.1687`；完整 context 扰动下分别为 `0.2199/0.2221/0.2030`。aux=.10 的 height failure 为 `0.9688`（无扰动）和 `0.8906`（扰动），aux=.05 为 `0.9688/0.8906`；tilt failure 则为 `0.0156/0.0469` vs `0.0313/0.0781`。增大 auxiliary weight 没有带来 return 提升，只略微改变失败模式，因此当前推荐值保持 `wm_aux_weight=0.05`，`.10` 作为已验证的负对照。

## 109. Policy-before-WM 在双 policy update 下仍无稳定优势

在 subject5 留出、训练 seed0、`policy_updates=2` 下，比较标准的 WM→policy 顺序与
`--policy_before_wm`。固定无扰动起点时，policy-before / 标准双更新 / 单更新 baseline 的
return_norm 为 `0.2184/0.2248/0.1969`；完整 context 扰动时为 `0.2399/0.2389/0.2155`。
policy-before 只在扰动条件下高出约 `0.001`，无扰动反而下降，且无扰动 height failure 为
`0.9844`、扰动为 `1.0000`。因此交换 optimizer 顺序没有形成稳定收益，默认仍采用先拟合
WM、再用新 WM 更新 policy；policy-before 保留为顺序敏感性负对照。

## 110. 更新顺序结论的跨 split 复核

在 subject4 留出、训练 seed0 上重复双更新顺序对照。policy-before / 标准 WM→policy / 单更新
baseline 的无扰动 return 为 `0.2461/0.2506/0.2280`，完整扰动为 `0.2599/0.2576/0.2394`。
与 subject5 的 `0.2184/0.2248` 和 `0.2399/0.2389` 结果一致：policy-before 在一个
context 下略有优势、另一个 context 下略有损失，且差异小于双更新本身相对单更新的收益。
因此跨两个留出 split，默认顺序仍采用 WM→policy；policy-before 只作为顺序敏感性对照。

## 111. Subject5 第三个训练 seed：policy_updates=2 继续胜出

在 subject5 留出集上新增训练 seed2，并严格使用同一 WM/训练配置比较
`policy_updates=2` 与 `policy_updates=1`。固定无扰动起点下，双更新/单更新 return_norm 为
`0.2124/0.1754`；完整 context 扰动下为 `0.2055/0.1788`，相对提升分别约 `21.1%` 和
`14.9%`。双更新的 first-failure step 也更晚（`44.25/40.70` 与 `44.58/43.02`）。
这使 subject5 三个训练 seed、subject4 两个训练 seed 的配对证据全部支持双 policy update；
当前推荐配置的跨 seed 稳定性进一步增强。

## 112. Subject5 seed2 的多起点扰动审计

对新增的 subject5 seed2 checkpoint 使用三个未参与训练的扰动 start seeds 984/985/986，双更新
与同 seed 单更新的 return_norm 分别为：`0.2187/0.1756`、`0.2090/0.1790`、
`0.2125/0.1833`。三组均由双更新胜出，均值为 `0.2134/0.1793`，相对提升约 `19.0%`。
这进一步表明 seed2 的收益不依赖单一固定起点，并与前两个 seed 的多起点审计方向一致。

## 113. 计算量配对：减少 WM batches 会破坏双更新收益

在 subject5、训练 seed0、1000 iterations 下，测试 `wm_batches=1 + policy_updates=2`，并与当前
`wm_batches=2 + policy_updates=2` 及单更新 baseline 比较。无扰动 return_norm 分别为
`0.1614/0.2031/0.1727`，完整 context 扰动为 `0.1812/0.2415/0.2054`。减少一次 WM
replay update 后，双 policy update 不再带来收益，且 tilt failure 上升、first-failure
step 提前。这说明当前正向结果不是“固定总 optimizer steps 后把算力转给 policy”即可复现；
两次 WM 拟合与两次 policy adaptation 的组合共同构成有效 recipe。

## 114. WM batches=3：无扰动微升但扰动回落

在 subject5、训练 seed0、`policy_updates=2` 下，将 WM replay batches 从 2 增加到 3。固定
起点评估中，wm3 / wm2 / 单更新 baseline 的无扰动 return 为 `0.2257/0.2243/0.1841`，
完整扰动为 `0.2184/0.2227/0.1877`。wm3 只在无扰动下微升，扰动下反而略低，且每轮训练
吞吐下降；没有稳定证据支持额外的第三次 WM 拟合。结合 wm1 的明显退化，当前
`wm_batches=2 + policy_updates=2` 是更合理的计算/鲁棒性折中。

## 115. Subject5 seed2 的 rollout-horizon 审计

为检查双更新收益是否只存在于 100-step 窗口，对同一 subject5 seed2 checkpoint 做不同长度
真实 rollout。50 steps 下，policy_updates=2/1 的 return_norm 为 `0.4495/0.3530`；
200 steps 无扰动为 `0.1178/0.0953`，200 steps 完整扰动为 `0.1100/0.0911`。双更新在
短窗口和长窗口均保持优势，且 first-failure step 更晚，说明收益并非只优化了某一固定评估
horizon。

## 116. Delayed second policy update：更平滑但没有 return 收益

新增 `--policy_updates_warmup_iters`，在指定 iteration 内只执行一次 policy update，再切换到
`policy_updates=2`。subject5 seed0 取 warmup=300：warmup/立即双更新/单更新的无扰动
return 为 `0.1925/0.1946/0.1691`，完整扰动为 `0.1903/0.1990/0.1711`。warmup 降低了
双更新 checkpoint 的 tilt failure，但两种 context 的 return 都略低；第二次 policy update
不需要延迟，当前默认仍在 WM-only warm-up 后立即启用双更新。该参数作为顺序/稳定性 ablation
保留。

## 117. 短 policy-update warmup 也不优于立即双更新

在 subject5、seed0 上测试 `--policy_updates_warmup_iters=100`。warm100/立即双更新/单更新
的无扰动 return 为 `0.2123/0.2273/0.1978`，完整扰动为 `0.2074/0.2246/0.1884`。
即使只延迟 100 iterations，return 仍下降；因此不是 300-step 延迟过长导致的单一现象。
当前推荐固定为 WM-only warm-up 结束后立即执行两次 policy update，warmup 参数仅用于负对照。

## 118. Policy BPTT truncation=8：subject5 小幅收益未跨 split 复现

在双更新候选上测试 `policy_bptt_truncation=8`。subject5 seed0 的 trunc8/trunc4 return 为
`0.2462/0.2438`（无扰动）和 `0.2399/0.2330`（扰动），但 subject4 seed0 复核为
`0.2534/0.2594` 与 `0.2429/0.2713`。trunc8 在 subject4 两种 context 都下降，尤其
扰动下降明显；因此 subject5 的小幅提升属于条件性结果，当前默认仍保持 truncation=4。

## 119. `pi_batch=64 + policy_updates=2`：条件性提升，扰动下不稳

在 subject5 seed0，batch64/batch32 双更新 return 为 `0.2188/0.2116`（无扰动）和
`0.2325/0.2291`（扰动）；但 subject4 seed0 复核为 `0.2759/0.2731` 与
`0.2309/0.2440`，扰动下 batch64 反而下降。更大 policy batch 没有跨 split 的稳定收益，
当前仍保留 `pi_batch=32`；batch64 作为方差/吞吐敏感性对照。

## 120. Bursty policy updates：平均次数相同仍明显退化

测试 `policy_updates=2, policy_period=2`，使平均 policy update 频率约等于每轮一次，但两次
更新集中在每隔一轮执行。subject5 seed0 的 period2/当前每轮双更新/单更新无扰动 return
为 `0.1810/0.2358/0.1932`，完整扰动为 `0.1804/0.2329/0.2085`。集中式更新明显退化，
说明双更新收益依赖每次 WM replay 拟合后立即做两次 policy adaptation，而不是平均 policy
optimizer steps 的数量。

## 121. Auxiliary weight=.02：弱约束没有改善双更新

在 subject5、seed0、`policy_updates=2` 下测试 `wm_aux_weight=.02`，并与当前 `.05` 和
nominal ramp 比较。无扰动 return 为 `.2045/.2085/.1789`，完整扰动为
`.2312/.2464/.2009`（aux=.02/.05/nominal）。降低 free-running auxiliary 权重没有
释放出更好的 policy adaptation，反而在两种 context 下略低；当前 `.05` 仍是更稳的默认值。

## 122. Auxiliary ramp=600：双更新下仍未跨 split 复现

在 subject5 seed0，ramp600/ramp300 的无扰动和扰动 return 为 `0.2369/0.2366` 与
`0.2552/0.2446`，看似略有优势；但 subject4 seed0 复核为 `0.2310/0.2373` 和
`0.2561/0.2675`，两种 context 均下降。较慢的 auxiliary 注入仍是 split-dependent，
当前默认保持 `wm_aux_ramp_iters=300`。

## 123. 双更新下的 16-step auxiliary：仍低于 8-step

在 subject5、seed0、当前双更新配方下，比较 `wm_aux_window=16, weight=.025` 与当前
8-step/.05。16-step/8-step/nominal 的无扰动 return 为 `0.2006/0.2146/0.1878`，
完整扰动为 `0.1772/0.1947/0.1657`。更长 auxiliary horizon 在双更新下仍损失 control
return，确认 8-step 是更合适的 deployment/control 折中。

## 124. Reward-sensitivity weighted WM loss：扰动略升但代价较高

在 subject5、seed0、双更新配方下启用 `wm_sensitivity_weight=.5`，按 tracking reward 对
真实 next-state 的局部敏感度加权 replay WM error。sensitivity/default 的无扰动 return
为 `0.2197/0.2208`，完整扰动为 `0.2341/0.2299`；训练吞吐约从 `3.66` 降至 `3.17`
iter/s。它只在一个扰动 batch 上略升，未显示稳定跨 context 收益，因此保留为诊断项，不
改变默认均匀 replay WM loss。

## 125. Auxiliary loss 与双 policy update 的交互项

为拆分机制，测试 `wm_aux_weight=0 + policy_updates=2`。subject5 seed0 的 no-aux/当前
aux=.05 双更新/单更新 baseline 无扰动 return 为 `0.2082/0.2270/0.1862`，完整扰动为
`0.2067/0.2305/0.1978`。去掉 auxiliary 后双更新仍略优于 nominal，但损失了当前配方的
主要增益，说明 free-running WM 约束与每轮双 policy adaptation 存在正交互；推荐 recipe
不是单独的 update-count trick，而是 `.05` auxiliary + 两次 policy update 的组合。

## 126. 双更新下去掉 teacher forcing：主 WM 误差和控制均退化

在当前双更新 recipe 中移除 `--wm_teacher_forcing`，让 4-step 主窗口也使用
autoregressive rollout。subject5 seed0 的 no-TF/current-TF/nominal return 为
`0.1921/0.2287/0.1966`（无扰动）和 `0.1985/0.2195/0.1947`（完整扰动）；no-TF
最终训练 `wm_err8` 约 `0.865`，显著高于 teacher-forced 配方。结果支持保留
teacher-forced 主 WM loss，再用低权重 8-step free-running auxiliary 连接部署目标。

## 127. 第二次 policy update 降低 LR：没有恢复扰动鲁棒性

新增 `--policy_second_update_scale`，只缩放每轮第二次 policy update 的 optimizer LR。取
`scale=.5` 的 subject5 seed0 结果为 `0.2023/0.2142`，低于等 LR 双更新的
`0.2173/0.2316`（无扰动/扰动）；tilt failure 也上升。降低第二次 update 的步长没有
缓解当前 recipe 的波动，默认继续使用 `scale=1.0`。

## 128. 第二次 policy update 提高 LR：同样退化

补测 `policy_second_update_scale=1.5`。subject5 seed0 的 1.5x/等 LR 双更新无扰动 return
为 `0.2179/0.2318`，完整扰动为 `0.2128/0.2259`；两种 context 均下降。结合
scale=.5 的结果，第二次 update 使用与第一次相同的 LR（scale=1.0）是当前稳定点。

## 129. Subject5 seed2 逐 clip 扰动分解

在 7 个测试动作、batch16、相同完整 context 扰动下，policy_updates=2/1 的 return 分别为：
`dance2 0.2549/0.2192`、`fallAndGetUp1 0.2266/0.1989`、`fight1 0.1978/0.1622`、
`jumps1 0.2910/0.2089`、`run1 0.1897/0.1303`、`walk1 0.2987/0.2300`、
`walk3 0.2328/0.2129`。双更新赢下全部 7/7，逐 clip 均值 `0.2417/0.1946`，提升约
`24.2%`，说明 seed2 的总体收益不是单一动作驱动。

## 130. 双更新 recipe 延长到 1500 iterations：1000 后仍有收益

从 subject5 seed0 的 1000-step checkpoint 继续训练到 1500，保持当前完整 recipe 不变。
在新固定起点下，1500/1000 的无扰动 return 为 `0.2657/0.2362`，完整扰动为
`0.2105/0.1990`；first-failure step 分别为 `48.08/46.95` 与 `39.80/40.92`。
1500-step 在两种 context 都提高 return（扰动提升较温和），没有显示 1000 后立即过拟合。
因此后续正式比较应把训练预算作为显式变量，当前 1000 是可复现基准而非已证明的最优停止点。

## 131. 1500-step 多起点扰动复核

对 subject5 seed0 的 1500-step continuation 与备份的 1000-step checkpoint，使用新 start
seeds 1029/1030/1031 做完整扰动评估。三组 return 分别为 `0.2456/0.2432`、
`0.2060/0.2028`、`0.2210/0.2195`（1500/1000），1500-step 三组全部胜出，均值
`0.2242/0.2218`，提升约 `1.1%`。长训练的收益较小但方向一致，支持把 1500 作为后续
预算扫描点，而不是把单次提升归因于固定起点噪声。

## 132. 2000-step continuation：收益继续增长

从 1500-step checkpoint 继续到 2000 iterations，在 subject5 seed0 的新起点上比较
2000/1500/1000。无扰动 return 为 `0.2766/0.2417/0.2346`，完整扰动为
`0.2647/0.2388/0.2329`。2000-step 在两种 context 都明显胜出，且 first-failure step
更晚；当前没有证据表明 1000 或 1500 是最优停止点。后续报告应同时给出固定训练预算
曲线，而不能只比较单一末端 checkpoint。

## 133. 2000-step 多起点复核

对 2000/1500/1000-step checkpoint 使用三个新扰动 start seeds 1034/1035/1036：
结果分别为 `0.2426/0.2215/0.2165`、`0.2658/0.2484/0.2234`、
`0.2507/0.2242/0.2137`。2000-step 在 3/3 起点胜出，均值
`0.2531/0.2314/0.2178`，相对 1000-step 提升约 `16.2%`，相对 1500-step 提升约
`9.4%`。这使长预算收益不再依赖单一起点。

## 134. 长预算收益跨 subject4 留出 split

将 subject4 seed0 的双更新 checkpoint 从 1000 continuation 到 2000。在两个新起点上，
2000/1000 的无扰动 return 为 `0.2888/0.2577`，完整扰动为 `0.3573/0.2979`；两种
context 均由 2000-step 胜出，first-failure step 也更晚。结合 subject5 的 2000-step
多起点结果，训练预算收益已跨两个留出 split 复现。

## 135. Subject4 2000-step 额外起点复核

在 subject4 上新增扰动 seeds 1039/1040，2000/1000 return 分别为
`0.2986/0.2415` 和 `0.3043/0.2664`；两组均由 2000-step 胜出，均值
`0.3015/0.2539`，提升约 `18.7%`。这进一步排除了 subject4 长预算收益的固定起点解释。

## 136. Subject5 第二个训练 seed 的 2000-step 复核

subject5 seed1 从 1000 continuation 到 2000 后，在新起点上的 return 为：无扰动
`0.2272/0.1967`、完整扰动 `0.2262/0.2064`（2000/1000）。两种 context 都由
2000-step 胜出，提升约 `15.5%/9.6%`；结合 seed0，subject5 的长预算收益已在两个
训练 seed 上复现。

## 137. Reproduction budget promoted to 2000 iterations

长预算在 subject4 seed0、subject5 seed0/seed1 上均通过新扰动起点复核，2000-step
checkpoint 一致高于各自 1000-step checkpoint。已将 `tools/run_concurrent_ramp.sh` 的
默认 `ITERATIONS` 从 1000 提升到 2000；设置 `ITERATIONS=1000` 仍可复现短预算 baseline。
这样脚本默认对应当前证据支持的 leading recipe，而不是较弱的早期预算。

## 138. Subject5 seed2 的公平长预算单更新对照

将 subject5 seed2 的单更新 baseline 与双更新 candidate 都从各自 1000-step checkpoint
继续到 2000 iterations，并用同一批扰动 seeds 1045/1046、同一评估命令做配对比较。
双更新/单更新 return 分别为 `0.2168/0.2139` 与 `0.2567/0.2376`；两组都支持双更新，
均值 `0.2367/0.2257`，提升约 `4.9%`。对应的单更新 1000-step checkpoint 在同一
评估中的均值仅为 `0.1729`，说明长预算本身有明显收益，但不能解释双更新相对单更新的
全部增益。该结果与 seed0/seed1 的长预算复核方向一致，进一步支持将双更新保留在默认
recipe 中，同时把训练预算固定为 2000 作为当前复现实验起点。

## 139. Subject4 公平长预算更新频率对照

subject4 seed0 的单更新 baseline 从 1000-step checkpoint 继续到 2000，与双更新 candidate
在相同留出 split 和新起点上配对评估。扰动 seeds 1047/1048 的双更新/单更新 return 为
`0.3160/0.2854`、`0.2890/0.2613`，均值 `0.3025/0.2734`，提升约 `10.6%`；无扰动
seeds 1049/1050 为 `0.2659/0.2659`、`0.2780/0.2609`，均值 `0.2720/0.2634`，提升
约 `3.3%`。因此双更新的长预算优势在 subject4 与 subject5 两个留出 split 上均保持，
且不依赖单一扰动起点。

## 140. 旧 checkpoint 恢复兼容性修复

旧版 `--no_critic` checkpoint 可能只保存 WM/policy 权重和 target WM，不含 critic 或优化器
状态。训练恢复逻辑现已对这些键采用条件加载，缺失时保留新初始化模块；这使旧的 1000-step
baseline 可以安全 continuation 到 2000-step，而不改变包含完整状态的新 checkpoint 行为。

## 141. Subject4 逐 clip 扰动审计

在 subject4 的 7 个测试 clip 上使用 batch16、100 steps、同一扰动 seed1051，双更新/单更新
return 依次为：dance2 `0.3106/0.2423`、fallAndGetUp1 `0.3701/0.3495`、
 fightAndSports1 `0.2978/0.2826`、run2 `0.2001/0.1839`、sprint1 `0.3819/0.3293`、
walk2 `0.3290/0.3391`、walk3 `0.3122/0.3169`。双更新在 5/7 clip 胜出，均值
`0.3145/0.2919`，提升约 `7.7%`；退化集中在两个 walk clip，提示后续可做动作条件化分析，
但不支持用单一失败 clip 否定总体收益。

## 142. Subject4 逐 clip 审计的第二扰动 seed

用独立扰动 seed1052 重复上一节的 7 个 clip 审计，双更新/单更新 return 为：dance2
`0.3168/0.2838`、fallAndGetUp1 `0.3440/0.3292`、fightAndSports1 `0.2719/0.2383`、
run2 `0.2131/0.1830`、sprint1 `0.3774/0.3402`、walk2 `0.2935/0.3215`、walk3
`0.3320/0.2831`。双更新在 6/7 clip 胜出，均值 `0.3070/0.2827`，提升约 `8.6%`。
walk2 的胜负随 seed 翻转，而 walk3 在两次都回退，说明动作条件差异部分依赖起点，
但总体提升并非单一扰动 seed 造成。

## 143. 双更新的 WM 误差与 imagined-support 机制审计

在 subject4 的 2000-step checkpoint 上，8-step replay 审计显示双更新的 teacher-forced WM
all-step error 为 `0.1307`，单更新为 `0.2288`；free-running error 为 `0.7445`，单更新为
`1.4992`。在留出 replay 环境上，双更新的 all-step error `0.7592`、train→audit final-step
差距 `0.2015`，均优于单更新的 `1.6799` 和 `2.0972`。这说明双更新不只是改变真实模拟器
中的策略回报，也改善了 policy 访问分布上的多步 WM 预测及其泛化。

imagined-support 审计（horizon16，独立随机采样，作机制参考而非主指标）显示双更新末步的
state/action support RMS-z 约为 `1.01/1.06`，单更新约为 `1.04/1.16`；双更新的 imagined
轨迹更接近 replay support，但 WM input norm 略高（约 `1.88` 对 `1.75`），提示它学习了
更有活性的动作分布，而不是简单收缩策略。该现象支持保留 free-running auxiliary loss，
不建议把输入范数直接当作 gate 信号。

## 144. Free-running auxiliary 的 2000-step 交互复核

subject5 seed0 的双更新 no-aux checkpoint 从 1000 continuation 到 2000，与当前 aux=`0.05`
recipe 做同预算评估。扰动 seeds 1053/1054 上 aux/no-aux return 为 `0.2582/0.2466`、
`0.2758/0.2769`，均值 `0.2670/0.2617`，提升约 `2.0%`；无扰动 seeds 1055/1056 为
`0.2896/0.2811`、`0.2435/0.2571`，均值 `0.2665/0.2691`，基本持平。对应 8-step WM
审计中 aux 的 free-running all-step error `0.5595`，no-aux 为 `0.8286`，teacher-forced
为 `0.0968/0.1106`。因此 auxiliary 在长预算下仍稳定改善多步 WM 校准，policy return 的
独立增益较小但没有系统性反转，保留 `.05` 作为默认值。

## 145. Subject5 跨 split WM 泛化审计

在 subject5 的 2000-step checkpoint 上，aux=`0.05` 的 8-step replay all-step error 为
`0.7019`（audit 环境 `0.5415`），no-aux 为 `0.9023`（audit `0.8635`）；final-step
train/audit 分别为 aux `1.3816/1.0527`、no-aux `1.9458/1.8395`。auxiliary 同时降低
多步误差和 train/audit 误差水平，且该方向与 subject4 一致，支持它是跨 split 的 WM 校准
机制，而非只对某个留出动作集有效。

## 146. 双 policy update 与 WM 精度的解耦

subject5 seed2 的 2000-step 公平 checkpoint 对照显示，双更新 policy 的 8-step
free-running WM all-step error 为 `0.6621`，单更新为 `0.6171`；audit replay all-step
为 `0.6538/0.6034`，双更新并没有带来更低的 WM 误差。此前相同 checkpoint 的真实
模拟器 return 仍由双更新胜出（扰动均值 `0.2367` 对 `0.2257`）。因此双 policy update
的收益不能解释为 WM 精度提升，更可能来自 policy 优化频率和行为适应；auxiliary 的
WM 校准作用与 policy update 频率的作用需要分开表述。

## 147. 默认复现实验入口 smoke check

用 `tools/run_concurrent_ramp.sh` 做 10-iteration 启动检查成功。生成的 `args.json` 确认默认
入口包含 `iterations=2000`（smoke 中通过环境变量覆盖为 10）、`policy_updates=2`、
`wm_aux_weight=0.05`、`wm_aux_ramp_iters=300`、`wm_batches=2`、teacher forcing 和
`no_critic`；脚本语法及 diff 检查均通过。

## 148. 双 policy update 的 batch 重用 ablation

新增 `--policy_reuse_batch` 控制双更新是否复用同一批 policy start/history。subject5 seed0
上，reuse 与默认独立重采样都训练 1000 steps、其余配置完全一致。四个扰动 seeds
1057/1058/1061/1062 的平均 return 为 `0.2061`（reuse）对 `0.2167`（独立重采样），
独立重采样在 4/4 起点胜出；无扰动 seeds 1059/1060 的均值为 `0.2037/0.2238`，同样
支持独立重采样。结论是第二个 policy update 的收益需要新的 replay batch，不能简化为对同一
批数据重复执行一次 optimizer step。

## 149. Batch reuse ablation 接入标准脚本

`tools/run_concurrent_ramp.sh` 现在支持 `POLICY_REUSE_BATCH=1`，会转发
`--policy_reuse_batch`；默认值 `0` 保持独立 replay 重采样。脚本在该环境变量下的
10-iteration smoke 成功，`args.json` 正确记录 `policy_reuse_batch=true`。

## 150. Subject4 长时域 200-step rollout

在 subject4 的 2000-step 双更新与单更新 checkpoint 上，用 batch64、完整扰动、seeds
1063/1064 做 200-step rollout。双更新/单更新 return 分别为 `0.1558/0.1477` 与
`0.1634/0.1425`，两个起点均胜出；均值 `0.1596/0.1451`，提升约 `10.0%`。平均
first-failure step 为 `57.5/53.7`，双更新也更晚失败。该结果把长时域收益从 subject5
扩展到 subject4。

## 151. Batch reuse 的 2000-step continuation 复核

将 subject5 seed0 的 batch-reuse checkpoint 从 1000 continuation 到 2000，与独立重采样的
2000-step candidate 做同预算比较。新扰动 seeds 1065/1066 上，reuse/independent 为
`0.2507/0.2562`、`0.2295/0.2476`，均值 `0.2401/0.2519`；无扰动 seeds 1067/1068
为 `0.2386/0.2596`、`0.2379/0.2427`，均值 `0.2383/0.2512`。独立重采样在 4/4
起点胜出，说明 batch reuse 的劣势在长预算下仍存在。

## 152. Batch diversity 的 action-support 负审计

对 subject5 两个 2000-step checkpoint 做 4096 个 replay start 的 policy-support 审计，reuse
与独立重采样的 action distance mean 分别为 `0.01368/0.01338`，p90 为 `0.02071/0.02026`；
与 one-step WM error 的相关性分别为 `0.0069/-0.0019`。两者 action support 几乎一致，
因此独立 batch 的 return 优势不能简单解释为策略更接近 replay action，后续应转向梯度
多样性和优化轨迹分析。

## 153. walk3 回退的终止模式审计

对 walk3_subject4 做 seeds 1069/1070、batch64、100 steps 的完整扰动诊断。双/单更新
return 为 `0.3313/0.3214` 与 `0.3165/0.3087`；双更新 first-failure 为 `52.8/48.8`
对单更新 `54.1/49.6`。两种策略的 tilt failure 都接近 0（双更新 `6.25%/0%`，单更新
`4.69%/1.56%`），失败主要来自 height failure（双更新 `79.69%/92.19%`，单更新
`84.38%/90.63%`）。因此 walk3 的动作差异主要是高度轨迹的细微变化，而非姿态稳定性
灾难；不支持加入 tilt-specific gate。

## 154. Policy gradient diversity audit

新增 `tools/policy_batch_gradient_audit.py`，在固定 checkpoint 上对两批独立 replay start
计算 64-step imagined-return policy gradient。subject5 seed0 的 2000-step 独立重采样
checkpoint，8 对 batch 的 cosine 均值为 `0.3256`（范围 `0.1622–0.3917`），reuse
checkpoint 为 `0.3620`（`0.2575–0.5031`）；两者 positive-dot rate 都是 `1.0`。
因此两批梯度总体同向，但独立重采样提供更大的方向多样性，与其更高的真实 rollout return
一致；该结果是机制支持，不单独构成因果证明。

## 155. Subject4 gradient diversity cross-split audit

在 subject4 的 2000-step checkpoint 上重复 8 对 batch 的 audit：双更新 checkpoint 的
独立 policy-gradient cosine 均值为 `0.3630`（范围 `0.1739–0.5586`），单更新 checkpoint
为 `0.4167`（`0.3060–0.5374`），两者 positive-dot rate 都是 `1.0`。双更新方向更
分散的现象与 subject5 一致，但 subject4 没有 batch-reuse 对照，因此这里只作为跨 split
支持性证据，不宣称独立因果识别。

## 156. Gradient diversity 的训练 seed 敏感性

subject5 seed1 的 2000-step 双更新 checkpoint 做同样的 8 对 batch audit，cosine 均值为
`0.4767`（范围 `0.2002–0.6844`），positive-dot rate 为 `1.0`。结合 subject5 seed0
的 `0.3256` 与 subject4 的 `0.3630`，跨 checkpoint 的梯度相似度有明显 seed 变化；
稳健结论是独立 batch 产生总体同向但多样性可变的 policy 方向，不应设定固定 cosine 阈值
作为训练 gate。

## 157. WM-policy 交错顺序 ablation

新增 `--interleave_policy_wm`，将默认的 `WM×2 → policy×2` 改为
`WM → policy1 → WM → policy2`。subject5 seed0、1000 steps 的四个扰动 seeds
1071/1072/1075/1076 上，标准/交错 return 分别为 `0.2333/0.2276`、`0.2452/0.2466`、
`0.2404/0.2361`、`0.2201/0.2117`；标准顺序 3/4 胜出，均值 `0.2347/0.2305`，
交错下降约 `1.8%`。无扰动 seeds 1073/1074 也略低（`0.2382/0.2328`）。因此当前
recipe 保持先完成 WM batches、再进行两次独立 policy update；交错刷新 WM 暂不启用。

## 158. 交错顺序的 2000-step continuation

将 interleave checkpoint 延长到 2000 后，subject5 新扰动 seeds 1077–1080 的交错/标准
return 为 `0.2323/0.2335`、`0.2375/0.2257`、`0.2772/0.2708`、`0.2454/0.2384`，
均值 `0.2481/0.2421`，各胜 2/4；无扰动 seeds 1081/1082 为 `0.2308/0.2320`、
`0.2706/0.2826`，均值 `0.2507/0.2573`。长预算结果依赖 context，未形成稳定优势，
因此保持标准 `WM×2→policy×2` 默认顺序。

## 159. `pi_batch=64, policy_updates=1` 的 2000-step proxy

将此前一次大 batch proxy 从 1000 continuation 到 2000，与标准 `pi_batch=32,
policy_updates=2` 做同预算比较。subject5 新扰动 seeds 1083–1086 的 proxy/双更新
return 为 `0.2639/0.2623`、`0.2657/0.2743`、`0.2508/0.2468`、`0.2498/0.2623`，
均值 `0.2576/0.2614`；无扰动 seeds 1087/1088 为 `0.2487/0.2584`、`0.2610/0.2604`，
均值 `0.2549/0.2594`。扩大一次 policy batch 仍不能稳定追上两次独立更新，支持“额外
sequential adaptation”而非单纯 batch size 的解释。

## 160. 两批 policy gradient 累积但单次 optimizer step

新增 `--policy_accumulate_updates`：双更新时对两个独立 replay batch 累积 policy gradient，
只执行一次 optimizer step。subject5 seed0、1000 steps 的四个扰动 seeds 1089–1092 上，
累积/标准 sequential return 为 `0.1901/0.2168`、`0.2192/0.2492`、`0.1740/0.2025`、
`0.1953/0.2431`，均值 `0.1946/0.2279`；无扰动 seeds 1093/1094 为
`0.1834/0.2218`、`0.2000/0.2385`，均值 `0.1917/0.2301`。累积方案在所有起点都明显
更差，说明 leading recipe 的收益来自 sequential policy adaptation，而不是把两批梯度
平均后做一次更大更新。该 flag 默认关闭。

## 161. 梯度累积的 2000-step continuation

将 accumulation checkpoint 延长到 2000 后，subject5 新扰动 seeds 1095–1098 的
accum/sequential return 为 `0.2434/0.2618`、`0.2313/0.2445`、`0.2409/0.2440`、
`0.2410/0.2421`，均值 `0.2392/0.2481`；无扰动 seeds 1099/1100 为
`0.2500/0.2610`、`0.2516/0.2535`，均值 `0.2508/0.2573`。六个起点全部支持
sequential。差距较 1000 steps 缩小，但未消失，说明长预算可部分补偿而不能替代连续
policy adaptation。

## 162. Accumulation ablation 的脚本复现检查

`tools/run_concurrent_ramp.sh` 在 `POLICY_ACCUMULATE_UPDATES=1` 下的 10-iteration smoke
成功，`args.json` 同时记录 `policy_updates=2`、`policy_accumulate_updates=true`、
`policy_reuse_batch=false`、`interleave_policy_wm=false`，确认 accumulation 是独立的
controlled ablation，不会意外启用其它更新顺序或 batch reuse。

## 163. Subject4 长时域 500-step rollout

subject4 双/单更新 2000-step checkpoint 在 batch64、完整扰动、500 steps 下继续比较。seeds
1101/1102 的 return 为 `0.0728/0.0642` 与 `0.0622/0.0535`，均值 `0.0675/0.0588`，
提升约 `14.8%`；平均 first-failure step `61.5/55.3`。双更新在两个起点均保持更晚失败，
长时域优势延伸到 500 steps。

## 164. Subject5 同种子 2000-step 单/双策略更新的 500-step rollout

为避免把不同训练预算混为公平对照，重新核对 checkpoint 元数据：`logs/loso5_polyupdates2_s0_1000/last.pt` 内部 `iteration=2000`，它是从 1000-step checkpoint 正确续训到 2000 步的双更新模型；单更新模型为同一 seed0 从头训练的 `logs/loso5_polyupdates1_s0_2000/last.pt`。在 subject5、两个扰动起点、500-step rollout 下，双/单更新 return_norm 分别为 `0.05590/0.05137` 与 `0.04587/0.03868`，均值 `0.05088/0.04502`，双更新相对提升约 `13.0%`。平均 first_failure_step 为 `45.53/43.48`，提升约 `4.7%`。这为双更新在更长 horizon 上的收益提供了同 seed 的额外支持，但仍是单个 subject、两个起点，不能替代跨 subject 复核。

## 165. Subject5 500-step 无扰动配对复核

使用同一 subject5、seed0、2000-step 双/单策略更新 checkpoint，在两个无扰动 rollout 起点上评估 500 步。双更新 return_norm 为 `0.05531/0.04662`，单更新为 `0.05129/0.03903`，均值 `0.05096/0.04516`，双更新提升约 `12.8%`。first_failure_step 均值为 `45.59/43.37`，提升约 `5.1%`。与扰动评估的约 13.0% 提升一致，说明该长时域收益并不依赖测试扰动。

## 166. Subject4 500-step 无扰动跨 subject 复核

subject4 的双更新 checkpoint `logs/loso4_polyupdates2_s0_1000/last.pt` 与单更新 checkpoint `logs/loso4_ramp_s0_1000/last.pt` 均核对为 `iteration=2000`。两个无扰动起点的 500-step rollout 中，双/单更新 return_norm 为 `0.07048/0.06364` 与 `0.06347/0.05426`，均值 `0.06698/0.05895`，双更新提升约 `13.6%`。first_failure_step 均值为 `60.41/55.35`，提升约 `9.1%`。结合 subject5 的无扰动与扰动结果，双更新的长时域收益已在两个 held-out subject 上复现。

## 167. 两个 subject 的 paired bootstrap 汇总

将 subject4/5 的四个无扰动 500-step 固定起点配成双更新/单更新 pairs，四个 pair 全部由双更新胜出。return_norm 的逐起点差值均值为 `+0.00691`；按四个起点做 paired bootstrap（200,000 次重采样），差值 95% 区间为 `[+0.00491,+0.00861]`，按两组均值计算的相对提升为 `13.28%`，bootstrap 95% 区间为 `[9.45%,17.997%]`。样本仍只有两个 subject、每个两个起点，区间只量化当前固定起点集合的不确定性。

## 168. Compute-matched 对照：单更新 4000 steps vs 双更新 2000 steps

为匹配总 optimizer update 量，在 subject5 seed0 上训练同配置单更新 4000 iterations（`logs/loso5_polyupdates1_s0_4000/last.pt`），与双更新 2000-step checkpoint 比较；两者均为每轮 `wm_batches=2`，因此单更新 4000 steps 与双更新 2000 steps 具有近似相同的 WM 更新数和 policy 更新数。500-step、两个无扰动起点：双更新 return `0.05531/0.04662`，单更新 4000 `0.05810/0.05246`，均值 `0.05096/0.05528`，单更新约高 `8.5%`；扰动起点：双更新 `0.05590/0.04587`，单更新 `0.05870/0.05224`，均值 `0.05088/0.05547`，单更新约高 `9.0%`。first_failure_step 也由单更新改善（无扰动均值 `48.43` vs `45.59`，扰动 `48.90` vs `45.53`）。因此双更新的正收益只在固定 iterations/固定环境交互预算下成立，不能宣称 compute-matched 优势；它更可能反映每轮 WM 后立即适应的优化路径，而非更高的总更新效率。

## 169. Compute-matched 曲线的 crossover 检查

训练日志按总 optimizer update 量对齐（双更新第 `k` 轮对应单更新第 `2k` 轮）后，subject5 seed0 的单更新 4000-step 曲线在所有可对齐点都不低于双更新 2000-step：sim reward 在 k=`1000/1200/1500/1800/2000` 时，双更新为 `0.474/0.479/0.486/0.513/0.520`，单更新对应为 `0.491/0.496/0.499/0.520/0.536`；episode return 也同方向。因而 compute-matched 反例不是只在最终 checkpoint 出现，而是从中段开始持续存在。双更新的优势应表述为固定 interaction/iteration budget 下的 sample-efficiency 候选，而不是 update-efficiency 或 wall-clock 优势。

## 170. 近似 wall-clock 对照：单更新 3000 vs 双更新 2000

subject5 seed0 日志中，双更新 2000 steps 用时约 `0.153 h`，单更新 3000 steps 用时约 `0.145 h`，因此补做 500-step rollout。无扰动两个起点的双/单更新 return 为 `0.05531/0.05725` 与 `0.04662/0.04662`，均值 `0.05096/0.05194`，单更新约高 `1.9%`；扰动为 `0.05590/0.05761` 与 `0.04587/0.04604`，均值 `0.05088/0.05182`，单更新约高 `1.8%`。first-failure 也基本持平、略偏向单更新。故在近似 wall-clock 对齐时，双更新没有稳定优势；其主要优势仍限于较少 environment iterations 下的早期样本效率。

## 171. 自适应 policy update schedule：前半程双更新、后半程单更新

新增 `--policy_updates_until_iter`，在指定 post-WM iteration 后把 policy update 数从 2 降为 1。subject5 seed0 采用前 1000 iterations 双更新、后 1000 iterations 单更新（总计 3000 次 policy update），500-step 无扰动 rollout 的双/自适应/单更新均值为 `0.05096/0.04896/0.04516`；扰动均值为 `0.05088/0.04867/0.04502`。自适应相对固定单更新提升约 `8.4%/8.1%`，但比固定双更新低约 `3.9%/4.3%`；first-failure 也处于中间。结论：第二次 update 的价值不只集中在前半程，简单的 1000-step cutoff 会损失固定双更新的收益；后续应测试基于收益/梯度/不确定性的动态 gate，而不是固定时间切换。

## 172. 动态 gate 信号审计

对 subject5 seed0 的固定双更新、固定单更新和前半双/后半单日志做初步信号审计。固定双更新中 `wm_err8` 与 `sim_rew` 的相关系数约 `-0.75`，而 `pi_gradnorm` 与 `sim_rew` 约 `-0.09`；单更新日志中 gradient norm 相关性方向甚至翻转（约 `+0.50`），说明 raw gradient norm 不能直接作为跨 schedule 的第二次更新 gate。WM error 是更稳定的控制质量指标，但它反映的是模型质量而非第二次 policy step 的边际收益。结合固定 cutoff 实验，下一步 gate 应测量“第一次 policy update 后的真实/想象收益增量”或 gradient agreement，而不是使用绝对 gradient norm 或固定 iteration。

## 173. Gradient-agreement 审计的初步结果

使用 `policy_batch_gradient_audit.py`、H=32/trunc=4、batch=16、4 对独立 replay batches：固定双更新 2000 checkpoint 的 cross-batch cosine 均值为 `0.231`（范围 `0.049–0.360`），前半双/后半单自适应 checkpoint 为 `0.235`（`0.171–0.321`），固定单更新 checkpoint 为 `0.303`（`0.137–0.551`）；三者 positive-dot rate 均为 `1.0`。自适应策略没有产生更高的 gradient agreement，且三种 schedule 的方向均为正。因此“cosine>0”本身过于宽松，不能解释固定双更新的收益；若继续做 gate，需要比较第一次 update 的实际 return delta 或更严格的 margin，而不是只用符号一致性。

## 174. 虚拟 policy-update delta 审计

新增 `tools/policy_update_delta_audit.py`：冻结 WM，在 replay 上克隆 policy，连续执行两个虚拟 Adam step，并在新 batch 上测每步 policy-loss delta。subject5 seed0、H=16、batch=16、6 pairs 的结果：固定双更新的 delta1/delta2 均值约 `-0.00001/-0.00032`，正向率 `0.50/0.33`；前半双/后半单为 `-0.00489/+0.00068`，正向率 `0.17/0.67`；固定单更新为 `-0.00217/-0.00005`，正向率 `0.17/0.50`。信号量级很小且 batch 方差大，不能直接作为 gate；但它提供了可执行的“边际 update value”测量接口。后续若使用，应改为多 batch 聚合并直接测 imagined return delta，而非单个 policy loss。

## 175. 加载真实 Adam 状态后的 delta 审计

修正 `policy_update_delta_audit.py` 以加载 checkpoint 的 `opt_pi` momentum/variance 状态。H=32、batch=32、8 pairs 的 held-out policy-loss 改变量（正值代表 loss 降低）为：固定双更新 delta1/delta2=`+0.00592/+0.00374`，正向率 `0.875/1.0`；自适应为 `+0.00235/-0.00251`，正向率 `0.75/0.0`；固定单更新为 `+0.00762/+0.00248`，正向率 `1.0/0.875`。真实 optimizer state 使固定双更新的第二步 delta 稳定为正，但自适应 checkpoint 的第二步为负；这仍不能直接预测最终 real-sim return，说明 held-out loss delta 与 deployed control gain 之间存在解耦。该审计可用于诊断，但暂不作为在线 gate。

## 176. Subject4 compute-matched 对照

subject4 seed0 上完成单更新 4000 iterations（`logs/loso4_polyupdates1_s0_4000/last.pt`），与双更新 2000 checkpoint 做 500-step rollout。无扰动两个起点：双/单 return=`0.07048/0.07926`、`0.06347/0.06909`，均值 `0.06698/0.07417`，单更新高约 `10.8%`；扰动：双/单=`0.07277/0.07979`、`0.06223/0.07025`，均值 `0.06750/0.07502`，单更新高约 `11.1%`。单更新 first-failure 也在四个起点中总体更晚。subject4 与 subject5 均复现：双更新的固定-iteration 优势在 compute-matched 4000-step 单更新下消失并反转。

## 177. Per-component WM weighting 接口 smoke

新增 `--wm_component_weights`，接受按 `position(3), rotation(3), linear_velocity(3), angular_velocity(3), q(29), qd(29)` 排列的 70 个正权重。默认空字符串保持原有 uniform loss；`--wm_root_height_weight` 在自定义向量上继续作为乘法修正。使用 70 个全 1 权重完成 1-iteration Isaac smoke，参数解析、WM loss 构造和 checkpoint 初始化均通过。该步骤只验证接口，不宣称任何控制性能收益；下一步需注入有来源的 error-propagation 权重再做 matched-budget ablation。

## 178. Kinematic sensitivity 权重生成器

新增 `tools/derive_kinematic_component_weights.py`，通过 G1 tracked-body position 对 q 的 Jacobian、以及 tracked-body angular velocity 对 qd 的 Jacobian 生成 70 维 WM component-weight 向量。q/qd 组内归一化为均值 1，并使用 `min_weight=0.25` 防止关节监督被完全抹掉。生成文件为 `docs/kinematic_component_weights.csv` 及对应 JSON metadata；q 权重范围 `0.234–2.500`，qd 范围 `0.673–1.780`。使用该向量完成 1-iteration Isaac smoke，参数解析和 loss 构造通过；尚无性能结论。

## 179. Jacobian-based WM weighting 的首轮性能筛选

subject5 seed0、1000 iterations、其它配置与 nominal 双更新 baseline 完全一致，仅替换 70 维 kinematic sensitivity 权重。500-step rollout 的 nominal/weighted return：无扰动 `0.05531/0.04621` 与 `0.04662/0.03544`，均值 `0.05096/0.04082`（weighted 下降约 `19.9%`）；扰动 `0.05590/0.04665` 与 `0.04587/0.03548`，均值 `0.05088/0.04106`（下降约 `19.3%`）。四个起点全部下降，first-failure 也更早。该结果说明单纯按 tracked-body position/angular-velocity Jacobian 重分配 WM loss 会牺牲均匀 dynamics fidelity；停止继续扫描该权重幅度，保留生成器作为可复现实验工具。

## 180. Kinematic reset-cluster imbalance audit

新增 `tools/audit_motion_clusters.py`，对 1-second motion bins 的 root local velocity/angular velocity、q、qd 特征做 16-cluster audit，不使用 policy/reward。subject5 train split 有 7210 bins，cluster count 范围 `114–1032`，entropy `2.641/2.773=95.2%` of maximum；subject4 train split 有 7196 bins，范围 `133–1573`，entropy `2.568/2.773=92.6%`。数据存在中等 cluster imbalance，但并非极端长尾；balanced reset sampling 值得作为低优先级 data-coverage ablation，不应假设会直接提升 concurrent WM-policy control。

## 181. Motion-cluster audit 可重复性

两个 subject 的 16-cluster audit 以固定 seed 重跑后，cluster counts 与 entropy 均逐字段一致。cluster imbalance 结论不是随机初始化偶然现象。

## 182. Mild Jacobian weighting 复核

为排除极端权重导致退化，将 Jacobian 权重裁剪到约 `[0.75,1.25]` 并保持 q/qd 组均值 1，在 subject5 seed0、1000 iterations 下复训。500-step rollout：无扰动 nominal/mild 均值 `0.05096/0.03909`（mild 下降约 `23.3%`）；扰动 `0.05088/0.03874`（下降约 `23.9%`）。退化没有因裁剪而消失，说明当前 downstream control 更依赖均匀 WM dynamics fidelity；停止继续探索该 weighting 分支。

## 183. Teacher-forced/free-running consistency target 的首轮筛选

新增 `--wm_consistency_weight`，在保持 nominal 4-step teacher-forced 主损失和 8-step auxiliary 的前提下，对 free-running rollout 与 detached teacher-forced rollout 增加一致性项。`.01` 权重的 300-step smoke：WM err8=`0.1610`，接近 ramp baseline 的约 `0.1598`；sim reward=`0.3821`，略高于 smoke baseline 的约 `0.3668`。随后在 subject4 两个 seed、1000 iterations 正式运行，训练末端 sim reward 为 `0.3972/0.396`，WM err8 为 `0.1493/0.1489`。由于当前固定起点 500-step rollout 尚未显示稳定的跨 seed 增益，且训练期 sim reward 的提升不足以替代真实 rollout 证据，本轮暂不纳入 leading recipe；保留该接口，下一步应直接用真实转移 target 或不确定性校准目标构造一致性，而不是继续扫描标量权重。

## 184. Termination-aware WM target 的真实 rollout 复核

新增 `--wm_termination_weight`，把 replay 中真实 `done` 事件与 WM 预测状态触发的 soft termination 做 BCE 对齐。`.05` 权重 smoke 稳定（300 iterations 的 WM err8=`0.1655`，sim reward=`0.373`），随后 subject4 两个 seed 各训练 1000 iterations。固定起点 500-step clean rollout 的 return_norm：seed0 termination/ramp=`0.15317/0.15150`（+1.1%），seed1=`0.03500/0.03635`（-3.7%）；合并均值约 `+0.2%`，first-failure 方向也相反。该 target 没有形成可重复控制收益，停止继续调权重，保留接口作为终止校准诊断。

## 185. EMA world model for policy imagination

测试已有 `--policy_ema_model --wm_ema_beta .995 --policy_ema_start 5`：nominal WM 继续用真实 replay 更新，policy imagination 改用 EMA WM。300-step smoke 的训练期指标看似改善（err8=`0.1473`、sim reward=`0.398`），但 EMA 分支触发 torch dynamo 重编译，1000-iteration wall-clock 约 `0.08 h`，普通 ramp 约 `0.05–0.06 h`。更关键的是固定起点 500-step clean rollout 反转：seed0 EMA/ramp=`0.12919/0.15150`（-14.7%），seed1=`0.03952/0.03635`（+8.7%），合并约下降 `10.2%`。这说明滞后的 EMA dynamics 会造成 imagined-policy 与部署时 nominal dynamics 的错配；停止该配置，不把训练期 sim reward 当作收益证据。

## 186. Occupancy-conditioned WM replay：三 seed、clean/perturb 真实复核

启用 `--wm_occupancy_weight .5 --wm_occupancy_sigma .05 --wm_occupancy_floor .25`，按当前 policy 在 replay window 起点的动作与历史动作距离，对真实 next-state WM loss 做 detached 重加权；不改变 policy objective，也不改变真实 target。三 seed 均为 1000 iterations，预算审计显示每个 checkpoint 都是 `sim_steps=160512`、`policy_steps=995`、`wm_steps=2000`、`total_steps=2995`，candidate wall-clock `0.051 h`，ramp baseline `0.048–0.062 h`。

固定起点 500-step rollout 的 return_norm：

| condition | seed0 | seed1 | seed2 | 三 seed均值相对提升 |
|---|---:|---:|---:|---:|
| clean candidate / ramp | 0.16730 / 0.15149 | 0.03694 / 0.03635 | 0.03223 / 0.03143 | **+7.8%** |
| perturb candidate / ramp | 0.16671 / 0.13538 | 0.03603 / 0.03569 | 0.03212 / 0.03129 | **+16.1%** |

六个 paired 条件全部 candidate 胜出，first-failure 在 seed0 clean/perturb 也更晚，其他两个 seed 基本持平。该结果是当前最强的 concurrent 候选：policy 通过 occupancy 改变 WM 学习分布，同时 nominal WM 仍由真实 replay fidelity 约束。下一步应做 subject5 跨 split 复核和 sigma/weight 小范围敏感性，而不是直接改主 loss。

## 187. Occupancy replay 的 subject5 跨 split 反例

按相同单 policy-update/ramp 配方在 subject5（训练 `s1234`、测试 `test_s5`）复核，seed0 500-step clean rollout：occupancy=`0.03881`，ramp baseline=`0.04095`，下降约 `5.2%`。因此 subject4 三 seed 的一致收益不能直接外推到未见 subject。

## 188. Occupancy replay 与双 policy update 的交互

在 subject5 使用当前 leading 的 `policy_updates=2` 重新训练 occupancy candidate，500-step clean rollout 为 `0.05439`，同 seed 双更新 baseline=`0.06710`，下降约 `19.0%`。occupancy replay 与双更新并非简单可叠加；当前证据支持将其视为 subject/context-conditioned coupling，而不是全局默认。

## 189. Mild occupancy replay：提高 uniform floor 的跨 split 复核

为缓解 subject5 上强 occupancy（weight=.5/floor=.25）的 replay 偏移，测试 `weight=.25, floor=.75, sigma=.05`。subject5 seed0 的 500-step rollout：clean=`0.04305/0.04095`、perturb=`0.04312/0.04114`（均约 +5%）。同配置 subject4 seed0：clean=`0.14760/0.15150`（-2.6%）、perturb=`0.14693/0.13538`（+8.5%）。提高 uniform floor 确实消除了 subject5 的严重退化，但没有产生跨 split、跨 context 的一致提升；保留为稳定性对照，不升级为默认。

## 190. Mild occupancy 的 subject4/5 配对结果

同一 mild 配方（weight=.25/floor=.75/sigma=.05）在两个 split 各完成至少一个 seed 的 clean/perturb 复核：subject4 `0.14760/0.15150` clean、`0.14693/0.13538` perturb；subject5 `0.04305/0.04095` clean、`0.04312/0.04114` perturb。它相较强 occupancy 更稳，但收益符号仍依赖 context，因此后续若要继续，应加入基于实时 WM validation error 的退化检测/回退，而不是继续盲扫 occupancy 标量。

## 191. Validation-error gated occupancy replay

新增 `--wm_occupancy_error_ratio`：在同一 replay batch 上比较 occupancy-weighted 与 uniform WM real-transition loss；前者超过 `ratio ×` 后者时，该 batch 回退 uniform。`weight=.25/floor=.75/ratio=1.05` 在 subject5 seed0 的 500-step rollout：clean=`0.04357/0.04094`（+6.4%），perturb=`0.04390/0.04114`（+6.7%）。但同配置 subject4 seed0：clean=`0.14005/0.15150`（-7.6%），perturb=`0.13989/0.13538`（+3.3%）。该 gate 能修复 subject5 的 replay 偏移，却会抑制 subject4 clean 的有效 coupling；保留为 adaptive-coupling 诊断接口，不设为全局默认。

## 192. Independent replay validation EMA for occupancy fallback

将 `--wm_occupancy_error_ratio` gate 改为独立 replay validation：每个 WM update 额外抽取未参与梯度的窗口，计算 teacher-forced WM error，并用 `--wm_occupancy_validation_beta=.9` 维护 EMA；当 validation error 相对上一 EMA 超过 ratio=`1.05` 时，当前 occupancy batch 回退 uniform。相比同 batch gate，这个信号不使用训练 batch 自证。

`weight=.25/floor=.75/ratio=1.05/beta=.9` 的 1000-iteration、单更新复核：subject5 seed0 clean=`0.04938/0.04095`（+20.6%），perturb=`0.04855/0.04114`（+18.0%）；subject4 seed0 clean=`0.14434/0.15149`（-4.7%），perturb=`0.14429/0.13538`（+6.6%）。独立 validation 明显改善 subject5，但 subject4 clean 仍有损失，说明 replay coupling 的收益依赖 context；该接口适合作为诊断与后续自适应研究，不作为全局默认。

## 193. Independent validation ratio=1.10 smoke

将独立 validation EMA gate 的 fallback ratio 从 `1.05` 放宽到 `1.10`，subject4/5 各做 300-step smoke。subject4 sim reward=`0.378`，subject5=`0.380`，均低于 ratio=`1.05` 同配方约 `0.402`。放宽阈值会重新允许较多高误差 occupancy 更新，早期 control 指标变差；因此不进入 1000-iteration 长跑，保留 `1.05` 作为当前诊断阈值。

## 194. Independent validation EMA 的 subject5 seed1 复核

为核对 subject5 seed0 的大幅收益，重新训练同配置 seed1，并新建同预算单更新 baseline。500-step test_s5 rollout：clean candidate/baseline=`0.01845/0.02064`（-10.6%），perturb=`0.01833/0.01996`（-8.1%）。这与 seed0 的 clean/perturb `+20.6%/+18.0%` 方向相反；因此独立 validation EMA 的 subject5 seed0 增益属于高方差现象，不能作为稳定跨 seed 结论。

## 195. 扩大 evaluation batch 后的方差审计

subject5 validation-EMA candidate 与同 seed baseline 使用 `batch=32`、4 个 evaluation seeds 重复 500-step test_s5 rollout。训练 seed0 的 candidate/baseline 均值约 `0.03976/0.03817`（+4.2%）；训练 seed1 约 `0.03254/0.03678`（-11.5%）。合并 8 个 evaluation 条件约 `0.03615/0.03748`（-3.5%）。因此此前 batch=4、单 evaluation seed 的 seed0 +20.6% 是起点评估方差放大的结果，不能支持方法收益。
