# SuperMimic concurrent WM-policy exploration status

## Current conclusion

Directly updating the nominal world model with imagined policy-return gradients
has not produced a stable control gain. The failures include action-path
coupling, residual adapters, EMA/target-only coupling, simple counterfactual
losses, uncertainty filtering, and kinematic reset balancing. Several variants
improve an imagined metric or a short probe while degrading a longer fixed-start
simulator rollout.

The safest architecture currently implemented is an isolated shadow policy
branch. It trains against the experimental target while the live policy remains
untouched. A candidate is promoted only after paired real-simulator probes.

## Evidence boundary

| Area | Current evidence | Decision |
|---|---|---|
| Policy/WM direct coupling | Long-budget and cross-subject regressions | Closed for the nominal WM |
| Kinematic-cluster reset sampling | WM err8 improved in a small matched probe, fixed 100-step return fell `0.0931 -> 0.0824` | Coverage diagnostic only |
| Action recovery | Mixed 1000-iteration cross-seed results; subject4 and subject5 directions are not stable | Diagnostic only |
| Shadow policy branch | Independent optimizers, checkpoints, promote/discard, and live-branch isolation verified | Safe experimental vehicle |
| Probe gate | Paired return LCB, survival, repeated horizons, and optional first-failure guard | Deployment gate |

## Recommended next experiment

Run the shadow branch on the established 4-step teacher-forced + 8-step
free-running recipe with `policy_updates=2`, two motion splits, and at least two
training seeds. Publish live and shadow checkpoints at the same iterations.
Use paired 100/500-step probes, two start seeds, and:

```text
max-relative-drop = 0.05
max-survival-drop = 0.05
max-first-failure-drop = 5 steps
```

Promote only when every horizon and split passes. Otherwise write
`shadow_discard`; do not interpret a higher imagined return or lower replay WM
error as evidence of deployment improvement.

## Reproducibility status

- The full selected container regression suite passes: 49 tests, one expected
  skip.
- The multi-horizon watcher has been exercised on a real shadow checkpoint and
  correctly wrote `shadow_discard` after both 100-step and 500-step probes
  regressed.
- The previously accepted shadow checkpoints still pass when the optional
  first-failure guard is enabled.
- `tools/validate_experiment_matrix.py` validates 7 rows across 3 matching
  budget axes.
- A fresh standard-recipe seed0 shadow run passed the 100/500-step gate at
  iterations 50 and 100 and ended in `shadow_promote`; this remains a
  single-split, short-budget signal pending replication.
- The seed1 replication passed at iteration 50 but failed only at the
  500-step horizon at iteration 100 (`-7.6%` versus `-1.4%` at 100 steps),
  correctly ending in `shadow_discard`. This supports keeping the long-horizon
  gate mandatory.
- A third seed reproduced the pattern on the other split (`-0.7%` at 100
  steps, `-11.5%` at 500 steps, with a 5.25-step earlier failure), again
  ending in `shadow_discard`.
- `shadow_checkpoint_audit.py` now excludes running normalizer buffers from
  policy-distance calculations; the previous near-one distance was an audit
  artifact, not a trainable-policy divergence.
- The corrected audit reports layerwise relative displacement; the standard
  recipe moves both encoder and decoder rather than only the action head.
- An opt-in whole-policy trust-region cap is available through
  `--shadow_policy_max_step_norm`; its container smoke test passes.
- A matched seed1 cap run (`--shadow_policy_max_step_norm 0.05`) passed both
  100/500-step probes at iteration 50 (+3.3% and +1.2%), but failed both at
  iteration 100 (−17.5% and −16.7%). The cap is therefore a useful safety
  mechanism, not a fix for long-horizon world-model/policy mismatch.
- To test the observed tilt-failure shift without changing the live baseline,
  `--shadow_policy_tilt_penalty` now adds a penalty only during shadow updates.
  With cap `.05` and tilt penalty `.02`, seed1 still passed at iteration 50
  (+2.4% mean over the two horizons), but iteration 100 regressed by −8.5%
  (100 steps) and −8.7% (500 steps). Tilt failures decreased, but return also
  fell, so this is a diagnostic arm rather than a fix.
- Added `tools/shadow_stability_audit.py` and exported
  `docs/shadow_stability_audit.json`, joining checkpoint distance with paired
  simulator outcomes. Across the standard runs, relative policy distance near
  `0.20` occurs in both a strong accept (seed0, +40.5% at 500 steps) and clear
  rejects (seed1, −7.6%; seed2, −11.5%). Parameter distance alone is therefore
  not a useful deployment gate. In the cap seed1 run, the rejected candidate's
  dominant signal was a simultaneous +0.47–0.55 tilt-failure delta and
  −0.16–0.18 return delta; failure-mode changes are more informative than
  displacement magnitude.
- `policy_probe_gate.py`, `run_probe_gate.py`, and `watch_probe_gate.py` now
  support optional `--max-height-failure-drop` and
  `--max-tilt-failure-drop` guards. Replaying the cap seed1 iteration-100
  candidate with a 5% tilt guard reports tilt deltas of `+0.51/+0.47` and
  rejects the candidate independently of the return gate.
- Replaying the previously accepted seed0 iteration-100 candidate with the
  same 5% subtype limits also rejects it: return improves by about 40% at both
  horizons, but the 100-step height-failure rate rises by 25 percentage points.
  This is a deliberate conservative boundary, not a gate bug; return gains
  with a new failure mode are not promotion-safe without an explicit tolerance.
- A threshold sweep is archived in `docs/shadow_gate_calibration.json`. No
  single subtype threshold separates all historical positive-return candidates
  from all rejects: a `.15` threshold catches the cap failure and accepts the
  cap iteration-50 candidate, but rejects the standard seed0 iteration-100
  candidate because its height-failure delta is `.25`. Subtype limits should
  therefore be treated as an explicit safety preference, not tuned as a hidden
  surrogate for return.
- The subtype gate now accepts `--failure-z` to compare limits against a
  conservative independent-binomial upper bound while retaining raw deltas.
  With `failure-z=1.96`, the cap seed1 long-horizon tilt deltas `+0.51/+0.47`
  become upper bounds `+0.61/+0.57`; the historical default (`0`) remains
  point-estimate behavior.
- The first subject5 adapter comparison was re-audited after finding a horizon
  mismatch (`horizon=16` candidate versus `horizon=32` nominal). A new matched
  nominal `horizon=16` run gives `.04433` versus adapter `.03094` at 500 steps
  (−30.2%), with survival `0.0078→0` and first failure 11.40 steps earlier.
  The corrected comparison strengthens, rather than weakens, the cross-split
  rejection; the matched JSON is `docs/policy_target_s5_300_eval_seed123.json`.
- Budget audit confirms the corrected nominal/adapter pair is interaction- and
  optimizer-matched (`sim_steps=25024`, `policy_steps=580`, `wm_steps=600`);
  the adapter only adds wall-clock cost (`0.020h` vs `0.011h`).
- Mechanism audit: the subject5 adapter was enabled on only 211/600 audits
  (35.2%), with final adapter norm `3.37` and audit error `89.0`; subject4 had
  similarly sparse target use. Thus frequent nominal fallback did not prevent
  deployment regression, and fallback coverage alone is not a sufficient gate.
- Strategic-error audit on the same held-out subject5 split measured nominal
  mean/p90/p99 WM error `1.479/1.667/1.678` and action RMS `.358`, versus
  adapter `1.820/2.087/2.198` and `.477` (error/action correlation `.159`).
  This mechanistically explains why fallback gating is insufficient: the
  adapter changes the action footprint toward strategically inaccurate states.
- Added the isolated `StrategicProbeBuffer` storage layer with score/source
  metadata, ring semantics, state round-trip, and optional score-prioritized
  sampling. Phase 0 is now wired behind opt-in trainer flags: replay windows
  contribute detached strategic-error × policy-compatibility scores to the
  separate buffer, and an additive one-step factual probe loss can be enabled.
  A 10-iteration Isaac smoke completed at 1.68 it/s with 160 stored probes;
  resume smoke restored the buffer and grew it to 176. The 49-test selected
  suite remains green. This is plumbing/replay bootstrap evidence only; the
  learned error critic and extra simulator action collection are still Phase 1.
- Added the Phase 1 offline `StrategicErrorCritic` and
  `tools/train_strategic_critic.py`. It predicts detached `log1p` factual WM
  error from state/action/context and ranks bounded action perturbations. A
  512-row smoke fit produced held-out RMSE `.300` (train `.339`); the output
  is a candidate list only and has not yet been executed in the simulator.
- Executed 32 ranked candidates from that critic in Isaac: score/real-error
  correlation was only `.176`, while candidate/base mean real errors were
  `275.30/275.39`. The critic is therefore not a validated active-collection
  signal yet; this negative calibration closes the shortcut of promoting it
  from low held-out regression error alone.
- Added same-state counterfactual collection via
  `tools/collect_strategic_dataset.py --action_candidates K`; a 320-label
  smoke and 20-epoch critic fit gave validation RMSE `1.448`, Isaac candidate
  error `294.35` versus base `294.20`, and score/error correlation `-.264`.
  The labels and evaluator work, but this small checkpoint does not support
  active collection yet.
- The active path now selects the top critic action per state and preloads its
  factual output through `--strategic_probe_dataset`. On a mature subject4
  checkpoint, two critic seeds reduced one-step real WM error by about 15%
  across 64 actions; the matched one-iteration trainer smoke restored 224
  probes and gave e8 `.259` versus `.295` control. This remains a mechanism
  result, not a policy-return result.
- A matched 100-iteration sweep showed probe weight `.1` harmed clean return
  (`.2238` vs `.2516` control), while `.01` preserved/improved clean return
  (`.2543` vs `.2516`) and perturb return (`.2499` vs `.2406`); end survival
  stayed identical. Keep `.01` for multi-seed validation only and close `.1`.
- Shared-start 500-step evaluation did not promote `.01`: seed0 return `.0468`
  vs `.0484` control and seed1 `.0396` vs `.0397`, all with zero end survival.
  The critic also exhibited OOD score extrapolation, so target normalization
  and candidate clipping were added as safeguards; active collection remains
  diagnostic.
- Candidate action-deviation penalty `50` reduced selected-action RMS
  `8.00→6.69` and normalized deviation `.470→.395`, but score/real-error
  correlation remained `-.059`; keep it diagnostic until control-relevant
  reward/safety labels are added.
- Added a reward-gated selector using WM-predicted tracking reward delta
  (`critic_reward_weight=2`). At probe weight `.01`, matched shared-start
  500-step clean evaluation was `.04840` vs `.04841` control (first failure
  `52.56` vs `52.63`); perturb evaluation was `.04887` vs `.04827` (first
  failure `53.06` vs `52.50`). Zero end survival and universal height failure
  remain. Reward gating repairs the ungated probe's long-horizon regression,
  but there is no promotion-level gain yet.
- Repeated the reward-gated continuation from seed1 with a matched control:
  clean `.03915` vs `.03919`, perturb `.03902` vs `.03907`; all four runs had
  zero end survival and height-only failure. The two-seed evidence supports
  retaining the selector as a neutral/stability-preserving diagnostic, not as
  a leading recipe.
- Extended the collector with deterministic seeds and executed simulator-reward
  diagnostics. For the same seed1 candidate set, selector reward deltas were
  `-0.5985` (critic only), `-0.1694` (reward weight `.5`), and `0.0` (weight
  `2`); the last selected the base action everywhere. This identifies the
  mechanism failure: one-step reward gating is a no-exploration fallback, not
  a positive intervention selector.
- Tested direct stacking of action recovery (`.05`) and reward-gated strategic
  probe (`.01`, reward weight `2`) from the subject4 seed1 action-recovery
  checkpoint. The combination reduced 500-step return from `.07223→.07003`
  clean and `.07211→.06890` under perturbation, with first failure earlier by
  `3.25/3.31` steps. Close this interaction; the two auxiliaries are not
  complementary at the tested weights. The resume path now resets only an optimizer whose parameter groups no longer
  match after adding a module, while retaining compatible model and replay state.
- Added a horizon-2 WM/policy rollout to the selector. On the same candidate
  set, reward weights `.05/.5` still had no positive immediate reward deltas
  (means `-.5668/0.0`), so raw short-horizon reward is not enough; use a
  bootstrapped advantage target if this line is revisited.
- Added a real-simulator candidate-rollout audit: 64 normalized-critic actions
  were executed for one step and then handed back to the policy. Candidate/base
  returns were `-4.244/-1.755` at 4 steps and `-1.845/-0.574` at 8 steps.
  This is a direct negative calibration result: WM-error ranking is not a
  control-advantage ranking, so it cannot select active interventions yet.
- Built a real-return advantage collection/training path. Horizon-4 noise `.05`
  yielded no positive candidates; local noise `.005` yielded `15.2%` positive
  candidates and only `+0.00053` best mean advantage. A 1280-row critic had
  validation correlation `.002`, so this is a data/diagnostic path, not yet a
  usable policy selector.
- Tested the advantage critic out of sample: trained on seed456, applied to
  fresh seed789 with a nonpositive-prediction base fallback. Mean selected
  advantage was `-4.6e-5`, with only `2.75%` positive selections. The local
  positive signal does not transfer, so this selector remains diagnostic.
- Added WM action-gradient candidate generation and real rollout validation.
  The negative predicted-reward gradient produced small positive deltas on 256
  seed1 train-split starts (`+.00029`/`+.00013` at 4/8 steps for epsilon `.005`,
  `+.00026` at epsilon `.01`), while the positive direction was slightly worse.
  Keep this as the next held-out/cross-seed candidate; it is not yet a training
  recipe and currently approximates `a_prev` due to legacy dataset format.
- Repeated the action-gradient audit from seed0: negative gradient `+.000060`
  and positive gradient `-.000493` at epsilon `.005` over 4 steps. The sign
  agrees with seed1 but the gain is tiny, so retain this as a local diagnostic
  only and do not couple it into training yet.
- Held-out test_s4 action-gradient validation failed to transfer the train-split
  gain: negative gradient `-7.7e-7` and positive gradient `-.000319` over 4
  steps at epsilon `.005`. Keep the audit for geometry analysis only; do not
  connect it to policy training.
- Fixed the action-gradient audit to persist true initial `a_prev`. The corrected
  held-out test_s4 negative-gradient rollout fell `-0.00398` at 4 steps,
  showing that the earlier weak positive signal was an artifact of the legacy
  approximation. Close action-gradient promotion; keep only the corrected audit
  tooling.
