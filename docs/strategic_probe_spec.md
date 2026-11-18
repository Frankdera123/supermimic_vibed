# Strategic-error probe buffer

This is the next coupling arm after direct policy/model coupling failed. The
goal is to collect factual transitions where the current policy is likely to
exploit the world model, while keeping the factual model objective controlled.

## Current implementation status

Phase 0 is now implemented behind three opt-in flags:
`--strategic_probe_capacity`, `--strategic_probe_weight`, and
`--strategic_probe_batch` (with `--strategic_probe_alpha` controlling score
sampling). Each WM update stores the factual first transition from the
current replay batch, scores it by detached strategic WM error multiplied by
policy-action compatibility when window indices are available, and optionally
adds a prioritized one-step probe loss. The probe state is checkpointed and
restored. This is a replay bootstrap, not yet a learned error critic or an
extra simulator-collection policy; those remain the next Phase 1 experiment.

The first Phase 1 offline component is now available as
`supermimic.strategic_critic.StrategicErrorCritic` and
`tools/train_strategic_critic.py`. It regresses detached `log1p` factual WM
error from `(state, action, xi)` and ranks bounded action perturbations around
replay actions. The tool reports a held-out RMSE and writes candidate states,
actions, and scores; it does not yet execute those candidates in the simulator.

On the 10-iteration smoke checkpoint, the critic's held-out RMSE was `.300`,
but executing 32 ranked candidates gave score-to-real-error correlation only
`.176`; candidate and base-action errors were effectively tied
(`275.30` vs `275.39`). This is a useful negative calibration result: replay
regression accuracy alone does not establish an active probe policy. The next
version must use a larger, policy-occupied dataset and multi-step simulator
validation before its scores can drive collection.

The collector now supports same-state counterfactual batches
(`tools/collect_strategic_dataset.py --action_candidates K`), so one state can
produce several simulator-grounded action/error labels. A 16-env, four-action
smoke produced 320 labels, but the resulting critic still had validation RMSE
`1.448` and ranked candidates slightly worse than the base action in Isaac
(`294.35` vs `294.20`, score/error correlation `-.264`). The infrastructure is
therefore valid, while this small/untrained checkpoint is not evidence for
active collection gains.

## Invariants

- `wm_nominal` is fit from the ordinary replay buffer; the optional probe loss
  is additive and never replaces ordinary replay.
- The probe policy never receives gradients from `wm_nominal` and cannot alter
  the deployable policy.
- Probe transitions are stored in a separate buffer with source metadata and
  are added through a small auxiliary factual loss, never by replacing the
  ordinary replay distribution.
- A probe update is eligible only after held-out one-step and horizon-4 errors
  pass, followed by the same paired 100/500-step simulator gate used for
  shadow promotion.

## Data flow

1. Train a small error critic on replay windows. Its target is the normalized
   one-step strategic error measured after executing the current policy action
   in the real simulator.
2. Roll the error critic's action policy only for state selection. Bound each
   probe by the existing action norm and recovery checks; do not update the
   deployable policy with the error reward.
3. Execute selected actions in the simulator and write `(s, a, s1, xi, f,
   strategic_error, probe_id)` to the isolated probe buffer.
4. Fit the nominal WM with uniform replay plus a detached, capped probe loss.
   Recompute the held-out factual audit before allowing another probe update.
5. Evaluate the resulting policy with paired fixed starts. Promotion requires
   no new height/tilt failure mode and non-degraded 500-step return on every
   held-out split.

## Required diagnostics

Each checkpoint must record probe fraction, probe action norm, strategic-error
quantiles, probe-buffer age, nominal one-step/horizon-4 error, and the ratio of
probe versus ordinary WM gradient norms. A high imagined return without a
corresponding simulator gain is classified as model exploitation.

The existing `adaptive_action_direction` path is not this algorithm: it uses
ensemble disagreement as a proxy before collection. It can provide the bounded
action/probe plumbing, but the new arm must replace that proxy with a learned
strategic-error critic and retain the factual next-state target.

The collector now accepts `--critic_checkpoint` and keeps the highest-scored
candidate per state. The trainer accepts `--strategic_probe_dataset` to preload
those factual transitions into the isolated probe buffer. On a mature subject4
checkpoint, two independent critic seeds reduced one-step real WM error by
about 15% across 64 ranked actions (`285.23→240.57`, `295.45→251.37`). A
matched one-iteration trainer continuation restored 224 probe rows and
reported WM e8 `.259` versus `.295` for a no-probe control. These are mechanism
and lifecycle signals only; the active actions have not yet been shown to
improve policy return.

Matched 100-iteration continuation clarifies the weight sensitivity. Probe
weight `.1` reduced WM e8 but degraded clean return (`.2238` vs control
`.2516`) and moved first failure earlier (`42.25` vs `47.19`). Reducing the
weight to `.01` avoided that degradation: clean return was `.2543` vs `.2516`,
and perturb return was `.2499` vs `.2406`, with unchanged end survival. These
are single-seed/100-step signals, so `.01` is a candidate for the next paired
multi-seed gate, not a promoted recipe.
The paired 500-step gate did not support promotion: seed0 clean return was
`.0468` versus `.0484` control and seed1 `.0396` versus `.0397`, with zero
end survival for all four runs. Critic output also showed severe action-space
extrapolation (predicted log-error near 24 while factual targets were
`0.06–5.15`); target normalization and candidate-range clipping are now in
the tooling, but this remains a diagnostic safeguard rather than evidence of
an improved probe policy.

An action-deviation penalty of `50` reduced the selected-action RMS from
`8.00` to `6.69` (normalized deviation `.470→.395`) and still lowered the
one-step candidate error in the diagnostic (`213.56→174.23` relative to its
base-action batch). However, score/real-error correlation was `-.059`, and no
policy rollout was run for this penalty. Keep this as a candidate selector
ablation only; a control-relevant reward/safety target is still missing.

Reward-gated collection was then tested by adding the WM-predicted tracking
reward delta to the critic score (`critic_reward_weight=2`) while retaining the
same action penalty. After 100 continuation iterations at probe weight `.01`,
the shared-start 500-step clean rollout was `.04840` versus `.04841` control
(first failure `52.56` versus `52.63`), and the perturbed rollout was `.04887`
versus `.04827` (first failure `53.06` versus `52.50`). All runs still ended in
height failure with zero end survival. Reward gating therefore removes the
long-horizon regression seen with the ungated `.01` probe, but the current
single-seed result is a stability repair rather than a demonstrated gain.

The same continuation was repeated from the independent seed1 checkpoint. Its
clean result was `.03915` versus `.03919` control and its perturbed result was
`.03902` versus `.03907`; both pairs had zero end survival and height-only
failures. Across two seeds, reward gating is reproducibly neutral to slightly
positive under perturbation, but it does not yet provide a promotion-level
return or survival improvement.

The collector now records the executed simulator reward of the selected
counterfactual relative to the unperturbed policy action. On a fixed seed1
candidate set, the critic-only selector (`reward_weight=0`) had mean immediate
reward delta `-0.5985`; adding reward weight `.5` reduced the damage to
`-0.1694` but still selected no positive-delta action, while weight `2` chose
the base action for every state (delta `0`). This explains the neutral rollout
result: the current reward gate suppresses exploration instead of discovering
better interventions. A future selector must use multi-step or advantage-like
targets rather than a one-step reward tie-break.

The first two-step implementation rolls the current policy once through the
predicted next state. On the same candidate set, horizon-2 reward weights `.05`
and `.5` still selected actions with no positive immediate reward deltas (means
`-.5668` and `0.0`). Shortening the tie-break horizon therefore does not by
itself create useful exploration; a multi-step target needs a bootstrapped
value/advantage signal rather than raw tracking reward.

A real-simulator rollout audit was added for critic-selected first actions. Using
64 seed1 candidates from the normalized critic, then handing control back to the
checkpoint policy, candidate return was `-4.244` versus `-1.755` for the base
action over 4 steps, and `-1.845` versus `-0.574` over 8 steps. End survival
was unchanged because this is a short horizon, but the return gap is strongly
negative. The current WM-error critic is therefore anti-aligned with control
advantage and must not drive active action selection without a new target.

A real-return advantage dataset was added via `tools/collect_advantage_dataset.py`.
For a 4-step continuation, candidate noise `.05` produced no positive
candidate relative to base; reducing noise to `.005` exposed a local regime
where `15.2%` of candidates were positive and the best candidate improved
mean normalized return by only `0.00053`. An offline advantage critic fit on
1280 rows had validation RMSE `.00193` but correlation `0.002`, so the signal
is not predictable from this small state/action sample. This supports local
search as a possible data source, but not a deployable selector yet.

The conservative advantage selector was tested out of sample. A critic trained
on the seed456 collection was applied to a fresh seed789 collection with the
base-action fallback whenever predicted advantage was nonpositive. It selected
only `2.75%` positive-return actions and had mean selected advantage `-4.6e-5`
relative to base. The in-sample local signal does not transfer, so this branch
remains an offline audit and is not connected to training.

A local action-gradient audit was added via
`tools/generate_action_gradient_candidates.py`. On 256 train-split seed1
starts, the negative WM reward-gradient direction slightly improved real
rollout return over the base action: `+0.00029` at epsilon `.005` over 4 steps,
`+0.00013` over 8 steps, and `+0.00026` at epsilon `.01` over 4 steps. The
positive direction was slightly worse. This is only a short-horizon,
single-checkpoint diagnostic; the gradient currently approximates `a_prev` with
the current action because legacy datasets do not store the previous action.
It is the next candidate for held-out and cross-seed validation, not a promoted
training objective.

The gradient audit was repeated from the independent seed0 checkpoint. At
epsilon `.005` over 4 real-simulator steps, the negative-gradient candidate
scored `+0.000060` over base, while the positive-gradient candidate scored
`-0.000493`. Together with seed1 (`+0.00029`), the sign is directionally
consistent but the effect is very small; this remains a diagnostic local
correction, not evidence for coupling it into policy training.

Held-out test_s4 validation did not transfer the train-split gain. For the seed1
checkpoint at epsilon `.005`, the negative gradient was effectively neutral
(`-7.7e-7` over 4 steps) and the positive gradient degraded return by
`-.000319`. The action-gradient line is therefore diagnostic only; its train
split sign should not be interpreted as a general control improvement.

The audit was corrected to store and use the true initial `a_prev` in newly
collected datasets. Repeating the held-out test_s4 negative-gradient audit with
this action-rate term gave `-0.00398` relative return over 4 steps, versus the
previous near-zero result obtained by substituting the current action for
`a_prev`. The earlier weak positive signal was therefore an artifact of the
legacy approximation; close action-gradient promotion and retain the corrected
pipeline for future diagnostics.
