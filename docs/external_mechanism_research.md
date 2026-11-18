# External mechanism review for concurrent WM-policy training

## Main finding

The current repository's occupancy coupling uses a detached distance between the current policy action and the replay action. That is only a local compatibility heuristic. It does not estimate the policy's transition occupancy, and the experiments show high seed/context variance.

A more principled reference is **TOM (Transition Occupancy Matching)**. TOM defines a policy's transition occupancy over `(state, action, next-state)` and derives a policy-aware model-learning objective as importance-weighted maximum likelihood. Its practical algorithm trains a transition discriminator and a policy-relevance dual Q function, then uses the resulting weights for dynamics-model regression. The paper explicitly distinguishes this from recency heuristics: relevant old transitions can receive larger weights than newer ones.

Reference: Ma et al., *Learning Policy-Aware Models for Model-Based Reinforcement Learning via Transition Occupancy Matching*, arXiv:2305.12663, especially Sections 4.1–4.3 and Algorithm 1.

A second external line is *Mismatched No More* (Eysenbach et al., NeurIPS 2022),
which optimizes a shared lower bound with a real-vs-model transition classifier:
the model is pushed toward realistic transitions while the policy is discouraged
from exploiting model errors. This is closer to the original concurrent-training
idea than replay importance weighting. The current SuperMimic implementation has
not enabled that classifier in policy loss yet; the next safe artifact is a
real-vs-model AUC audit before adding its gradient to training.

That audit is now implemented as `tools/model_realism_audit.py`. Using 4,096 real
and 4,096 one-step nominal-WM transitions, held-out discriminator AUC was `0.624`
on subject4 seed1 and `0.707` on subject5 seed1. The signal is real but modest,
which is the right regime for a small detached policy penalty; the classifier
must be independently calibrated and trained on fresh fake transitions before
any concurrent gradient is enabled.

The saved discriminator is now accepted by `train.py` via
`--policy_realism_checkpoint`; `--policy_realism_weight=.01` completed a
30-iteration subject4 smoke without shape or gradient errors. The penalty is
currently diagnostic only and has no rollout claim.

The full subject4 test closes this naive implementation: a 1000-iteration run
with realism weight `.01` lost all 4 paired 500-step conditions, averaging
`-13.49%` clean and `-14.92%` perturb. The offline classifier signal does not
transfer as a direct policy penalty; a successful MnM-style version needs fresh
online fake samples, a jointly trained discriminator, and a normalized
lower-bound objective rather than this fixed penalty.

A tenfold smaller realism weight (`.001`) was also run for 1000 iterations. It
remained slightly negative on subject4 (`-2.07%` clean, `-2.17%` perturb; 2/4
wins in each), so the failure is not just penalty magnitude. The fixed offline
discriminator path is closed; any future MnM test must train the discriminator
online against the current WM and policy distribution.

## Relevance to SuperMimic

- The existing `wm_occupancy_weight` branch is **not TOM**. It weights a replay window from action distance only and has no transition discriminator, Bellman-flow/dual-Q term, or next-state occupancy estimate.
- The subject4/subject5 reversals are therefore evidence against the heuristic, not evidence against policy-aware model learning in general.
- A faithful TOM reproduction would require a stochastic policy or an explicit action-noise density, a transition discriminator over `(s, a, s1)`, and a stable dual-Q objective. Adding a scalar weight to the current branch would not be a faithful test.

## Minimal staged experiment

1. Add a replay-only transition classifier with detached features `(invariant(s), action, xi, normalized transition target)` and a held-out classifier audit. Positive samples must come from a fresh current-policy buffer; negatives come from historical replay matched by episode/frame coverage.
2. Calibrate classifier odds on an independent replay split. Clip and normalize the odds before using them as WM-loss weights.
3. Compare uniform WM, current action-distance occupancy, and discriminator-weighted WM under identical interaction and optimizer budgets. Report one-step error, 8-step error, classifier AUROC/calibration, and batch32 multi-seed real rollouts.
4. Reject the arm if classifier AUROC is near chance, if held-out WM error rises by >25%, or if real rollout gains do not survive at least two training seeds and four evaluation seeds.

This is the next research line; no claim is made that TOM transfers to the current deterministic G1 setup until these audits pass.

## First classifier audit

新增 `tools/transition_occupancy_audit.py`，在保存 replay 的 subject5 checkpoint 上用最新/最旧 25% buffer rows 作为 current-policy proxy，训练 transition-tuple classifier，并在留出样本上评估：

| checkpoint | AUROC | AP | Brier |
|---|---:|---:|---:|
| occupancy seed0 | 0.857 | 0.864 | 0.186 |
| occupancy seed1 | 0.846 | 0.854 | 0.181 |
| uniform seed0 | 0.903 | 0.906 | 0.164 |
| uniform seed1 | 0.811 | 0.843 | 0.197 |

这证明 transition tuple 中存在可学习的新旧 footprint，但标签由时间片构造，不能直接解释为 policy occupancy；下一版必须做 age-matched / coverage-matched negatives，避免把 recency 当成 policy relevance。

## Age/coverage-matched classifier audit

`transition_occupancy_audit.py --matched` now matches old negatives to newest positives by coarse frame-modulo and episode-step buckets and removes absolute frame/episode-step features from the classifier. On subject5 checkpoints, held-out AUROC remained high: occupancy seed0 `0.927`, occupancy seed1 `0.947`, uniform seed0 `0.936`, uniform seed1 `0.918`. This is stronger evidence that transition tuples contain a learnable policy/trajectory footprint beyond a trivial age feature, although it is still not TOM's dual-Q occupancy estimator. The next safe step is an offline discriminator-odds WM weighting smoke with independent calibration.

## Offline odds-weighting smoke

The audit tool now supports `--save_model`; the saved classifier is loaded by
`train.py --wm_transition_weight_checkpoint` as a frozen, clipped, normalized odds
weight on replay WM windows. A 30-iteration subject5 smoke completed with the
matched seed0 classifier (`AUROC=0.936`) and produced normal WM/policy logs. This
only validated plumbing and gradient isolation. The full subject5 evaluation now
uses three training seeds, each with four paired 500-step clean/perturb starts.
Relative to the corresponding nominal ramp checkpoints, weighted WM averaged
`+6.65%` clean and `+6.29%` perturb return over 12 paired conditions, winning
10/12 pairs. Per-training-seed gains remain unequal, so this is a promising
candidate rather than a default recipe.
The classifier remains a proxy for TOM, since it has no dual-Q/Bellman-flow term.

## Cross-split check on subject4

The same procedure was run on the subject4 held-out split (training seed1,
matched classifier AUROC `0.840`, four paired 500-step clean/perturb starts).
Odds weighting lost on average `4.07%` clean and `4.38%` perturb return, with only
2/4 paired wins in each context. Thus the subject5 gain does not transfer across
splits; this mechanism is now classified as split-dependent diagnostic evidence,
not a global recipe.

Reducing the odds scale to `0.25` did not rescue subject4: clean/perturb remained
`-3.44%/-2.18%`, with only 1/4 paired wins in each context. The failure is not
explained by an overly large scalar weight, so further scalar sweeps are stopped;
the next step is independent classifier calibration or a true occupancy target.

Held-out temperature calibration was also tested (subject4 temperature `0.485`).
The calibrated run lost all 8 clean/perturb paired conditions, with mean changes
of `-7.03%/-7.95%`. This closes the classifier-odds line for now: the remaining
path is a genuine occupancy estimator (for example a dual-Q/Bellman-flow target),
not more odds or temperature tuning.

## Dual-Q audit prototype

`tools/dual_q_occupancy_audit.py` now implements the SMODICE/TOM chi-divergence
dual objective as an offline audit. On subject4 and subject5 seed1 replay it
produced effective sample fractions `0.477` and `0.487`, with P95 weights about
`2.60/2.47` and maxima `13.19/15.07`. This is too high-variance to wire directly
into WM. The prototype currently uses the observed replay action for the
one-step `q(s')` proxy; a strict TOM integration must replace that with the
current policy action and then pass independent calibration before WM weighting.

The policy-action version is now implemented via `--use_policy_action`. On
subject4, early Q training was stable at 5/10/20 epochs (ESS `0.892/0.885/0.859`,
max weights `1.53/1.67/2.11`), but diverged by 40/80 epochs (ESS `0.548/0.329`,
max weights `6.48/16.53`). Subject5 at 80 epochs showed the same high-variance
pattern (ESS `0.358`, max `14.60`). This establishes an actionable gate: Q
training needs early stopping or a target-network/regularized objective before
any WM weighting experiment.

Regularization checks confirm the issue is structural: target-Q with
`tau=0.001–0.05` gave ESS only `0.33–0.38`; Q L2=`0.01` improved ESS to `0.355`
but gradient clipping worsened it to `0.300`. No 80-epoch configuration passed
the ESS/maximum-weight gate, so dual-Q remains an early-stopped audit only.

The early-stopped table can now be consumed by `train.py
--wm_dual_q_weight_checkpoint`. A subject4 30-iteration smoke using the 10-epoch
policy-action table completed normally (`wm_err8=0.351` at iteration 30), so the
offline-to-WM plumbing is validated. The table is computed from a prior replay
checkpoint; no real-rollout gain is claimed yet.

The first full subject4 rollout used the 10-epoch policy-action table for 1000
iterations. Across four paired 500-step clean/perturb starts, dual-Q weighting
was `-1.44%` clean and `-2.34%` perturb versus nominal, with 2/4 wins in each
context. This is less damaging than classifier-odds weighting, but still not a
positive result; it remains an exploratory audit rather than a recipe.

A stale-weight check used three 300-iteration segments, recomputing the
policy-action dual-Q table from the latest replay checkpoint before each resume.
Subject4 still dropped `3.29%` clean and `3.42%` perturb (1/4 paired wins), so
stale weights are not the main failure; the dual-Q occupancy target itself is
misaligned with this tracking WM objective.

## Online realism discriminator check

A fresh online discriminator was added to train on each WM update using current
replay transitions as real samples and current nominal-WM one-step predictions as
fake samples. Its policy penalty used the running standardized logit, avoiding a
fixed offline classifier. The subject4 1000-iteration run completed normally,
but four paired 500-step evaluations were uniformly worse than nominal:
`-9.50%` clean and `-9.08%` perturb, with `0/4` wins in both contexts. Thus the
simple online realism-gradient route is also closed for the current tracking
objective; a future Mismatched-No-More style attempt would need a different
lower-bound or joint objective, not just a realism penalty.

## Action-recovery / action-discriminative WM auxiliary

A new state-space diagnostic follows the action-discriminative WM idea: an inverse
head predicts the executed action from the replay current state and the WM
predicted next state. Gradients through the predicted next state encourage the
WM to preserve action-dependent transition differences; the inverse head is
auxiliary and is discarded at deployment. Subject4 weight `.1` was neutral
(clean `+0.69%`, perturb `+0.02%`, 2/4 wins each). On subject5, weight `.1`
won all eight paired starts with `+15.13%/+16.17%` clean/perturb; a weaker
weight `.05` also won all eight with `+25.64%/+25.83%`. Ordinary held-out WM
fidelity did not improve for `.05` (all-step `0.822 -> 0.937`, audit-final gap
`0.145 -> 0.372`), so the result must be interpreted as improved action
selection geometry rather than a generic prediction-error reduction. This is
now the leading exploratory coupling, pending a second training seed and a
subject4 weight check.
