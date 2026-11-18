# SuperMimic concurrent world-model/policy research protocol

## Evidence boundary

The current mini-testbed consistently shows that an unrestricted policy-return gradient
into a nominal dynamics model is not a stable learning signal. Scalar coupling, gradient
projection, conflict gating, norm matching, gradient EMA, update-period changes, delayed
starts, output/parameter trust regions, L2 anchors, capacity reduction, and short task
prefixes all fail to beat the uniform H=64/trunc4 policy baseline. Norm matching can give
a short-budget gain, but it degrades with longer training. The reliable gain remains
policy BPTT truncation with a replay-trained model.

This is consistent with the stable Dreamer design: the world model is trained from replay,
while actor/critic are trained on imagined trajectories with the model parameters held
fixed for that update. “Concurrent” means the channels run in the same outer loop, not
that the policy loss is allowed to rewrite the nominal simulator.

## Recommended optimization channels

Let `f_phi` be the nominal WM, `r_omega` a policy-sensitive residual member, and
`pi_theta` the policy.

1. **Nominal channel.** Update `phi` only with real replay transitions and held-out audit
   monitoring. Keep its normalizer frozen/shared for cross-seed comparisons.
2. **Policy channel.** Update `theta` through a frozen snapshot of `f_phi` (or a frozen
   ensemble snapshot) using truncated BPTT. The actor gradient must not enter `phi`.
3. **Residual channel.** If a residual is studied, train `omega` from real transitions,
   bootstrap/environment splits, or an explicitly calibrated uncertainty target. Do not
   use raw imagined return as its only supervision.
4. **Coupling channel.** Couple the channels through replay selection, target-model
   snapshots, ensemble uncertainty, or audit-calibrated rollout gating. A scalar joint
   loss is not sufficient evidence of useful coupling.

## Minimum viable next experiment

Train two bootstrap residual members from the same frozen nominal checkpoint. For each
member, report train/audit one-step and 8-step error, residual-vs-nominal error delta, and
coverage-risk curves. During policy learning, use the frozen nominal model for the first
half of each rollout and an uncertainty-gated residual/ensemble only where the threshold
was calibrated on train environments. Always retain nominal fallback.

The run is accepted only if all of the following hold on two seeds:

- real simulator episode tail is at least the uniform H=64/trunc4 baseline;
- held-out 8-step error does not increase relative to nominal;
- uncertainty coverage is reported with a train-calibrated threshold and audit confidence
  interval;
- the imagined-to-real return gap does not widen;
- removing the residual/uncertainty member removes the claimed gain.

A method that improves imagined return but fails any of these checks is classified as model
exploitation, regardless of its joint-loss value.
