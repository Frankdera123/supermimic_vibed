# SuperMimic concurrent WM/policy leading recipe

## Current default

```text
4-step teacher-forced WM loss
+ 8-step free-running WM auxiliary, weight 0.05 ramped over 300 iterations
+ 2 WM replay batches per iteration
+ 2 policy updates after the WM batches
+ independently sampled policy replay batches
+ policy BPTT truncation 4
+ no critic
+ 2000 training iterations
```

Run it with:

```bash
SEED=0 LOG_DIR=logs/concurrent_ramp_s0 tools/run_concurrent_ramp.sh
```

The script keeps `POLICY_REUSE_BATCH=0` and `INTERLEAVE_POLICY_WM=0` by default. Those
flags expose negative or mixed ablations only.

## Evidence

- On subject4 and subject5 held-out splits, 2000-step double-update checkpoints outperform
  matched 1000-step checkpoints under new perturbed starts.
- Fair 2000-step single-update versus double-update comparisons favor double updates on both
  held-out splits; the improvement is also present in 200-step rollouts.
- Free-running auxiliary loss consistently lowers multi-step WM error on both splits, although
  its extra policy-return gain at 2000 steps is modest.
- The second policy update improves return without consistently lowering WM error. Its effect is
  therefore policy optimization/behavior adaptation, not simply better WM fit.
- Independent policy-batch resampling beats same-batch reuse at both 1000 and 2000 steps.
  Action-support distances are nearly identical, while cross-batch gradient directions are more
  diverse for independent sampling. This is supporting mechanism evidence, not a causal gate.

## Boundaries

- Direct policy gradients into the nominal WM, residual adapters, uncertainty gates, and
  WM-policy interleaving have no stable improvement and remain diagnostics.
- Interleaving `WM -> policy1 -> WM -> policy2` is context-dependent and does not replace grouped
  `WM x2 -> policy x2`.
- The available data contains only subject4 and subject5 held-out splits; no additional subject
  is present for external validation.

## Next experiments

Keep the nominal replay objective fixed. The useful open questions are the gradient geometry of
sequential policy adaptation, action-conditioned height failures, and longer-horizon calibration.
Every candidate should be judged by paired held-out simulator rollouts, not replay WM error alone.

## Recent boundary checks

- Interleaving WM and policy updates (`WM → policy1 → WM → policy2`) is mixed at 2000 steps
  and slightly worse at 1000; it does not replace grouped WM batches.
- A single policy update with `pi_batch=64` remains slightly below two independent updates at
  2000 steps, so the gain is not explained by one larger batch.
- Same-batch reuse is weaker at both 1000 and 2000 steps. Keep independent replay sampling.
- Gradient cosine is positive but seed-dependent; use the gradient audit diagnostically, not as
  a fixed training gate.

The current default therefore remains the grouped WM×2 → policy×2 recipe. The main open
mechanistic question is how sequential policy adaptation benefits from independent batches while
leaving WM fitting unchanged.

- Exact two-batch gradient accumulation followed by one optimizer step is substantially worse
  than two sequential policy steps at 1000 steps. This is direct evidence for sequential policy
  adaptation, not just a larger effective batch or averaged gradient.

The same conclusion survives a 2000-step continuation: gradient accumulation remains below
sequential updates on six new starts, although the gap narrows with training budget.

The double-update advantage also persists in subject4 500-step perturbed rollouts, with higher
return and later first failure than the matched single-update checkpoint.

### Adapter implementation status (2026-11-06)

The policy-sensitive target adapter is now executable behind an explicit audit gate. Runtime smoke confirms both branches: strict replay audit rejects the adapter, while a permissive audit enables it and persists adapter/optimizer state. No claim of policy improvement is made; the next scientific test remains a held-out, pessimistic objective with trust-region acceptance.
