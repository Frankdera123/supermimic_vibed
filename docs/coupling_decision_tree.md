# SuperMimic concurrent coupling decision tree

This note records the current evidence boundary and prevents reopening variants
that already failed the long-budget or cross-subject checks.

## Closed branches

- Direct policy-return gradients into nominal WM: scalar, action-path,
  last-layer, projection, conflict gate, norm matching, EMA, delayed and
  sparse schedules all fail to produce a stable 600-step gain.
- EMA/target WM for policy imagination: one-step and horizon-2 replay gates
  give short-run gains but reverse at 600 iterations (subject4 horizon-2:
  `0.1834` vs nominal `0.2174`).
- Action-conditioned residual adapter: teacher-forced error can improve, but
  free-running error changes are effectively zero; unconstrained multi-step
  fitting can diverge.
- Ensemble disagreement: final 4-step error correlation is `-0.186` on
  subject4 and `0.132` on subject5, so it has no global gate semantics.
- Action-recovery auxiliary and gradient alignment/projection: gains are split
  dependent; alignment-gated subject5 fell `0.1684 -> 0.1461`.

## Required properties for the next arm

1. `wm_nominal` is replay-only and remains the factual reference.
2. A separate `wm_task` is used only for policy imagination and has an
   explicitly bounded policy-sensitive adapter.
3. The adapter update is a lower-bound or pessimistic policy objective, not a
   raw imagined-return maximization. Its update must be clipped in parameter
   space and accepted only when held-out multi-step factual error does not
   increase.
4. The policy update must report nominal and task-model returns separately,
   along with adapter norm, held-out 1/4-step error, fallback coverage, and
   real simulator return.
5. Promotion requires both subject4 and subject5 improvement at 600 iterations
   with `policy_updates=2`; any short-run-only gain is classified as model
   exploitation.

## Next implementation order

1. Add a low-rank action adapter to `PolicyTargetWorldModel`, initialized at
   zero and excluded from nominal EMA updates.
2. Compute a pessimistic policy objective using the minimum of nominal and task
   model returns, while keeping the adapter gradient separate from the policy
   gradient.
3. Apply a held-out replay trust-region check to the proposed adapter step;
   reject the step and fall back to nominal when either the 1-step or 4-step
   error ratio exceeds its budget.
4. Run a 300-step smoke, then the matched 600-step subject4/subject5 pair.

The existing audit gate, fixed-policy evaluator, budget audit, and checkpoint
state tracking already cover the required measurement infrastructure.

## 2026-11-06 implementation update

The required pessimistic objective and relative held-out audit constraint are now available as explicit opt-in flags. The pessimistic return uses the task-model gradient only where it is below the detached nominal reference; the relative audit compares independent multi-step factual error against the EMA nominal rollout. Long-budget subject4/subject5 promotion is still pending.
