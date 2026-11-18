# External SuperMimic implementation gap table

| Mechanism observed in `allmimic` | Current repository status | Evidence / next action |
|---|---|---|
| No-exploration-noise collection fraction | Implemented as `--deterministic_collect_fraction`; first matched 0.5 data-coverage ablation is negative | subject5 seed0, 1000 steps: 500-step return fell ~22.7% clean and ~22.2% perturb; stop unless a different coverage hypothesis is proposed. |
| Per-joint error-propagation WM weights | Not implemented as a general joint-weight vector | Candidate WM-only ablation. Preserve normalized total loss and evaluate one-step error, 8-step error, and real return. |
| Balanced reset sampling by rare kinematic clusters | Implemented as opt-in `--reset_sampling_mode kinematic` (default remains return-based) | Matched 20-iteration probe lowered WM err8 but fixed 100-step return (0.0824 vs 0.0931) and had zero survival for both; retain as coverage diagnostic, not a leading mechanism. |
| Capped hard-example reset sampling | Partially covered by current return-based reset logic | Compare only after a kinematic-cluster implementation exists; otherwise the mechanisms are confounded. |
| Residual upper-body / absolute-leg actions | Outside current state/action parameterization | Requires changing the policy interface and is a separate reproduction branch, not a drop-in WM-policy concurrency test. |
| Wider policy latent | Architecture ablation, unrelated to WM-policy update coupling | Defer until the matched-budget WM data experiments are complete. |

Per-joint WM weighting and kinematic-cluster reset sampling have now both been
tested as isolated coverage/WM diagnostics and were not promoted on matched
real-rollout probes.
