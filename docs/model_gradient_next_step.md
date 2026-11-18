# Model-gradient concurrent training: next experiment

## Why this direction

The direct concurrent variants tested in `current_frontier.md` let policy imagination gradients update the world model. Even when the gradient was restricted to the action pathway, the 100-iteration schedule-sampling run reduced fixed-start return on both seeds and increased tilt failures. The failure is therefore not just excessive gradient magnitude; it is that the task-return gradient is an unreliable supervision signal for the factual transition model.

The next experiment should use the model-gradient idea: train the model so that the policy-gradient signal it induces is locally compatible with the signal supported by simulator transitions. The world model remains factual in its forward prediction, while the policy-facing objective constrains the local action geometry. This is different from adding policy return as a transition target.

Relevant references:

- Jia et al., *Model gradient: unified model and policy learning in model-based reinforcement learning*, Frontiers of Computer Science 18, 184339 (2024), DOI `10.1007/s11704-023-3150-5`. The paper motivates model learning by matching gradients from generated and real data.
- Ma et al., *Do Transformer World Models Give Better Policy Gradients?*, ICML 2024, PMLR 235:33855–33879. The paper shows that long-horizon model architecture can create circuitous policy-gradient paths and proposes action-conditioned paths.

## Concrete SuperMimic formulation

For a factual replay state `s` and action direction `d`, use the simulator counterfactual pair already stored by `generate_counterfactual_dataset.py`:

```text
g_sim(s,d) = normalize( target(s_plus) - target(s_minus) )
g_wm (s,d) = normalize( wm_rollout(s, a+d) - wm_rollout(s, a-d) )
L_geom = 1 - cosine(g_wm, stopgrad(g_sim))
L_mag  = smooth_l1(log(||Δ_wm||), log(||Δ_sim||))
```

For h4 data, `wm_rollout` must use four repeated actions, matching the dataset metadata. Apply the loss to a detached/local geometry head or to the WM action Jacobian, while leaving the nominal state prediction loss factual. Do not backpropagate policy return through the same parameters in the first test.

The first implementation should expose three switches:

1. `--wm_counterfactual_geometry_weight`: cosine direction loss;
2. `--wm_counterfactual_geometry_mag_weight`: log-magnitude loss;
3. `--wm_counterfactual_geometry_detach_state`: compute the geometry head from the WM action path while blocking state-reconstruction gradients.

The existing endpoint/effect/norm h4 losses are not equivalent: they minimize state error and can trade height against tilt. The geometry loss directly tests whether the WM gives the policy the right local action direction and scale.

## Promotion protocol

Use the same h4 simulator datasets from seeds 1 and 303, train seeds 0 and 1, and compare against the schedule-sampling baseline. Before any policy coupling, require:

- held-out simulator/WM action-effect cosine not lower than baseline by more than 0.02;
- log-magnitude error not worse than baseline;
- fixed-start clean and perturb return no worse than baseline by 2%;
- no increase in tilt-failure rate above 3 percentage points;
- survival is evaluated on the same 64 paired starts.

Only after this WM-only geometry test passes should a small action-path policy gradient be enabled. If geometry improves but control does not, the result distinguishes gradient-path quality from policy optimization and prevents another ambiguous return-weight sweep.

