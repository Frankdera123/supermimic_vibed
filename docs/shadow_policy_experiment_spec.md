# Shadow-policy concurrent training experiment

## Motivation

The current online target path can be detected and rolled back, but the seed5
audit showed that disabling at iteration 50 cannot undo policy updates already
made. A shadow branch removes that exposure: target coupling updates a copy of
the policy and critic, while the live branch continues with nominal WM.

## State and update rule

At each checkpoint boundary maintain:

- `pi_live`, `critic_live`: the deployed branch, trained only with nominal WM;
- `pi_shadow`, `critic_shadow`: copied from the last accepted live checkpoint;
- target WM and shadow optimizers: used only for shadow policy updates.

Collect simulator data with `pi_live`. Fit the factual WM from the shared
replay. Run the normal policy update on `pi_live` through nominal WM. In a
separate update, run `pi_shadow` through the target WM using the same replay
starts and actions; do not write shadow gradients into `pi_live`, WM, or the
live critic.

## Promotion gate

Every `ckpt_every` iterations:

1. Save the live checkpoint and shadow checkpoint.
2. Run paired `fixed_policy_eval.py` on subject4 and subject5 with seeds 123
   and 456, horizon 500, batch128.
3. Compare each shadow checkpoint with the live checkpoint from the same
   iteration using `watch_probe_gate.py --shadow-mode --min-steps 500`, with
   both `--max-relative-drop .05` and `--max-survival-drop .05`.
4. If the lower confidence bound and survival gate pass, copy shadow
   policy/critic and optimizer state into live. Otherwise discard shadow and
   restart it from the current live branch.

## Required evidence

The experiment is promotable only if, across at least three training seeds:

- shadow promotion never reduces either held-out split beyond the 5% gate;
- at least one seed produces a positive dual-split improvement over nominal;
- rejected shadows leave the live policy numerically unchanged except for its
  nominal update;
- final 500-step evaluation is reported against same-seed nominal baselines.

The current rollback tools provide the probe and acceptance machinery. The
second policy/critic optimizer branch is wired into the training loop as an
opt-in prototype. The state-isolation primitive is implemented as
`supermimic.shadow_policy.ShadowPolicyBranch` and covered by
`tools/test_shadow_policy.py`; it also exposes checkpoint
`state_dict`/`load_state_dict` and independent Adam optimizer construction, so
promotion experiments can persist shadow state across probe boundaries.

## Implementation status

The prototype is now wired behind `--shadow_policy`. It requires
`--policy_target_audit_max_error > 0`, keeps the live policy on nominal WM,
updates an isolated policy/critic branch through the target WM, and stores the
shadow weights and optimizers in checkpoints. Promotion remains manual by
default (`ShadowPolicyBranch.promote()`), but a running job can use the
probe-gated watcher:

For a conservative candidate, set `--shadow_policy_lr_scale` below `1`; this
controls only the isolated policy/critic optimizers.

When shadow mode is enabled, `train_log.csv` also records
`policy_param_norm` and `shadow_policy_param_norm`, which should be inspected
before interpreting a probe result.

```bash
python tools/watch_probe_gate.py ... --shadow-mode --promote-on-accept
```

After an accepted paired probe, the watcher writes `shadow_promote`; training
consumes that command on its next iteration and promotes or discards the
isolated branch.
When running with `--shadow_policy`, point the watcher at the generated shadow
artifacts so the gate evaluates the branch that will be promoted or discarded:

```bash
python tools/watch_probe_gate.py ... \
  --checkpoint-pattern 'shadow_ckpt_*.pt' \
  --require-shadow-artifact --shadow-mode --promote-on-accept
```

In shadow mode, rejection writes `shadow_discard` and leaves the live branch
untouched; it does not use the live rollback path.
