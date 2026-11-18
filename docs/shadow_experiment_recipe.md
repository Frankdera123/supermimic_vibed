# Shadow-policy experiment recipe

This recipe evaluates concurrent world-model/policy training without updating
the deployable policy from the target world model until a real-simulator probe
accepts the candidate.

## 1. Train the isolated branch

Use the same seed, motion split, interaction budget, and nominal optimizer
settings as the reference run. Start with a conservative shadow learning rate:

```bash
python supermimic/train.py \
  --log_dir logs/shadow_s0 \
  --shadow_policy \
  --shadow_policy_lr_scale 0.1 \
  --shadow_policy_updates 2 \
  --policy_target_audit_max_error 999 \
  --policy_target_control_file logs/shadow_s0/control \
  --ckpt_every 200
```

The run publishes both `ckpt_*.pt` (live) and `shadow_ckpt_*.pt` (candidate)
artifacts. Keep the live checkpoint as the rollback source.

## 2. Probe the shadow artifact

Use paired fixed starts with at least 500 simulator steps, at least two start
seeds, and every held-out motion split. Point the watcher at the shadow glob and
require the metadata check:

```bash
python tools/watch_probe_gate.py \
  --checkpoint-dir logs/shadow_s0 \
  --checkpoint-pattern 'shadow_ckpt_*.pt' \
  --require-shadow-artifact \
  --shadow-mode \
  --reference logs/nominal_s0/ckpt_001000.pt \
  --motion-dir data/lafan1_g1 \
  --seed 123 --seed 456 \
  --out-dir logs/shadow_s0/probes \
  --control-file logs/shadow_s0/control \
  --steps 100 --steps 500 --min-steps 100 \
  --max-relative-drop 0.05 \
  --max-survival-drop 0.05 \
  --max-first-failure-drop 5 \
  --max-tilt-failure-drop 0.05 \
  --promote-on-accept
```

In `--shadow-mode`, each `shadow_ckpt_N` is compared with the live
`ckpt_N` from the same iteration. An accepted probe writes `shadow_promote`; a
rejected probe writes `shadow_discard`, leaving the live branch untouched. The
trainer consumes the command at an iteration boundary. Repeating `--steps`
requires every listed horizon to pass independently, so a short-horizon gain
cannot hide a long-horizon regression.

The optional `--max-height-failure-drop` and `--max-tilt-failure-drop` guards
reject a candidate when a failure subtype increases even if return or end
survival has not crossed its threshold. They default to zero (disabled) for
backward compatibility. When subtype estimates are noisy, `--failure-z 1.96`
compares the limits against a conservative independent-binomial upper bound;
the raw deltas and upper bounds are both retained in the gate output.

## 3. Required evidence

For each candidate, retain the paired JSON files, gate output, parameter norms,
and the live/shadow checkpoint pair. A candidate is eligible for promotion only
when the lower confidence bound passes the return threshold and no held-out
split loses more than the survival threshold. Training-time imagined return or
replay WM error cannot replace the simulator probe.
