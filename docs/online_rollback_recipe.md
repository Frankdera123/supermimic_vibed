# Online target coupling with rollback

This is the currently validated procedure for experimenting with policy-target
coupling. It keeps target updates externally gated and restores the last
accepted policy when a long-horizon probe rejects a checkpoint.

1. Train with periodic checkpoints and a control file:

```bash
python -m supermimic.train \
  --iterations 300 --ckpt_every 50 \
  --policy_target_audit_max_error 1000000 \
  --policy_target_audit_ratio .25 \
  --policy_target_control_file logs/target.control \
  --log_dir logs/target_run
```

2. Run the watcher in a separate process. Use at least two held-out motion
   splits and two fixed-start seeds. The default 500-step horizon is required
   because 100-step probes missed a 32–40% long-horizon degradation in the
   seed4/seed5 audits.

```bash
python tools/watch_probe_gate.py \
  --checkpoint-dir logs/target_run \
  --reference logs/baseline_same_seed/last.pt \
  --motion-dir data/lafan1_g1_test_s4 \
  --motion-dir data/lafan1_g1_test_s5 \
  --seed 123 --seed 456 \
  --out-dir logs/target_probes \
  --control-file logs/target.control \
  --batch 128 --steps 500 --max-relative-drop .05 \
  --stop-on-reject
```

When a checkpoint passes, the control file contains `enable`. When a later
checkpoint fails, the watcher writes `rollback:<previous_checkpoint>`; the
trainer restores policy, critic, and available optimizer state outside
`torch.compile`, then continues with nominal WM. A same-seed nominal baseline
is mandatory for interpreting the final return; replay or imagination metrics
are not valid substitutes.
