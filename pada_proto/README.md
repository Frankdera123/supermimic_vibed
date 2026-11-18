# pada_proto — PADA-0 isolated experiment workspace

PADA-0 (Policy-Shift Data Efficiency Audit): does a fixed simulator query
budget spent on **policy-shift-directed probes** improve the world model's
control-relevant fidelity more than the same budget spent on **random probes**?

Everything here reuses the CRGF toolchain (`crgf_proto/`, unmodified) on top of
the OPCL proto package variant, per `../docs/pada/PADA_SOURCE_AUDIT.md`.
Protocol (pre-registered): `../docs/pada/PADA0_EXPERIMENT_PROTOCOL.md`.

## Layout

| path | content |
|---|---|
| `pada_bridge.py` | path resolution, namespace-package import (proto variant + crgf core), diag-pair bundle loading/validation |
| `tools/pada_gen_directions.py` | all direction sets: random (per-start, seeded) + policy-shift deltas from **real verified single-update diag pairs** |
| `tools/pada_sim_collect.py` | one deterministic sim session (<=4 envs): probe branches (both arms) + shared blind eval set; records per-step trajectories/rewards |
| `tools/pada_finetune_wm.py` | M0 -> M1 (random probes) / M2 (policy probes), identical protocol, snapshots every 250 steps |
| `tools/pada_eval.py` | Level 1-4 metrics, reliability gate, paired bootstrap, verdict-check inputs |
| `tools/test_pada.py` | 22 CPU synthetic tests (fairness invariants, metric correctness, gate behavior) |

## Reproduce (RTX 5090, `supermimic:latest` container, <=4 GPU-h total)

```bash
# 0. CPU synthetic tests (host or container)
cd pada_proto && python tools/test_pada.py          # 22/22 must pass

# 1. direction sets (CPU or GPU, ~1 min)
python tools/pada_gen_directions.py \
    --out_dir ../results/pada/dirs --device cuda:0

# 2. probe + eval collection (GPU sim, deterministic mode, ~30 min)
docker exec -w /workspace/supermimic/pada_proto crgf_smoke \
    python tools/pada_sim_collect.py \
    --dirs_dir ../results/pada/dirs --out_dir ../results/pada/collect \
    --max_minutes 75

# 3. fine-tune M1/M2 from M0 (GPU, ~5 min)
python tools/pada_finetune_wm.py \
    --probes ../results/pada/collect/probes.pt \
    --out_dir ../results/pada/ckpts --steps 1500

# 4. evaluation (GPU WM-side autograd, ~40 min)
python tools/pada_eval.py \
    --dirs_dir ../results/pada/dirs --collect_dir ../results/pada/collect \
    --ckpt_dir ../results/pada/ckpts --out_dir ../results/pada/eval
```

Outputs land in `../results/pada/`: `directions_meta.json` (all pair SHA256s),
`collect.budget.jsonl` (sim query ledger), `probes.pt`/`eval_data.pt` (raw
trajectories), `M_*.pt` (snapshot checkpoints + provenance), `eval_rows.csv`,
`eval_summary.json`, `level1.json`.

## Key conventions (do not weaken when editing)

* Policy-shift directions are OPEN-LOOP shared-observation deltas
  `pi_new(o)-pi_old(o)` from real diag-pair bundles (`VALID_SINGLE_POLICY_UPDATE`,
  `pi_updates N->N+1`); never present them as closed-loop deltas.
* The two probe arms differ ONLY in direction sets; starts, base actions
  (pi_old rolled closed-loop in M0), eps, branch count, seeds, sim config,
  fine-tune schedule and eval protocol are shared.
* Fail-closed: pair bundles failing validation (normalizer changed, not a
  verified single update, strict-load mismatch) are rejected, not repaired.
* The reliability gate is computed from sim data only (model-independent), so
  all models are scored on identical gated samples.
