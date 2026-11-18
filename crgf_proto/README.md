# crgf_proto — CRGF isolated research workspace (published snapshot)

This directory is the **published snapshot** of the isolated working copy used
for the Control-Relevant Gradient Fidelity (CRGF) study. The narrative reports
live in [`../docs/crgf/`](../docs/crgf/) (10 reports, synced from
`crgf_proto/reports/`, which is not duplicated here).

## Layout

| path | content |
|---|---|
| `supermimic/` | isolated package copy used by the audits. `crgf.py` is the CRGF core (three-path decomposition, dual-protocol adapter, fail-closed gate). `robot.py` carries a CPU guard around the isaaclab imports — diff against the repo-root version: `results/crgf/code_diff_robot.patch` |
| `tools/` | the 9 CRGF tools only (the other audit tools live in the repo-root `tools/`) |
| `results/` | raw evidence: `b0b1b2_provenance.json` + `b0b1b2_sha256.txt` (checkpoint provenance), `results/crgf/` (980 simulator FD rows, WM-side gradient CSVs, gain sweep, verdict JSONs, figures, direction sets) |

The `.pt` files under `results/crgf/` are **direction sets** (unit action-sequence
perturbation directions), force-included via a `.gitignore` exception because
every audit consumes exactly these files. Model checkpoints are NOT included
(read-only inputs, see `results/b0b1b2_sha256.txt` for hashes).

## Running

```bash
cd crgf_proto
ln -s ../data data          # motion data (repo-root data/)
python tools/test_crgf.py   # 18/18 CPU synthetic tests
```

Tools resolve `supermimic` via `parents[1]`, i.e. from this directory — run
them with `crgf_proto` as the working directory. GPU tools
(`crgf_phase0_smoke.py`, `crgf_sim_fd_audit.py`) additionally require the
Isaac Lab / MuJoCo-Warp stack and read-only checkpoints; see
`../docs/crgf/GPU_EXPERIMENT_BUDGET.md` for the protocol and
`../docs/crgf/PHASE1_EXECUTION_DECISION.md` for the deterministic-mode
constraints (≤4 envs, run_to_run).

## Reproduction entry points

| artifact | produced by |
|---|---|
| `results/crgf/wm_audit_*.csv` | `tools/crgf_wm_directional_audit.py` |
| `results/crgf/pol_path_*.csv` | `tools/crgf_policy_path_audit.py` |
| `results/crgf/sim_fd_h8_v3/h32/h64*.csv` | `tools/crgf_sim_fd_audit.py` |
| `results/crgf/fidelity_verdict_*.json` | `tools/crgf_analyze_fidelity.py` |
| `results/crgf/gain_sweep_summary.json` | `tools/crgf_wm_directional_audit.py --action_gain` |
| `results/crgf/figures/*.png` | `tools/crgf_figures.py` |
