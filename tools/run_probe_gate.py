"""Run paired fixed-policy probes and update the training control file.

This runner is intended to execute inside the SuperMimic environment.  It
keeps all starts paired by evaluating reference and candidate checkpoints in a
single ``fixed_policy_eval.py`` invocation for each split and seed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--motion-dir", action="append", required=True)
    ap.add_argument("--seed", action="append", type=int, required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--control-file", required=True, type=Path)
    ap.add_argument("--rollback-checkpoint", type=Path, default=None)
    ap.add_argument("--promote-on-accept", action="store_true",
                    help="write shadow_promote after an accepted probe")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, action="append", default=None,
                    help="probe horizon; repeat to require every listed horizon")
    ap.add_argument("--min-steps", type=int, default=None,
                    help="minimum horizon accepted by the gate (defaults to 500 for one horizon, or the shortest explicit horizon)")
    ap.add_argument("--max-relative-drop", type=float, default=0.05)
    ap.add_argument("--max-survival-drop", type=float, default=0.05)
    ap.add_argument("--max-first-failure-drop", type=float, default=0.0)
    ap.add_argument("--max-height-failure-drop", type=float, default=0.0)
    ap.add_argument("--max-tilt-failure-drop", type=float, default=0.0)
    ap.add_argument("--failure-z", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not args.motion_dir or not args.seed:
        ap.error("at least one motion split and seed are required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    horizons = args.steps or [500]
    if any(h < 1 for h in horizons):
        ap.error("--steps must be positive")
    min_steps = args.min_steps if args.min_steps is not None else (500 if args.steps is None else min(horizons))
    if min_steps < 0:
        ap.error("--min-steps must be nonnegative")
    eval_script = Path(__file__).with_name("fixed_policy_eval.py")
    probes_by_horizon: dict[int, list[Path]] = {h: [] for h in horizons}
    for horizon in horizons:
        for split_idx, motion_dir in enumerate(args.motion_dir):
            for seed in args.seed:
                out = args.out_dir / f"probe_h{horizon}_split{split_idx}_seed{seed}.json"
                cmd = [sys.executable, str(eval_script), "--batch", str(args.batch),
                       "--steps", str(horizon), "--seed", str(seed),
                       "--deterministic", "--device", args.device,
                       "--eval_motion_dir", motion_dir, "--json_out", str(out),
                       str(args.reference), str(args.candidate)]
                subprocess.run(cmd, check=True)
                probes_by_horizon[horizon].append(out)

    # A single horizon keeps the original subprocess/control behavior.  With
    # multiple horizons, evaluate each horizon independently and require all of
    # them to pass; a short optimistic probe cannot dilute a long-horizon fail.
    decisions = []
    gate_script = Path(__file__).with_name("policy_probe_gate.py")
    for horizon, probes in probes_by_horizon.items():
        gate = [sys.executable, str(gate_script), *map(str, probes),
                "--reference", str(args.reference), "--candidate", str(args.candidate),
                "--max-relative-drop", str(args.max_relative_drop),
                "--max-survival-drop", str(args.max_survival_drop),
                "--max-first-failure-drop", str(args.max_first_failure_drop),
                "--max-height-failure-drop", str(args.max_height_failure_drop),
                "--max-tilt-failure-drop", str(args.max_tilt_failure_drop),
                "--failure-z", str(args.failure_z),
                "--min-steps", str(max(min_steps, horizon))]
        if len(horizons) == 1:
            gate += ["--control-file", str(args.control_file)]
            if args.rollback_checkpoint is not None:
                gate += ["--rollback-checkpoint", str(args.rollback_checkpoint)]
            result = subprocess.run(gate, check=False)
            decisions.append(result.returncode == 0)
            if result.returncode == 0 and args.promote_on_accept:
                args.control_file.write_text("shadow_promote\n", encoding="utf-8")
            raise SystemExit(result.returncode)
        result = subprocess.run(gate, check=False, capture_output=True, text=True)
        decisions.append(result.returncode == 0)
        if result.stdout:
            try:
                print(json.loads(result.stdout))
            except json.JSONDecodeError:
                print(result.stdout, end="")
    passed = all(decisions)
    if passed:
        token = "shadow_promote\n" if args.promote_on_accept else "enable\n"
    elif args.rollback_checkpoint is not None:
        token = f"rollback:{args.rollback_checkpoint}\n"
    else:
        token = "disable\n"
    args.control_file.write_text(token, encoding="utf-8")
    print(json.dumps({"passed": passed, "horizons": horizons,
                      "per_horizon_passed": decisions}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
