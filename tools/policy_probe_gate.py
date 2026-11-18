"""Decide whether a candidate policy checkpoint passes repeated real-sim probes.

Each JSON is produced by ``fixed_policy_eval.py`` with the reference and
candidate checkpoints in the same invocation, so their starts are paired.  A
candidate passes only when the lower confidence bound of its paired return
difference is above the allowed relative degradation margin.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _return(data: dict, needle: str) -> float:
    for row in data["results"]:
        if needle in row["checkpoint"]:
            return float(row["return_norm"])
    raise KeyError(f"checkpoint {needle!r} not found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probes", nargs="+", type=Path,
                    help="paired fixed_policy_eval JSON files")
    ap.add_argument("--reference", required=True,
                    help="reference checkpoint basename or path substring")
    ap.add_argument("--candidate", required=True,
                    help="candidate checkpoint basename or path substring")
    ap.add_argument("--max-relative-drop", type=float, default=0.05,
                    help="allowed relative drop in the confidence bound")
    ap.add_argument("--max-survival-drop", type=float, default=0.05,
                    help="allowed absolute end-survival drop when probe rows provide survival_at_end")
    ap.add_argument("--max-first-failure-drop", type=float, default=0.0,
                    help="allowed absolute drop in mean first-failure step (0 disables this gate)")
    ap.add_argument("--max-height-failure-drop", type=float, default=0.0,
                    help="allowed absolute increase in height-failure rate (0 disables this gate)")
    ap.add_argument("--max-tilt-failure-drop", type=float, default=0.0,
                    help="allowed absolute increase in tilt-failure rate (0 disables this gate)")
    ap.add_argument("--failure-z", type=float, default=0.0,
                    help="optional normal margin for subtype-delta upper bounds (0 uses point estimates)")
    ap.add_argument("--z", type=float, default=1.96,
                    help="normal quantile used for the lower confidence bound")
    ap.add_argument("--min-steps", type=int, default=0,
                    help="reject probe files whose fixed-policy horizon is shorter")
    ap.add_argument("--control-file", type=Path, default=None,
                    help="optional file to atomically write enable/disable")
    ap.add_argument("--rollback-checkpoint", type=Path, default=None,
                    help="checkpoint path to encode on rejection for policy rollback")
    args = ap.parse_args()
    if (args.max_relative_drop < 0 or args.max_survival_drop < 0
            or args.max_first_failure_drop < 0 or args.max_height_failure_drop < 0
            or args.max_tilt_failure_drop < 0 or args.failure_z < 0
            or args.z < 0 or args.min_steps < 0):
        ap.error("gate thresholds, --z, and --min-steps must be nonnegative")
    deltas = []
    survival_drops = []
    first_failure_drops = []
    height_failure_drops = []
    tilt_failure_drops = []
    height_failure_upper = []
    tilt_failure_upper = []
    for path in args.probes:
        data = json.loads(path.read_text())
        if args.min_steps and int(data.get("steps", 0)) < args.min_steps:
            raise SystemExit(
                f"probe {path} has {data.get('steps', 0)} steps; "
                f"minimum required is {args.min_steps}")
        ref = _return(data, args.reference)
        cand = _return(data, args.candidate)
        if ref <= 0:
            raise ValueError(f"non-positive reference return in {path}")
        deltas.append((cand - ref) / ref)
        ref_row = next(row for row in data["results"] if args.reference in row["checkpoint"])
        cand_row = next(row for row in data["results"] if args.candidate in row["checkpoint"])
        if "survival_at_end" in ref_row and "survival_at_end" in cand_row:
            survival_drops.append(float(ref_row["survival_at_end"]) - float(cand_row["survival_at_end"]))
        if ("first_failure_step_mean" in ref_row
                and "first_failure_step_mean" in cand_row):
            first_failure_drops.append(
                float(ref_row["first_failure_step_mean"])
                - float(cand_row["first_failure_step_mean"]))
        if "height_failure_rate" in ref_row and "height_failure_rate" in cand_row:
            ref_rate = float(ref_row["height_failure_rate"])
            cand_rate = float(cand_row["height_failure_rate"])
            delta = cand_rate - ref_rate
            height_failure_drops.append(delta)
            n = max(int(data.get("batch", 1)), 1)
            se = math.sqrt((ref_rate * (1 - ref_rate) + cand_rate * (1 - cand_rate)) / n)
            height_failure_upper.append(delta + args.failure_z * se)
        if "tilt_failure_rate" in ref_row and "tilt_failure_rate" in cand_row:
            ref_rate = float(ref_row["tilt_failure_rate"])
            cand_rate = float(cand_row["tilt_failure_rate"])
            delta = cand_rate - ref_rate
            tilt_failure_drops.append(delta)
            n = max(int(data.get("batch", 1)), 1)
            se = math.sqrt((ref_rate * (1 - ref_rate) + cand_rate * (1 - cand_rate)) / n)
            tilt_failure_upper.append(delta + args.failure_z * se)
    mean = sum(deltas) / len(deltas)
    variance = sum((x - mean) ** 2 for x in deltas) / max(len(deltas) - 1, 1)
    stderr = math.sqrt(variance / len(deltas))
    lower = mean - args.z * stderr
    survival_passed = all(drop <= args.max_survival_drop for drop in survival_drops)
    failure_passed = (args.max_first_failure_drop <= 0.0
                      or all(drop <= args.max_first_failure_drop for drop in first_failure_drops))
    height_passed = (args.max_height_failure_drop <= 0.0
                     or all(drop <= args.max_height_failure_drop for drop in height_failure_upper))
    tilt_passed = (args.max_tilt_failure_drop <= 0.0
                   or all(drop <= args.max_tilt_failure_drop for drop in tilt_failure_upper))
    passed = (lower >= -args.max_relative_drop and survival_passed and failure_passed
              and height_passed and tilt_passed)
    if args.control_file is not None:
        args.control_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.control_file.with_name(args.control_file.name + ".tmp")
        if passed:
            token = "enable\n"
        elif args.rollback_checkpoint is not None:
            token = f"rollback:{args.rollback_checkpoint}\n"
        else:
            token = "disable\n"
        tmp.write_text(token)
        tmp.replace(args.control_file)
    print(json.dumps({
        "passed": passed,
        "n_probes": len(deltas),
        "relative_deltas": deltas,
        "mean_relative_delta": mean,
        "lower_confidence_bound": lower,
        "max_relative_drop": args.max_relative_drop,
        "survival_drops": survival_drops,
        "max_survival_drop": args.max_survival_drop,
        "first_failure_drops": first_failure_drops,
        "max_first_failure_drop": args.max_first_failure_drop,
        "height_failure_drops": height_failure_drops,
        "max_height_failure_drop": args.max_height_failure_drop,
        "height_failure_upper_bounds": height_failure_upper,
        "tilt_failure_drops": tilt_failure_drops,
        "max_tilt_failure_drop": args.max_tilt_failure_drop,
        "tilt_failure_upper_bounds": tilt_failure_upper,
        "failure_z": args.failure_z,
    }, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
