"""Apply a strict real-simulator acceptance gate to saved evaluator JSON."""
from __future__ import annotations
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("reference", type=Path)
ap.add_argument("candidate", type=Path)
ap.add_argument("--reference_index", type=int, default=0)
ap.add_argument("--candidate_index", type=int, default=0)
ap.add_argument("--return_tolerance", type=float, default=0.0)
ap.add_argument("--first_failure_tolerance", type=float, default=0.0)
args = ap.parse_args()
ref = json.loads(args.reference.read_text())["results"][args.reference_index]
cand = json.loads(args.candidate.read_text())["results"][args.candidate_index]
checks = {
    "return": cand["return_norm"] >= ref["return_norm"] - args.return_tolerance,
    "survival": cand["survival_at_end"] >= ref["survival_at_end"],
    "failure_rate": cand["failure_rate"] <= ref["failure_rate"],
    "height_failure_rate": cand["height_failure_rate"] <= ref["height_failure_rate"],
    "tilt_failure_rate": cand["tilt_failure_rate"] <= ref["tilt_failure_rate"],
    "first_failure_step": cand["first_failure_step_mean"] >= ref["first_failure_step_mean"] - args.first_failure_tolerance,
}
out = {"pass": all(checks.values()), "checks": checks,
       "reference": ref, "candidate": cand,
       "tolerances": {"return": args.return_tolerance,
                      "first_failure_step": args.first_failure_tolerance}}
print(json.dumps(out, indent=2))
