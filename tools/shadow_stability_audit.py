"""Join shadow checkpoint distance with paired simulator failure signals.

The audit is intentionally descriptive: it does not promote or reject a
candidate.  It answers whether parameter displacement, return change, and
failure-mode changes move together across checkpoints and horizons.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from shadow_checkpoint_audit import policy_distance, layerwise_distance
import torch


_ITER_RE = re.compile(r"shadow_ckpt_(\d+)$")


def _probe_rows(probe_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(probe_dir.glob("probe_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pair = payload.get("results", [])
        if len(pair) != 2:
            continue
        live, shadow = pair
        def delta(name: str) -> float:
            return float(shadow[name]) - float(live[name])
        def rel_return() -> float:
            base = max(abs(float(live["return_norm"])), 1e-12)
            return (float(shadow["return_norm"]) - float(live["return_norm"])) / base
        rows.append({
            "probe": str(path),
            "steps": int(payload["steps"]),
            "seed": int(payload["seed"]),
            "relative_return_delta": rel_return(),
            "survival_delta": delta("survival_at_end"),
            "failure_delta": delta("failure_rate"),
            "height_failure_delta": delta("height_failure_rate"),
            "tilt_failure_delta": delta("tilt_failure_rate"),
            "first_failure_delta": delta("first_failure_step_mean"),
        })
    return rows


def audit_run(run: Path) -> list[dict]:
    output = []
    for probe_dir in sorted((run / "probes").glob("shadow_ckpt_*")):
        match = _ITER_RE.match(probe_dir.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        live_path = run / f"ckpt_{iteration:06d}.pt"
        shadow_path = run / f"shadow_ckpt_{iteration:06d}.pt"
        if not live_path.exists() or not shadow_path.exists():
            continue
        live = torch.load(live_path, map_location="cpu", weights_only=False)
        shadow = torch.load(shadow_path, map_location="cpu", weights_only=False)
        distance, live_norm, shadow_norm, rel_shadow = policy_distance(live["pi"], shadow["pi"])
        report = {
            "run": str(run),
            "iteration": iteration,
            "policy_l2_distance": distance,
            "live_policy_l2_norm": live_norm,
            "shadow_policy_l2_norm": shadow_norm,
            "policy_relative_distance_to_live": distance / max(live_norm, 1e-12),
            "policy_relative_distance_to_shadow": rel_shadow,
            "layerwise_relative_distance": layerwise_distance(live["pi"], shadow["pi"]),
            "probes": _probe_rows(probe_dir),
        }
        output.append(report)
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, type=Path,
                    help="shadow run directory; repeat for multiple runs")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    reports = [row for run in args.run for row in audit_run(run)]
    rendered = json.dumps(reports, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
