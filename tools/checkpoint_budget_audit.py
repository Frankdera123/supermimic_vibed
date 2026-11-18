#!/usr/bin/env python3
"""Audit saved SuperMimic checkpoints for actual interaction/optimizer budgets."""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import torch


def audit(path: str) -> dict[str, object]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck.get("args", {}) or {}
    iteration = int(ck.get("iteration", -1))
    policy_updates = int(args.get("policy_updates", 1))
    wm_batches = int(args.get("wm_batches", 0))
    policy_period = max(int(args.get("policy_period", 1)), 1)
    wm_only_iters = int(args.get("wm_only_iters", 0))
    post_wm_iters = max(iteration - wm_only_iters, 0)
    policy_due = (post_wm_iters + policy_period - 1) // policy_period
    # The count is the configured optimizer-step budget. Warm-up policy steps
    # and interleave/replace variants are reported separately as flags because
    # their exact schedule depends on the training loop.
    policy_steps = policy_due * policy_updates
    wm_steps = iteration * wm_batches
    return {
        "checkpoint": str(path),
        "run": Path(path).parent.name,
        "iteration": iteration,
        "sim_steps": int(ck.get("sim_steps", -1)),
        "policy_updates": policy_updates,
        "policy_period": policy_period,
        "policy_optimizer_steps_est": policy_steps,
        "wm_batches": wm_batches,
        "wm_optimizer_steps_est": wm_steps,
        "total_optimizer_steps_est": policy_steps + wm_steps,
        "wall_hours": float(ck.get("wall_time", 0.0)) / 3600.0,
        "motion_dir": args.get("motion_dir", ""),
        "aux_window": int(args.get("wm_aux_window", 0)),
        "aux_weight": float(args.get("wm_aux_weight", 0.0)),
        "aux_ramp_iters": int(args.get("wm_aux_ramp_iters", 0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="checkpoint paths or glob patterns")
    ap.add_argument("--csv", default=None, help="optional CSV output path")
    ns = ap.parse_args()
    paths: list[str] = []
    for item in ns.paths:
        matches = sorted(glob.glob(item, recursive=True))
        paths.extend(matches or [item])
    rows = [audit(p) for p in paths if os.path.isfile(p)]
    if not rows:
        raise SystemExit("no checkpoint files found")
    fields = list(rows[0])
    if ns.csv:
        with open(ns.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
    print("run,iteration,sim_steps,policy_steps,wm_steps,total_steps,wall_h,aux")
    for r in rows:
        print(f"{r['run']},{r['iteration']},{r['sim_steps']},"
              f"{r['policy_optimizer_steps_est']},{r['wm_optimizer_steps_est']},"
              f"{r['total_optimizer_steps_est']},{r['wall_hours']:.3f},"
              f"{r['aux_window']}@{r['aux_weight']:.4g}/r{r['aux_ramp_iters']}")


if __name__ == "__main__":
    main()
