"""Compare SuperMimic learning curves, not only final checkpoint rows.

Example:
  python tools/analyze_curves.py --metric ep_ret_norm --tail 3 \
      logs/long500_baseline_s0 logs/long500_baseline_s1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


def rows(run: str) -> list[dict[str, float]]:
    with (Path(run) / "train_log.csv").open(newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def auc(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return ys[0] if ys else float("nan")
    return sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) * 0.5
               for i in range(1, len(xs))) / (xs[-1] - xs[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories containing train_log.csv")
    p.add_argument("--metric", default="ep_ret_norm")
    p.add_argument("--tail", type=int, default=3, help="number of final logged rows for tail mean")
    args = p.parse_args()
    if args.tail < 1:
        p.error("--tail must be positive")
    summaries = []
    for run in args.runs:
        rs = rows(run)
        if not rs or args.metric not in rs[0]:
            raise ValueError(f"missing metric {args.metric} in {run}")
        x = [r["iteration"] for r in rs]
        y = [r[args.metric] for r in rs]
        tail = y[-args.tail:]
        summaries.append((run, y[-1], statistics.mean(tail), max(y), auc(x, y)))
    print("run,last,tail_mean,best,curve_auc")
    for run, last, tail, best, area in summaries:
        print(f"{run},{last:.6g},{tail:.6g},{best:.6g},{area:.6g}")
    if len(summaries) > 1:
        print("aggregate_tail_mean,%.6g" % statistics.mean(s[2] for s in summaries))


if __name__ == "__main__":
    main()
