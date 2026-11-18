"""Summarize paired SuperMimic ablation logs.

Example:
  python tools/analyze_ablation.py \
    --baseline logs/long_baseline logs/seed1_baseline logs/seed2_baseline \
    --variant logs/long_quantile logs/seed1_quantile logs/seed2_quantile
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from pathlib import Path


def final_row(run_dir: str) -> dict[str, float]:
    path = Path(run_dir) / "train_log.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows in {path}")
    row = rows[-1]
    return {k: float(v) for k, v in row.items() if k not in {"iteration"}}


def bootstrap_mean(values: list[float], draws: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws)]
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", nargs="+", required=True)
    p.add_argument("--variant", nargs="+", required=True)
    args = p.parse_args()
    if len(args.baseline) != len(args.variant):
        p.error("baseline and variant must have the same number of runs")
    b = [final_row(x) for x in args.baseline]
    v = [final_row(x) for x in args.variant]
    for metric in ("ep_ret_norm", "wm_err8", "sim_rew"):
        diffs = [v[i][metric] - b[i][metric] for i in range(len(b))]
        lo, hi = bootstrap_mean(diffs)
        print(f"{metric}: baseline_mean={statistics.mean(x[metric] for x in b):.6g} "
              f"variant_mean={statistics.mean(x[metric] for x in v):.6g} "
              f"paired_mean={statistics.mean(diffs):+.6g} "
              f"paired_std={statistics.stdev(diffs) if len(diffs) > 1 else 0.0:.6g} "
              f"bootstrap95=[{lo:+.6g},{hi:+.6g}] "
              f"wins={sum(d > 0 for d in diffs)}/{len(diffs)}")


if __name__ == "__main__":
    main()
