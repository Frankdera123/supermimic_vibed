"""Validate the hand-curated matched-budget experiment matrix."""
from __future__ import annotations

import argparse, csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path,
                    default=Path(__file__).resolve().parents[1] / "docs" / "experiment_matrix.csv")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.path.open()))
    required = {"interaction", "optimizer", "wallclock"}
    axes = {r["matching_axis"] for r in rows}
    if not required <= axes:
        raise SystemExit(f"missing matching axes: {sorted(required - axes)}")
    for i, row in enumerate(rows, 2):
        a = float(row["return_double_or_method"])
        b = float(row["return_baseline_or_compare"])
        claimed = float(row["relative_change"])
        actual = a / b - 1.0
        if abs(actual - claimed) > args.tol:
            raise SystemExit(f"row {i}: relative_change={claimed} but recomputed {actual}")
    print(f"validated {len(rows)} rows across {len(axes)} matching axes")


if __name__ == "__main__":
    main()
