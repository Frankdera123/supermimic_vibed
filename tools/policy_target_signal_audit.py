"""Audit whether training-time metrics predict fixed-policy control quality.

This is deliberately descriptive: it joins one training log and one or more
fixed-policy JSON files by run name, then reports the final training metrics and
the mean held-out return.  It is used to prevent replay/imagination metrics from
being treated as a deployment-value fallback without evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("sim_rew", "imag_rew", "imag_survival", "ep_ret_norm", "ep_len", "term_frac")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="directory containing run subdirectories")
    ap.add_argument("--run", action="append", required=True,
                    help="run name; may be supplied more than once")
    ap.add_argument("--eval", action="append", required=True, dest="eval_files",
                    help="fixed-policy JSON for each run, in the same order as --run")
    args = ap.parse_args()
    if len(args.run) != len(args.eval_files):
        ap.error("--run and --eval must have the same number of entries")
    print("run,sim_rew,imag_rew,imag_survival,ep_ret_norm,ep_len,term_frac,fixed_return")
    for run, eval_file in zip(args.run, args.eval_files):
        rows = list(csv.DictReader((args.root / run / "train_log.csv").open()))
        if not rows:
            raise RuntimeError(f"empty training log: {run}")
        row = rows[-1]
        result = json.loads(Path(eval_file).read_text())["results"]
        fixed_return = sum(float(x["return_norm"]) for x in result) / len(result)
        vals = [row.get(k, "") for k in METRICS]
        print(",".join([run, *vals, f"{fixed_return:.8g}"]))


if __name__ == "__main__":
    main()
