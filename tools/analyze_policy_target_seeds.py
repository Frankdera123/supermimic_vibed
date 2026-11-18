"""Compare policy-target audit traces across matched training logs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = ("sim_rew", "policy_target_audit_h1", "policy_target_audit_h4",
          "policy_target_adapter_norm", "policy_target_accepts")


def load(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {k: np.asarray([float(r[k]) for r in rows], dtype=float) for k in FIELDS}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = {}
    for path in args.logs:
        values, rows = load(path / "train_log.csv")
        corr = {}
        for field in FIELDS[1:]:
            if np.std(values[field]) < 1e-12 or np.std(values["sim_rew"]) < 1e-12:
                corr[field] = None
            else:
                corr[field] = float(np.corrcoef(values[field], values["sim_rew"])[0, 1])
        result[str(path)] = {
            "iterations": len(rows),
            "sim_reward_first": float(values["sim_rew"][0]),
            "sim_reward_last": float(values["sim_rew"][-1]),
            "audit_accepts_last": int(values["policy_target_accepts"][-1]),
            "audit_rejects_last": int(float(rows[-1]["policy_target_rejects"])),
            "adapter_norm_last": float(values["policy_target_adapter_norm"][-1]),
            "reward_correlation": corr,
        }
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
