"""Merge compatible counterfactual datasets along their sampled-state axis."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="+", type=Path)
ap.add_argument("--output", required=True, type=Path)
args = ap.parse_args()
if len(args.datasets) < 2:
    ap.error("provide at least two datasets")
loaded = [torch.load(p, map_location="cpu", weights_only=False) for p in args.datasets]
required = {"xi", "action", "directions", "pairs"}
if any(not required.issubset(d) for d in loaded):
    ap.error("all datasets must contain xi/action/directions/pairs")
shape = loaded[0]["directions"].shape[1:]
if any(d["directions"].shape[1:] != shape for d in loaded):
    ap.error("direction dimensions must match")
out = dict(loaded[0])
out["xi"] = torch.cat([d["xi"] for d in loaded], 0)
out["action"] = torch.cat([d["action"] for d in loaded], 0)
out["directions"] = torch.cat([d["directions"] for d in loaded], 0)
pair_count = len(loaded[0]["pairs"])
if any(len(d["pairs"]) != pair_count for d in loaded):
    ap.error("datasets must contain the same number of delta pairs")
out["pairs"] = []
for pair_idx in range(pair_count):
    src = [d["pairs"][pair_idx] for d in loaded]
    deltas = [float(p["delta"]) for p in src]
    if any(x != deltas[0] for x in deltas[1:]):
        ap.error("corresponding pairs must have identical delta values")
    merged = {"delta": src[0]["delta"]}
    for key in ("s0", "plus", "minus"):
        merged[key] = torch.cat([p[key] for p in src], 0)
    out["pairs"].append(merged)
if all("frames" in d for d in loaded):
    out["frames"] = torch.cat([d["frames"] for d in loaded], 0)
out["source_datasets"] = [str(p) for p in args.datasets]
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(out, args.output)
print({"output": str(args.output), "states": int(out["xi"].shape[0]),
       "pairs": len(out["pairs"]), "directions": int(out["directions"].shape[1])})
