"""K-fold environment-ID calibration for ensemble disagreement thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from ensemble_coverage import collect, model_from_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if len(args.checkpoints) < 2 or args.folds < 2:
        ap.error("provide >=2 checkpoints and folds>=2")
    torch.manual_seed(args.seed)
    cks = [torch.load(p, map_location=args.device) for p in args.checkpoints]
    state = cks[0].get("buffer")
    if state is None:
        ap.error("first checkpoint must contain replay buffer state")
    T, N = state["data"]["s"].shape[:2]
    folds = min(args.folds, N)
    buffer = ReplayBuffer(T, N, args.device); buffer.load_state_dict(state)
    wms = []
    for ck in cks:
        m = model_from_state(ck["wm"], args.device); m.load_state_dict(ck["wm"]); m.eval(); wms.append(m)
    out = {"checkpoints": [str(p) for p in args.checkpoints], "folds": folds, "results": []}
    edges = torch.linspace(0, N, folds + 1, device=args.device).round().long()
    for f in range(folds):
        lo, hi = int(edges[f]), int(edges[f + 1])
        audit_ids = torch.arange(lo, hi, device=args.device)
        train_ids = torch.cat((torch.arange(0, lo, device=args.device),
                               torch.arange(hi, N, device=args.device)))
        train_d, _ = collect(wms, buffer, train_ids, args.window, args.batch, args.batches)
        audit_d, audit_e = collect(wms, buffer, audit_ids, args.window, args.batch, args.batches)
        t = torch.quantile(train_d, .5)
        sel = audit_d <= t
        selected = audit_e[sel]
        out["results"].append({
            "fold": f, "audit_env_range": [lo, hi - 1],
            "threshold": float(t), "audit_selection_rate": float(sel.float().mean()),
            "selected_mean_error": float(selected.mean()) if sel.any() else None,
            "selected_p90_error": float(torch.quantile(selected, .9)) if sel.any() else None,
            "selected_error_le_2": float((selected <= 2).float().mean()) if sel.any() else None,
            "audit_mean_error_all": float(audit_e.mean()),
        })
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
