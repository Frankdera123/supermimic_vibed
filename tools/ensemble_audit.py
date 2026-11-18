"""Offline disagreement calibration for independently trained WM checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.networks import WorldModel


def model_from_state(state, device):
    key = "residual_net.0.weight"
    hidden = int(state[key].shape[0]) if key in state else 0
    return WorldModel(residual_hidden=hidden).to(device)


def rollout(wm, w, k, use_residual=False):
    state = unflatten_state(w["s"][:, 0])
    preds = []
    for j in range(k):
        state = wm.step(state, w["a"][:, j], normalize_xi(w["xi"][:, j]), use_residual=use_residual)
        preds.append(state)
    return tuple(torch.stack([x[i] for x in preds], 1) for i in range(6))


def metric(wms, w, k, use_residual=False):
    with torch.no_grad():
        preds = [rollout(wm, w, k, use_residual=use_residual) for wm in wms]
        target = unflatten_state(w["s1"])
        std = torch.sqrt(wms[0].out_norm.var + wms[0].out_norm.eps**2).clamp_min(1e-4)
        # Pairwise state disagreement, averaged across model pairs and all
        # valid transitions. For two members this is one normalized distance.
        dis = []
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                dis.append((wms[0].state_error(preds[i], preds[j]) / std).square().mean(-1))
        disagreement = torch.stack(dis).mean(0)
        actual = (wms[0].state_error(preds[0], target) / std).square().mean(-1)
        mask = w["mask"].bool()
        # final-step values are the main audit statistic.
        d, e = disagreement[:, -1][mask[:, -1]], actual[:, -1][mask[:, -1]]
        d0, e0 = disagreement[:, 0][mask[:, 0]], actual[:, 0][mask[:, 0]]
        def corr(x, y):
            x, y = x - x.mean(), y - y.mean()
            return (x * y).mean() / (x.square().mean().sqrt() * y.square().mean().sqrt()).clamp_min(1e-8)
        return torch.stack((corr(d0, e0), corr(d, e), d.mean(), e.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--reference", type=int, default=0)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--use_residual", action="store_true")
    args = ap.parse_args()
    if len(args.checkpoints) < 2:
        ap.error("provide at least two checkpoints")
    cks = [torch.load(p, map_location=args.device, weights_only=False) for p in args.checkpoints]
    ref = cks[args.reference]
    state = ref.get("buffer")
    if state is None:
        ap.error("reference checkpoint must contain replay buffer state")
    T, N = state["data"]["s"].shape[:2]
    buffer = ReplayBuffer(T, N, args.device)
    buffer.load_state_dict(state)
    if buffer.size < args.window:
        ap.error("replay buffer is shorter than the requested window")
    wms = []
    for ck in cks:
        wm = model_from_state(ck["wm"], args.device)
        wm.load_state_dict(ck["wm"])
        wm.eval()
        wms.append(wm)
    vals = []
    for _ in range(args.batches):
        w = buffer.sample_windows(args.batch, args.window)
        vals.append(metric(wms, w, args.window, use_residual=args.use_residual))
    out = torch.stack(vals).mean(0)
    print(json.dumps({
        "checkpoints": [str(p) for p in args.checkpoints],
        "window": args.window, "members": len(wms), "use_residual": args.use_residual,
        "one_step_corr": float(out[0]), "final_step_corr": float(out[1]),
        "mean_disagreement_final": float(out[2]),
        "mean_member0_error_final": float(out[3]),
    }, indent=2))


if __name__ == "__main__":
    main()
