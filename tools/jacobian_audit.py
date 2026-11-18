"""Estimate local state/action Jacobian norms of a saved world model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.mdp import invariant_features, normalize_xi, unflatten_state
from supermimic.networks import WorldModel


def model_from_state(state, device):
    key = "residual_net.0.weight"
    hidden = int(state[key].shape[0]) if key in state else 0
    return WorldModel(residual_hidden=hidden).to(device)


def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return (x * y).mean() / (x.square().mean().sqrt() * y.square().mean().sqrt()).clamp_min(1e-8)


def collect(wm, buf, device, batches, batch, residual):
    js, ja, err = [], [], []
    with torch.enable_grad():
        std = torch.maximum(wm.out_norm.std(), torch.full_like(wm.out_norm.std(), 1e-4))
        for _ in range(batches):
            w = buf.sample_windows(batch, 1)
            sflat = w["s"][:, 0].detach().requires_grad_(True)
            action = w["a"][:, 0].detach().requires_grad_(True)
            xi = normalize_xi(w["xi"][:, 0])
            S = unflatten_state(sflat)
            pred = wm.step(S, action, xi, use_residual=residual)
            y = invariant_features(pred)
            probe = torch.randn_like(y)
            gs, ga = torch.autograd.grad((y * probe).sum(), (sflat, action), retain_graph=False)
            js.append(gs.square().mean(-1).sqrt().detach())
            ja.append(ga.square().mean(-1).sqrt().detach())
            target = unflatten_state(w["s1"][:, 0])
            err.append((wm.state_error(pred, target) / std).square().mean(-1).detach())
    return torch.cat(js), torch.cat(ja), torch.cat(err)


def summarize(x):
    return {"mean": float(x.mean()), "p50": float(torch.quantile(x, .5)),
            "p90": float(torch.quantile(x, .9)), "p99": float(torch.quantile(x, .99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    for path in args.checkpoints:
        ck = torch.load(path, map_location=args.device)
        state = ck.get("buffer")
        if state is None:
            ap.error(f"{path} must contain replay buffer state")
        T, N = state["data"]["s"].shape[:2]
        buf = ReplayBuffer(T, N, args.device); buf.load_state_dict(state)
        wm = model_from_state(ck["wm"], args.device); wm.load_state_dict(ck["wm"]); wm.eval()
        residual = "residual_net.0.weight" in ck["wm"]
        # Reset sampling so checkpoints from the same run are audited on the
        # same replay transitions when multiple paths are supplied.
        torch.manual_seed(0)
        js, ja, er = collect(wm, buf, args.device, args.batches, args.batch, residual)
        print(json.dumps({"checkpoint": str(path), "residual_branch": residual,
                          "samples": int(er.numel()), "state_jacobian": summarize(js),
                          "action_jacobian": summarize(ja), "one_step_error": summarize(er),
                          "state_jacobian_error_corr": float(corr(js, er)),
                          "action_jacobian_error_corr": float(corr(ja, er))}, indent=2))


if __name__ == "__main__":
    main()
