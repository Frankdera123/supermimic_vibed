"""Measure where a learned WM residual improves nominal prediction error."""

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
    if key not in state:
        raise ValueError("checkpoint has no residual_net")
    return WorldModel(residual_hidden=int(state[key].shape[0])).to(device)


def rollout(wm, w, k, residual):
    s = unflatten_state(w["s"][:, 0]); out = []
    for j in range(k):
        s = wm.step(s, w["a"][:, j], normalize_xi(w["xi"][:, j]), use_residual=residual)
        out.append(s)
    return tuple(torch.stack([x[i] for x in out], 1) for i in range(6))


def collect(wm, buf, ids, k, batch, batches):
    us, en, er, dr = [], [], [], []
    with torch.no_grad():
        std = torch.sqrt(wm.out_norm.var + wm.out_norm.eps ** 2).clamp_min(1e-4)
        for _ in range(batches):
            w = buf.sample_windows(batch, k, env_ids=ids)
            nom, res = rollout(wm, w, k, False), rollout(wm, w, k, True)
            target = unflatten_state(w["s1"])
            en0 = (wm.state_error(nom, target) / std).square().mean(-1)[:, -1]
            er0 = (wm.state_error(res, target) / std).square().mean(-1)[:, -1]
            d = (wm.state_error(res, nom) / std).square().mean(-1)[:, -1]
            # Use disagreement magnitude as the gate score; all values are
            # restricted to valid final transitions.
            ok = w["mask"][:, -1].bool()
            us.append(d[ok]); en.append(en0[ok]); er.append(er0[ok]); dr.append((er0 - en0)[ok])
    return torch.cat(us), torch.cat(en), torch.cat(er), torch.cat(dr)


def summarize(u, en, er, q):
    t = torch.quantile(u, q); sel = u >= t
    return summarize_threshold(u, en, er, t, q)


def summarize_threshold(u, en, er, t, q=None):
    sel = u >= t
    if not bool(sel.any()):
        return {"uncertainty_quantile": q, "threshold": float(t),
                "selected_rate": 0.0, "nominal_error": 0.0,
                "residual_error": 0.0,
                "error_delta_residual_minus_nominal": 0.0,
                "residual_improves_fraction": 0.0}
    return {"uncertainty_quantile": q, "threshold": float(t),
            "selected_rate": float(sel.float().mean()),
            "nominal_error": float(en[sel].mean()), "residual_error": float(er[sel].mean()),
            "error_delta_residual_minus_nominal": float((er[sel] - en[sel]).mean()),
            "residual_improves_fraction": float((er[sel] < en[sel]).float().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--buffer_checkpoint", type=Path, default=None,
                    help="optional checkpoint whose replay buffer is evaluated")
    ap.add_argument("--norm_checkpoint", type=Path, default=None,
                    help="optional checkpoint supplying WM input/output normalization")
    ap.add_argument("--audit_fraction", type=float, default=.25)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location=args.device)
    buffer_ck = torch.load(args.buffer_checkpoint, map_location=args.device) if args.buffer_checkpoint else ck
    state = buffer_ck.get("buffer")
    if state is None:
        ap.error("checkpoint must contain replay buffer")
    T, N = state["data"]["s"].shape[:2]
    buf = ReplayBuffer(T, N, args.device); buf.load_state_dict(state)
    na = max(1, min(N - 1, round(N * args.audit_fraction)))
    wm = model_from_state(ck["wm"], args.device); wm.load_state_dict(ck["wm"]); wm.eval()
    if args.norm_checkpoint:
        norm_ck = torch.load(args.norm_checkpoint, map_location=args.device)
        ns = norm_ck["wm"]
        for name in ("in_norm", "out_norm"):
            module = getattr(wm, name)
            for field in ("mean", "var", "count"):
                key = f"{name}.{field}"
                if key in ns:
                    getattr(module, field).copy_(ns[key])
    out = {"checkpoint": str(args.checkpoint),
           "buffer_checkpoint": str(args.buffer_checkpoint or args.checkpoint),
           "norm_checkpoint": str(args.norm_checkpoint or args.checkpoint),
           "window": args.window, "folds": {}}
    collected = {}
    for name, ids in (("train", torch.arange(0, N - na, device=args.device)),
                      ("audit", torch.arange(N - na, N, device=args.device))):
        u, en, er, delta = collect(wm, buf, ids, args.window, args.batch, args.batches)
        collected[name] = (u, en, er)
        out["folds"][name] = {"samples": int(u.numel()),
                               "all_nominal_error": float(en.mean()),
                               "all_residual_error": float(er.mean()),
                               "all_delta": float(delta.mean()),
                               "bins": [summarize(u, en, er, q) for q in (.5, .75, .9)]}
    # Deployment-style calibration: thresholds are estimated on the training
    # environments and reused unchanged on held-out environments.
    u_tr, en_tr, er_tr = collected["train"]
    u_au, en_au, er_au = collected["audit"]
    out["calibrated_on_train"] = {}
    for q in (.5, .75, .9):
        threshold = torch.quantile(u_tr, q)
        out["calibrated_on_train"][str(q)] = {
            "train": summarize_threshold(u_tr, en_tr, er_tr, threshold, q),
            "audit": summarize_threshold(u_au, en_au, er_au, threshold, q),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
