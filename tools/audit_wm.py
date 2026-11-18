"""Evaluate a saved world model on disjoint replay environment IDs.

The checkpoint must be a ``last.pt`` containing the replay buffer state.  This
is an audit only: it never updates normalization statistics or model weights.
"""

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


def rollout(wm: WorldModel, w: dict, k: int, use_residual: bool = False):
    state = unflatten_state(w["s"][:, 0])
    preds = []
    for j in range(k):
        state = wm.step(state, w["a"][:, j], normalize_xi(w["xi"][:, j]), use_residual=use_residual)
        preds.append(state)
    return tuple(torch.stack([x[i] for x in preds], dim=1) for i in range(6))


def errors(wm: WorldModel, w: dict, k: int, use_residual: bool = False):
    with torch.no_grad():
        pred = rollout(wm, w, k, use_residual=use_residual)
        target = unflatten_state(w["s1"])
        std = torch.sqrt(wm.out_norm.var + wm.out_norm.eps**2).clamp_min(1e-4)
        e = (wm.state_error(pred, target) / std).square().mean(-1)
        mask = w["mask"]
        denom = mask.sum().clamp_min(1.0)
        one = (e[:, 0] * mask[:, 0]).sum() / mask[:, 0].sum().clamp_min(1.0)
        final = (e[:, -1] * mask[:, -1]).sum() / mask[:, -1].sum().clamp_min(1.0)
        all_steps = (e * mask).sum() / denom
    return one, final, all_steps


def evaluate(wm, buffer, env_ids, k, batch, batches, use_residual=False):
    vals = []
    for _ in range(batches):
        w = buffer.sample_windows(batch, k, env_ids=env_ids)
        vals.append(errors(wm, w, k, use_residual=use_residual))
    return torch.stack([torch.stack(v) for v in vals]).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--audit_fraction", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--use_residual", action="store_true",
                    help="evaluate the optional residual branch in addition to nominal WM")
    args = ap.parse_args()
    if not 0.0 < args.audit_fraction < 1.0:
        ap.error("audit_fraction must be strictly between 0 and 1")
    ck = torch.load(args.checkpoint, map_location=args.device)
    if "buffer" not in ck:
        ap.error("checkpoint does not contain replay buffer state; use last.pt")
    state = ck["buffer"]
    T, N = state["data"]["s"].shape[:2]
    buffer = ReplayBuffer(T, N, args.device)
    buffer.load_state_dict(state)
    if buffer.size < args.window:
        ap.error(f"buffer has {buffer.size} steps, need {args.window}")
    n_audit = max(1, int(round(N * args.audit_fraction)))
    if n_audit >= N:
        n_audit = N - 1
    train_ids = torch.arange(0, N - n_audit, device=args.device)
    audit_ids = torch.arange(N - n_audit, N, device=args.device)
    wm = model_from_state(ck["wm"], args.device)
    wm.load_state_dict(ck["wm"])
    wm.eval()
    train = evaluate(wm, buffer, train_ids, args.window, args.batch, args.batches, args.use_residual)
    audit = evaluate(wm, buffer, audit_ids, args.window, args.batch, args.batches, args.use_residual)
    gap = audit - train
    result = {
        "checkpoint": str(args.checkpoint),
        "window": args.window, "use_residual": args.use_residual,
        "train_envs": [int(train_ids[0]), int(train_ids[-1])],
        "audit_envs": [int(audit_ids[0]), int(audit_ids[-1])],
        "train_one_step": float(train[0]), "audit_one_step": float(audit[0]),
        "train_final_step": float(train[1]), "audit_final_step": float(audit[1]),
        "train_all_steps": float(train[2]), "audit_all_steps": float(audit[2]),
        "audit_minus_train_final": float(gap[1]),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
