"""Audit an independent action-conditioned WM uncertainty predictor.

The nominal world model is frozen.  A small network receives the same
deployment-time input as the WM and predicts its one-step normalized squared
error.  It is fit on one environment fold and evaluated on held-out
environments with the train low-risk thresholds reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.networks import WorldModel
from supermimic.mdp import normalize_xi, unflatten_state


def model_from_state(state, device):
    key = "residual_net.0.weight"
    hidden = int(state[key].shape[0]) if key in state else 0
    return WorldModel(residual_hidden=hidden).to(device)


def collect(wm, buf, ids, window, batch_size, batches, seed, trajectory_features=False):
    xs, errors = [], []
    torch.manual_seed(seed)
    with torch.no_grad():
        std = torch.maximum(wm.out_norm.std(), torch.full_like(wm.out_norm.std(), 1e-4))
        for _ in range(batches):
            w = buf.sample_windows(batch_size, window, env_ids=ids)
            state = unflatten_state(w["s"][:, 0])
            seq_x = []
            for j in range(window):
                seq_x.append(wm.in_norm(wm.inputs(
                    state, w["a"][:, j], normalize_xi(w["xi"][:, j]))))
                state = wm.step(state, w["a"][:, j], normalize_xi(w["xi"][:, j]))
            x0 = seq_x[0]
            if trajectory_features:
                seq = torch.stack(seq_x, 1)
                # Summary statistics preserve action-conditioned rollout
                # evolution without introducing a second recurrent model.
                x = torch.cat((x0, seq.mean(1), seq.abs().amax(1), seq[:, -1]), -1)
            else:
                x = x0
            pred = state
            target = unflatten_state(w["s1"][:, window - 1])
            error = (wm.state_error(pred, target) / std).square().mean(-1)
            valid = w["mask"][:, window - 1].bool()
            xs.append(x[valid]); errors.append(error[valid])
    return torch.cat(xs), torch.cat(errors)


def collect_sequence(wm, buf, ids, window, batch_size, batches, seed):
    """Collect the full nominal input sequence for a recurrent risk model."""
    xs, errors = [], []
    torch.manual_seed(seed)
    with torch.no_grad():
        std = torch.maximum(wm.out_norm.std(), torch.full_like(wm.out_norm.std(), 1e-4))
        for _ in range(batches):
            w = buf.sample_windows(batch_size, window, env_ids=ids)
            state = unflatten_state(w["s"][:, 0])
            seq = []
            for j in range(window):
                seq.append(wm.in_norm(wm.inputs(
                    state, w["a"][:, j], normalize_xi(w["xi"][:, j]))))
                state = wm.step(state, w["a"][:, j], normalize_xi(w["xi"][:, j]))
            target = unflatten_state(w["s1"][:, window - 1])
            error = (wm.state_error(state, target) / std).square().mean(-1)
            valid = w["mask"][:, window - 1].bool()
            xs.append(torch.stack(seq, 1)[valid]); errors.append(error[valid])
    return torch.cat(xs), torch.cat(errors)
from supermimic.buffer import ReplayBuffer


def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return (x * y).mean() / (x.square().mean().sqrt() * y.square().mean().sqrt()).clamp_min(1e-8)


def fit_model(x, error, epochs, lr, weight_decay):
    model = nn.Sequential(nn.Linear(x.shape[-1], 128), nn.SiLU(),
                          nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1)).to(x)
    # Log-error regression prevents a few catastrophic windows dominating the
    # uncertainty fit and yields a monotone risk score at deployment.
    target = error.clamp_min(1e-6).log().unsqueeze(-1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        pred = model(x)
        loss = nn.functional.smooth_l1_loss(pred, target)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model.eval()


def fit_sequence_model(x, error, epochs, lr, weight_decay):
    model = nn.GRU(x.shape[-1], 128, batch_first=True)
    head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
    model = nn.ModuleDict({"rnn": model, "head": head}).to(x)
    target = error.clamp_min(1e-6).log().unsqueeze(-1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        _, h = model["rnn"](x)
        pred = model["head"](h[-1])
        loss = nn.functional.smooth_l1_loss(pred, target)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model.eval()


def risk_score(model, x):
    if isinstance(model, nn.ModuleDict):
        _, h = model["rnn"](x)
        return model["head"](h[-1]).squeeze(-1)
    return model(x).squeeze(-1)


def summarize(model, x, error, q):
    with torch.no_grad():
        score = risk_score(model, x)
    threshold = torch.quantile(score, q)
    # Low predicted risk is the selected/ftrusted subset.
    sel = score <= threshold
    return {"train_quantile": q, "threshold": float(threshold),
            "selected_rate": float(sel.float().mean()),
            "selected_error": float(error[sel].mean()) if sel.any() else None,
            "selected_error_p90": float(torch.quantile(error[sel], .9)) if sel.any() else None,
            "all_error": float(error.mean()),
            "coverage_error_le_2": float((error[sel] <= 2).float().mean()) if sel.any() else None}


def summarize_fixed(model, x, error, threshold):
    with torch.no_grad():
        score = risk_score(model, x)
    sel = score <= threshold
    return {"threshold": float(threshold), "selected_rate": float(sel.float().mean()),
            "selected_error": float(error[sel].mean()) if sel.any() else None,
            "selected_error_p90": float(torch.quantile(error[sel], .9)) if sel.any() else None,
            "coverage_error_le_2": float((error[sel] <= 2).float().mean()) if sel.any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--audit_fraction", type=float, default=.25)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trajectory_features", action="store_true",
                    help="use mean/max/final nominal-rollout features instead of only the initial input")
    ap.add_argument("--sequence_model", action="store_true",
                    help="fit a GRU on the full nominal input sequence")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not 0 < args.audit_fraction < 1:
        ap.error("audit_fraction must be in (0,1)")
    ck = torch.load(args.checkpoint, map_location=args.device)
    if "buffer" not in ck:
        ap.error("checkpoint must contain replay buffer state; use last.pt")
    state = ck["buffer"]; T, N = state["data"]["s"].shape[:2]
    buf = ReplayBuffer(T, N, args.device); buf.load_state_dict(state)
    wm = model_from_state(ck["wm"], args.device); wm.load_state_dict(ck["wm"]); wm.eval()
    na = max(1, min(N - 1, round(N * args.audit_fraction)))
    tr_ids = torch.arange(0, N - na, device=args.device)
    au_ids = torch.arange(N - na, N, device=args.device)
    if args.sequence_model:
        xtr, entr = collect_sequence(wm, buf, tr_ids, args.window, args.batch, args.batches, args.seed)
        xau, enau = collect_sequence(wm, buf, au_ids, args.window, args.batch, args.batches, args.seed + 1)
        model = fit_sequence_model(xtr, entr, args.epochs, args.lr, args.weight_decay)
    else:
        xtr, entr = collect(wm, buf, tr_ids, args.window, args.batch, args.batches, args.seed,
                            args.trajectory_features)
        xau, enau = collect(wm, buf, au_ids, args.window, args.batch, args.batches, args.seed + 1,
                            args.trajectory_features)
    with torch.no_grad():
        str_ = risk_score(model, xtr); sau = risk_score(model, xau)
    out = {"checkpoint": str(args.checkpoint), "window": args.window,
           "trajectory_features": args.trajectory_features,
           "sequence_model": args.sequence_model,
           "train_samples": int(xtr.shape[0]), "audit_samples": int(xau.shape[0]),
           "train_error_mean": float(entr.mean()), "audit_error_mean": float(enau.mean()),
           "train_risk_error_corr": float(corr(str_, entr.log1p())),
           "audit_risk_error_corr": float(corr(sau, enau.log1p())),
           "calibrated_on_train": {}}
    for q in (.5, .75, .9):
        t = torch.quantile(str_, q)
        out["calibrated_on_train"][str(q)] = {
            "train": summarize_fixed(model, xtr, entr, t),
            "audit": summarize_fixed(model, xau, enau, t),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
