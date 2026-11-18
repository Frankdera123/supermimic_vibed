"""Audit an independent classifier for residual-world-model validity.

The classifier only sees deployment-time features (the normalized WM input and
time/context scalars).  Labels are generated from replay targets by comparing
one-step nominal and residual prediction error.  It is fit on one environment
fold and evaluated, with the same decision threshold, on a held-out fold.  The
script intentionally does not alter the policy or WM checkpoint; it tests
whether a separately learned gate has a transferable signal before it is
introduced into concurrent training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.networks import WorldModel


def model_from_state(state, device):
    key = "residual_net.0.weight"
    if key not in state:
        raise ValueError("checkpoint has no residual_net; classifier labels need a residual branch")
    return WorldModel(residual_hidden=int(state[key].shape[0])).to(device)


def batch(wm, w):
    s0 = unflatten_state(w["s"][:, 0])
    a = w["a"][:, 0]
    xi = normalize_xi(w["xi"][:, 0])
    x = wm.in_norm(wm.inputs(s0, a, xi))
    nom = wm.step(s0, a, xi, use_residual=False)
    res = wm.step(s0, a, xi, use_residual=True)
    target = unflatten_state(w["s1"][:, 0])
    std = torch.maximum(wm.out_norm.std(), torch.full_like(wm.out_norm.std(), 1e-4))
    en = (wm.state_error(nom, target) / std).square().mean(-1)
    er = (wm.state_error(res, target) / std).square().mean(-1)
    # The final two features are available at deployment and help prevent a
    # classifier from treating very early/late clip positions identically.
    # Fixed scales make the feature representation identical during offline
    # calibration and deployment; per-batch standardization would leak audit
    # fold statistics into the gate.
    f = w["f"][:, 0].float() / 1000.0
    step = w["ep_step"][:, 0].float() if "ep_step" in w else torch.zeros_like(f)
    step = step / 1000.0
    feat = torch.cat((x, f[:, None], step[:, None]), -1)
    valid = w["mask"][:, 0].bool()
    disagreement = (wm.state_error(res, nom) / std).square().mean(-1)
    return feat[valid], (er - en)[valid], en[valid], er[valid], disagreement[valid]


def collect(wm, buf, ids, window, batch_size, batches, seed):
    gen = torch.Generator(device=buf.device)
    gen.manual_seed(seed)
    xs, ds, ens, ers, us = [], [], [], [], []
    # ReplayBuffer samples from the global RNG.  Resetting the global seed at
    # each fold makes the audit deterministic without changing model state.
    torch.manual_seed(seed)
    with torch.no_grad():
        for _ in range(batches):
            w = buf.sample_windows(batch_size, window, env_ids=ids)
            x, d, en, er, u = batch(wm, w)
            xs.append(x); ds.append(d); ens.append(en); ers.append(er); us.append(u)
    return torch.cat(xs), torch.cat(ds), torch.cat(ens), torch.cat(ers), torch.cat(us)


def fit_gate(x, delta, epochs, lr, weight_decay):
    model = nn.Sequential(nn.Linear(x.shape[-1], 64), nn.SiLU(), nn.Linear(64, 1)).to(x)
    y = (delta < 0).float().unsqueeze(-1)
    pos = y.sum().clamp_min(1.0); neg = y.numel() - pos
    weights = torch.where(y > 0.5, neg / pos, torch.ones_like(y))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        logits = model(x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y, weight=weights)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model.eval(), float(y.mean())


def summarize(model, x, delta, en, er, threshold, score=None):
    if score is None:
        with torch.no_grad():
            score = model(x).squeeze(-1).sigmoid()
    sel = score >= threshold
    if not bool(sel.any()):
        return {"threshold": float(threshold), "selected_rate": 0.0,
                "delta_residual_minus_nominal": 0.0, "improves_fraction": 0.0,
                "nominal_error": 0.0, "residual_error": 0.0}
    return {"threshold": float(threshold), "selected_rate": float(sel.float().mean()),
            "delta_residual_minus_nominal": float(delta[sel].mean()),
            "improves_fraction": float((delta[sel] < 0).float().mean()),
            "nominal_error": float(en[sel].mean()), "residual_error": float(er[sel].mean())}


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
    ap.add_argument("--quantile", type=float, default=.75,
                    help="fraction rejected; selected gate is the upper tail")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save_model", type=Path, default=None,
                    help="optional path for the fitted gate checkpoint")
    args = ap.parse_args()
    if not 0 < args.audit_fraction < 1 or not 0 < args.quantile < 1:
        ap.error("audit_fraction and quantile must be in (0,1)")
    ck = torch.load(args.checkpoint, map_location=args.device)
    if "buffer" not in ck:
        ap.error("checkpoint must contain replay buffer state; use last.pt")
    state = ck["buffer"]; T, N = state["data"]["s"].shape[:2]
    buf = ReplayBuffer(T, N, args.device); buf.load_state_dict(state)
    if buf.size < args.window:
        ap.error(f"buffer has {buf.size} steps, need {args.window}")
    wm = model_from_state(ck["wm"], args.device); wm.load_state_dict(ck["wm"]); wm.eval()
    na = max(1, min(N - 1, round(N * args.audit_fraction)))
    tr_ids = torch.arange(0, N - na, device=args.device)
    au_ids = torch.arange(N - na, N, device=args.device)
    with torch.no_grad():
        xtr, dtr, entr, ertr, utr = collect(wm, buf, tr_ids, args.window, args.batch, args.batches, args.seed)
        xau, dau, enau, erau, uau = collect(wm, buf, au_ids, args.window, args.batch, args.batches, args.seed + 1)
    gate, train_rate = fit_gate(xtr, dtr, args.epochs, args.lr, args.weight_decay)
    with torch.no_grad():
        train_score = gate(xtr).squeeze(-1).sigmoid()
        train_threshold = torch.quantile(train_score, args.quantile)
        disagreement_threshold = torch.quantile(utr, args.quantile)
    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": gate.state_dict(), "input_dim": int(xtr.shape[-1]),
                    "threshold": float(train_threshold),
                    "quantile": float(args.quantile),
                    "checkpoint": str(args.checkpoint)}, args.save_model)
    out = {"checkpoint": str(args.checkpoint), "window": args.window,
           "train_envs": [int(tr_ids[0]), int(tr_ids[-1])],
           "audit_envs": [int(au_ids[0]), int(au_ids[-1])],
           "train_samples": int(xtr.shape[0]), "audit_samples": int(xau.shape[0]),
           "train_residual_improves_rate": train_rate,
           "audit_residual_improves_rate": float((dau < 0).float().mean()),
           "calibrated_on_train": {"quantile": args.quantile,
                                   "threshold": float(train_threshold),
                                   "train": summarize(gate, xtr, dtr, entr, ertr, train_threshold),
                                   "audit": summarize(gate, xau, dau, enau, erau, train_threshold)},
           "disagreement_baseline": {
               "threshold": float(disagreement_threshold),
               "train": summarize(gate, xtr, dtr, entr, ertr, disagreement_threshold, score=utr),
               "audit": summarize(gate, xau, dau, enau, erau, disagreement_threshold, score=uau)}}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
