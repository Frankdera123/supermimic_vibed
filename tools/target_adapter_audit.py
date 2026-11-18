"""Offline audit for an action-conditioned policy-sensitive WM adapter.

The adapter is fitted on one replay split and evaluated on a disjoint split.
It predicts a normalized frame-local transition correction on top of a frozen
nominal WM; no simulator or policy update is performed here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.networks import WorldModel
from supermimic.rotations import so3_exp, yaw_mat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--trust_weight", type=float, default=1e-3)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=("env", "time"), default="time")
    ap.add_argument("--objective", choices=("one_step", "free"), default="free")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    data = ck["buffer"]["data"]
    wm = WorldModel().to(dev)
    wm.load_state_dict(ck["wm"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    # Build a disjoint split. Time splitting keeps environment identity fixed
    # while testing temporal generalization; env splitting is retained as a
    # stricter cross-environment diagnostic.
    n_env = data["s"].shape[0]
    n_time = data["s"].shape[1]
    cut = max(1, n_time // 2)
    if args.split == "env":
        env_cut = max(1, n_env // 2)
        train_env = torch.arange(env_cut, device=dev)
        audit_env = torch.arange(env_cut, n_env, device=dev)
    else:
        train_env = torch.arange(n_env, device=dev)
        audit_env = torch.arange(n_env, device=dev)
    in_dim = wm.in_dim
    adapter = nn.Sequential(nn.Linear(in_dim, args.hidden), nn.SiLU(), nn.Linear(args.hidden, 70)).to(dev)
    nn.init.zeros_(adapter[-1].weight); nn.init.zeros_(adapter[-1].bias)
    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    std = wm.out_norm.std().detach().clamp_min(1e-5)

    def batch(env_ids, train=True):
        e = env_ids[torch.randint(len(env_ids), (args.batch,), device=dev)]
        if args.split == "time":
            lo, hi = (0, cut) if train else (cut, n_time)
        else:
            lo, hi = 0, n_time
        t = torch.randint(lo, max(lo + 1, hi), (args.batch,), device=dev)
        s0 = data["s"][e, t].to(dev); s1 = data["s1"][e, t].to(dev)
        a = data["a"][e, t].to(dev); xi = normalize_xi(data["xi"][e, t].to(dev))
        return unflatten_state(s0), unflatten_state(s1), a, xi

    def features(S, a, xi):
        return wm.in_norm(wm.inputs(S, a, xi))

    def adapted_delta(S, a, xi):
        x = features(S, a, xi)
        base = wm.step(S, a, xi)
        base_y = wm.targets(S, base)
        return base_y + adapter(x) * std

    def apply_delta(S, delta):
        p, R, v, w, q, qd = S
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 29, 29), -1)
        return (p + (H @ dp.unsqueeze(-1)).squeeze(-1), R @ so3_exp(dth),
                v + (R @ dv.unsqueeze(-1)).squeeze(-1),
                w + (R @ dw.unsqueeze(-1)).squeeze(-1), q + dq, qd + dqd)

    def free_error(env_ids):
        T = data["s"].shape[1]; h = min(args.horizon, T)
        if args.split == "time":
            lo, hi = (cut, max(cut + 1, T - h + 1))
        else:
            lo, hi = 0, max(1, T - h + 1)
        e = env_ids[torch.randint(len(env_ids), (args.batch,), device=dev)]
        t = torch.randint(lo, hi, (args.batch,), device=dev)
        S0 = unflatten_state(data["s"][e, t].to(dev))
        S_base, S_adapt = S0, S0
        vals_base, vals_adapt = [], []
        with torch.no_grad():
            for k in range(h):
                a = data["a"][e, t + k].to(dev); xi = normalize_xi(data["xi"][e, t + k].to(dev))
                truth = unflatten_state(data["s1"][e, t + k].to(dev))
                S_base = wm.step(S_base, a, xi)
                S_adapt = apply_delta(S_adapt, adapted_delta(S_adapt, a, xi))
                vals_base.append((wm.state_error(S_base, truth) / std).square().mean(-1))
                vals_adapt.append((wm.state_error(S_adapt, truth) / std).square().mean(-1))
        return torch.stack(vals_base).mean(), torch.stack(vals_adapt).mean()

    def error(env_ids, train=False):
        # Aggregate a contiguous replay horizon while teacher-forcing the
        # observed state at each step. This measures whether the correction is
        # useful across the deployment horizon without conflating it with a
        # free-running state integrator.
        T = data["s"].shape[1]
        h = min(args.horizon, T)
        e = env_ids[torch.randint(len(env_ids), (args.batch,), device=dev)]
        if args.split == "time":
            lo, hi = (0, max(1, cut - h + 1)) if train else (cut, max(cut + 1, T - h + 1))
        else:
            lo, hi = 0, max(1, T - h + 1)
        t = torch.randint(lo, hi, (args.batch,), device=dev)
        vals = []
        with torch.no_grad():
            for k in range(h):
                s0 = data["s"][e, t + k].to(dev); s1 = data["s1"][e, t + k].to(dev)
                a = data["a"][e, t + k].to(dev); xi = normalize_xi(data["xi"][e, t + k].to(dev))
                S, S1 = unflatten_state(s0), unflatten_state(s1)
                vals.append(((adapted_delta(S, a, xi) - wm.targets(S, S1)) / std).square().mean())
        return torch.stack(vals).mean()

    base_audit = float(error(audit_env, train=False))
    for _ in range(args.steps):
        if args.objective == "one_step":
            S, S1, a, xi = batch(train_env, train=True)
            base_y = wm.targets(S, wm.step(S, a, xi)).detach()
            true_y = wm.targets(S, S1).detach()
            correction = adapter(features(S, a, xi))
            fit = (((base_y + correction * std - true_y) / std).square().mean())
        else:
            T = data["s"].shape[1]; h = min(args.horizon, T)
            if args.split == "time":
                lo, hi = 0, max(1, cut - h + 1)
            else:
                lo, hi = 0, max(1, T - h + 1)
            e = train_env[torch.randint(len(train_env), (args.batch,), device=dev)]
            t = torch.randint(lo, hi, (args.batch,), device=dev)
            S_pred = unflatten_state(data["s"][e, t].to(dev)); losses = []; corrections = []
            for k in range(h):
                a = data["a"][e, t + k].to(dev); xi = normalize_xi(data["xi"][e, t + k].to(dev))
                truth = unflatten_state(data["s1"][e, t + k].to(dev))
                correction = adapter(features(S_pred, a, xi)); corrections.append(correction)
                S_pred = apply_delta(S_pred, wm.targets(S_pred, wm.step(S_pred, a, xi)) + correction * std)
                losses.append((wm.state_error(S_pred, truth) / std).square().mean(-1))
            fit = torch.stack(losses).mean()
            trust = torch.stack(corrections).square().mean()
        if args.objective == "one_step":
            trust = correction.square().mean()
        loss = fit + args.trust_weight * trust
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); opt.step()
    adapted_audit = float(error(audit_env, train=False))
    free_base, free_adapt = free_error(audit_env)
    print({"base_audit_horizon": base_audit, "adapter_audit_horizon": adapted_audit,
           "relative_change": adapted_audit / max(base_audit, 1e-8) - 1.0,
           "base_free_running": float(free_base), "adapter_free_running": float(free_adapt),
           "free_running_relative_change": float(free_adapt / free_base.clamp_min(1e-8) - 1.0),
           "split": args.split, "objective": args.objective,
           "train_envs": int(len(train_env)), "audit_envs": int(len(audit_env)),
           "horizon": int(min(args.horizon, data["s"].shape[1])),
           "evaluation": "teacher-forced and free-running contiguous replay horizons"})


if __name__ == "__main__":
    main()
