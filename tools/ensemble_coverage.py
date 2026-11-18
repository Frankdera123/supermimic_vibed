"""Train/audit coverage-risk calibration for independent WM disagreement."""

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
    out = []
    for j in range(k):
        state = wm.step(state, w["a"][:, j], normalize_xi(w["xi"][:, j]), use_residual=use_residual)
        out.append(state)
    return tuple(torch.stack([x[i] for x in out], 1) for i in range(6))


def scores(wms, w, k, step=-1, use_residual=False):
    with torch.no_grad():
        preds = [rollout(m, w, k, use_residual=use_residual) for m in wms]
        target = unflatten_state(w["s1"])
        std = torch.sqrt(wms[0].out_norm.var + wms[0].out_norm.eps ** 2).clamp_min(1e-4)
        pair = []
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                pair.append((wms[0].state_error(preds[i], preds[j]) / std).square().mean(-1))
        disagreement = torch.stack(pair).mean(0)
        error = (wms[0].state_error(preds[0], target) / std).square().mean(-1)
        mask = w["mask"].bool()
        return disagreement[:, step][mask[:, step]], error[:, step][mask[:, step]]


def collect(wms, buffer, env_ids, k, batch, batches, step=-1, use_residual=False):
    ds, es = [], []
    for _ in range(batches):
        d, e = scores(wms, buffer.sample_windows(batch, k, env_ids=env_ids), k,
                      step=step, use_residual=use_residual)
        ds.append(d); es.append(e)
    return torch.cat(ds), torch.cat(es)


def bootstrap_ci(values: torch.Tensor, statistic, repeats: int, seed: int = 0):
    """Percentile bootstrap CI over audit windows (not simulator trajectories)."""
    if values.numel() == 0:
        return None
    gen = torch.Generator(device=values.device).manual_seed(seed)
    n = values.numel()
    out = []
    for start in range(0, repeats, 128):
        b = min(128, repeats - start)
        idx = torch.randint(n, (b, n), generator=gen, device=values.device)
        out.append(statistic(values[idx]))
    boot = torch.cat(out).flatten()
    return [float(torch.quantile(boot, .025)), float(torch.quantile(boot, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--audit_fraction", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap repetitions for audit confidence intervals")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed for reproducible train/audit windows")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--use_residual", action="store_true")
    ap.add_argument("--horizons", default="",
                    help="comma-separated rollout steps to calibrate (default: final step only)")
    args = ap.parse_args()
    if len(args.checkpoints) < 2 or not 0 < args.audit_fraction < 1:
        ap.error("provide >=2 checkpoints and 0<audit_fraction<1")
    torch.manual_seed(args.seed)
    cks = [torch.load(p, map_location=args.device, weights_only=False) for p in args.checkpoints]
    state = cks[0].get("buffer")
    if state is None:
        ap.error("first checkpoint must contain replay buffer state")
    T, N = state["data"]["s"].shape[:2]
    buffer = ReplayBuffer(T, N, args.device); buffer.load_state_dict(state)
    na = max(1, min(N - 1, int(round(N * args.audit_fraction))))
    train_ids = torch.arange(0, N - na, device=args.device)
    audit_ids = torch.arange(N - na, N, device=args.device)
    wms = []
    for ck in cks:
        m = model_from_state(ck["wm"], args.device); m.load_state_dict(ck["wm"]); m.eval(); wms.append(m)
    horizons = [args.window - 1]
    if args.horizons:
        horizons = sorted(set(int(x) - 1 for x in args.horizons.split(",") if x.strip()))
        if not horizons or min(horizons) < 0 or max(horizons) >= args.window:
            ap.error("--horizons must contain 1-based steps within --window")
    result = {"checkpoints": [str(p) for p in args.checkpoints], "window": args.window,
              "use_residual": args.use_residual, "horizons": {}}
    for step in horizons:
        train_d, train_e = collect(wms, buffer, train_ids, args.window, args.batch, args.batches,
                                   step=step, use_residual=args.use_residual)
        audit_d, audit_e = collect(wms, buffer, audit_ids, args.window, args.batch, args.batches,
                                   step=step, use_residual=args.use_residual)
        thresholds = []
        for q in (0.50, 0.75, 0.90, 0.95):
            t = torch.quantile(train_d, q)
            sel = audit_d <= t
            selected = audit_e[sel]
            # Bootstrap the per-window audit statistics while keeping the threshold
            # fixed from train environments, which matches intended deployment.
            select_ci = bootstrap_ci(sel.float(), lambda x: x.mean(1), args.bootstrap)
            mean_ci = bootstrap_ci(selected, lambda x: x.mean(1), args.bootstrap)
            coverage_ci = bootstrap_ci((selected <= 2).float(), lambda x: x.mean(1), args.bootstrap)
            thresholds.append({
                "train_disagreement_quantile": q, "threshold": float(t),
                "audit_selection_rate": float(sel.float().mean()),
                "audit_selection_rate_ci95": select_ci,
                "audit_mean_error_selected": float(selected.mean()) if sel.any() else None,
                "audit_mean_error_selected_ci95": mean_ci,
                "audit_p90_error_selected": float(torch.quantile(selected, .90)) if sel.any() else None,
                "audit_error_coverage_le_2": float((selected <= 2).float().mean()) if sel.any() else None,
                "audit_error_coverage_le_2_ci95": coverage_ci,
            })
        result["horizons"][str(step + 1)] = {
            "train_samples": int(train_d.numel()),
            "audit_samples": int(audit_d.numel()),
            "thresholds": thresholds,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
