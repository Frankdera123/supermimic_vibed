"""Fit an offline strategic-error critic from a saved SuperMimic checkpoint.

The target is simulator-grounded one-step WM error on factual replay rows. The
critic is only a candidate selector; it never changes the policy or WM.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.networks import WorldModel
from supermimic.strategic_critic import StrategicErrorCritic


def _model_from_checkpoint(ck, device):
    state = ck["wm"]
    key = "residual_net.0.weight"
    hidden = int(state[key].shape[0]) if key in state else 0
    wm = WorldModel(residual_hidden=hidden).to(device)
    wm.load_state_dict(state)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    return wm


@torch.no_grad()
def _targets(wm, s, a, xi, s1, batch):
    std = torch.maximum(wm.out_norm.std(), torch.full_like(wm.out_norm.std(), 1e-4))
    out = []
    for lo in range(0, len(s), batch):
        hi = min(len(s), lo + batch)
        pred = wm.step(unflatten_state(s[lo:hi]), a[lo:hi], normalize_xi(xi[lo:hi]))
        err = (wm.state_error(pred, unflatten_state(s1[lo:hi])) / std).square().mean(-1)
        out.append(torch.log1p(err))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--dataset", type=Path, default=None,
                    help="optional policy-occupied dataset from collect_strategic_dataset.py")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_fraction", type=float, default=.2)
    ap.add_argument("--candidate_count", type=int, default=256)
    ap.add_argument("--candidate_noise", type=float, default=.5,
                    help="action noise std in normalized action units")
    ap.add_argument("--candidate_action_penalty", type=float, default=0.0,
                    help="penalty on squared candidate deviation from the factual action")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.epochs < 1 or args.batch < 1 or not 0.0 < args.val_fraction < 1.0:
        ap.error("epochs/batch must be positive and val_fraction must be in (0,1)")
    if args.candidate_count < 1 or args.candidate_noise < 0.0 or args.candidate_action_penalty < 0.0:
        ap.error("candidate_count positive; noise and action penalty nonnegative")
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    if args.dataset is None and "buffer" not in ck:
        ap.error("checkpoint must contain serialized replay buffer")
    wm = _model_from_checkpoint(ck, dev)
    if args.dataset is not None:
        dataset = torch.load(args.dataset, map_location=dev, weights_only=False)
        s, s1, a, xi = (dataset[k].to(dev) for k in ("s", "s1", "a", "xi"))
        frames = dataset.get("f", torch.zeros(len(s), dtype=torch.long)).to(dev)
    else:
        buf = ck["buffer"]
        size, ptr = int(buf["size"]), int(buf["ptr"])
        data = buf["data"]
        if size < 4:
            ap.error("replay buffer needs at least four time rows")
        T = data["s"].shape[0]
        ages = torch.arange(size, device=dev)
        rows = (ptr - size + ages) % T
        # Preserve chronological order, then flatten environments. Zero rows
        # can occur only in old checkpoints with incomplete warmup.
        s = data["s"][rows].reshape(-1, data["s"].shape[-1]).to(dev)
        s1 = data["s1"][rows].reshape_as(s).to(dev)
        a = data["a"][rows].reshape(-1, data["a"].shape[-1]).to(dev)
        xi = data["xi"][rows].reshape(-1, data["xi"].shape[-1]).to(dev)
        frames = data["f"][rows].reshape(-1).to(dev)
    valid = (s.square().sum(-1) + s1.square().sum(-1)) > 1e-8
    s, s1, a, xi, frames = s[valid], s1[valid], a[valid], xi[valid], frames[valid]
    if len(s) < 4:
        ap.error("replay buffer has fewer than four nonzero transitions")
    targets = _targets(wm, s, a, xi, s1, args.batch)
    critic = StrategicErrorCritic(hidden=args.hidden).to(dev)
    raw_x = critic.features(s, a, xi)
    perm = torch.randperm(len(s), device=dev)
    n_val = max(1, int(len(s) * args.val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    mean, std = raw_x[train_idx].mean(0), raw_x[train_idx].std(0).clamp_min(1e-3)
    x = (raw_x - mean) / std
    target_mean = targets[train_idx].mean()
    target_std = targets[train_idx].std().clamp_min(1e-3)
    target_min, target_max = targets[train_idx].min(), targets[train_idx].max()
    target_norm = (targets - target_mean) / target_std
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.epochs):
        order = train_idx[torch.randperm(len(train_idx), device=dev)]
        for start in range(0, len(order), args.batch):
            idx = order[start:start + args.batch]
            pred = critic(x[idx])
            loss = F.smooth_l1_loss(pred, target_norm[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        train_pred = critic(x[train_idx]) * target_std + target_mean
        val_pred = critic(x[val_idx]) * target_std + target_mean
        train_mse = float((train_pred - targets[train_idx]).square().mean().sqrt())
        val_mse = float((val_pred - targets[val_idx]).square().mean().sqrt())
        # Query bounded action perturbations around factual actions. This is a
        # ranking diagnostic for the future simulator collection phase.
        n = min(args.candidate_count, len(val_idx))
        base_idx = val_idx[:n]
        base_a = a[base_idx]
        noise = torch.randn(n, 8, base_a.shape[-1], device=dev) * args.candidate_noise * 20.0
        candidates = (base_a[:, None] + noise).clamp(-20.0, 20.0)
        cs = s[base_idx, None].expand(-1, candidates.shape[1], -1).reshape(-1, s.shape[-1])
        cx = xi[base_idx, None].expand(-1, candidates.shape[1], -1).reshape(-1, xi.shape[-1])
        cf = critic((critic.features(cs, candidates.reshape(-1, candidates.shape[-1]), cx) - mean) / std)
        cf = (cf * target_std + target_mean).clamp(target_min, target_max)
        cf = cf.reshape(n, -1)
        if args.candidate_action_penalty > 0.0:
            cf = cf - args.candidate_action_penalty * (
                (candidates - base_a[:, None]) / 20.0).square().mean(-1)
        best = cf.argmax(-1)
        chosen = candidates[torch.arange(n, device=dev), best]
    payload = {
        "checkpoint": str(args.checkpoint), "samples": len(s),
        "train_rmse_log1p": train_mse, "val_rmse_log1p": val_mse,
        "candidate_count": n, "candidate_width": int(candidates.shape[1]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": critic.state_dict(), "mean": mean.cpu(), "std": std.cpu(),
                "input_dim": critic.input_dim, "hidden": args.hidden,
                "metadata": payload, "candidate_states": s[base_idx].cpu(),
                "candidate_actions": chosen.cpu(), "candidate_base_actions": base_a.cpu(),
                "candidate_frames": frames[base_idx].cpu(),
                "candidate_scores": cf.max(-1).values.cpu(),
                "target_mean": target_mean.cpu(), "target_std": target_std.cpu(),
                "target_min": target_min.cpu(), "target_max": target_max.cpu()}, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
