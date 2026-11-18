"""Train/evaluate a simulator-supervised local WM intervention adapter."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.mdp import flatten_state, normalize_xi, unflatten_state
from supermimic.networks import WorldModel

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", type=Path)
ap.add_argument("dataset", type=Path)
ap.add_argument("--eval_dataset", type=Path, default=None)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--iterations", type=int, default=2000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--effect_weight", type=float, default=0.0,
                help="weight paired (a+ - a-) action-effect loss")
ap.add_argument("--gram_weight", type=float, default=0.0,
                help="weight multi-direction response Gram loss")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
dev = args.device
ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
st = ck["wm"]; hidden = int(st["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in st else 0
wm = WorldModel(residual_hidden=hidden).to(dev); wm.load_state_dict(st); wm.eval()
for p in wm.parameters(): p.requires_grad_(False)

def make_data(path):
    d = torch.load(path, map_location=dev, weights_only=False)
    s0f = d["pairs"][0]["s0"].to(dev)
    s0 = unflatten_state(s0f)
    xi = normalize_xi(d["xi"].to(dev))
    action = d["action"].to(dev); dirs = d["directions"].to(dev)
    horizon = int(d.get("counterfactual_horizon", 1))
    xs, ys, bs = [], [], []
    for pair in d["pairs"]:
        delta = float(pair["delta"])
        plus = pair["plus"].to(dev); minus = pair["minus"].to(dev)
        x_pair, y_pair, b_pair = [], [], []
        for k in range(dirs.shape[1]):
            ap = action + delta * dirs[:, k]; am = action - delta * dirs[:, k]
            with torch.no_grad():
                pp = wm.step(s0, ap, xi, use_residual=hidden > 0)
                pm = wm.step(s0, am, xi, use_residual=hidden > 0)
                pd = torch.cat((wm.state_error(pp, s0), wm.state_error(pm, s0)), 0)
                bp, bm = pp, pm
                for _ in range(horizon - 1):
                    bp = wm.step(bp, ap, xi, use_residual=hidden > 0)
                    bm = wm.step(bm, am, xi, use_residual=hidden > 0)
            target = torch.cat((
                wm.state_error(unflatten_state(plus[:, k]), s0),
                wm.state_error(unflatten_state(minus[:, k]), s0)), 0)
            baseline = torch.cat((wm.state_error(bp, s0), wm.state_error(bm, s0)), 0)
            sf = s0f.repeat(2, 1); af = torch.cat((ap, am), 0)
            x_pair.append(torch.cat((sf, af, pd), -1).view(2, s0f.shape[0], -1).transpose(0, 1))
            y_pair.append(target.view(2, s0f.shape[0], -1).transpose(0, 1))
            b_pair.append(baseline.view(2, s0f.shape[0], -1).transpose(0, 1))
        # [batch, direction, +/- , feature], preserving groups for paired losses.
        xs.append(torch.stack(x_pair, 1)); ys.append(torch.stack(y_pair, 1)); bs.append(torch.stack(b_pair, 1))
    xg, yg, bg = torch.cat(xs, 0), torch.cat(ys, 0), torch.cat(bs, 0)
    return (xg.reshape(-1, xg.shape[-1]), yg.reshape(-1, yg.shape[-1]),
            bg.reshape(-1, bg.shape[-1]), xg.shape[0], dirs.shape[1])

X, Y, B, n_pairs, n_dirs = make_data(args.dataset)
Xe, Ye, Be = (make_data(args.eval_dataset)[0:3]
              if args.eval_dataset is not None else (None, None, None))
mean, std = X.mean(0), X.std(0).clamp_min(1e-4)
Xn = (X - mean) / std
model = nn.Sequential(nn.Linear(X.shape[-1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 70)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
gen = torch.Generator(device=dev).manual_seed(0)
perm = torch.randperm(X.shape[0], generator=gen, device=dev)
n_train_pairs = max(1, int(.8 * n_pairs))
rows_per_pair = 2 * n_dirs
n_train = n_train_pairs * rows_per_pair
# Keep all +/- directions from a simulator state together. This prevents the
# Gram/effect objective from leaking a direction into the held-out split.
train_idx = torch.arange(n_train, device=dev)
hold_idx = torch.arange(n_train, X.shape[0], device=dev)
for _ in range(args.iterations):
    idx = train_idx[torch.randint(train_idx.numel(), (min(args.batch, train_idx.numel()),), generator=gen, device=dev)]
    pred = model(Xn[idx])
    loss = (pred - Y[idx]).square().mean()
    if args.effect_weight > 0.0 or args.gram_weight > 0.0:
        # Full grouped batch keeps the +/- and multi-direction pairing intact.
        pred_all = model(Xn[train_idx]).view(n_train_pairs, n_dirs, 2, 70)
        true_all = Y[train_idx].view(n_train_pairs, n_dirs, 2, 70)
        pred_eff = pred_all[:, :, 0] - pred_all[:, :, 1]
        true_eff = true_all[:, :, 0] - true_all[:, :, 1]
        if args.effect_weight > 0.0:
            loss = loss + args.effect_weight * (pred_eff - true_eff).square().mean()
        if args.gram_weight > 0.0:
            pred_n = pred_eff / pred_eff.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            true_n = true_eff / true_eff.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            pred_gram = pred_n @ pred_n.transpose(-1, -2)
            true_gram = true_n @ true_n.transpose(-1, -2)
            loss = loss + args.gram_weight * (pred_gram - true_gram).square().mean()
    opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
with torch.no_grad():
    base = ((B[train_idx] - Y[train_idx]) ** 2).mean().sqrt()
    adapted = (model(Xn[train_idx]) - Y[train_idx]).square().mean().sqrt()
    hold_base = ((B[hold_idx] - Y[hold_idx]) ** 2).mean().sqrt()
    hold_adapt = (model(Xn[hold_idx]) - Y[hold_idx]).square().mean().sqrt()
    out = {"train_samples": int(train_idx.numel()), "holdout_samples": int(hold_idx.numel()),
           "baseline_rmse": float(base), "adapter_rmse": float(adapted),
           "holdout_baseline_rmse": float(hold_base), "holdout_adapter_rmse": float(hold_adapt)}
    if Xe is not None:
        pred_base = ((Be - Ye) ** 2).mean().sqrt()
        pred_adapt = (model((Xe - mean) / std) - Ye).square().mean().sqrt()
        out.update({"eval_samples": int(Xe.shape[0]), "eval_baseline_rmse": float(pred_base), "eval_adapter_rmse": float(pred_adapt)})
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": model.state_dict(), "input_mean": mean.cpu(), "input_std": std.cpu(), "input_dim": X.shape[-1], "output_dim": 70, "source_checkpoint": str(args.checkpoint), "metrics": out}, args.output)
print(json.dumps(out, indent=2))
