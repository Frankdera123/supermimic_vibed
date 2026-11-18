"""Fit a small offline recovery-head selector from paired policy traces."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("trace", type=Path)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--history", type=int, default=1)
a = ap.parse_args()
torch.manual_seed(a.seed)
dev = torch.device(a.device)
d = torch.load(a.trace, map_location=dev, weights_only=False)
feat = d["features"].float()[:, :, :, :4]  # policy, time, start, state risk features
rew = d["rewards"].float()
T = rew.shape[1]
ret = torch.flip(torch.cumsum(torch.flip(rew, (1,)), 1), (1,))
better_tilt = (ret[1] > ret[0]).float()
h = max(1, a.history)
idx = torch.arange(T, device=dev)
windows = torch.stack([feat[:, (idx - (h - 1 - k)).clamp_min(0), :, :]
                       for k in range(h)], dim=-2)
x = windows.permute(0, 2, 1, 3, 4).reshape(-1, 4 * h)
# Each policy's own state is a training example; the label says whether the
# tilt head won from this start/time. The duplicated state trajectories keep
# the selector causal at deployment.
y = better_tilt.unsqueeze(0).expand(2, -1, -1).reshape(-1)
mean, std = x.mean(0), x.std(0).clamp_min(1e-3)
xn = (x - mean) / std
perm = torch.randperm(len(y), device=dev)
nval = max(1, len(y) // 5)
vi, ti = perm[:nval], perm[nval:]
model = nn.Sequential(nn.Linear(4 * h, 64), nn.Tanh(), nn.Linear(64, 1)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
for _ in range(a.epochs):
    order = ti[torch.randperm(len(ti), device=dev)]
    for j in range(0, len(order), 1024):
        idx = order[j:j+1024]
        loss = F.binary_cross_entropy_with_logits(model(xn[idx]).squeeze(-1), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    prob = model(xn[vi]).squeeze(-1).sigmoid()
    acc = float(((prob >= .5) == (y[vi] >= .5)).float().mean())
    base = float(torch.maximum(y[vi].mean(), 1-y[vi].mean()))
out = {"samples": len(y), "validation_accuracy": acc, "majority_baseline": base,
       "positive_rate": float(y.mean()), "history": h}
print(out)
a.output.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": model.state_dict(), "mean": mean.cpu(), "std": std.cpu(), "history": h, "metadata": out}, a.output)
