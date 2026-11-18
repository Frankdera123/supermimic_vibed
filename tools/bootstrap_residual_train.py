"""Train an audit-only residual member from a frozen nominal WM and replay bootstrap mask."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.networks import WorldModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nominal", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bootstrap_prob", type=float, default=.632)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    ck = torch.load(args.nominal, map_location=args.device)
    bs = ck.get("buffer")
    if bs is None:
        ap.error("nominal checkpoint must contain replay buffer")
    data = bs["data"]; T, N = data["s"].shape[:2]
    buf = ReplayBuffer(T, N, args.device); buf.load_state_dict(bs)
    base = ck["wm"]
    wm = WorldModel(residual_hidden=args.hidden).to(args.device)
    # Load nominal trunk and normalization; the residual head stays zero-init.
    missing, unexpected = wm.load_state_dict(base, strict=False)
    if any(not k.startswith("residual_net") for k in missing):
        raise RuntimeError(f"unexpected missing nominal keys: {missing}")
    for p in wm.net.parameters(): p.requires_grad_(False)
    for p in wm.in_norm.parameters(): p.requires_grad_(False)
    for p in wm.out_norm.parameters(): p.requires_grad_(False)
    wm.train()
    opt = torch.optim.Adam(wm.residual_net.parameters(), lr=args.lr)
    for it in range(args.steps):
        w = buf.sample_windows(args.batch, args.window)
        S = unflatten_state(w["s"][:, 0])
        pred = wm.step(S, w["a"][:, 0], normalize_xi(w["xi"][:, 0]), use_residual=True)
        target = unflatten_state(w["s1"][:, 0])
        std = torch.maximum(wm.out_norm.std(), torch.full_like(wm.out_norm.std(), 1e-4))
        err = (wm.state_error(pred, target) / std).square().mean(-1)
        boot = (torch.rand_like(err) < args.bootstrap_prob).float()
        loss = (err * boot).sum() / boot.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.residual_net.parameters(), 1.0); opt.step()
    out = dict(ck)
    out["wm"] = wm.state_dict()
    out["args"] = {"source": str(args.nominal), **vars(args)}
    out["bootstrap_member"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(json.dumps({"output": str(args.output), "steps": args.steps,
                      "hidden": args.hidden, "bootstrap_prob": args.bootstrap_prob}, indent=2))

if __name__ == "__main__":
    main()
