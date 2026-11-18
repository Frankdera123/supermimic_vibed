"""Audit kinematic imbalance across one-second motion bins."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_dir")
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    kin = G1Kinematics(args.device)
    m = MotionLib.from_dir(args.motion_dir, kin, args.device)
    starts = m.bin_start
    # Compact bin-level kinematics; no policy or reward signal is used.
    x = torch.cat((m.root_v_local[starts], m.root_w_local[starts],
                   m.q[starts], 0.1 * m.qd[starts]), -1)
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-4)
    g = torch.Generator(device=args.device).manual_seed(0)
    centers = x[torch.randperm(x.shape[0], generator=g, device=args.device)[:args.clusters]].clone()
    for _ in range(args.steps):
        d = (x[:, None] - centers[None]).square().mean(-1)
        labels = d.argmin(-1)
        for k in range(args.clusters):
            sel = labels == k
            if sel.any(): centers[k] = x[sel].mean(0)
    counts = torch.bincount(labels, minlength=args.clusters)
    probs = counts.float() / counts.sum()
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    out = {"motion_dir": args.motion_dir, "bins": int(len(starts)),
           "clusters": args.clusters, "counts": counts.cpu().tolist(),
           "min_count": int(counts.min()), "max_count": int(counts.max()),
           "entropy": float(entropy), "max_entropy": float(torch.tensor(args.clusters, dtype=torch.float32).log())}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
