"""Derive a physically grounded WM component-weight vector from G1 FK sensitivity.

The generated weights are a kinematic proxy: each joint's q weight is based on
the Jacobian of tracked-body positions, and each qd weight on tracked-body
angular velocities. They are normalized within q and qd groups and are meant
for a controlled WM-loss ablation, not as a claimed optimal weighting.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.kinematics import G1Kinematics
from supermimic.robot import default_joint_pos_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-weight", type=float, default=0.25)
    args = ap.parse_args()
    kin = G1Kinematics(args.device)
    q0 = torch.tensor(default_joint_pos_list(), dtype=torch.float32, device=args.device)
    qd0 = torch.zeros_like(q0)
    p = torch.zeros(1, 3, device=args.device)
    R = torch.eye(3, device=args.device).unsqueeze(0)
    v = torch.zeros_like(p); w = torch.zeros_like(p)

    def pos_fn(q):
        return kin.forward(p, R, v, w, q.reshape(1, -1), qd0.reshape(1, -1))[0].reshape(-1)

    def angvel_fn(qd):
        return kin.forward(p, R, v, w, q0.reshape(1, -1), qd.reshape(1, -1))[3].reshape(-1)

    jq = torch.autograd.functional.jacobian(pos_fn, q0).detach()
    jqd = torch.autograd.functional.jacobian(angvel_fn, qd0).detach()
    qw = jq.square().sum(0).sqrt().clamp_min(1e-4)
    qdw = jqd.square().sum(0).sqrt().clamp_min(1e-4)
    qw = (qw / qw.mean()).clamp_min(args.min_weight); qw = qw / qw.mean()
    qdw = (qdw / qdw.mean()).clamp_min(args.min_weight); qdw = qdw / qdw.mean()
    weights = torch.cat((torch.ones(12, device=qw.device), qw, qdw)).cpu().tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(",".join(f"{x:.8g}" for x in weights) + "\n")
    meta = args.out.with_suffix(".json")
    meta.write_text(json.dumps({"position_rotation_velocity_weights": 12,
                                "q_weights": qw.cpu().tolist(),
                                "qd_weights": qdw.cpu().tolist(),
                                "definition": "q: tracked-position Jacobian norm; qd: tracked-angular-velocity Jacobian norm",
                                "normalization": "mean one within q and qd groups after minimum-weight floor",
                                "min_weight": args.min_weight}, indent=2) + "\n")
    print(json.dumps({"weights": str(args.out), "metadata": str(meta),
                      "q_min": float(qw.min()), "q_max": float(qw.max()),
                      "qd_min": float(qdw.min()), "qd_max": float(qdw.max())}, indent=2))


if __name__ == "__main__":
    main()
