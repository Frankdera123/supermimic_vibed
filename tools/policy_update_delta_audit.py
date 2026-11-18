"""Estimate marginal policy-update value on held-out replay batches.

This is an offline diagnostic: the world model is frozen, a cloned policy takes
one virtual Adam step on one replay batch, and its loss change is measured on a
fresh batch. A second virtual step is then measured in the same way.
"""
from __future__ import annotations

import argparse, copy, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_batch_gradient_audit import policy_loss, sample_valid
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list
from supermimic.mdp import unflatten_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--trunc", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    for path in args.checkpoints:
        ck = torch.load(path, map_location=args.device, weights_only=False)
        a = ck["args"]; dev = args.device; state = ck["buffer"]
        T, N = state["data"]["s"].shape[:2]
        buf = ReplayBuffer(T, N, dev); buf.load_state_dict(state)
        kin = G1Kinematics(dev)
        motion = MotionLib.from_dir(a.get("motion_dir", "data/lafan1_g1"), kin, dev)
        mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
        wm_state = ck["wm"]
        hidden = int(wm_state["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in wm_state else 0
        wm = WorldModel(residual_hidden=hidden).to(dev); wm.load_state_dict(wm_state); wm.eval()
        for p in wm.parameters(): p.requires_grad_(False)
        pi0 = Policy(OBS_DIM, 73 * int(a.get("goal_frames", 10)), hist=int(a.get("hist", 10))).to(dev)
        pi0.load_state_dict(ck["pi"]); pi0.eval()
        lr = float(a.get("pi_lr", a.get("lr", 3e-4)))
        deltas1, deltas2 = [], []
        for i in range(args.pairs):
            torch.manual_seed(7000 + i)
            p = copy.deepcopy(pi0); p.train()
            opt = torch.optim.Adam(p.parameters(), lr=lr)
            if ck.get("opt_pi") is not None:
                # Match the checkpoint's optimizer momentum/variance state;
                # fresh Adam would measure a different virtual update.
                opt.load_state_dict(ck["opt_pi"])
            train1 = sample_valid(buf, motion, args.batch, int(a.get("hist", 10)), args.horizon)
            eval1 = sample_valid(buf, motion, args.batch, int(a.get("hist", 10)), args.horizon)
            before = float(policy_loss(p, wm, mdp, motion, eval1, int(a.get("hist", 10)), args.horizon, float(a.get("gamma", .99)), args.trunc, int(a.get("goal_frames", 10))).detach())
            opt.zero_grad(set_to_none=True)
            policy_loss(p, wm, mdp, motion, train1, int(a.get("hist", 10)), args.horizon, float(a.get("gamma", .99)), args.trunc, int(a.get("goal_frames", 10))).backward()
            torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0); opt.step()
            after1 = float(policy_loss(p, wm, mdp, motion, eval1, int(a.get("hist", 10)), args.horizon, float(a.get("gamma", .99)), args.trunc, int(a.get("goal_frames", 10))).detach())
            train2 = sample_valid(buf, motion, args.batch, int(a.get("hist", 10)), args.horizon)
            eval2 = sample_valid(buf, motion, args.batch, int(a.get("hist", 10)), args.horizon)
            before2 = float(policy_loss(p, wm, mdp, motion, eval2, int(a.get("hist", 10)), args.horizon, float(a.get("gamma", .99)), args.trunc, int(a.get("goal_frames", 10))).detach())
            opt.zero_grad(set_to_none=True)
            policy_loss(p, wm, mdp, motion, train2, int(a.get("hist", 10)), args.horizon, float(a.get("gamma", .99)), args.trunc, int(a.get("goal_frames", 10))).backward()
            torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0); opt.step()
            after2 = float(policy_loss(p, wm, mdp, motion, eval2, int(a.get("hist", 10)), args.horizon, float(a.get("gamma", .99)), args.trunc, int(a.get("goal_frames", 10))).detach())
            deltas1.append(before - after1); deltas2.append(before2 - after2)
        out = {"checkpoint": str(path), "pairs": args.pairs, "batch": args.batch,
               "horizon": args.horizon, "delta1_mean": sum(deltas1)/len(deltas1),
               "delta2_mean": sum(deltas2)/len(deltas2),
               "delta1_positive_rate": sum(x > 0 for x in deltas1)/len(deltas1),
               "delta2_positive_rate": sum(x > 0 for x in deltas2)/len(deltas2)}
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
