"""Audit policy/replay action support and its relation to WM error."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list


def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return (x * y).mean() / (x.square().mean().sqrt() * y.square().mean().sqrt()).clamp_min(1e-8)


def model_from_state(state, device):
    key = "residual_net.0.weight"
    hidden = int(state[key].shape[0]) if key in state else 0
    return WorldModel(residual_hidden=hidden).to(device)


def collect(ck, buf, mdp, motion, device, batches, batch, goal_frames, hist):
    wm = model_from_state(ck["wm"], device); wm.load_state_dict(ck["wm"]); wm.eval()
    pi = Policy(OBS_DIM, 73 * goal_frames, hist=hist).to(device)
    pi.load_state_dict(ck["pi"]); pi.eval()
    ds, es = [], []
    with torch.no_grad():
        std = torch.maximum(wm.out_norm.std(), torch.full_like(wm.out_norm.std(), 1e-4))
        for _ in range(batches):
            s, h = buf.sample_starts(batch, hist)
            state = unflatten_state(s["s"])
            obs = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
            goal = motion.goal(s["f"], goal_frames)
            policy_action = pi(obs, goal)
            # Action distance is normalized by the policy's physical action scale.
            dist = ((policy_action - s["a"]) / 20.0).square().mean(-1).sqrt()
            pred = wm.step(state, s["a"], normalize_xi(s["xi"]))
            target = unflatten_state(s["s1"])
            err = (wm.state_error(pred, target) / std).square().mean(-1)
            ds.append(dist); es.append(err)
    return torch.cat(ds), torch.cat(es)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    for path in args.checkpoints:
        ck = torch.load(path, map_location=args.device)
        if "buffer" not in ck:
            ap.error(f"{path} must contain replay buffer state")
        a = ck.get("args", {})
        goal_frames, hist = int(a.get("goal_frames", 4)), int(a.get("hist", 4))
        state = ck["buffer"]; T, N = state["data"]["s"].shape[:2]
        buf = ReplayBuffer(T, N, args.device); buf.load_state_dict(state)
        kin = G1Kinematics(args.device)
        motion = MotionLib.from_dir(a.get("motion_dir", "data/lafan1_g1"), kin, args.device)
        mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device))
        d, e = collect(ck, buf, mdp, motion, args.device, args.batches, args.batch,
                       goal_frames, hist)
        out = {"checkpoint": str(path), "samples": int(d.numel()),
               "action_distance_mean": float(d.mean()),
               "action_distance_p90": float(torch.quantile(d, .9)),
               "wm_error_mean": float(e.mean()), "wm_error_p90": float(torch.quantile(e, .9)),
               "action_distance_error_corr": float(corr(d, e))}
        for q in (.5, .75, .9):
            t = torch.quantile(d, q); sel = d <= t
            out[f"support_q{q}"] = {"threshold": float(t), "selected_rate": float(sel.float().mean()),
                                     "selected_wm_error": float(e[sel].mean())}
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
