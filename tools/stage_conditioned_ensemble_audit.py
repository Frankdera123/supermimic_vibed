"""Measure bootstrap disagreement drift across policy-training stages."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import OBS_DIM, MDP, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list


def model_from_state(state, device):
    key = "residual_net.0.weight"
    hidden = int(state[key].shape[0]) if key in state else 0
    return WorldModel(residual_hidden=hidden).to(device)


def tail_from_log(path: Path):
    log = path.parent / "train_log.csv"
    if not log.exists():
        return None
    with log.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    if "ep_ret_norm" in rows[-1]:
        return float(rows[-1]["ep_ret_norm"])
    # A legacy resumed run has a truncated header but still keeps the standard
    # column order; ep_ret_norm is column 13 (zero-based).
    values = list(rows[-1].values())
    return float(values[13]) if len(values) > 13 else None


def audit(policy_path, models, args, mdp, motion):
    ck = torch.load(policy_path, map_location=args.device, weights_only=False)
    state = ck["buffer"]
    T, N = state["data"]["s"].shape[:2]
    buf = ReplayBuffer(T, N, args.device); buf.load_state_dict(state)
    a = ck.get("args", {})
    goal_frames, hist = int(a.get("goal_frames", 4)), int(a.get("hist", 4))
    pi = Policy(OBS_DIM, 73 * goal_frames, hist=hist).to(args.device)
    pi.load_state_dict(ck["pi"]); pi.eval()
    all_d, all_r = [], [[] for _ in models]
    with torch.no_grad():
        for _ in range(args.batches):
            s, h = buf.sample_starts(args.batch, hist)
            states = [unflatten_state(s["s"]) for _ in models]
            xi, xi_n, f = s["xi"], normalize_xi(s["xi"]), s["f"]
            obs_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
            goal, a_prev = motion.goal(f, goal_frames), s["a_prev"]
            total = [torch.zeros(args.batch, device=args.device) for _ in models]
            for _ in range(args.horizon):
                act = pi(obs_hist, goal)
                nxt = []
                for j, wm in enumerate(models):
                    ns = wm.step(states[j], act, xi_n, use_residual=(j > 0))
                    total[j] += mdp.reward(ns, f, act, a_prev)
                    nxt.append(ns)
                std = torch.sqrt(models[0].out_norm.var + models[0].out_norm.eps ** 2).clamp_min(1e-4)
                d = (models[0].state_error(nxt[0], nxt[1]) / std).square().mean(-1)
                if len(nxt) > 2:
                    d = 0.5 * (d + (models[0].state_error(nxt[0], nxt[2]) / std).square().mean(-1))
                all_d.append(d)
                obs = mdp.obs(nxt[0], act, xi, f + 1)
                obs_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
                goal, a_prev, f, states = motion.goal(f + 1, goal_frames), act, f + 1, nxt
            for j in range(len(models)):
                all_r[j].append(total[j] / args.horizon)
    d = torch.cat(all_d)
    returns = [torch.cat(x) for x in all_r]
    qs = {f"p{q}": float(torch.quantile(d, q / 100)) for q in (10, 25, 50, 75, 90)}
    return {
        "policy_checkpoint": str(policy_path),
        "train_episode_tail": tail_from_log(policy_path),
        "samples": int(d.numel()),
        "disagreement": {"mean": float(d.mean()), **qs},
        "imagined_return_mean": [float(x.mean()) for x in returns],
        "imagined_return_p10": [float(torch.quantile(x, .1)) for x in returns],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy_checkpoints", nargs="+")
    ap.add_argument("--model_checkpoints", nargs=3, required=True)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    cks = [torch.load(p, map_location=args.device, weights_only=False) for p in args.model_checkpoints]
    models = []
    for ck in cks:
        wm = model_from_state(ck["wm"], args.device); wm.load_state_dict(ck["wm"]); wm.eval(); models.append(wm)
    ref = cks[0]["args"]
    dev = args.device
    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(ref.get("motion_dir", "data/lafan1_g1"), kin, dev)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    out = [audit(Path(p), models, args, mdp, motion) for p in args.policy_checkpoints]
    print(json.dumps({"models": args.model_checkpoints, "horizon": args.horizon, "stages": out}, indent=2))


if __name__ == "__main__":
    main()
