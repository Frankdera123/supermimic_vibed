"""Evaluate critic-selected first actions with short real-simulator rollouts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import add_launcher_args, launch_simulation
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", type=Path)
ap.add_argument("candidates", type=Path)
ap.add_argument("--horizon", type=int, default=4)
ap.add_argument("--eval_motion_dir", default=None)
ap.add_argument("--output", type=Path, default=None)
add_launcher_args(ap)
args = ap.parse_args(); dev = args.device
if args.horizon < 1: ap.error("--horizon must be positive")
ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
cand = torch.load(args.candidates, map_location=dev, weights_only=False)
frames = cand["candidate_frames"].to(dev).long()
a0 = cand["candidate_actions"].to(dev)
b0 = cand.get("candidate_base_actions", a0).to(dev)
n = int(frames.numel())
base = ck.get("args", {}); motion_dir = args.eval_motion_dir or base.get("motion_dir")
kin = G1Kinematics(dev); motion = MotionLib.from_dir(motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
goal_frames, hist = int(base.get("goal_frames", 10)), int(base.get("hist", 10))
policy = Policy(OBS_DIM, 73 * goal_frames, hist=hist).to(dev); policy.load_state_dict(ck["pi"]); policy.eval()
sim_cfg = make_sim(dev, SimpleNamespace(num_envs=2*n))
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, 2*n, motion, mdp, gamma=float(base.get("gamma", .99)), eval_mode=True, reset_noise=False)
    env.reset(torch.arange(2*n, device=dev), frames.repeat(2)); env.refresh()
    obs_hist = torch.zeros(2*n, hist, OBS_DIM, device=dev)
    alive = torch.ones(2*n, dtype=torch.bool, device=dev)
    ret = torch.zeros(2*n, device=dev)
    first_fail = torch.full((2*n,), args.horizon, dtype=torch.long, device=dev)
    with torch.no_grad():
        for t in range(args.horizon):
            S = env.state()
            o = mdp.obs(S, env.a_prev, env.xi, env.f)
            fresh = env.ep_step == 0
            obs_hist = torch.where(fresh[:, None, None], o[:, None].expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o[:, None]), 1))
            if t == 0:
                act = torch.cat((a0, b0), 0)
            else:
                act = policy(obs_hist, motion.goal(env.f, goal_frames)).clamp(-20, 20)
            act = torch.where(alive[:, None], act, torch.zeros_like(act))
            out = env.step(act)
            ret += out["r"] * alive.float()
            fail = out["term"] & alive
            first_fail[fail & (first_fail == args.horizon)] = t + 1
            alive &= ~out["done"]
    c, b = ret[:n], ret[n:]
    result = {"horizon": args.horizon, "samples": n,
              "candidate_return_mean": float((c / (args.horizon*.1)).mean()),
              "base_return_mean": float((b / (args.horizon*.1)).mean()),
              "candidate_minus_base": float(((c-b)/(args.horizon*.1)).mean()),
              "candidate_survival": float(alive[:n].float().mean()),
              "base_survival": float(alive[n:].float().mean()),
              "candidate_first_failure": float(first_fail[:n].float().mean()),
              "base_first_failure": float(first_fail[n:].float().mean())}
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text+"\n")
