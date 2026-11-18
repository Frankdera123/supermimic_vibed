"""Execute critic-ranked action candidates in the real simulator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import add_launcher_args, launch_simulation
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import WorldModel
from supermimic.robot import default_joint_pos_list


ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", type=Path)
ap.add_argument("candidates", type=Path)
ap.add_argument("--eval_motion_dir", type=str, default=None)
ap.add_argument("--output", type=Path, default=None)
add_launcher_args(ap)
args = ap.parse_args(); dev = args.device
ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
cand = torch.load(args.candidates, map_location=dev, weights_only=False)
frames = cand.get("candidate_frames")
actions = cand.get("candidate_actions")
base_actions = cand.get("candidate_base_actions")
if frames is None or actions is None:
    ap.error("candidate file must contain candidate_frames and candidate_actions")
if base_actions is None:
    base_actions = actions
base = ck.get("args", {})
motion_dir = args.eval_motion_dir or base.get("motion_dir")
if not motion_dir:
    ap.error("--eval_motion_dir is required when checkpoint args have no motion_dir")
n = int(frames.numel())
if n < 1 or actions.shape[0] != n:
    ap.error("candidate frame/action counts do not match")
kin = G1Kinematics(dev); motion = MotionLib.from_dir(motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
st = ck["wm"]; key = "residual_net.0.weight"
rh = int(st[key].shape[0]) if key in st else 0
wm = WorldModel(residual_hidden=rh).to(dev); wm.load_state_dict(st); wm.eval()
sim_cfg = make_sim(dev, SimpleNamespace(num_envs=n))
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, n, motion, mdp,
                      gamma=float(base.get("gamma", .99)), eval_mode=True,
                      reset_noise=False)
    env.reset(torch.arange(n, device=dev), frames.long())
    env.refresh(); s0 = env.state(); xi = env.xi.clone()
    env.step(actions.detach()); s1 = env.state()
    with torch.no_grad():
        std = torch.maximum(wm.out_norm.std(), torch.full_like(wm.out_norm.std(), 1e-4))
        pred = wm.step(s0, actions, normalize_xi(xi))
        base_pred = wm.step(s0, base_actions, normalize_xi(xi))
        err = (wm.state_error(pred, s1) / std).square().mean(-1)
        base_err = (wm.state_error(base_pred, s1) / std).square().mean(-1)
        score = cand.get("candidate_scores", torch.zeros(n, device=dev)).to(dev).float()
        corr = torch.corrcoef(torch.stack((score, err)))[0, 1] if n > 1 else torch.tensor(float("nan"), device=dev)
        out = {"samples": n, "candidate_error_mean": float(err.mean()),
               "base_error_mean": float(base_err.mean()),
               "candidate_minus_base": float((err - base_err).mean()),
               "score_actual_error_corr": float(corr),
               "candidate_error_p90": float(torch.quantile(err, .9)),
               "base_error_p90": float(torch.quantile(base_err, .9))}
    payload = json.dumps(out, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(payload + "\n")
    print(payload)
