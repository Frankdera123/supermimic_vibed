"""Measure WM error on actions selected by a saved policy in the real simulator.

This is a small Error-MDP diagnostic: it evaluates the transition error in the
state/action region the current policy actually chooses, rather than replaying
behavior actions or sampling random action perturbations.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import add_launcher_args, launch_simulation
from supermimic.env import TrackingSim, make_sim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list

ap = argparse.ArgumentParser()
ap.add_argument("checkpoints", nargs="+", type=Path)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--seed", type=int, default=17)
ap.add_argument("--eval_motion_dir", type=str, default="data/lafan1_g1_test_s4")
ap.add_argument("--output", type=Path, default=None,
                help="optional JSON output path")
add_launcher_args(ap)
args = ap.parse_args(); dev = args.device
cks = [torch.load(p, map_location=dev, weights_only=False) for p in args.checkpoints]
base = cks[0].get("args", {})
goal_frames = int(base.get("goal_frames", 10)); hist = int(base.get("hist", 10))
kin = G1Kinematics(dev); motion = MotionLib.from_dir(args.eval_motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
gen = torch.Generator(device=dev).manual_seed(args.seed)
valid = torch.where(motion.clip_len >= 8)[0]
clip = valid[torch.randint(valid.numel(), (args.batch,), generator=gen, device=dev)]
off = torch.randint(1, 4, (args.batch,), generator=gen, device=dev)
frames = motion.clip_start[clip] + off

def load_policy(ck):
    p = Policy(OBS_DIM, 73 * goal_frames, hist=hist).to(dev)
    p.load_state_dict(ck["pi"]); p.eval()
    st = ck["wm"]; rh = int(st["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in st else 0
    wm = WorldModel(residual_hidden=rh).to(dev); wm.load_state_dict(st); wm.eval()
    return p, wm, rh > 0

total = args.batch * len(cks); sim_cfg = make_sim(dev, SimpleNamespace(num_envs=total))
with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, total, motion, mdp,
                      gamma=float(base.get("gamma", .99)), eval_mode=True, reset_noise=False)
    env.reset(torch.arange(total, device=dev), frames.repeat(len(cks)))
    env.refresh(); S0_all = env.state()
    xi_all = env.xi.clone(); f_all = env.f.clone(); a_prev = env.a_prev.clone()
    obs = mdp.obs(S0_all, a_prev, xi_all, f_all)
    histories = torch.zeros(total, hist, OBS_DIM, device=dev); histories[:, -1] = obs
    goals = motion.goal(f_all, goal_frames)
    actions = []
    for j, ck in enumerate(cks):
        sl = slice(j * args.batch, (j + 1) * args.batch)
        p, _, _ = load_policy(ck)
        with torch.no_grad():
            actions.append(p(histories[sl], goals[sl]).clamp(-20, 20))
    act = torch.cat(actions, 0)
    # Physics bindings require a leaf tensor without an autograd graph; the
    # policy action itself is retained above for the strategic-error report.
    env.step(act.detach()); S1_all = env.state()
    out = {"batch": args.batch, "seed": args.seed, "models": []}
    for j, (path, ck) in enumerate(zip(args.checkpoints, cks)):
        sl = slice(j * args.batch, (j + 1) * args.batch)
        p, wm, residual = load_policy(ck)
        S0 = tuple(x[sl] for x in S0_all); S1 = tuple(x[sl] for x in S1_all)
        with torch.no_grad():
            pred = wm.step(S0, act[sl], normalize_xi(xi_all[sl]), use_residual=residual)
            std = torch.maximum(wm.out_norm.std(), torch.full_like(wm.out_norm.std(), 1e-4))
            err = (wm.state_error(pred, S1) / std).square().mean(-1)
        out["models"].append({
            "checkpoint": str(path), "policy_action_rms": float(act[sl].square().mean().sqrt().detach()),
            "strategic_wm_error_mean": float(err.mean().detach()),
            "strategic_wm_error_p90": float(torch.quantile(err, .9).detach()),
            "strategic_wm_error_p99": float(torch.quantile(err, .99).detach()),
            "error_action_corr": float(torch.corrcoef(torch.stack((err, act[sl].square().mean(-1))))[0, 1].detach()),
        })
    payload = json.dumps(out, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
