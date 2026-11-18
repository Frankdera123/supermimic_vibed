"""Compare WM local action effects with one-step Isaac simulation finite differences."""
from __future__ import annotations
import argparse, json, sys
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
ap.add_argument("checkpoints", nargs="+", type=Path)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--seed", type=int, default=17)
ap.add_argument("--delta", type=float, default=.5)
ap.add_argument("--action_scale", type=float, default=2.0,
                help="std of the random base action used for the local finite difference")
ap.add_argument("--eval_motion_dir", type=str, default="data/lafan1_g1_test_s4")
add_launcher_args(ap)
args = ap.parse_args()
dev = args.device
cks = [torch.load(p, map_location=dev, weights_only=False) for p in args.checkpoints]
base = cks[0].get("args", {})
kin = G1Kinematics(dev)
motion = MotionLib.from_dir(args.eval_motion_dir, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
gen = torch.Generator(device=dev).manual_seed(args.seed)
valid = torch.where(motion.clip_len >= 8)[0]
clip = valid[torch.randint(valid.numel(), (args.batch,), generator=gen, device=dev)]
max_off = (motion.clip_len[clip] - 2).clamp_min(1)
off = (torch.rand(args.batch, generator=gen, device=dev) * max_off.float()).long() + 1
frames = motion.clip_start[clip] + off
total = 3 * args.batch
sim_args = SimpleNamespace(num_envs=total)
sim_cfg = make_sim(dev, sim_args)

def model_from_ck(ck):
    st = ck["wm"]
    hidden = int(st["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in st else 0
    m = WorldModel(residual_hidden=hidden).to(dev)
    m.load_state_dict(st); m.eval()
    return m, hidden > 0

with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, total, motion, mdp,
                      gamma=float(base.get("gamma", .99)), eval_mode=True, reset_noise=False)
    env.reset(torch.arange(total, device=dev), frames.repeat(3))
    xi = torch.zeros(args.batch, 33, device=dev)
    xi[:, 0] = 1.0
    env.xi.copy_(xi.repeat(3, 1)); env.refresh()
    direction = torch.randn(args.batch, 29, generator=gen, device=dev)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    action = torch.randn(args.batch, 29, generator=gen, device=dev) * args.action_scale
    acts = torch.cat((action, action + args.delta * direction, action - args.delta * direction), 0)
    S0_all = env.state()
    env.step(acts)
    S1_all = env.state()
    S0 = tuple(x[:args.batch] for x in S0_all)
    Splus = tuple(x[args.batch:2*args.batch] for x in S1_all)
    Sminus = tuple(x[2*args.batch:] for x in S1_all)
    true_vec = WorldModel.state_error(Splus, Sminus) / (2.0 * args.delta)
    true_norm = true_vec.square().mean(-1).sqrt()
    out = {"batch": args.batch, "delta": args.delta, "seed": args.seed,
           "true_sensitivity_mean": float(true_norm.mean()), "models": []}
    for path, ck in zip(args.checkpoints, cks):
        wm, residual = model_from_ck(ck)
        a = action; xi_n = normalize_xi(xi)
        pred_plus = wm.step(S0, a + args.delta * direction, xi_n, use_residual=residual)
        pred_minus = wm.step(S0, a - args.delta * direction, xi_n, use_residual=residual)
        pred_vec = wm.state_error(pred_plus, pred_minus) / (2.0 * args.delta)
        pred_norm = pred_vec.square().mean(-1).sqrt()
        c = (pred_vec * true_vec).sum(-1) / (pred_vec.square().sum(-1).sqrt() * true_vec.square().sum(-1).sqrt()).clamp_min(1e-8)
        gain = (pred_vec * true_vec).sum() / pred_vec.square().sum().clamp_min(1e-8)
        component_gain = (pred_vec * true_vec).sum(0) / pred_vec.square().sum(0).clamp_min(1e-8)
        sample_gain = (pred_vec * true_vec).sum(-1) / pred_vec.square().sum(-1).clamp_min(1e-8)
        action_norm = action.square().mean(-1).sqrt()
        qs = torch.quantile(action_norm, torch.tensor([0.25, 0.5, 0.75], device=dev))
        bins = [action_norm <= qs[0], (action_norm > qs[0]) & (action_norm <= qs[1]),
                (action_norm > qs[1]) & (action_norm <= qs[2]), action_norm > qs[2]]
        bin_stats = []
        for mask in bins:
            bin_stats.append({"n": int(mask.sum()),
                              "action_norm_mean": float(action_norm[mask].mean().detach()),
                              "gain": float(sample_gain[mask].mean().detach()),
                              "sensitivity_ratio": float((pred_norm[mask] / true_norm[mask].clamp_min(1e-6)).mean().detach()),
                              "vector_cosine": float(c[mask].mean().detach())})
        out["models"].append({"checkpoint": str(path), "pred_sensitivity_mean": float(pred_norm.mean().detach()),
                              "sensitivity_ratio": float((pred_norm / true_norm.clamp_min(1e-6)).mean().detach()),
                              "least_squares_gain": float(gain.detach()),
                              "component_gain": component_gain.detach().cpu().tolist(),
                              "vector_cosine_mean": float(c.mean().detach()),
                              "vector_cosine_p10": float(torch.quantile(c, .1).detach()),
                              "action_norm_quartile_bins": bin_stats})
    print(json.dumps(out, indent=2))
