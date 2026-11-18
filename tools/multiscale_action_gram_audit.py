"""Multi-scale simulator-vs-WM action-response Gram audit."""
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
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--directions", type=int, default=4)
ap.add_argument("--seed", type=int, default=23)
ap.add_argument("--action_scale", type=float, default=2.0)
ap.add_argument("--delta", type=float, nargs="+", default=[.1, .5, 1.0])
ap.add_argument("--eval_motion_dir", type=str, default="data/lafan1_g1_test_s4")
ap.add_argument("--gate_reference", type=Path, default=None,
                help="reference checkpoint for held-out geometry acceptance")
ap.add_argument("--gate_max_gram_increase", type=float, default=0.02,
                help="allowed absolute Gram discrepancy increase per delta")
ap.add_argument("--gate_max_cosine_drop", type=float, default=0.01,
                help="allowed response-vector cosine drop per delta")
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
off = torch.randint(1, 4, (args.batch,), generator=gen, device=dev)
frames = motion.clip_start[clip] + off
K, B = args.directions, args.batch
total = (1 + 2 * K) * B
sim_cfg = make_sim(dev, SimpleNamespace(num_envs=total))

def model_from_ck(ck):
    st = ck["wm"]
    hidden = int(st["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in st else 0
    m = WorldModel(residual_hidden=hidden).to(dev); m.load_state_dict(st); m.eval()
    return m, hidden > 0

def stack_group(S, start, n):
    return tuple(x[start:start+n] for x in S)

def gram(V):
    G = torch.einsum("bkd,bld->bkl", V, V)
    return G / G.square().sum((-2, -1), keepdim=True).sqrt().clamp_min(1e-8)

with launch_simulation(sim_cfg, args):
    env = TrackingSim(sim_cfg, total, motion, mdp,
                      gamma=float(base.get("gamma", .99)), eval_mode=True, reset_noise=False)
    xi = torch.zeros(B, 33, device=dev); xi[:, 0] = 1.0
    directions = torch.randn(B, K, 29, generator=gen, device=dev)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    action = torch.randn(B, 29, generator=gen, device=dev) * args.action_scale
    out = {"batch": B, "directions": K, "deltas": args.delta, "models": {}}
    for delta in args.delta:
        env.reset(torch.arange(total, device=dev), frames.repeat(1 + 2*K))
        env.xi.copy_(xi.repeat(1 + 2*K, 1)); env.refresh()
        acts = [action]
        for k in range(K):
            acts.extend((action + delta * directions[:, k], action - delta * directions[:, k]))
        env.step(torch.cat(acts, 0))
        S0 = stack_group(env.state(), 0, B)
        true_resp = []
        for k in range(K):
            sp = stack_group(env.state(), (1 + 2*k)*B, B)
            sm = stack_group(env.state(), (2 + 2*k)*B, B)
            true_resp.append(WorldModel.state_error(sp, sm) / (2.0 * delta))
        true_v = torch.stack(true_resp, 1)
        true_g = gram(true_v)
        for path, ck in zip(args.checkpoints, cks):
            wm, residual = model_from_ck(ck)
            xi_n = normalize_xi(xi)
            pred_resp = []
            for k in range(K):
                pp = wm.step(S0, action + delta * directions[:, k], xi_n, use_residual=residual)
                pm = wm.step(S0, action - delta * directions[:, k], xi_n, use_residual=residual)
                pred_resp.append(wm.state_error(pp, pm) / (2.0 * delta))
            pred_v = torch.stack(pred_resp, 1)
            pred_g = gram(pred_v)
            gd = (pred_g - true_g).square().sum((-2, -1)).sqrt()
            vc = torch.nn.functional.cosine_similarity(pred_v, true_v, dim=-1)
            ratio = pred_v.square().sum(-1).sqrt() / true_v.square().sum(-1).sqrt().clamp_min(1e-8)
            key = str(path)
            out["models"].setdefault(key, []).append({
                "delta": float(delta), "gram_frobenius_discrepancy": float(gd.mean().detach()),
                "gram_flat_cosine": float(torch.nn.functional.cosine_similarity(
                    pred_g.flatten(1), true_g.flatten(1), dim=-1).mean().detach()),
                "response_vector_cosine": float(vc.mean().detach()),
                "sensitivity_ratio": float(ratio.mean().detach()),
            })
    if args.gate_reference is not None:
        ref_key = str(args.gate_reference)
        if ref_key not in out["models"]:
            raise ValueError("--gate_reference must be one of the positional checkpoints")
        ref_rows = out["models"][ref_key]
        gate = {}
        for key, rows in out["models"].items():
            if key == ref_key:
                continue
            checks = []
            for ref_row, row in zip(ref_rows, rows):
                gram_ok = row["gram_frobenius_discrepancy"] <= ref_row["gram_frobenius_discrepancy"] + args.gate_max_gram_increase
                cosine_ok = row["response_vector_cosine"] >= ref_row["response_vector_cosine"] - args.gate_max_cosine_drop
                checks.append({"delta": row["delta"], "gram_ok": gram_ok,
                               "cosine_ok": cosine_ok, "pass": gram_ok and cosine_ok,
                               "gram_delta": row["gram_frobenius_discrepancy"] - ref_row["gram_frobenius_discrepancy"],
                               "cosine_delta": row["response_vector_cosine"] - ref_row["response_vector_cosine"]})
            gate[key] = {"pass": all(x["pass"] for x in checks), "checks": checks,
                         "max_gram_increase": args.gate_max_gram_increase,
                         "max_cosine_drop": args.gate_max_cosine_drop}
        out["gate"] = {"reference": ref_key, "candidates": gate}
    print(json.dumps(out, indent=2))
