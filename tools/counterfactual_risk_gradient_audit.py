"""Compare simulator and WM action gradients for h4 height/tilt risk.

The audit uses paired +/- action directions from a simulator counterfactual
dataset.  It measures whether the nominal WM predicts the *direction* in which
the endpoint root height/tilt tracking risk changes, rather than only matching
the endpoint state.  This is the observable target suggested by model-gradient
joint learning.
"""
from __future__ import annotations

import argparse, json, sys
from types import SimpleNamespace
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import add_launcher_args, launch_simulation
from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.networks import WorldModel
from supermimic.mdp import unflatten_state, normalize_xi
from supermimic.env import make_sim

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint", type=Path)
ap.add_argument("datasets", nargs="+", type=Path)
add_launcher_args(ap)
args = ap.parse_args()
dev = args.device

def risk(S, target_p, target_R):
    p, R = S[0], S[1]
    z = ((p[:, 2] - target_p[:, 2]) / 0.10).square()
    # R[2,2] is the same upright/root-tilt observable used by the evaluator.
    tilt = ((R[:, 2, 2] - target_R[:, 2, 2]) / 0.25).square()
    return z + tilt

with launch_simulation(make_sim(dev, SimpleNamespace(num_envs=1)), args):
    ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    st = ck["wm"]
    hidden = int(st["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in st else 0
    wm = WorldModel(residual_hidden=hidden).to(dev)
    wm.load_state_dict(st); wm.eval()
    out = {"checkpoint": str(args.checkpoint), "datasets": {}}
    for path in args.datasets:
        d = torch.load(path, map_location=dev, weights_only=False)
        motion = MotionLib.from_dir(d["eval_motion_dir"], G1Kinematics(dev), dev)
        horizon = int(d.get("counterfactual_horizon", 1))
        frames = d["frames"].to(dev).long()
        xi = normalize_xi(d["xi"].to(dev))
        true_proj, pred_proj = [], []
        true_delta, pred_delta = [], []
        act_all = d["action"].to(dev)
        dirs_all = d["directions"].to(dev)
        for pair in d["pairs"]:
            s0 = unflatten_state(pair["s0"].to(dev))
            frame = (frames + horizon).clamp(max=motion.root_p.shape[0] - 1)
            tp = motion.root_p[frame]
            tR = motion.root_R[frame]
            act = act_all
            dirs = dirs_all
            delta = float(pair["delta"])
            for k in range(dirs.shape[1]):
                a_plus = act + delta * dirs[:, k]
                a_minus = act - delta * dirs[:, k]
                s_true_p = unflatten_state(pair["plus"][:, k].to(dev))
                s_true_m = unflatten_state(pair["minus"][:, k].to(dev))
                s_pred_p, s_pred_m = s0, s0
                for _ in range(horizon):
                    s_pred_p = wm.step(s_pred_p, a_plus, xi)
                    s_pred_m = wm.step(s_pred_m, a_minus, xi)
                rt = (risk(s_true_p, tp, tR) - risk(s_true_m, tp, tR)) / (2 * delta)
                rp = (risk(s_pred_p, tp, tR) - risk(s_pred_m, tp, tR)) / (2 * delta)
                true_delta.append(rt.detach()); pred_delta.append(rp.detach())
                true_proj.append(torch.sign(rt).eq(torch.sign(rp)).float().detach())
        t = torch.cat(true_delta); p = torch.cat(pred_delta); hit = torch.cat(true_proj)
        corr = torch.corrcoef(torch.stack((t, p)))[0, 1] if t.numel() > 1 else torch.tensor(float("nan"))
        out["datasets"][str(path)] = {
            "n": int(t.numel()),
            "risk_gradient_correlation": float(corr),
            "risk_gradient_sign_accuracy": float(hit.mean()),
            "true_gradient_std": float(t.std()),
            "pred_gradient_std": float(p.std()),
            "true_positive_fraction": float((t > 0).float().mean()),
            "pred_positive_fraction": float((p > 0).float().mean()),
        }
    print(json.dumps(out, indent=2))
