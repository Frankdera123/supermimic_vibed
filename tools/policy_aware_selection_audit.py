"""Audit replay scores for policy-aware world-model data selection.

The audit is deliberately offline: it does not update a model or run physics.
For replay transitions it measures (1) action compatibility with the current
policy, (2) factual one-step WM error, and optionally (3) disagreement between
two frozen WMs.  It then reports whether score-selected rows actually contain
more policy-compatible high-error transitions than uniform replay.
"""
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


def make_wm(state, device):
    key = "residual_net.0.weight"
    hidden = int(state[key].shape[0]) if key in state else 0
    model = WorldModel(residual_hidden=hidden).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def collect(ck, wm, pi, mdp, motion, buf, samples, batch, sigma, device,
            member=None):
    deltas, errors, disagreements, masks = [], [], [], []
    goal_frames = int(ck.get("args", {}).get("goal_frames", 10))
    hist = int(ck.get("args", {}).get("hist", 10))
    for _ in range((samples + batch - 1) // batch):
        b = min(batch, samples - sum(x.numel() for x in errors))
        if b <= 0:
            break
        start, history = buf.sample_starts(b, hist)
        with torch.no_grad():
            obs = mdp.obs(
                unflatten_state(history["s"]),
                history["a_prev"], history["xi"], history["f"],
            )
            pred_a = pi(obs, motion.goal(start["f"], goal_frames))
            delta = ((pred_a - start["a"]) / 20.0).square().mean(-1)
            compat = torch.exp(-delta / (2.0 * sigma * sigma))
            s0 = unflatten_state(start["s"])
            pred = wm.step(s0, start["a"], normalize_xi(start["xi"]))
            target = unflatten_state(start["s1"])
            std = torch.sqrt(wm.out_norm.var + wm.out_norm.eps ** 2).clamp_min(1e-4)
            error = (wm.state_error(pred, target) / std).square().mean(-1)
            if member is None:
                dis = torch.zeros_like(error)
            else:
                pred_m = member.step(s0, start["a"], normalize_xi(start["xi"]))
                dis = (wm.state_error(pred, pred_m) / std).square().mean(-1)
            # sample_starts returns a transition start without a done field;
            # one-step rows are valid for this diagnostic by construction.
            valid = torch.ones_like(error)
        deltas.append(delta.detach()); errors.append(error.detach())
        disagreements.append(dis.detach()); masks.append(valid.detach())
    return tuple(torch.cat(x) for x in (deltas, errors, disagreements, masks))


def summary(name, score, error, compat, valid, fraction):
    n = max(1, int(round(score.numel() * fraction)))
    idx = torch.argsort(score, descending=True)[:n]
    v = valid[idx]
    denom = v.sum().clamp_min(1.0)
    return {
        "selected": int(n),
        "valid_fraction": float(v.mean()),
        "mean_error": float((error[idx] * v).sum() / denom),
        "p90_error": float(torch.quantile(error[idx], .9)),
        "mean_action_distance": float((compat[idx].clamp_min(1e-8).log().neg().mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("member", type=Path, nargs="?",
                    help="optional independent WM checkpoint for disagreement")
    ap.add_argument("--samples", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=.05)
    ap.add_argument("--top_fraction", type=float, default=.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--motion_dir", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.samples < 1 or args.sigma <= 0 or not 0 < args.top_fraction <= 1:
        ap.error("samples, sigma, and top_fraction are invalid")
    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ck.get("buffer")
    if state is None:
        ap.error("checkpoint must contain replay buffer state")
    T, N = state["data"]["s"].shape[:2]
    buf = ReplayBuffer(T, N, args.device); buf.load_state_dict(state)
    base = ck.get("args", {})
    motion_dir = args.motion_dir or base.get("motion_dir", "data/lafan1_g1")
    kin = G1Kinematics(args.device)
    motion = MotionLib.from_dir(motion_dir, kin, args.device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=args.device))
    pi = Policy(OBS_DIM, 73 * int(base.get("goal_frames", 10)),
                hist=int(base.get("hist", 10))).to(args.device)
    pi.load_state_dict(ck["pi"]); pi.eval()
    wm = make_wm(ck["wm"], args.device)
    member = None
    if args.member is not None:
        mck = torch.load(args.member, map_location=args.device, weights_only=False)
        member = make_wm(mck["wm"], args.device)
    delta, error, disagreement, valid = collect(
        ck, wm, pi, mdp, motion, buf, args.samples, args.batch,
        args.sigma, args.device, member)
    compat = torch.exp(-delta / (2.0 * args.sigma * args.sigma))
    compat_n = compat / compat.mean().clamp_min(1e-6)
    err_n = error / error.mean().clamp_min(1e-6)
    unc_n = disagreement / disagreement.mean().clamp_min(1e-6) if member is not None else torch.zeros_like(error)
    scores = {
        "uniform": torch.zeros_like(error),
        "policy_compat": compat_n,
        "compat_error": compat_n * err_n,
        "compat_error_uncertainty": compat_n * err_n * (1.0 + .5 * unc_n),
    }
    out = {
        "checkpoint": str(args.checkpoint),
        "member": str(args.member) if args.member else None,
        "samples": int(error.numel()), "sigma": args.sigma,
        "top_fraction": args.top_fraction,
        "overall_error": float(error.mean()),
        "overall_policy_compatibility": float(compat.mean()),
        "overall_disagreement": float(disagreement.mean()),
        "selection": {k: summary(k, s, error, compat, valid, args.top_fraction)
                       for k, s in scores.items()},
    }
    payload = json.dumps(out, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
