"""Estimate multi-step WM sensitivity to the imagined action sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, invariant_features, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list


def model_from_state(state, device):
    key = "residual_net.0.weight"
    hidden = int(state[key].shape[0]) if key in state else 0
    return WorldModel(residual_hidden=hidden).to(device)


def sample_valid(buf, motion, B, hist, horizon):
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buf.sample_starts(2 * B, hist)
        valid = s["f"] + horizon < motion.clip_end[motion.clip_id[s["f"]]]
        idx = valid.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()})
        hists.append({k: v[idx] for k, v in h.items()})
        n += int(idx.numel())
    return ({k: torch.cat([x[k] for x in starts]) for k in starts[0]},
            {k: torch.cat([x[k] for x in hists]) for k in hists[0]})


def collect(ck, buf, mdp, motion, device, horizon, batches, batch, goal_frames, hist):
    wm = model_from_state(ck["wm"], device); wm.load_state_dict(ck["wm"]); wm.eval()
    pi = Policy(OBS_DIM, 73 * goal_frames, hist=hist).to(device); pi.load_state_dict(ck["pi"]); pi.eval()
    residual = "residual_net.0.weight" in ck["wm"]
    vals = []
    for _ in range(batches):
        s, h = sample_valid(buf, motion, batch, hist, horizon)
        S = unflatten_state(s["s"])
        xi, f = s["xi"], s["f"]
        obs_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        goal = motion.goal(f, goal_frames)
        actions = []
        with torch.no_grad():
            for _t in range(horizon):
                act = pi(obs_hist, goal).detach()
                # Keep policy action values fixed but expose WM sensitivity to
                # each action as an independent control perturbation.
                actions.append(act.requires_grad_(True))
                S = wm.step(S, actions[-1], normalize_xi(xi), use_residual=residual)
                f = f + 1
                obs = mdp.obs(S, actions[-1], xi, f)
                obs_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
                goal = motion.goal(f, goal_frames)
        # Re-run with gradients: the policy/history construction above is not
        # part of the WM Jacobian and should not enter the graph.
        S = unflatten_state(s["s"])
        for act in actions:
            S = wm.step(S, act, normalize_xi(xi), use_residual=residual)
        y = invariant_features(S)
        probe = torch.randn_like(y)
        grads = torch.autograd.grad((y * probe).sum(), actions, allow_unused=True)
        norms = torch.stack([(g.square().mean(-1).sqrt() if g is not None else torch.zeros(batch, device=device))
                             for g in grads], 1)
        vals.append(norms.detach())
    return torch.cat(vals)


def stats(x):
    return {"mean": float(x.mean()), "p50": float(torch.quantile(x, .5)),
            "p90": float(torch.quantile(x, .9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    for path_s in args.checkpoints:
        path = Path(path_s); ck = torch.load(path, map_location=args.device)
        state = ck.get("buffer")
        if state is None:
            ap.error(f"{path} must contain replay buffer state")
        a = ck.get("args", {}); device = args.device
        goal_frames, hist = int(a.get("goal_frames", 4)), int(a.get("hist", 4))
        T, N = state["data"]["s"].shape[:2]
        buf = ReplayBuffer(T, N, device); buf.load_state_dict(state)
        kin = G1Kinematics(device)
        motion = MotionLib.from_dir(a.get("motion_dir", "data/lafan1_g1"), kin, device)
        mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
        torch.manual_seed(0)
        g = collect(ck, buf, mdp, motion, device, args.horizon, args.batches, args.batch,
                    goal_frames, hist)
        cumulative = g.square().sum(1).sqrt()
        print(json.dumps({"checkpoint": str(path), "horizon": args.horizon,
                          "residual_branch": "residual_net.0.weight" in ck["wm"],
                          "samples": int(g.shape[0]), "per_step": [stats(g[:, t]) for t in range(args.horizon)],
                          "cumulative_action_sensitivity": stats(cumulative)}, indent=2))


if __name__ == "__main__":
    main()
