"""Measure support drift along policy/world-model imagined trajectories."""

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


def qstats(x):
    return {"mean": float(x.mean()), "p50": float(torch.quantile(x, .5)),
            "p90": float(torch.quantile(x, .9)), "p99": float(torch.quantile(x, .99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location=args.device)
    if "buffer" not in ck:
        ap.error("checkpoint must contain replay buffer state")
    a = ck.get("args", {}); device = args.device
    goal_frames, hist = int(a.get("goal_frames", 4)), int(a.get("hist", 4))
    state = ck["buffer"]; T, N = state["data"]["s"].shape[:2]
    buf = ReplayBuffer(T, N, device); buf.load_state_dict(state)
    kin = G1Kinematics(device)
    motion = MotionLib.from_dir(a.get("motion_dir", "data/lafan1_g1"), kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    wm = model_from_state(ck["wm"], device); wm.load_state_dict(ck["wm"]); wm.eval()
    pi = Policy(OBS_DIM, 73 * goal_frames, hist=hist).to(device); pi.load_state_dict(ck["pi"]); pi.eval()
    residual = "residual_net.0.weight" in ck["wm"]

    # Replay support statistics use all currently buffered states/actions.  The
    # invariant state coordinates match the WM input's state representation.
    with torch.no_grad():
        replay_s = unflatten_state(state["data"]["s"][:buf.size].reshape(-1, 71))
        replay_z = invariant_features(replay_s)
        zmean, zstd = replay_z.mean(0), replay_z.std(0).clamp_min(1e-3)
        replay_a = state["data"]["a"][:buf.size].reshape(-1, 29)
        amean, astd = replay_a.mean(0), replay_a.std(0).clamp_min(1e-3)
        state_scores, action_scores, wm_norms = [], [], []
        for _ in range(args.batches):
            starts, hists, n = [], [], 0
            while n < args.batch:
                ss, hh = buf.sample_starts(2 * args.batch, hist)
                valid = (ss["f"] + args.horizon < motion.clip_end[motion.clip_id[ss["f"]]])
                idx = valid.nonzero(as_tuple=False).squeeze(-1)[: args.batch - n]
                starts.append({k: v[idx] for k, v in ss.items()})
                hists.append({k: v[idx] for k, v in hh.items()})
                n += int(idx.numel())
            s = {k: torch.cat([x[k] for x in starts]) for k in starts[0]}
            h = {k: torch.cat([x[k] for x in hists]) for k in hists[0]}
            S = unflatten_state(s["s"])
            xi, f = s["xi"], s["f"]
            obs_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
            goal = motion.goal(f, goal_frames)
            zs, aas, wn = [], [], []
            for _t in range(args.horizon):
                act = pi(obs_hist, goal)
                xi_n = normalize_xi(xi)
                wm_x = wm.in_norm(wm.inputs(S, act, xi_n))
                S = wm.step(S, act, xi_n, use_residual=residual)
                f = f + 1
                obs = mdp.obs(S, act, xi, f)
                obs_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
                goal = motion.goal(f, goal_frames)
                zs.append(((invariant_features(S) - zmean) / zstd).square().mean(-1).sqrt())
                aas.append(((act - amean) / astd).square().mean(-1).sqrt())
                wn.append(wm_x.square().mean(-1).sqrt())
            state_scores.append(torch.stack(zs, 1)); action_scores.append(torch.stack(aas, 1)); wm_norms.append(torch.stack(wn, 1))
    state_scores = torch.cat(state_scores); action_scores = torch.cat(action_scores); wm_norms = torch.cat(wm_norms)
    out = {"checkpoint": str(args.checkpoint), "horizon": args.horizon,
           "residual_branch": residual, "samples": int(state_scores.shape[0]),
           "per_step": []}
    for t in range(args.horizon):
        out["per_step"].append({"step": t + 1,
                                "state_support_rms_z": qstats(state_scores[:, t]),
                                "action_support_rms_z": qstats(action_scores[:, t]),
                                "wm_input_rms": qstats(wm_norms[:, t])})
    out["final"] = out["per_step"][-1]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
