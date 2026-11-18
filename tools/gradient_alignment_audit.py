"""Measure policy-task/replay-fidelity gradient geometry on a saved checkpoint."""

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


def model_from_state(state, device):
    hidden = int(state["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in state else 0
    return WorldModel(residual_hidden=hidden).to(device)


def flatten(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for p, g in zip(params, grads)])


def cosine(a, b):
    return (a * b).sum() / (a.square().sum().sqrt() * b.square().sum().sqrt()).clamp_min(1e-8)


def valid_starts(buf, motion, B, hist, horizon):
    ss, hh, n = [], [], 0
    while n < B:
        s, h = buf.sample_starts(2 * B, hist)
        ok = s["f"] + horizon < motion.clip_end[motion.clip_id[s["f"]]]
        ix = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        ss.append({k: v[ix] for k, v in s.items()}); hh.append({k: v[ix] for k, v in h.items()}); n += int(ix.numel())
    return ({k: torch.cat([x[k] for x in ss]) for k in ss[0]},
            {k: torch.cat([x[k] for x in hh]) for k in hh[0]})


def task_loss(wm, pi, mdp, motion, buf, device, horizon, batch, goal_frames, hist, params):
    s, h = valid_starts(buf, motion, batch, hist, horizon)
    S = unflatten_state(s["s"]); xi, f = s["xi"], s["f"]
    obs_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
    goal = motion.goal(f, goal_frames); a_prev = s["a_prev"]
    returns, alive = torch.zeros(batch, device=device), torch.ones(batch, device=device)
    for t in range(horizon):
        act = pi(obs_hist, goal)
        S = wm.step(S, act, normalize_xi(xi), use_residual=wm.residual_net is not None)
        f = f + 1
        r = mdp.reward(S, f, act, a_prev)
        d = mdp.soft_termination(S, f)
        returns = returns + (0.99 ** t) * alive * r
        alive = alive * (1 - d)
        obs = mdp.obs(S, act, xi, f)
        obs_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        goal = motion.goal(f, goal_frames); a_prev = act
    return -returns.mean()


def replay_loss(wm, buf, window):
    w = buf.sample_windows(64, window)
    S = unflatten_state(w["s"][:, 0])
    preds = []
    for j in range(window):
        S = wm.step(S, w["a"][:, j], normalize_xi(w["xi"][:, j]), use_residual=wm.residual_net is not None)
        preds.append(S)
    pred = tuple(torch.stack([x[i] for x in preds], 1) for i in range(6))
    target = unflatten_state(w["s1"])
    std = torch.maximum(wm.out_norm.std(), torch.full_like(wm.out_norm.std(), 1e-4))
    e = (wm.state_error(pred, target) / std).square().mean(-1)
    return (e * w["mask"]).sum() / w["mask"].sum().clamp_min(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--replay_window", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; task/replay batches use seed and seed+1")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location=args.device)
    if "buffer" not in ck:
        ap.error("checkpoint must contain a replay buffer")
    a = ck.get("args", {}); device = args.device
    state = ck["buffer"]; T, N = state["data"]["s"].shape[:2]
    buf = ReplayBuffer(T, N, device); buf.load_state_dict(state)
    kin = G1Kinematics(device); motion = MotionLib.from_dir(a.get("motion_dir", "data/lafan1_g1"), kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    wm = model_from_state(ck["wm"], device); wm.load_state_dict(ck["wm"]); wm.train()
    pi = Policy(OBS_DIM, 73 * int(a.get("goal_frames", 4)), hist=int(a.get("hist", 4))).to(device)
    pi.load_state_dict(ck["pi"]); pi.eval()
    for p in pi.parameters(): p.requires_grad_(False)
    for p in wm.in_norm.parameters(): p.requires_grad_(False)
    for p in wm.out_norm.parameters(): p.requires_grad_(False)
    # Residual checkpoints isolate the adapter geometry; nominal checkpoints
    # fall back to the dynamics network itself for a complete audit.
    params = list(wm.residual_net.parameters()) if wm.residual_net is not None else list(wm.net.parameters())
    for p in params: p.requires_grad_(True)
    out = {"checkpoint": str(args.checkpoint), "results": []}
    for horizon in (4, 8, 16):
        torch.manual_seed(args.seed)
        tl = task_loss(wm, pi, mdp, motion, buf, device, horizon, args.batch,
                       int(a.get("goal_frames", 4)), int(a.get("hist", 4)), params)
        gt = torch.autograd.grad(tl, params, retain_graph=False, allow_unused=True)
        torch.manual_seed(args.seed + 1)
        rl = replay_loss(wm, buf, args.replay_window)
        gr = torch.autograd.grad(rl, params, retain_graph=False, allow_unused=True)
        vt, vr = flatten(gt, params), flatten(gr, params)
        dot = (vt * vr).sum()
        out["results"].append({"horizon": horizon, "task_loss": float(tl.detach()),
                               "replay_loss": float(rl.detach()),
                               "task_grad_norm": float(vt.norm()), "replay_grad_norm": float(vr.norm()),
                               "dot": float(dot), "cosine": float(cosine(vt, vr)),
                               "task_projection_on_replay": float(dot / vr.square().sum().clamp_min(1e-8))})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
