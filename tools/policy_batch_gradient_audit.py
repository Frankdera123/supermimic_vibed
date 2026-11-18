"""Measure policy-gradient diversity across independently sampled replay batches."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list


def flat_grad(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
                      for p, g in zip(params, grads)])


def cosine(a, b):
    return (a * b).sum() / (a.square().sum().sqrt() * b.square().sum().sqrt()).clamp_min(1e-8)


def sample_valid(buf, motion, batch, hist, horizon):
    ss, hh, n = [], [], 0
    while n < batch:
        s, h = buf.sample_starts(2 * batch, hist)
        ok = s["f"] + horizon < motion.clip_end[motion.clip_id[s["f"]]]
        ix = ok.nonzero(as_tuple=False).squeeze(-1)[: batch - n]
        ss.append({k: v[ix] for k, v in s.items()})
        hh.append({k: v[ix] for k, v in h.items()})
        n += int(ix.numel())
    return ({k: torch.cat([x[k] for x in ss]) for k in ss[0]},
            {k: torch.cat([x[k] for x in hh]) for k in hh[0]})


def policy_loss(pi, wm, mdp, motion, batch, hist, horizon, gamma, trunc, goal_frames):
    s, h = batch
    B = s["f"].shape[0]
    S = unflatten_state(s["s"])
    xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
    xi_n = normalize_xi(xi)
    obs_hist = mdp.add_noise(mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"]))
    goal = motion.goal(f, goal_frames)
    rs, ds = [], []
    for t in range(horizon):
        act = pi(obs_hist, goal)
        S = wm.step(S, act, xi_n)
        f = f + 1
        rs.append(mdp.reward(S, f, act, a_prev))
        ds.append(mdp.soft_termination(S, f))
        obs = mdp.obs(S, act, xi, f)
        obs_hist = torch.cat((obs_hist[:, 1:], obs[:, None]), 1)
        goal = motion.goal(f, goal_frames)
        a_prev = act
        if trunc > 0 and (t + 1) % trunc == 0 and t + 1 < horizon:
            S = tuple(x.detach() for x in S)
            obs_hist = obs_hist.detach(); a_prev = a_prev.detach()
    r, d = torch.stack(rs), torch.stack(ds)
    alive = torch.cumprod(1 - d, 0)
    c = torch.cat((torch.ones(1, B, device=r.device), alive[:-1]), 0)
    disc = gamma ** torch.arange(horizon, device=r.device, dtype=torch.float32)[:, None]
    return -(disc * c * r).sum(0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", type=Path)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--trunc", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    for path in args.checkpoints:
        ck = torch.load(path, map_location=args.device, weights_only=False)
        a = ck.get("args", {}); dev = args.device
        state = ck.get("buffer")
        if state is None: ap.error(f"{path} must contain replay buffer state")
        T, N = state["data"]["s"].shape[:2]
        buf = ReplayBuffer(T, N, dev); buf.load_state_dict(state)
        kin = G1Kinematics(dev); motion = MotionLib.from_dir(a.get("motion_dir", "data/lafan1_g1"), kin, dev)
        mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
        wm_state = ck["wm"]; hidden = int(wm_state["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in wm_state else 0
        wm = WorldModel(residual_hidden=hidden).to(dev); wm.load_state_dict(wm_state); wm.eval()
        pi = Policy(OBS_DIM, 73 * int(a.get("goal_frames", 10)), hist=int(a.get("hist", 10))).to(dev)
        pi.load_state_dict(ck["pi"]); pi.train()
        for p in wm.parameters(): p.requires_grad_(False)
        params = list(pi.parameters())
        vals, norms, dots = [], [], []
        for i in range(args.pairs):
            torch.manual_seed(1000 + i)
            b1 = sample_valid(buf, motion, args.batch, int(a.get("hist", 10)), args.horizon)
            b2 = sample_valid(buf, motion, args.batch, int(a.get("hist", 10)), args.horizon)
            pi.zero_grad(set_to_none=True)
            l1 = policy_loss(pi, wm, mdp, motion, b1, int(a.get("hist", 10)), args.horizon, float(a.get("gamma", .99)), args.trunc, int(a.get("goal_frames", 10)))
            g1 = flat_grad(torch.autograd.grad(l1, params, retain_graph=False), params)
            pi.zero_grad(set_to_none=True)
            l2 = policy_loss(pi, wm, mdp, motion, b2, int(a.get("hist", 10)), args.horizon, float(a.get("gamma", .99)), args.trunc, int(a.get("goal_frames", 10)))
            g2 = flat_grad(torch.autograd.grad(l2, params, retain_graph=False), params)
            vals.append(float(cosine(g1, g2))); norms.append((float(g1.norm()), float(g2.norm())))
            dots.append(float((g1 * g2).sum() > 0))
        out = {"checkpoint": str(path), "pairs": args.pairs, "batch": args.batch, "horizon": args.horizon,
               "trunc": args.trunc, "independent_gradient_cosine_mean": sum(vals) / len(vals),
               "independent_gradient_cosine_min": min(vals), "independent_gradient_cosine_max": max(vals),
               "positive_dot_rate": sum(dots) / len(dots),
               "gradient_norm_mean": [sum(x[j] for x in norms) / len(norms) for j in (0, 1)]}
        print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
