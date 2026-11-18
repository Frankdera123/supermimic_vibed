"""Offline SMODICE/TOM-style dual-Q occupancy audit.

This is deliberately an audit, not a training default.  It uses the newest
replay slice as the current-policy proxy and older rows as historical replay,
then fits a non-negative Q with the chi f-divergence Bellman residual.  The
derived ``relu(e_v + 1)`` weights are reported with ESS and held-out statistics
before any WM integration is attempted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_rows(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    buf = ck.get("buffer")
    if not buf:
        raise RuntimeError("checkpoint has no serialized replay buffer")
    data = buf["data"]
    ptr, size, cap = int(buf["ptr"]), int(buf["size"]), data["s"].shape[0]
    rows = (ptr - size + torch.arange(size)) % cap
    nenv = data["s"].shape[1]
    def flat(x):
        tail = x.shape[2:]
        return x[rows].reshape(-1, *tail).float() if tail else x[rows].reshape(-1).float()
    return ck, {k: flat(data[k]) for k in ("s", "a", "s1", "a_prev", "xi", "f", "ep_step")}, size, nenv


class NonnegativeQ(torch.nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.SiLU(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
                                       torch.nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.nn.functional.softplus(self.net(x)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--positive_fraction", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target_tau", type=float, default=0.0,
                    help="Polyak coefficient for a target Q (0 keeps the direct bootstrap)")
    ap.add_argument("--q_l2", type=float, default=0.0,
                    help="L2 regularization on Q parameters")
    ap.add_argument("--grad_clip", type=float, default=0.0,
                    help="optional global Q gradient norm clip")
    ap.add_argument("--weight_clip", type=float, default=0.0,
                    help="optional upper clip on derived occupancy weights before renormalization")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--save_weights", type=Path, default=None,
                    help="save age-by-environment dual-Q weights for replay WM experiments")
    ap.add_argument("--use_policy_action", action="store_true",
                    help="reconstruct pi(s') from the checkpoint instead of reusing observed action")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    ck, d, size, nenv = load_rows(args.checkpoint)
    m = max(1, int(size * args.positive_fraction) * nenv)
    # Current-policy proxy is the newest slice; historical replay is oldest.
    pos = {k: v[-m:] for k, v in d.items()}
    neg = {k: v[:m] for k, v in d.items()}
    xdim = d["s"].shape[-1] + d["a"].shape[-1]
    q = NonnegativeQ(xdim, args.hidden)
    q_target = NonnegativeQ(xdim, args.hidden) if args.target_tau > 0.0 else q
    if q_target is not q:
        q_target.load_state_dict(q.state_dict())
        for p in q_target.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(q.parameters(), lr=args.lr, weight_decay=args.q_l2)
    # The SMODICE/TOM no-discriminator audit uses reward 1 for current-policy
    # transitions and 0 for historical transitions.
    state = torch.cat((pos["s"], neg["s"]))
    action = torch.cat((pos["a"], neg["a"]))
    next_state = torch.cat((pos["s1"], neg["s1"]))
    next_action = action.clone()
    if args.use_policy_action:
        from supermimic.kinematics import G1Kinematics
        from supermimic.mdp import MDP, OBS_DIM, unflatten_state
        from supermimic.motion import MotionLib
        from supermimic.networks import Policy
        from supermimic.robot import default_joint_pos_list
        dev = "cpu"
        motion_dir = ck.get("args", {}).get("motion_dir", "data/lafan1_g1")
        kin = G1Kinematics(dev)
        motion = MotionLib.from_dir(motion_dir, kin, dev)
        mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
        goal_frames = int(ck.get("args", {}).get("goal_frames", 10))
        hist = int(ck.get("args", {}).get("hist", 10))
        pi = Policy(OBS_DIM, 73 * goal_frames, hist=hist).to(dev)
        pi.load_state_dict(ck["pi"]); pi.eval()
        with torch.no_grad():
            xi = torch.cat((pos["xi"], neg["xi"])) if "xi" in pos else torch.zeros(len(state), 33)
            f = torch.cat((pos["f"], neg["f"])).long()
            aprev = torch.cat((pos.get("a_prev", pos["a"]), neg.get("a_prev", neg["a"])))
            obs = mdp.obs(unflatten_state(state), aprev, xi, f)
            obs1 = mdp.obs(unflatten_state(next_state), action, xi, f)
            oh = torch.zeros(len(state), hist, OBS_DIM)
            oh[:, -1] = obs
            oh1 = torch.zeros_like(oh); oh1[:, -1] = obs1
            next_action = pi(oh1, motion.goal(f, goal_frames))
    reward = torch.cat((torch.ones(m), torch.zeros(m)))
    initial = pos["s"]
    n = state.shape[0]
    for _ in range(args.epochs):
        order = torch.randperm(n)
        for idx in order.split(args.batch):
            q_now = q(torch.cat((state[idx], action[idx]), -1))
            q_next = q_target(torch.cat((next_state[idx], next_action[idx]), -1)).detach()
            e_v = reward[idx] + args.gamma * q_next - q_now
            q_init = q(torch.cat((initial[idx % len(initial)],
                                  action[idx % len(initial)]), -1))
            loss = ((0.5 * e_v.square() + e_v).mean()
                    + (1.0 - args.gamma) * q_init.mean())
            opt.zero_grad(); loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(q.parameters(), args.grad_clip)
            opt.step()
        if q_target is not q:
            with torch.no_grad():
                for pt, p in zip(q_target.parameters(), q.parameters()):
                    pt.mul_(1.0 - args.target_tau).add_(p, alpha=args.target_tau)
    with torch.no_grad():
        e = reward + args.gamma * q_target(torch.cat((next_state, next_action), -1)) - q(torch.cat((state, action), -1))
        weights = (e + 1.0).clamp_min(0.0)
        if args.weight_clip > 0.0:
            weights = weights.clamp_max(args.weight_clip)
        weights = weights / weights.mean().clamp_min(1e-6)
        if args.weight_clip > 0.0:
            # Alternating projection keeps both the mean-one convention and a
            # practical upper bound after renormalization.
            for _ in range(8):
                weights = weights.clamp_max(args.weight_clip)
                weights = weights / weights.mean().clamp_min(1e-6)
        ess = float(weights.sum().square() / weights.square().sum().clamp_min(1e-6))
        result = {
            "checkpoint": str(args.checkpoint), "iteration": int(ck.get("iteration", -1)),
            "rows": int(n), "epochs": args.epochs, "gamma": args.gamma,
            "target_tau": args.target_tau, "q_l2": args.q_l2,
            "grad_clip": args.grad_clip,
            "weight_clip": args.weight_clip,
            "weight_mean": float(weights.mean()), "weight_std": float(weights.std()),
            "weight_p95": float(weights.quantile(.95)), "weight_max": float(weights.max()),
            "effective_sample_fraction": ess / n,
            "positive_weight_mean": float(weights[:m].mean()),
            "historical_weight_mean": float(weights[m:].mean()),
        }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.save_weights:
        args.save_weights.parent.mkdir(parents=True, exist_ok=True)
        table = torch.ones(size, nenv)
        q = m // nenv
        table[:q] = weights[m:].reshape(q, nenv).cpu()
        table[-q:] = weights[:m].reshape(q, nenv).cpu()
        torch.save({
            "weights": table,
            "buffer_rows": size, "envs": nenv, "checkpoint": str(args.checkpoint),
            "epochs": args.epochs, "use_policy_action": args.use_policy_action,
            "result": result,
        }, args.save_weights)


if __name__ == "__main__":
    main()
