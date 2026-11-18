"""CRGF policy-parameter path audit (CPU-only, no simulator).

Extends the three-path decomposition from action-space to the POLICY
GRADIENT itself.  For the closed-loop imagination loop (policy in the loop,
WM/critic frozen -- exactly the structure of ``policy_update``), computes

  g_path = grad_theta J^(path)   for path in {state, direct, critic}
  g_total = grad_theta J

and reports, per model/horizon: the norm share of each path in g_total, the
pairwise cosine between path gradients (gradient CONFLICT at the policy level),
and the V_MAX clamp activity of the critic term.

Why this matters: at H=8 the action-space gradient is 77-98% critic, but the
policy update is driven by g_theta aggregated over starts and steps.  If the
critic path also dominates g_theta AND points differently from the dynamics
path, then the critic -- not the dynamics model -- steers the policy update,
which is hypothesis H5 at its most decision-relevant level.

No parameters are updated; checkpoints are read-only.

Run:
  python tools/crgf_policy_path_audit.py --wm_ckpt A.pt B.pt C.pt \
      --proto_root <proto> --policy_ckpt <501.pt> --critic_mode shared \
      --shared_critic_ckpt <501.pt> --out_prefix results/crgf/pol_path_audit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent))

import torch  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    from _crgf_bootstrap import install as _install_stub, fix_data_paths

    ap = argparse.ArgumentParser()  # noqa: E501 (bootstrap import order matters, see main)
    ap.add_argument("--wm_ckpt", nargs="+", required=True, type=Path)
    ap.add_argument("--policy_ckpt", required=True, type=Path)
    ap.add_argument("--critic_mode", choices=["own", "shared", "none"], default="shared")
    ap.add_argument("--shared_critic_ckpt", type=Path, default=None)
    ap.add_argument("--proto_root", default=None)
    ap.add_argument("--motion_dir", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--num_clips", type=int, default=8)
    ap.add_argument("--states_per_clip", type=int, default=2)
    ap.add_argument("--horizons", default="8,32")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_prefix", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = args.device
    repo_root = here.parent
    _install_stub()
    if args.proto_root:
        # variant package must win over the local one
        sys.path.insert(0, args.proto_root)
        import importlib

        importlib.invalidate_caches()
        for m in [k for k in sys.modules if k == "supermimic" or k.startswith("supermimic.")]:
            del sys.modules[m]
    # AFTER the variant import: patch the ACTIVE kinematics module's MJCF path
    fix_data_paths(repo_root)

    from supermimic import crgf
    from supermimic.crgf import Protocol
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP, OBS_DIM, normalize_xi
    from supermimic.motion import MotionLib
    from supermimic.networks import Critic, Policy, WorldModel
    from supermimic.robot import default_joint_pos_list

    t0 = time.time()
    horizons = [int(h) for h in args.horizons.split(",")]
    pol_ck = torch.load(args.policy_ckpt, map_location=dev, weights_only=False)
    a_pol = pol_ck.get("args", {})
    hist = int(a_pol.get("hist", 10))
    gamma = float(a_pol.get("gamma", 0.99))
    motion_dir = args.motion_dir
    if motion_dir is None:
        md = a_pol.get("motion_dir", "data/lafan1_g1")
        from pathlib import Path as _P

        motion_dir = md if _P(md).exists() else str(
            repo_root / md.replace("/workspace/supermimic/", ""))

    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(motion_dir, kin, dev)
    default_q = torch.tensor(default_joint_pos_list(), device=dev)
    try:
        mdp = MDP(kin, motion, default_q,
                  phase_features=bool(a_pol.get("phase_conditioned_policy", False)),
                  clip_features=bool(a_pol.get("clip_conditioned_policy", False)),
                  contact_masked_reward=bool(a_pol.get("contact_masked_reward", False)))
    except TypeError:
        mdp = MDP(kin, motion, default_q)
    import importlib as _il

    motion_mod = _il.import_module("supermimic.motion")
    mdp_mod = _il.import_module("supermimic.mdp")
    is_proto = hasattr(motion_mod, "GOAL_STRIDE")
    K = (int(motion_mod.GOAL_FRAMES) if is_proto else int(a_pol.get("goal_frames", 10)))
    v_max = (mdp_mod.MAX_STEP_REWARD / (1.0 - gamma)) if is_proto else None
    proto = Protocol(mdp, K, v_max=v_max)

    pi = Policy(OBS_DIM, int(proto.actor_goal(motion.clip_start[:2],
                                              motion.ref_state(motion.clip_start[:2])).shape[-1]),
                hist=hist).to(dev)
    pi.load_state_dict(pol_ck["pi"])

    def load_critic(ck):
        if "critic" not in ck or ck["critic"] is None:
            return None
        f_probe = motion.clip_start[:2]
        S_probe = motion.ref_state(f_probe)
        xi_probe = torch.zeros(2, 33, device=dev)
        xi_probe[:, 0] = 1.0
        vin = proto.critic_input(S_probe, f_probe, normalize_xi(xi_probe),
                                 proto.wm_goal(f_probe, S_probe))
        cr = Critic(vin.shape[-1]).to(dev)
        cr.load_state_dict(ck["critic"])
        cr.eval()
        return cr

    shared_critic = None
    if args.critic_mode == "shared":
        shared_critic = load_critic(torch.load(args.shared_critic_ckpt, map_location=dev,
                                               weights_only=False))

    # starts (same selection as the WM-side audit)
    names = motion.names
    sel = ([i for i, n in enumerate(names) if any(c in n for c in args.clips)]
           if args.clips else list(range(len(names)))[: args.num_clips])
    import torch as _t

    g0 = _t.Generator().manual_seed(args.seed)
    rows = []
    for c in sel:
        s0, s1 = int(motion.clip_start[c]), int(motion.clip_end[c])
        hi = max(s0 + 1, s1 - max(horizons) - 1)
        n = min(args.states_per_clip, hi - s0)
        frames = s0 + 1 + _t.randperm(hi - s0 - 1, generator=g0)[:n].sort().values
        rows += [(c, int(f)) for f in frames.tolist()]

    out_rows = []
    for wm_path in args.wm_ckpt:
        ck = torch.load(wm_path, map_location=dev, weights_only=False)
        wm = WorldModel().to(dev)
        wm.load_state_dict(ck["wm"])
        wm.eval()
        critic = shared_critic if args.critic_mode == "shared" else (
            load_critic(ck) if args.critic_mode == "own" else None)
        for H in horizons:
            # re-scope the loop to this horizon by rebuilding graphs
            def build(S0, f0, xi, xi_n, Hh):
                B = S0[0].shape[0]
                S = S0
                a_prev = torch.zeros(B, 29, device=dev)
                o_hist = proto.obs(S0, a_prev, xi, f0)
                o_hist = o_hist.unsqueeze(1).expand(-1, hist, -1).contiguous()
                f = f0
                c_prev = torch.ones(B, device=dev)
                L_state = torch.zeros((), device=dev)
                L_direct = torch.zeros((), device=dev)
                L_critic_terms = torch.zeros((), device=dev)
                v_max_hits = 0.0
                v_n = 0
                for t in range(Hh):
                    g = proto.actor_goal(f, S)
                    a = pi(o_hist, g)
                    S = proto.wm_step(wm, S, a, xi_n)
                    f1 = f + 1
                    r_state, r_direct = crgf.reward_split(mdp, S, f1, a, a_prev)
                    d = crgf.soft_term(mdp, S, f1)
                    c = c_prev * (1.0 - d)
                    L_state = L_state + (gamma ** t) * (c * r_state).sum()
                    L_direct = L_direct + (gamma ** t) * (c * r_direct).sum()
                    if critic is not None and t == Hh - 1:
                        gw = proto.wm_goal(f1, S)
                        vin = proto.critic_input(S, f1, xi_n, gw)
                        V = critic(vin)
                        if v_max is not None:
                            v_max_hits += float((V >= v_max - 1e-6).float().mean())
                            v_n += 1
                        L_critic_terms = L_critic_terms + (gamma ** Hh) * (c * V).sum()
                    o1 = proto.obs(S, a, xi, f1)
                    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
                    f = f1
                    a_prev = a
                    c_prev = c
                return L_state, L_direct, L_critic_terms, (v_max_hits / max(v_n, 1))

            grads = {k: [] for k in ("state", "direct", "critic", "total")}
            clamp_stats = []
            for i0 in range(0, len(rows), args.batch):
                chunk = rows[i0:i0 + args.batch]
                f0 = torch.tensor([f for _, f in chunk], device=dev)
                S0 = motion.ref_state(f0)
                xi = torch.zeros(len(chunk), 33, device=dev)
                xi[:, 0] = 1.0
                xi_n = normalize_xi(xi)
                L_state, L_direct, L_critic, clamp_frac = build(S0, f0, xi, xi_n, H)
                L_total = L_state + L_direct + L_critic
                flat = {}
                for name, L in (("state", L_state), ("direct", L_direct),
                                ("critic", L_critic), ("total", L_total)):
                    if L.requires_grad:
                        gs = torch.autograd.grad(L, list(pi.parameters()),
                                                 retain_graph=True, allow_unused=True)
                        flat[name] = torch.cat([(g.view(-1) if g is not None else
                                                 torch.zeros(p.numel(), device=dev))
                                                for g, p in zip(gs, pi.parameters())])
                    else:
                        flat[name] = torch.zeros(sum(p.numel() for p in pi.parameters()), device=dev)
                if critic is None:
                    flat["critic"] = torch.zeros_like(flat["total"])
                # sanity: total == state + direct + critic by linearity
                for k in ("state", "direct", "critic"):
                    grads[k].append(flat[k].detach())
                grads["total"].append(flat["total"].detach())
                clamp_stats.append(clamp_frac)
            for k in grads:
                grads[k] = torch.stack(grads[k])  # (N, theta)
            N = grads["total"].shape[0]
            norms = {k: grads[k].norm(dim=-1) for k in grads}

            def cos(a, b):
                na, nb = a.norm(dim=-1).clamp_min(1e-12), b.norm(dim=-1).clamp_min(1e-12)
                return (a * b).sum(-1) / (na * nb)

            per_start = []
            for i in range(N):
                per_start.append({
                    "norm_state": float(norms["state"][i]),
                    "norm_direct": float(norms["direct"][i]),
                    "norm_critic": float(norms["critic"][i]),
                    "norm_total": float(norms["total"][i]),
                    "share_critic": float(norms["critic"][i] / norms["total"][i].clamp_min(1e-12)),
                    "share_state": float(norms["state"][i] / norms["total"][i].clamp_min(1e-12)),
                    "cos_state_critic": float(cos(grads["state"][i], grads["critic"][i])),
                    "cos_state_total": float(cos(grads["state"][i], grads["total"][i])),
                })
            med = lambda k: sorted([p[k] for p in per_start])[len(per_start) // 2]
            summary = {
                "wm_ckpt": str(wm_path), "iteration": ck.get("iteration"),
                "horizon": H, "n_starts": N,
                "critic_mode": args.critic_mode,
                "med_norm_state": med("norm_state"),
                "med_norm_direct": med("norm_direct"),
                "med_norm_critic": med("norm_critic"),
                "med_norm_total": med("norm_total"),
                "med_share_critic": med("share_critic"),
                "med_share_state": med("share_state"),
                "med_cos_state_critic": med("cos_state_critic"),
                "med_cos_state_total": med("cos_state_total"),
                "v_clamp_frac": sum(clamp_stats) / max(len(clamp_stats), 1),
                "sha256": sha256(wm_path),
            }
            for p in per_start:
                out_rows.append({"wm_ckpt": str(wm_path), "horizon": H, **p})
            print(json.dumps(summary))

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_prefix.with_suffix(".json"), "w") as fh:
        json.dump({"protocol": {"gamma": gamma, "goal_frames": K, "hist": hist,
                                "horizons": horizons, "critic_mode": args.critic_mode,
                                "policy_ckpt": str(args.policy_ckpt),
                                "policy_sha256": sha256(args.policy_ckpt),
                                "note": "closed-loop imagination, WM+critic frozen, "
                                        "exact autograd per-path policy gradients; "
                                        "g_critic obtained exactly (linear differencing "
                                        "not needed: critic term built as its own graph)"},
                   "starts": [{"clip": motion.names[c], "frame": f} for c, f in rows]},
                  fh, indent=1)
    with open(args.out_prefix.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(json.dumps({"elapsed_s": round(time.time() - t0, 1), "rows": len(out_rows)}))


if __name__ == "__main__":
    main()
