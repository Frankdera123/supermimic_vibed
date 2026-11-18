"""Generate policy-update action-delta directions (delta_a) for CRGF audits.

For a real policy-update pair (theta_old, theta_new), roll the OLD policy
closed-loop in a designated world model to obtain a shared observation
sequence, then evaluate BOTH policies teacher-forced on that same observation
sequence:

    delta_a_t = pi_new(o_hist_t) - pi_old(o_hist_t)

The result is a pure action-space direction field on a fixed nominal
trajectory: exactly the direction a real policy update moves the action
sequence, without any simulator interaction and without updating any
parameters (CPU-only, read-only on checkpoints).

Usage:
  python tools/crgf_delta_action_directions.py \
      --ckpt_old <old.pt> --ckpt_new <new.pt> [--rollout_wm <wm.pt>] \
      --motion_dir data/lafan1_g1 --clips walk1_subject1 ... --out delta.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_PROTO_ROOT = None
for _i, _a in enumerate(sys.argv):
    if _a == "--proto_root":
        _PROTO_ROOT = sys.argv[_i + 1]
    elif _a.startswith("--proto_root="):
        _PROTO_ROOT = _a.split("=", 1)[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _crgf_bootstrap import install as _install_stub

_install_stub()
if _PROTO_ROOT:
    # inserted after the repo root so the variant package takes precedence
    sys.path.insert(0, _PROTO_ROOT)

import torch

from _crgf_bootstrap import fix_data_paths

fix_data_paths(Path(__file__).resolve().parents[1])
from supermimic.crgf import Protocol, base_action_rollout, normalize_dir_seqs
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import Policy, WorldModel
from supermimic.robot import default_joint_pos_list


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remap_data_path(path: str, repo_root: Path) -> str:
    """Checkpoint args recorded container paths (/workspace/supermimic/...)."""
    if not path:
        return path
    p = Path(path)
    if p.exists():
        return str(p)
    if str(p).startswith("/workspace/supermimic"):
        candidate = repo_root / str(p).replace("/workspace/supermimic/", "")
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"motion_dir {path} not found locally")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_old", required=True, type=Path)
    ap.add_argument("--ckpt_new", required=True, type=Path)
    ap.add_argument("--rollout_wm", type=Path, default=None,
                    help="world model used to generate the shared obs sequence "
                         "(default: ckpt_old's own WM)")
    ap.add_argument("--motion_dir", default=None,
                    help="default: ckpt_old args motion_dir remapped to this repo")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="motion clip names (substring match); default: all")
    ap.add_argument("--num_clips", type=int, default=8)
    ap.add_argument("--states_per_clip", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--goal_frames", type=int, default=None)
    ap.add_argument("--hist", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--proto_root", default=None,
                    help="supermimic package variant matching these checkpoints")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    dev = args.device
    repo_root = Path(__file__).resolve().parents[1]

    ck_old = torch.load(args.ckpt_old, map_location=dev, weights_only=False)
    ck_new = torch.load(args.ckpt_new, map_location=dev, weights_only=False)
    a_old = ck_old.get("args", {})
    hist = args.hist or int(a_old.get("hist", 10))
    motion_dir = args.motion_dir or remap_data_path(a_old.get("motion_dir", "data/lafan1_g1"), repo_root)

    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(motion_dir, kin, dev)
    default_q = torch.tensor(default_joint_pos_list(), device=dev)
    try:
        mdp = MDP(kin, motion, default_q,
                  phase_features=bool(a_old.get("phase_conditioned_policy", False)),
                  clip_features=bool(a_old.get("clip_conditioned_policy", False)),
                  contact_safe_reference=bool(a_old.get("contact_safe_reward", False)),
                  contact_masked_reward=bool(a_old.get("contact_masked_reward", False)),
                  contact_features=bool(a_old.get("contact_conditioned_policy", False)),
                  current_contact_features=bool(a_old.get("current_contact_policy", False)))
    except TypeError:
        mdp = MDP(kin, motion, default_q)

    def load_wm(ck):
        st = ck["wm"]
        try:
            m = WorldModel(residual_hidden=0).to(dev)
        except TypeError:
            m = WorldModel().to(dev)  # variant WM without residual support
        m.load_state_dict(st)
        m.eval()
        return m

    def load_pi(ck):
        pi = Policy(OBS_DIM, 73 * K, hist=hist).to(dev)
        pi.load_state_dict(ck["pi"])
        pi.eval()
        return pi

    from supermimic.motion import GOAL_FRAMES  # proto hardcodes this global

    is_proto = _PROTO_ROOT is not None
    K = args.goal_frames or (GOAL_FRAMES if is_proto else int(a_old.get("goal_frames", 10)))
    wm = load_wm(torch.load(args.rollout_wm, map_location=dev, weights_only=False)
                 if args.rollout_wm else ck_old)
    proto = Protocol(mdp, K)
    pi_old, pi_new = load_pi(ck_old), load_pi(ck_new)

    # ---- start selection: deterministic frames with horizon margin ----
    names = motion.names
    if args.clips:
        sel = [i for i, n in enumerate(names) if any(c in n for c in args.clips)]
        if not sel:
            ap.error(f"no clip matches {args.clips}")
    else:
        sel = list(range(len(names)))[: args.num_clips]
    starts = []
    g = torch.Generator().manual_seed(args.seed)
    H = args.horizon
    for c in sel:
        s0, s1 = int(motion.clip_start[c]), int(motion.clip_end[c])
        hi = max(s0 + 1, s1 - H - 1)  # keep rewards inside the clip
        n = min(args.states_per_clip, hi - s0)
        frames = s0 + 1 + torch.randperm(hi - s0 - 1, generator=g)[:n].sort().values
        starts.append((c, frames))

    rows = []
    with torch.no_grad():
        for c, frames in starts:
            for f0 in frames.tolist():
                f0t = torch.tensor([f0], device=dev)
                S0 = motion.ref_state(f0t)
                xi = torch.zeros(1, 33, device=dev)
                xi[:, 0] = 1.0
                obs0 = proto.obs(S0, torch.zeros(1, 29, device=dev), xi, f0t)
                obs_hist0 = obs0.unsqueeze(1).expand(-1, hist, -1).contiguous()
                # shared obs sequence generated by pi_old in wm
                S = S0
                o_hist = obs_hist0
                f = f0t
                acts_old, acts_new = [], []
                for _ in range(H):
                    g_goal = proto.actor_goal(f, S)
                    a_old_t = pi_old(o_hist, g_goal)
                    acts_old.append(a_old_t)
                    acts_new.append(pi_new(o_hist, g_goal))
                    S = proto.wm_step(wm, S, torch.clamp(a_old_t, -20, 20), normalize_xi(xi))
                    f1 = f + 1
                    o1 = proto.obs(S, a_old_t, xi, f1)
                    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
                    f = f1
                rows.append({
                    "clip": motion.names[c], "frame": f0,
                    "a_old": torch.stack(acts_old, 1).cpu(),
                    "a_new": torch.stack(acts_new, 1).cpu(),
                })

    a_old_seq = torch.cat([r["a_old"] for r in rows], 0)   # (N, H, 29)
    a_new_seq = torch.cat([r["a_new"] for r in rows], 0)
    dirs, norms = normalize_dir_seqs(a_new_seq - a_old_seq), (a_new_seq - a_old_seq).reshape(len(rows), -1).norm(dim=-1)
    out = {
        "meta": {
            "ckpt_old": str(args.ckpt_old), "sha256_old": sha256(args.ckpt_old),
            "ckpt_new": str(args.ckpt_new), "sha256_new": sha256(args.ckpt_new),
            "rollout_wm": str(args.rollout_wm) if args.rollout_wm else str(args.ckpt_old),
            "motion_dir": motion_dir, "goal_frames": K, "hist": hist,
            "horizon": H, "seed": args.seed,
            "clips": [r["clip"] for r in rows], "frames": [r["frame"] for r in rows],
            "note": "shared obs sequence from pi_old in rollout_wm; both policies "
                    "evaluated teacher-forced on the same obs history (open-loop)",
        },
        "delta_actions": (a_new_seq - a_old_seq).cpu(),
        "delta_norms": norms.cpu(),
        "dirs_unit": dirs.cpu(),
        "a_old": a_old_seq.cpu(),
        "a_new": a_new_seq.cpu(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(json.dumps({"out": str(args.out), "n": len(rows), "horizon": H,
                      "delta_norm_mean": float(norms.mean()),
                      "delta_norm_p10": float(torch.quantile(norms.cpu(), 0.1)),
                      "delta_norm_max": float(norms.max())}, indent=1))


if __name__ == "__main__":
    main()
