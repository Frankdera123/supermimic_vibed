"""CRGF world-model-side directional derivative audit (CPU or GPU, no simulator).

For each world-model checkpoint, computes the EXACT autograd directional
derivatives of the imagined policy objective w.r.t. the open-loop action
sequence, decomposed into three paths:

  D_state   dynamics Jacobian x analytic state-mediated reward (+ termination)
  D_direct  direct action-reward path (action-rate; model-free, c detached)
  D_critic  critic bootstrap through the WM state Jacobian at the horizon end
  D_total   full objective; cross residual = D_total - D_state - D_direct - D_critic

Direction sets: random per-step / repeated unit directions, real policy-update
action deltas (from tools/crgf_delta_action_directions.py), and the model's own
steepest-ascent direction of the full objective.

This tool produces the WM side of the fidelity comparison.  The simulator side
(central finite differences of open-loop action sequences) is implemented in
tools/crgf_sim_fd_audit.py and requires GPU authorization.

Usage:
  python tools/crgf_wm_directional_audit.py --wm_ckpt A.pt B.pt [C.pt] \
      --policy_ckpt P.pt [--critic_mode own|shared] [--shared_critic_ckpt S.pt] \
      --horizons 8,32 --directions random,repeated,delta,ascent \
      --delta_dirs_pt delta.pt --out_prefix logs/crgf/wm_audit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

# --proto_root must be resolved BEFORE importing supermimic: the OPCL-proto
# checkpoints (B0/B1/B2) were trained with a different WM (35-dim
# acceleration-level output) and different obs/goal conventions, so this tool
# can run against either code variant.
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

from supermimic import crgf
from _crgf_bootstrap import fix_data_paths

fix_data_paths(Path(__file__).resolve().parents[1])
from supermimic.crgf import (ascent_dirs, base_action_rollout,
                             open_loop_paths, ori_cosine_stats,
                             path_directional_derivatives, random_dir_seqs)
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, normalize_xi
from supermimic.motion import MotionLib
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import default_joint_pos_list


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remap_data_path(path: str, repo_root: Path) -> str:
    if not path:
        return path
    p = Path(path)
    if p.exists():
        return str(p)
    if str(p).startswith("/workspace/supermimic"):
        candidate = repo_root / str(p).replace("/workspace/supermimic/", "")
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"motion_dir {path} not found locally; pass --motion_dir")


def build_mdp(motion_dir, args_ns):
    kin = G1Kinematics(args_ns.device)
    motion = MotionLib.from_dir(motion_dir, kin, args_ns.device)
    kwargs = dict(
        phase_features=bool(args_ns.phase_conditioned_policy),
        clip_features=bool(args_ns.clip_conditioned_policy),
        contact_safe_reference=bool(args_ns.contact_safe_reward),
        contact_masked_reward=bool(args_ns.contact_masked_reward),
        contact_features=bool(args_ns.contact_conditioned_policy),
        current_contact_features=bool(args_ns.current_contact_policy))
    default_q = torch.tensor(default_joint_pos_list(), device=args_ns.device)
    try:
        mdp = MDP(kin, motion, default_q, **kwargs)
    except TypeError:
        # variant MDP without the feature-flag interface (e.g. OPCL proto)
        mdp = MDP(kin, motion, default_q)
    return kin, motion, mdp


def select_starts(motion, clips, states_per_clip, horizon, seed):
    names = motion.names
    if clips:
        sel = [i for i, n in enumerate(names) if any(c in n for c in clips)]
        if not sel:
            raise SystemExit(f"no clip matches {clips}")
    else:
        sel = list(range(len(names)))
    starts = []
    g = torch.Generator().manual_seed(seed)
    for c in sel:
        s0, s1 = int(motion.clip_start[c]), int(motion.clip_end[c])
        hi = max(s0 + 1, s1 - horizon - 1)
        n = min(states_per_clip, hi - s0)
        frames = s0 + 1 + torch.randperm(hi - s0 - 1, generator=g)[:n].sort().values
        starts.append((c, frames.tolist()))
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_ckpt", nargs="+", required=True, type=Path,
                    help="world-model checkpoints (full ckpts or WM-only probes)")
    ap.add_argument("--policy_ckpt", type=Path, required=True,
                    help="checkpoint providing the shared policy pi")
    ap.add_argument("--critic_mode", choices=["own", "shared", "none"], default="own",
                    help="'own': each WM ckpt's own critic if present; "
                         "'shared': one critic from --shared_critic_ckpt for all; "
                         "'none': pure-return objective (no critic path)")
    ap.add_argument("--shared_critic_ckpt", type=Path, default=None)
    ap.add_argument("--motion_dir", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--num_clips", type=int, default=8)
    ap.add_argument("--states_per_clip", type=int, default=4)
    ap.add_argument("--horizons", default="8,32")
    ap.add_argument("--directions", default="random,repeated,delta,ascent")
    ap.add_argument("--delta_dirs_pt", type=Path, default=None)
    ap.add_argument("--gamma", type=float, default=None, help="default: ckpt args")
    ap.add_argument("--tz", type=float, default=0.02)
    ap.add_argument("--tt", type=float, default=0.04)
    ap.add_argument("--batch", type=int, default=8, help="starts per autograd batch")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--proto_root", default=None,
                    help="path to a supermimic package variant (e.g. the OPCL "
                         "proto) whose modules evaluate these checkpoints")
    ap.add_argument("--out_prefix", required=True, type=Path)
    ap.add_argument("--save_base_actions", type=Path, default=None,
                    help="save the shared base action sequences so the simulator-side "
                         "FD audit (crgf_sim_fd_audit.py) perturbs exactly the same actions")
    ap.add_argument("--frames_pt", type=Path, default=None,
                    help="explicit (clip_name, frame) start list, e.g. saved by a "
                         "dense verification run; overrides --clips selection")
    ap.add_argument("--action_gain", type=float, default=1.0,
                    help="open-loop equivalent of wm_action_effect_gain: WM input "
                         "actions are scaled by this factor; rewards keep the "
                         "original actions (Route A gain-response sweep)")
    ap.add_argument("--save_dirs", type=Path, default=None,
                    help="save the unit direction sequences per set/model so the "
                         "simulator-side FD audit evaluates the SAME directions "
                         "(ascent sets are tagged ascent_<ckpt-stem>)")
    ap.add_argument("--seed", type=int, default=0)
    # MDP feature flags (defaults mirror train.py; override per checkpoint config)
    ap.add_argument("--phase_conditioned_policy", action="store_true")
    ap.add_argument("--clip_conditioned_policy", action="store_true")
    ap.add_argument("--contact_safe_reward", action="store_true")
    ap.add_argument("--contact_masked_reward", action="store_true")
    ap.add_argument("--contact_conditioned_policy", action="store_true")
    ap.add_argument("--current_contact_policy", action="store_true")
    args = ap.parse_args()
    dev = args.device
    repo_root = Path(__file__).resolve().parents[1]
    horizons = [int(h) for h in args.horizons.split(",")]
    direction_sets = args.directions.split(",")
    t_start = time.time()

    pol_ck = torch.load(args.policy_ckpt, map_location=dev, weights_only=False)
    a_pol = pol_ck.get("args", {})
    hist = int(a_pol.get("hist", 10))
    gamma = args.gamma if args.gamma is not None else float(a_pol.get("gamma", 0.99))
    motion_dir = args.motion_dir or remap_data_path(a_pol.get("motion_dir", "data/lafan1_g1"), repo_root)

    kin, motion, mdp = build_mdp(motion_dir, args)
    import importlib

    motion_mod = importlib.import_module("supermimic.motion")
    mdp_mod = importlib.import_module("supermimic.mdp")
    # the proto variant hardcodes GOAL_FRAMES/GOAL_STRIDE in motion.py
    is_proto_variant = hasattr(motion_mod, "GOAL_STRIDE")
    K = (int(motion_mod.GOAL_FRAMES) if is_proto_variant
         else int(a_pol.get("goal_frames", 10)))
    v_max = (mdp_mod.MAX_STEP_REWARD / (1.0 - gamma)) if is_proto_variant else None
    proto = crgf.Protocol(mdp, K, v_max=v_max)

    # goal / critic-input dims probed through the protocol adapter (works for
    # both the frame-index goal and the proto state-relative goal)
    f_probe = motion.clip_start[:2]
    S_probe = motion.ref_state(f_probe)
    xi_probe = torch.zeros(2, 33, device=dev)
    xi_probe[:, 0] = 1.0
    goal_dim = int(proto.actor_goal(f_probe, S_probe).shape[-1])
    cin_dim = int(proto.critic_input(S_probe, f_probe, normalize_xi(xi_probe),
                                     proto.wm_goal(f_probe, S_probe)).shape[-1])

    pi = Policy(OBS_DIM, goal_dim, hist=hist).to(dev)
    pi.load_state_dict(pol_ck["pi"])
    pi.eval()

    def load_critic(ck):
        if "critic" not in ck or ck["critic"] is None:
            return None
        cr = Critic(cin_dim).to(dev)
        cr.load_state_dict(ck["critic"])
        cr.eval()
        return cr

    shared_critic = None
    if args.critic_mode == "shared":
        if args.shared_critic_ckpt is None:
            ap.error("--critic_mode shared requires --shared_critic_ckpt")
        shared_critic = load_critic(torch.load(args.shared_critic_ckpt, map_location=dev,
                                               weights_only=False))

    if args.frames_pt:
        fr = torch.load(args.frames_pt, map_location=dev, weights_only=False)
        name_to_idx = {n: i for i, n in enumerate(motion.names)}
        rows = [(name_to_idx[c], int(f)) for c, f in fr["pairs"]]
        starts = [(c, [f]) for c, f in rows]
    else:
        starts = select_starts(motion, args.clips, args.states_per_clip, max(horizons), args.seed)
        rows = [(c, f) for c, fs in starts for f in fs]

    delta_data = None
    if "delta" in direction_sets:
        if args.delta_dirs_pt is None:
            ap.error("--directions delta requires --delta_dirs_pt")
        delta_data = torch.load(args.delta_dirs_pt, map_location=dev, weights_only=False)

    base_actions_all = {}
    base_starts_all = []
    dirs_bookkeeping = {}
    out_rows = []
    summary = {"protocol": {
        "gamma": gamma, "tz": args.tz, "tt": args.tt, "goal_frames": K, "hist": hist,
        "motion_dir": motion_dir, "horizons": horizons, "direction_sets": direction_sets,
        "starts": [{"clip": motion.names[c], "frames": fs} for c, fs in starts],
        "policy_ckpt": str(args.policy_ckpt), "policy_sha256": sha256(args.policy_ckpt),
        "critic_mode": args.critic_mode,
        "shared_critic_ckpt": str(args.shared_critic_ckpt) if args.shared_critic_ckpt else None,
        "note": "WM-side exact autograd directional derivatives; open-loop action "
                "sequences; soft termination; obs histories repeat the start frame "
                "(synthetic step 0, matching the 2026-11-18 OPCL pilot protocol)",
    }, "models": []}

    for wm_path in args.wm_ckpt:
        ck = torch.load(wm_path, map_location=dev, weights_only=False)
        wm_st = ck["wm"]
        hidden = int(wm_st["residual_net.0.weight"].shape[0]) if "residual_net.0.weight" in wm_st else 0
        try:
            wm = WorldModel(residual_hidden=hidden).to(dev)
        except TypeError:
            wm = WorldModel().to(dev)  # variant WM without residual support
            hidden = 0
        wm.load_state_dict(wm_st)
        wm.eval()
        critic = None
        if args.critic_mode == "own":
            critic = load_critic(ck)
        elif args.critic_mode == "shared":
            critic = shared_critic
        base_wm = wm  # base action sequence generated in this WM

        for H in horizons:
            # ---- shared base action sequence for this horizon ----
            A_base, meta_starts = [], []
            for i0 in range(0, len(rows), args.batch):
                chunk = rows[i0:i0 + args.batch]
                f0 = torch.tensor([f for _, f in chunk], device=dev)
                S0 = motion.ref_state(f0)
                xi = torch.zeros(len(chunk), 33, device=dev)
                xi[:, 0] = 1.0
                obs0 = proto.obs(S0, torch.zeros(len(chunk), 29, device=dev), xi, f0)
                obs_hist0 = obs0.unsqueeze(1).expand(-1, hist, -1).contiguous()
                acts = base_action_rollout(wm, pi, mdp, S0, f0, xi, normalize_xi(xi),
                                           obs_hist0, K, H)
                A_base.append(acts)
                meta_starts += [(motion.names[c], f) for c, f in chunk]
            A_base = torch.cat(A_base, 0)  # (N, H, 29)
            N = A_base.shape[0]
            base_actions_all[H] = A_base.detach().cpu()
            if not base_starts_all:
                base_starts_all = meta_starts

            # ---- gradients (chunked) ----
            grads = {k: [] for k in ("state", "direct", "critic", "total", "cross")}
            diag = {"J_return": [], "J_total": [], "ori_clamp": [], "hard_done_any": []}
            for i0 in range(0, N, args.batch):
                f0 = torch.tensor([f for _, f in meta_starts[i0:i0 + args.batch]], device=dev)
                S0 = motion.ref_state(f0)
                xi = torch.zeros(len(f0), 33, device=dev)
                xi[:, 0] = 1.0
                xi_n = normalize_xi(xi)
                a_seq = A_base[i0:i0 + args.batch].clone().requires_grad_(True)
                paths = open_loop_paths(wm, critic, mdp, S0, f0, a_seq,
                                        torch.zeros(len(f0), 29, device=dev, dtype=a_seq.dtype),
                                        xi_n, gamma=gamma, tz=args.tz, tt=args.tt,
                                        goal_frames=K, record_states=True, v_max=v_max,
                                        action_gain=args.action_gain)
                g = path_directional_derivatives(paths, a_seq)
                for k in grads:
                    grads[k].append(g[k].detach())
                diag["J_return"].extend(paths["J_return_vec"].tolist())
                diag["J_total"].extend(paths["J_total_vec"].tolist())
                # orientation acos-clamp activity on the baseline trajectory
                clamp_per = torch.zeros(len(f0), dtype=torch.bool)
                with torch.no_grad():
                    for t, St in enumerate(paths["states"]):
                        f1 = f0 + t + 1
                        _, hi_flag, lo_flag = ori_cosine_stats(mdp, St, f1)
                        clamp_per |= (hi_flag | lo_flag)
                diag["ori_clamp"].extend(clamp_per.tolist())
                diag["hard_done_any"].extend((paths["hard_done"].any(0)).tolist())

            grads = {k: torch.cat(v, 0) for k, v in grads.items()}  # (N, H*29)
            NH = H * 29
            dirs = {}
            if "random" in direction_sets:
                dirs["random"] = random_dir_seqs(N, H, 29, seed=args.seed + 1,
                                                 mode="per_step", device=dev)
            if "repeated" in direction_sets:
                dirs["repeated"] = random_dir_seqs(N, H, 29, seed=args.seed + 2,
                                                   mode="repeated", device=dev)
            if "delta" in direction_sets and delta_data is not None:
                dd = delta_data["dirs_unit"]
                if dd.shape[1] >= H:
                    ddH = crgf.normalize_dir_seqs(dd[:, :H].contiguous())
                else:
                    # pad the shorter real-update direction with zeros on the
                    # tail steps and renormalize (same convention as the sim side)
                    ddH = crgf.normalize_dir_seqs(
                        torch.cat([dd, torch.zeros(dd.shape[0], H - dd.shape[1],
                                                   dd.shape[2], device=dd.device)], 1).contiguous())
                if ddH.shape[0] >= N:
                    dirs["delta"] = ddH[:N].to(dev)
                else:
                    reps = (N + ddH.shape[0] - 1) // ddH.shape[0]
                    dirs["delta"] = ddH.repeat(reps, 1, 1)[:N].to(dev)
            if "ascent" in direction_sets:
                dirs["ascent"] = ascent_dirs(grads["total"], N, H, 29)

            for dname, dseq in dirs.items():
                dirs_bookkeeping.setdefault(
                    "ascent" if dname == "ascent" else dname, {})[str(H)] = dseq.detach().cpu()
                model_tag = Path(wm_path).parent.name or Path(wm_path).stem
                dirs_bookkeeping.setdefault(f"ascent_{model_tag}", {})
                if dname == "ascent":
                    dirs_bookkeeping[f"ascent_{model_tag}"][str(H)] = dseq.detach().cpu()
                proj = {k: (grads[k] * dseq.reshape(N, -1)).sum(-1) for k in grads}
                for idx, (clip_name, frame) in enumerate(meta_starts):
                    out_rows.append({
                        "model": str(wm_path), "horizon": H, "direction_set": dname,
                        "clip": clip_name, "frame": frame,
                        "D_state": float(proj["state"][idx]),
                        "D_direct": float(proj["direct"][idx]),
                        "D_critic": float(proj["critic"][idx]),
                        "D_total": float(proj["total"][idx]),
                        "cross_residual": float(proj["cross"][idx]),
                        "J_return": diag["J_return"][idx],
                        "J_total": diag["J_total"][idx],
                        "ori_clamp_frac": diag["ori_clamp"][idx],
                        "hard_done_any": diag["hard_done_any"][idx],
                    })
            model_summary = {
                "wm_ckpt": str(wm_path), "sha256": sha256(wm_path),
                "iteration": ck.get("iteration"), "action_gain": args.action_gain,
                "residual_branch": hidden > 0,
                "critic_source": (str(args.shared_critic_ckpt) if args.critic_mode == "shared"
                                  else ("own" if critic is not None else "none")),
                "per_horizon": {},
            }
            hs = {"J_return_mean": sum(diag["J_return"]) / len(diag["J_return"]),
                  "J_total_mean": sum(diag["J_total"]) / len(diag["J_total"]),
                  "ori_clamp_frac": sum(1.0 for x in diag["ori_clamp"] if x) / len(diag["ori_clamp"]),
                  "hard_done_any_frac": sum(1.0 for x in diag["hard_done_any"] if x) / len(diag["hard_done_any"]),
                  "grad_norm": {k: float(grads[k].norm()) for k in grads},
                  "n_direction_sets": {k: int(v.shape[0]) for k, v in dirs.items()}}
            model_summary["per_horizon"][str(H)] = hs
            summary["models"].append(model_summary)
            del grads, dirs

    if args.save_dirs:
        args.save_dirs.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"horizons": horizons, "note": "unit direction sequences "
                    "(N, H, 29); ascent_* keys are per-model", **dirs_bookkeeping},
                   args.save_dirs)
    if args.save_base_actions:
        args.save_base_actions.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"base_actions": base_actions_all, "starts": base_starts_all,
                    "horizons": horizons, "gamma": gamma, "goal_frames": K, "hist": hist,
                    "policy_ckpt": str(args.policy_ckpt),
                    "policy_sha256": sha256(args.policy_ckpt),
                    "note": "shared base action sequences per horizon; keys are horizon ints"},
                   args.save_base_actions)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out_prefix.with_suffix(".json")
    csv_path = args.out_prefix.with_suffix(".csv")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=1)
    if out_rows:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path),
                      "rows": len(out_rows), "elapsed_s": round(time.time() - t_start, 1),
                      "models": [m["wm_ckpt"] for m in summary["models"]]}, indent=1))


if __name__ == "__main__":
    main()
