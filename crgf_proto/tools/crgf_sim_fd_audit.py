"""CRGF simulator-side central finite differences -- v2 sequential-branch design.

GPU REQUIRED (Isaac Lab / MuJoCo-Warp), deterministic mode.  Phase 0 smoke
(results/crgf/phase0_smoke.json) established:
  * NewtonCfg.deterministic_mode="run_to_run" + disable_sensors=True gives
    BIT-EXACT repeat runs at <=4 envs per context (repeat_range = 0.0);
  * >=8 envs crashes on warp's int32 workspace limit (linear in env count),
    so the v1 parallel-branch wave design is replaced by v2:

  ONE SimulationContext with --envs (default 4) reused for every triplet (no
  scene rebuild); in each wave, every env sequentially executes ONE triplet's
  plus/minus/base branches, resetting between branches.  Under deterministic
  mode the branch ordering cannot contaminate results (verified in Phase 0
  and independently by the OPCL reproducibility fix).

    D_sim(path, d) = [J_sim^(path)(A + eps*d) - J_sim^(path)(A - eps*d)] / (2*eps)

with paths decomposed by the SAME reward_split as the WM side:
  J_state   discounted soft-termination-masked state-mediated reward
  J_direct  discounted action-rate term
  J_critic  gamma^H * c_{H-1} * V(s_H) on REAL simulator states per branch
            (ground truth for the critic path; requires --critic_ckpt)

D_sim is model-independent; per-model D_WM comes from the WM-side audits.

Fail-closed gate (crgf.reliability_gate) per (H, direction_set, start) over
pooled eps x repeats; per-eps means are recorded for failure attribution.

Run ONLY with an explicit Phase 1/2 GPU grant (PHASE1_EXECUTION_DECISION.md).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_ckpt", required=True, type=Path)
    ap.add_argument("--wm_ckpt", type=Path, required=True,
                    help="world model used to roll the base actions (read-only)")
    ap.add_argument("--critic_ckpt", type=Path, default=None)
    ap.add_argument("--proto_root", default=None)
    ap.add_argument("--motion_dir", default="data/lafan1_g1")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--num_clips", type=int, default=8)
    ap.add_argument("--states_per_clip", type=int, default=2)
    ap.add_argument("--horizons", default="8,32,64")
    ap.add_argument("--direction_sets", default="random,repeated,delta,ascent")
    ap.add_argument("--dirs_pt", nargs="*", default=None,
                    help="direction tensor files from crgf_wm_directional_audit "
                         "--save_dirs (ascent_<model>.pt); delta comes from "
                         "--delta_dirs_pt")
    ap.add_argument("--delta_dirs_pt", type=Path, default=None)
    ap.add_argument("--eps", default="0.01,0.03,0.1")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tz", type=float, default=0.02)
    ap.add_argument("--tt", type=float, default=0.04)
    ap.add_argument("--envs", type=int, default=4,
                    help="<= Phase 0 deterministic limit (4)")
    ap.add_argument("--max_minutes", type=float, default=110.0)
    ap.add_argument("--out_prefix", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=17)
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()
    dev = "cuda:0"
    repo_root = here.parent
    from _crgf_bootstrap import install as _install_stub, fix_data_paths

    _install_stub()
    if args.proto_root:
        sys.path.insert(0, args.proto_root)
        import importlib

        importlib.invalidate_caches()
        for m in [k for k in sys.modules if k == "supermimic" or k.startswith("supermimic.")]:
            del sys.modules[m]
    fix_data_paths(repo_root)

    horizons = [int(h) for h in args.horizons.split(",")]
    eps_values = [float(e) for e in args.eps.split(",")]
    direction_sets = args.direction_sets.split(",")
    ledger = args.out_prefix.with_suffix(".budget.jsonl")
    t0 = time.time()

    def log(event, **kw):
        with open(ledger, "a") as fh:
            fh.write(json.dumps({"t": round(time.time() - t0, 1), "event": event, **kw}) + "\n")

    stop_reason = []

    def stop(reason):
        log("STOP", reason=reason)
        stop_reason.append(reason)

    from supermimic.crgf import (Protocol, base_action_rollout, random_dir_seqs,
                                 reliability_gate, normalize_dir_seqs, reward_split,
                                 soft_term)
    from supermimic.env import TrackingSim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP, OBS_DIM, normalize_xi
    from supermimic.motion import MotionLib
    from supermimic.networks import Critic, Policy, WorldModel
    from supermimic.robot import default_joint_pos_list

    pol_ck = torch.load(args.policy_ckpt, map_location=dev, weights_only=False)
    a_pol = pol_ck.get("args", {})
    hist = int(a_pol.get("hist", 10))
    gamma = args.gamma
    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(args.motion_dir, kin, dev)
    default_q = torch.tensor(default_joint_pos_list(), device=dev)
    try:
        mdp = MDP(kin, motion, default_q,
                  phase_features=bool(a_pol.get("phase_conditioned_policy", False)),
                  clip_features=bool(a_pol.get("clip_conditioned_policy", False)),
                  contact_masked_reward=bool(a_pol.get("contact_masked_reward", False)))
    except TypeError:
        mdp = MDP(kin, motion, default_q)
    proto = Protocol(mdp, int(a_pol.get("goal_frames", 10)))
    f_probe = motion.clip_start[:2]
    S_probe = motion.ref_state(f_probe)
    xi_probe = torch.zeros(2, 33, device=dev)
    xi_probe[:, 0] = 1.0
    goal_dim = int(proto.actor_goal(f_probe, S_probe).shape[-1])
    pi = Policy(OBS_DIM, goal_dim, hist=hist).to(dev)
    pi.load_state_dict(pol_ck["pi"])
    pi.eval()

    critic = None
    if args.critic_ckpt is not None:
        ck = torch.load(args.critic_ckpt, map_location=dev, weights_only=False)
        vin_probe = proto.critic_input(S_probe, f_probe, normalize_xi(xi_probe),
                                       proto.wm_goal(f_probe, S_probe))
        critic = Critic(vin_probe.shape[-1]).to(dev)
        critic.load_state_dict(ck["critic"])
        critic.eval()

    # ---- starts (identical protocol to the WM-side audit for exact pairing) ----
    names = motion.names
    sel = ([i for i, n in enumerate(names) if any(c in n for c in args.clips)]
           if args.clips else list(range(len(names)))[: args.num_clips])
    g0 = torch.Generator().manual_seed(args.seed)
    # FIXED 64-frame margin so the start frames match the WM-side audits
    # exactly (which select with the largest horizon of their protocol)
    starts = []
    for c in sel:
        s0, s1 = int(motion.clip_start[c]), int(motion.clip_end[c])
        hi = max(s0 + 1, s1 - 64 - 1)
        n = min(args.states_per_clip, hi - s0)
        frames = s0 + 1 + torch.randperm(hi - s0 - 1, generator=g0)[:n].sort().values
        starts.append((c, frames.tolist()))
    meta_starts = [(c, f) for c, fs in starts for f in fs]
    N = len(meta_starts)
    f0_all = torch.tensor([f for _, f in meta_starts], device=dev)
    log("protocol", n_starts=N, horizons=horizons, eps=eps_values,
        direction_sets=direction_sets, repeats=args.repeats, envs=args.envs)

    # ---- base actions per horizon (rolled in the given WM; shared everywhere) ----
    ck_wm = torch.load(args.wm_ckpt, map_location=dev, weights_only=False)
    wm = WorldModel().to(dev)
    wm.load_state_dict(ck_wm["wm"])
    wm.eval()
    base_by_h = {}
    for H in horizons:
        S0 = motion.ref_state(f0_all)
        xi = torch.zeros(N, 33, device=dev)
        xi[:, 0] = 1.0
        obs0 = proto.obs(S0, torch.zeros(N, 29, device=dev), xi, f0_all)
        obs_hist0 = obs0.unsqueeze(1).expand(-1, hist, -1).contiguous()
        base_by_h[H] = base_action_rollout(wm, pi, mdp, S0, f0_all, xi,
                                           normalize_xi(xi), obs_hist0, proto.K, H)
        log("base_actions", horizon=H)
    del wm

    # ---- direction bookkeeping ----
    delta_data = (torch.load(args.delta_dirs_pt, map_location=dev, weights_only=False)
                  if args.delta_dirs_pt else None)
    dirs_files = {}  # tag -> {str(H): unit dirs (N,H,29)}
    for p in (args.dirs_pt or []):
        d = torch.load(p, map_location=dev, weights_only=False)
        for k, v in d.items():
            if k.startswith("ascent_") and isinstance(v, dict):
                dirs_files[k] = v

    def get_dirs(H, dset):
        """(N, H, 29) unit direction sequences for one direction set."""
        if dset == "random":
            return random_dir_seqs(N, H, 29, seed=args.seed + 1, device=dev)
        if dset == "repeated":
            return random_dir_seqs(N, H, 29, seed=args.seed + 2, mode="repeated", device=dev)
        if dset == "delta":
            if delta_data is None:
                return None
            dd = delta_data["dirs_unit"]
            if dd.shape[1] >= H:
                dd = dd[:, :H]
            else:  # pad the shorter update direction with zeros, renormalize
                dd = torch.cat([dd, torch.zeros(dd.shape[0], H - dd.shape[1], 29,
                                                device=dd.device)], 1)
            dd = normalize_dir_seqs(dd)
            reps = (N + dd.shape[0] - 1) // dd.shape[0]
            return dd.repeat(reps, 1, 1)[:N].to(dev)
        # ascent_<model> tags come from --dirs_pt files ({str(H): tensor})
        if dset in dirs_files:
            dirs = dirs_files[dset][str(H)].to(dev)
            if dirs.shape[0] >= N:
                return dirs[:N]
            reps = (N + dirs.shape[0] - 1) // dirs.shape[0]
            return dirs.repeat(reps, 1, 1)[:N]
        return None

    from isaaclab.app import launch_simulation
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    import isaaclab.sim as sim_utils

    def det_cfg():
        return sim_utils.SimulationCfg(
            dt=0.02, device=dev,
            physics=NewtonCfg(
                deterministic_mode="run_to_run",
                solver_cfg=MJWarpSolverCfg(njmax=500, nconmax=128, cone="pyramidal",
                                           integrator="implicitfast",
                                           use_mujoco_contacts=True,
                                           disable_sensors=True),
                num_substeps=4))

    A = 29
    out_rows = []

    # triplet schedule: (H, dset, start_idx, eps, rep)
    schedule = []
    for H in horizons:
        for dset in direction_sets:
            if get_dirs(H, dset) is None:
                log("skip_direction_set", horizon=H, dset=dset)
                continue
            for n in range(N):
                for eps in eps_values:
                    for rep in range(args.repeats):
                        schedule.append((H, dset, n, eps, rep))
    log("schedule", triplets=len(schedule))
    stop_secs = args.max_minutes * 60

    sim_args = SimpleNamespace(num_envs=args.envs)
    with launch_simulation(det_cfg(), args):
        env = TrackingSim(det_cfg(), args.envs, motion, mdp, gamma=gamma,
                          eval_mode=True, reset_noise=False)
        xi_one = torch.zeros(33, device=dev)
        xi_one[0] = 1.0
        xi_n_one = normalize_xi(xi_one.unsqueeze(0)).squeeze(0)

        def run_branch(env_id, frame, act_seq):
            """One branch of H prescribed steps in env env_id (reset first)."""
            env.reset(torch.tensor([env_id], device=dev),
                      torch.tensor([frame], device=dev))
            env.xi[env_id] = xi_one
            env.refresh()
            a_prev = torch.zeros(1, A, device=dev)
            c_prev = torch.ones(1, device=dev)
            J_state = 0.0
            J_direct = 0.0
            S_last = f_last = None
            H = act_seq.shape[0]
            for t in range(H):
                a = act_seq[t:t + 1]
                env.step(a)
                f1 = env.f[env_id:env_id + 1].clone()
                S1 = tuple(x[env_id:env_id + 1] for x in env.state())
                r_s, r_d = reward_split(mdp, S1, f1, a, a_prev)
                d_soft = soft_term(mdp, S1, f1, tz=args.tz, tt=args.tt)
                c_prev = c_prev * (1 - d_soft)
                J_state += float((gamma ** t * c_prev * r_s)[0])
                J_direct += float((gamma ** t * c_prev * r_d)[0])
                S_last, f_last = S1, f1
                a_prev = a
            J_critic = None
            if critic is not None:
                gw = proto.wm_goal(f_last, S_last)
                vin = proto.critic_input(S_last, f_last, xi_n_one.unsqueeze(0), gw)
                J_critic = float(((gamma ** H) * c_prev * critic(vin))[0])
            return J_state, J_direct, J_critic

        wave_i = 0
        # group the schedule by (H, dset): within a group all triplets share the
        # horizon, so a wave of --envs triplets runs its plus/minus/base phases
        # as BATCHED env steps (4-env determinism verified in Phase 0)
        from collections import defaultdict as _dd

        sched_groups = _dd(list)
        for (H, dset, n, eps, rep) in schedule:
            sched_groups[(H, dset)].append((n, eps, rep))
        for (H, dset), items in sorted(sched_groups.items()):
            if stop_reason:
                break
            dirs = get_dirs(H, dset)
            if dirs is None:
                log("skip_direction_set", horizon=H, dset=dset)
                continue
            A_base = base_by_h[H]
            for c0 in range(0, len(items), args.envs):
                if (time.time() - t0) > stop_secs:
                    stop(f"wall-clock budget {args.max_minutes} min reached")
                    break
                chunk = items[c0:c0 + args.envs]
                E = len(chunk)
                acts = torch.zeros(E, 3, H, A, device=dev)
                for i, (n, eps, rep) in enumerate(chunk):
                    acts[i, 0] = A_base[n] + eps * dirs[n]
                    acts[i, 1] = A_base[n] - eps * dirs[n]
                    acts[i, 2] = A_base[n]
                if E < args.envs:
                    # the sim context has a fixed env count; pad trailing envs
                    # with a repeated triplet whose results are discarded
                    pad = acts[:1].expand(args.envs - E, 3, H, A)
                    acts = torch.cat([acts, pad.clone()], 0)
                frames = torch.tensor([meta_starts[n][1] for n, _, _ in chunk], device=dev)
                res = {}
                for b, bname in enumerate(("plus", "minus", "base")):
                    env.reset(torch.arange(E, device=dev), frames)
                    env.xi[:E] = 0
                    env.xi[:E, 0] = 1.0
                    env.refresh()
                    a_prev = torch.zeros(args.envs, A, device=dev)
                    c_prev = torch.ones(args.envs, device=dev)
                    J_state = torch.zeros(args.envs, device=dev)
                    J_direct = torch.zeros(args.envs, device=dev)
                    S_last = f_last = None
                    for t in range(H):
                        a = acts[:, b, t]
                        env.step(a)
                        f1 = env.f.clone()
                        S1 = env.state()
                        r_s, r_d = reward_split(mdp, S1, f1, a, a_prev)
                        d_soft = soft_term(mdp, S1, f1, tz=args.tz, tt=args.tt)
                        c_prev = c_prev * (1 - d_soft)
                        J_state += (gamma ** t) * c_prev * r_s
                        J_direct += (gamma ** t) * c_prev * r_d
                        S_last, f_last = S1, f1
                        a_prev = a
                    J_critic = None
                    if critic is not None:
                        gw = proto.wm_goal(f_last, S_last)
                        vin = proto.critic_input(S_last, f_last,
                                                 normalize_xi(env.xi), gw)
                        J_critic = (gamma ** H) * c_prev * critic(vin)
                    res[bname] = (J_state, J_direct, J_critic)
                for i, (n, eps, rep) in enumerate(chunk):
                    clip_name, frame = motion.names[meta_starts[n][0]], meta_starts[n][1]
                    row = {"horizon": H, "direction_set": dset, "clip": clip_name,
                           "frame": frame, "eps": eps, "rep": rep,
                           "J_base_state": float(res["base"][0][i]),
                           "J_base_direct": float(res["base"][1][i]),
                           "J_base_total": float(res["base"][0][i] + res["base"][1][i]
                                                 + (res["base"][2][i] if res["base"][2] is not None else 0.0))}
                    for pi_, p in enumerate(("state", "direct", "critic")):
                        jp = res["plus"][pi_]
                        jm = res["minus"][pi_]
                        row[f"D_sim_{p}"] = (None if jp is None
                                             else float((jp[i] - jm[i]) / (2 * eps)))
                    out_rows.append(row)
                wave_i += 1
                if wave_i % 50 == 0:
                    log("progress", waves=wave_i, rows=len(out_rows),
                        mem_gb=round(torch.cuda.memory_reserved() / 2**30, 2))
        log("waves_done", waves=wave_i, rows=len(out_rows))

    # ---- fail-closed gate per (H, dset, start) over pooled eps x reps ----
    from collections import defaultdict

    groups = defaultdict(list)
    for r in out_rows:
        groups[(r["horizon"], r["direction_set"], r["clip"], r["frame"])].append(r)
    for key, rs in groups.items():
        base = [r["J_base_total"] for r in rs]
        for p in ("state", "direct", "critic"):
            vals = [r[f"D_sim_{p}"] for r in rs if r[f"D_sim_{p}"] is not None]
            if len(vals) >= 2:
                ok, why = reliability_gate(torch.tensor(base), torch.tensor(vals))
            else:
                ok, why = False, "insufficient_repeats"
            for r in rs:
                r[f"gate_{p}"] = ok
                r[f"gate_reason_{p}"] = why
            for e in eps_values:
                ev = [r[f"D_sim_{p}"] for r in rs
                      if r["eps"] == e and r[f"D_sim_{p}"] is not None]
                r[f"D_mean_{p}_eps{e}"] = (sum(ev) / len(ev)) if ev else None

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_prefix.with_suffix(".csv")
    if out_rows:
        fields = sorted({k for r in out_rows for k in r})
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
    gated = sum(1 for r in out_rows if r.get("gate_state"))
    log("done", rows=len(out_rows), gated=gated)
    print(json.dumps({"csv": str(csv_path), "rows": len(out_rows),
                      "gated_state_ok": gated,
                      "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
