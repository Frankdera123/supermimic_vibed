"""PADA-0 simulator collection: probe branches (both arms) + shared eval set.

GPU REQUIRED (Isaac Lab / MuJoCo-Warp, deterministic mode, <=4 envs) -- same
constraints and wave design as crgf_proto/tools/crgf_sim_fd_audit.py (Phase 0:
4 envs bit-exact repeats, >=8 envs int32 workspace limit).

One sim session records, for every prescribed open-loop branch, the PER-STEP
trajectory (s1 flat 71, f, a, a_prev, r_state, r_direct, soft_term, hard_done)
so that ALL downstream metrics are computed offline:
  * probe transitions -> M1/M2 fine-tuning data (per arm);
  * eval plus/minus trajectories -> Level-2 state FD + Level-3 return FD
    (H=32 full and H=8 truncated sums), model-independent ground truth;
  * eval base branches (2 reps) -> gate baseline + Level-1 free-running eval.

Fairness (protocol §2): the two probe arms share identical starts, base
actions, eps, branch count, H, sim config and seeds; ONLY the direction set
differs.  Direction sets come from pada_gen_directions.py outputs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
sys_path_fixed = False


def _fix_sys_path():
    global sys_path_fixed
    if not sys_path_fixed:
        import sys

        sys.path.insert(0, str(_HERE.parent))
        sys_path_fixed = True


_fix_sys_path()
import pada_bridge as bridge  # noqa: E402

bridge.import_stack()
bridge.fix_paths()

import torch  # noqa: E402

from supermimic.crgf import Protocol, base_action_rollout, reward_split, soft_term  # noqa: E402
from supermimic.mdp import normalize_xi  # noqa: E402


def collect_wave(env, mdp, proto, items, H, gamma, tz, tt):
    """items: list of dicts {env_frame, act_seq (H,29), key}; returns {key: traj}.

    Each env executes its own branch concurrently (CRGF wave pattern: reset
    first, deterministic mode makes branch order irrelevant)."""
    from supermimic.mdp import flatten_state

    E = len(items)
    out = {}
    dev = items[0]["act_seq"].device
    env.reset(torch.arange(E, device=dev),
              torch.tensor([it["env_frame"] for it in items], device=dev))
    env.xi[:E] = 0
    env.xi[:E, 0] = 1.0
    env.refresh()
    S0 = tuple(x[:E] for x in env.state())
    s0_flat = flatten_state(S0).detach().cpu()                   # (E, 71)
    recs = [{k: [] for k in ("f", "r_state", "r_direct", "soft_d", "done")}
            for _ in range(E)]
    s1_steps = []
    a = torch.stack([it["act_seq"] for it in items], 0)          # (E, H, 29)
    a_prev = torch.zeros(E, 29, device=dev)
    for t in range(H):
        env.step(a[:, t])
        f1 = env.f.clone()
        S1 = env.state()
        s1_steps.append(flatten_state(S1).detach().cpu())        # (E, 71)
        r_s, r_d = reward_split(mdp, S1, f1, a[:, t], a_prev)
        d_soft = soft_term(mdp, S1, f1, tz=tz, tt=tt)
        hard = mdp.termination(S1, f1)
        for i in range(E):
            recs[i]["f"].append(int(f1[i]))
            recs[i]["r_state"].append(float(r_s[i]))
            recs[i]["r_direct"].append(float(r_d[i]))
            recs[i]["soft_d"].append(float(d_soft[i]))
            recs[i]["done"].append(bool(hard[i]))
        a_prev = a[:, t].clone()
    s1_all = torch.stack(s1_steps, 1)                            # (E, H, 71)
    for i, it in enumerate(items):
        rec = recs[i]
        out[it["key"]] = {
            "s0": s0_flat[i],
            "s1": s1_all[i],
            "f": torch.tensor(rec["f"]),
            "a": it["act_seq"].detach().cpu(),
            "a_prev": torch.cat((torch.zeros(1, 29), it["act_seq"][:-1].cpu()), 0),
            "r_state": torch.tensor(rec["r_state"]),
            "r_direct": torch.tensor(rec["r_direct"]),
            "soft_d": torch.tensor(rec["soft_d"]),
            "done": torch.tensor(rec["done"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tz", type=float, default=0.02)
    ap.add_argument("--tt", type=float, default=0.04)
    ap.add_argument("--eps_probe", type=float, default=1.0)
    ap.add_argument("--eps_eval", default="0.03,0.1")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--base_reps", type=int, default=2)
    ap.add_argument("--max_minutes", type=float, default=75.0)
    ap.add_argument("--skip_probes", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--train_run", default="online_B0")
    ap.add_argument("--train_iter", type=int, default=2100)
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()
    dev = getattr(args, "device", None) or "cuda:0"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ledger = args.out_dir / "collect.budget.jsonl"

    def log(event, **kw):
        with open(ledger, "a") as fh:
            fh.write(json.dumps({"t": round(time.time() - t0, 1), "event": event, **kw}) + "\n")

    stop_reason = []

    def stop(reason):
        log("STOP", reason=reason)
        stop_reason.append(reason)

    meta = json.load(open(args.dirs_dir / "directions_meta.json"))
    H = int(meta["horizon"])
    eps_eval = [float(e) for e in args.eps_eval.split(",")]
    probe_starts = meta["probe_starts"]
    eval_starts = meta["eval_starts"]
    S_p, S_e = len(probe_starts), len(eval_starts)
    log("protocol", S_probe=S_p, S_eval=S_e, H=H, eps_probe=args.eps_probe,
        eps_eval=eps_eval, envs=args.envs, gamma=args.gamma)

    pair = bridge.load_pair_bundle(args.train_run, args.train_iter, device=dev)
    world = bridge.setup_world(device=dev, pair=pair)
    motion, mdp, proto = world["motion"], world["mdp"], world["proto"]
    pi_old, wm0 = pair["pi_old"], pair["wm"]
    hist = pair["hist"]

    def probe_frame(si):
        return probe_starts[si]["frame"]

    def eval_frame(si):
        return eval_starts[si]["frame"]

    # ---- base actions: pi_old closed-loop in M0 (shared across everything) ----
    base_probe = base_eval = None
    if not args.skip_probes:
        f_all = torch.tensor([probe_frame(i) for i in range(S_p)], device=dev)
        S0 = motion.ref_state(f_all)
        xi = torch.zeros(S_p, 33, device=dev)
        xi[:, 0] = 1.0
        obs0 = proto.obs(S0, torch.zeros(S_p, 29, device=dev), xi, f_all)
        obs_hist0 = obs0.unsqueeze(1).expand(-1, hist, -1).contiguous()
        base_probe = base_action_rollout(wm0, pi_old, mdp, S0, f_all, xi,
                                         normalize_xi(xi), obs_hist0, proto.K, H)
        log("base_actions", kind="probe", n=S_p)
    if not args.skip_eval:
        f_all = torch.tensor([eval_frame(i) for i in range(S_e)], device=dev)
        S0 = motion.ref_state(f_all)
        xi = torch.zeros(S_e, 33, device=dev)
        xi[:, 0] = 1.0
        obs0 = proto.obs(S0, torch.zeros(S_e, 29, device=dev), xi, f_all)
        obs_hist0 = obs0.unsqueeze(1).expand(-1, hist, -1).contiguous()
        base_eval = base_action_rollout(wm0, pi_old, mdp, S0, f_all, xi,
                                        normalize_xi(xi), obs_hist0, proto.K, H)
        log("base_actions", kind="eval", n=S_e)

    def load_dirs(name):
        d = torch.load(args.dirs_dir / f"{name}.pt", map_location=dev, weights_only=False)
        return d["dirs"].to(dev)

    # ---- build the branch schedule (all items are single branches) ----
    probe_items, eval_items = [], []
    if not args.skip_probes:
        probe_sets = {"random": load_dirs("probe_random"),
                      "policy": load_dirs("probe_policy")}
        for arm, dirs in probe_sets.items():
            for si in range(S_p):
                for ji in range(dirs.shape[1]):
                    for br in ("plus", "minus"):
                        sgn = 1.0 if br == "plus" else -1.0
                        act = base_probe[si] + sgn * args.eps_probe * dirs[si, ji]
                        probe_items.append({"arm": arm, "si": si, "ji": ji, "branch": br,
                                            "env_frame": probe_frame(si), "act_seq": act,
                                            "key": f"{arm}_{si}_{ji}_{br}"})
        log("schedule", kind="probe", branches=len(probe_items))
    if not args.skip_eval:
        eval_sets = {name: load_dirs(name) for name in
                     ("eval_random", "eval_pol_seen", "eval_pol_B", "eval_pol_C")}
        for set_name, dirs in eval_sets.items():
            for si in range(S_e):
                for ji in range(dirs.shape[1]):
                    for eps in eps_eval:
                        for br in ("plus", "minus"):
                            sgn = 1.0 if br == "plus" else -1.0
                            act = base_eval[si] + sgn * eps * dirs[si, ji]
                            eval_items.append({"set": set_name, "si": si, "ji": ji,
                                               "eps": eps, "branch": br,
                                               "env_frame": eval_frame(si),
                                               "act_seq": act,
                                               "key": f"{set_name}_{si}_{ji}_{eps}_{br}"})
        for si in range(S_e):
            for rep in range(args.base_reps):
                eval_items.append({"set": "base", "si": si, "ji": -1, "eps": 0.0,
                                   "branch": f"rep{rep}", "env_frame": eval_frame(si),
                                   "act_seq": base_eval[si], "key": f"base_{si}_{rep}"})
        log("schedule", kind="eval", branches=len(eval_items))

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

    stop_secs = args.max_minutes * 60
    probe_out, eval_out = {}, {}

    with launch_simulation(det_cfg(), args):
        from supermimic.env import TrackingSim

        env = TrackingSim(det_cfg(), args.envs, motion, mdp, gamma=args.gamma,
                          eval_mode=True, reset_noise=False)
        wave_i = 0

        def run_all(items, sink):
            nonlocal wave_i
            for c0 in range(0, len(items), args.envs):
                if (time.time() - t0) > stop_secs:
                    stop(f"wall-clock budget {args.max_minutes} min reached")
                    break
                chunk = items[c0:c0 + args.envs]
                if len(chunk) < args.envs:
                    pad = dict(chunk[0])
                    pad["key"] = "__pad__"
                    chunk = chunk + [pad] * (args.envs - len(chunk))
                res = collect_wave(env, mdp, proto, chunk, H, args.gamma, args.tz,
                                   args.tt)
                for it in chunk:
                    if it["key"] != "__pad__":
                        sink[it["key"]] = res[it["key"]]
                wave_i += 1
                if wave_i % 25 == 0:
                    log("progress", waves=wave_i, done=len(sink),
                        mem_gb=round(torch.cuda.memory_reserved() / 2**30, 2))

        if probe_items:
            run_all(probe_items, probe_out)
        if not stop_reason and eval_items:
            run_all(eval_items, eval_out)
        log("waves_done", waves=wave_i,
            probe_branches=len(probe_out), eval_branches=len(eval_out))

    # ---- persist ----
    save_meta = {"pair": {"run": pair["run"], "iteration": pair["iteration"],
                          "sha256": pair["sha256"],
                          "pi_updates": [pair["pi_updates_before"], pair["pi_updates_after"]]},
                 "gamma": args.gamma, "tz": args.tz, "tt": args.tt,
                 "eps_probe": args.eps_probe, "eps_eval": eps_eval, "H": H,
                 "xi": "nominal friction=1 offsets=0 (eval_mode)",
                 "probe_starts": probe_starts, "eval_starts": eval_starts,
                 "base_reps": args.base_reps}
    if probe_out:
        arms = {}
        for arm in ("random", "policy"):
            arms[arm] = {k: v for k, v in probe_out.items() if k.startswith(arm + "_")}
        torch.save({"meta": save_meta,
                    "base_actions": base_probe.cpu(),
                    "branches": {k: v for k, v in probe_out.items()}},
                   args.out_dir / "probes.pt")
    if eval_out:
        torch.save({"meta": save_meta,
                    "base_actions": (base_eval.cpu() if base_eval is not None else None),
                    "branches": eval_out},
                   args.out_dir / "eval_data.pt")
    print(json.dumps({"out_dir": str(args.out_dir), "probe_branches": len(probe_out),
                      "eval_branches": len(eval_out), "waves": wave_i,
                      "elapsed_s": round(time.time() - t0, 1),
                      "stop_reason": stop_reason}, indent=1))


if __name__ == "__main__":
    main()
