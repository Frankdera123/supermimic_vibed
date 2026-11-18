"""CRGF Phase 0 determinism smoke probe (GPU; REQUIRES explicit authorization).

Answers the two questions that decide the Phase 1/2 budget (GPU_EXPERIMENT_BUDGET.md):

  Q1  What is the largest stable env batch under NewtonCfg.deterministic_mode
      = "run_to_run" + MJWarpSolverCfg.disable_sensors=True?  (The OPCL proto
      hit an int32 workspace limit at 8 envs; this repo may differ.)
  Q2  What is the baseline-return noise floor and the epsilon-derivative
      stability on a contact-heavy start?  (Decides whether the fail-closed
      gate passes enough samples for the pilot.)

Protocol (fail-closed, mirrors crgf.reliability_gate):
  for batch in [2, 4, 8, 16, 32, 64]:
      R repeats of the SAME base action sequence from ONE shared reset;
      record per-env returns and the cross-repeat range.
  then on the largest stable batch:
      directional FD at eps in {0.01, 0.03} on a contact-heavy clip start
      (fallAndGetUp1_subject1-type) and a benign walk start; 3 repeats each.

Outputs: results/crgf/phase0_smoke.{json,csv} + budget ledger.  No parameter
updates, no policy perturbation, read-only checkpoints.  Hard stop after
--max_minutes (default 25) or when GPU memory exceeds --max_mem_gb.

Run (ONLY after authorization):
  python tools/crgf_phase0_smoke.py --policy_ckpt <ckpt> --wm_ckpt <ckpt> \
      --motion_dir data/lafan1_g1 --out_prefix results/crgf/phase0_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, str(here.parent))

import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_ckpt", required=True, type=Path)
    ap.add_argument("--wm_ckpt", required=True, type=Path)
    ap.add_argument("--motion_dir", default="data/lafan1_g1")
    ap.add_argument("--start_frames", nargs="*", type=int, default=None,
                    help="explicit global reference frames; default: first frame "
                         "of fallAndGetUp1_subject1 (contact-heavy) and walk1_subject1")
    ap.add_argument("--clip_names", nargs="*", default=["fallAndGetUp1_subject1", "walk1_subject1"])
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batches", default="2,4,8,16,32,64")
    ap.add_argument("--eps", default="0.01,0.03")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--max_minutes", type=float, default=25.0)
    ap.add_argument("--max_mem_gb", type=float, default=20.0)
    ap.add_argument("--proto_root", default=None)
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
    # AFTER the variant import: patch the ACTIVE kinematics MJCF path
    fix_data_paths(repo_root)
    t0 = time.time()
    ledger = args.out_prefix.with_suffix(".budget.jsonl")

    def log(event, **kw):
        with open(ledger, "a") as fh:
            fh.write(json.dumps({"t": round(time.time() - t0, 1), "event": event, **kw}) + "\n")

    def memory_gb():
        return torch.cuda.memory_reserved() / 2**30

    def stop(reason):
        log("STOP", reason=reason)
        raise SystemExit(f"[phase0] STOP: {reason}")

    from supermimic.crgf import Protocol, base_action_rollout, reliability_gate
    from supermimic.env import TrackingSim
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP, OBS_DIM, normalize_xi
    from supermimic.motion import MotionLib
    from supermimic.networks import Policy, WorldModel
    from supermimic.robot import default_joint_pos_list

    kin = G1Kinematics(dev)
    motion = MotionLib.from_dir(args.motion_dir, kin, dev)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    proto = Protocol(mdp, 10)
    pol_ck = torch.load(args.policy_ckpt, map_location=dev, weights_only=False)
    a_pol = pol_ck.get("args", {})
    hist = int(a_pol.get("hist", 10))
    pi = Policy(OBS_DIM, int(proto.actor_goal(motion.clip_start[:1],
                                              motion.ref_state(motion.clip_start[:1])).shape[-1]),
                hist=hist).to(dev)
    pi.load_state_dict(pol_ck["pi"])
    pi.eval()
    ck_wm = torch.load(args.wm_ckpt, map_location=dev, weights_only=False)
    wm = WorldModel().to(dev)
    wm.load_state_dict(ck_wm["wm"])
    wm.eval()

    # start frames
    if args.start_frames:
        frames = torch.tensor(args.start_frames, device=dev)
    else:
        frames = []
        for name in args.clip_names:
            idx = motion.names.index(name)
            frames.append(int(motion.clip_start[idx]) + 1)
        frames = torch.tensor(frames, device=dev)

    # shared base action sequence
    S0 = motion.ref_state(frames)
    xi = torch.zeros(len(frames), 33, device=dev)
    xi[:, 0] = 1.0
    obs0 = proto.obs(S0, torch.zeros(len(frames), 29, device=dev), xi, frames)
    obs_hist0 = obs0.unsqueeze(1).expand(-1, hist, -1).contiguous()
    A_base = base_action_rollout(wm, pi, mdp, S0, frames, xi, normalize_xi(xi),
                                 obs_hist0, 10, args.horizon)  # (F, H, 29)
    log("base_actions", frames=frames.tolist(), horizon=args.horizon)

    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation

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

    results = {"batch_scaling": [], "eps_probe": [], "verdict": {}}
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    sim_args = SimpleNamespace(num_envs=0)
    with launch_simulation(det_cfg(), args):
        largest_stable = 0
        for B in [int(b) for b in args.batches.split(",")]:
            if (time.time() - t0) / 60 > args.max_minutes:
                stop("time budget")
            envs = B * len(frames)
            try:
                env = TrackingSim(det_cfg(), envs, motion, mdp, gamma=args.gamma,
                                  eval_mode=True, reset_noise=False)
                if envs > 4:
                    # probe whether deterministic mode can even launch at this
                    # size (warp int32 workspace limit, cf. OPCL repro fix)
                    ids = torch.arange(envs, device=dev)
                    env.reset(ids, frames.repeat(B))
                    env.step(A_base.repeat(B, 1, 1).reshape(envs, args.horizon, 29)[:, 0])
            except Exception as exc:
                log("batch_failed", B=B, error=str(exc)[:200])
                try:
                    del env
                except Exception:
                    pass
                try:
                    import isaaclab.sim as _sim_utils

                    _sim_utils.SimulationContext.clear_instance()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                continue
            rets = []
            for r in range(args.repeats):
                ids = torch.arange(envs, device=dev)
                env.reset(ids, frames.repeat(B))
                xi_b = torch.zeros(envs, 33, device=dev)
                xi_b[:, 0] = 1.0
                env.xi.copy_(xi_b)
                env.refresh()
                a_prev = torch.zeros(envs, 29, device=dev)
                # straight rollout: env e uses A_base[e % F]
                a_prev = torch.zeros(envs, 29, device=dev)
                total = torch.zeros(envs, device=dev)
                for t in range(args.horizon):
                    idx = torch.arange(envs, device=dev) % len(frames)
                    a = A_base[idx, t]
                    out = env.step(a)
                    total += out["r"]
                    a_prev = a
                rets.append(total.reshape(B, len(frames)).mean(-1))  # (B,) mean over start copies
            rets = torch.stack(rets)  # (R, B)
            rng = float((rets.max(0).values - rets.min(0).values).max())
            mem = memory_gb()
            results["batch_scaling"].append({"batch_envs": envs, "repeat_range": rng,
                                             "mem_gb": round(mem, 2)})
            log("batch_scaling", B=B, repeat_range=rng, mem_gb=mem)
            if rng <= 1e-5 and mem <= args.max_mem_gb:
                largest_stable = envs
            del env
            # allow the next batch size to construct a fresh SimulationContext
            try:
                import isaaclab.sim as _sim_utils

                _sim_utils.SimulationContext.clear_instance()
            except Exception as exc:
                log("clear_context_failed", error=str(exc)[:150])
            torch.cuda.empty_cache()
            time.sleep(2.0)
            if mem > args.max_mem_gb:
                stop(f"memory {mem:.1f} GB > {args.max_mem_gb} GB")

        # eps probe on the largest stable batch; each triplet needs 3 parallel
        # branch envs, so the probe uses as many starts as the env budget allows
        F_probe = max(1, largest_stable // 3)
        if largest_stable >= 3:
            env = TrackingSim(det_cfg(), 3 * F_probe, motion, mdp, gamma=args.gamma,
                              eval_mode=True, reset_noise=False)
            F = F_probe
            for eps in [float(e) for e in args.eps.split(",")]:
                d = torch.randn(F, args.horizon, 29, generator=gen)
                d = d / d.reshape(F, -1).norm(dim=-1).view(-1, 1, 1).clamp_min(1e-8)
                d = d.to(dev)
                branches = torch.cat([A_base[:F] + eps * d, A_base[:F] - eps * d, A_base[:F]], 0)  # (3F, H, 29)
                Ds, base_rets = [], []
                for r in range(args.repeats):
                    env.reset(torch.arange(3 * F, device=dev), frames[:F].repeat(3))
                    xi_b = torch.zeros(3 * F, 33, device=dev)
                    xi_b[:, 0] = 1.0
                    env.xi.copy_(xi_b)
                    env.refresh()
                    a_prev = torch.zeros(3 * F, 29, device=dev)
                    total = torch.zeros(3 * F, device=dev)
                    for t in range(args.horizon):
                        out = env.step(branches[:, t])
                        total += out["r"]
                        a_prev = branches[:, t]
                    J = total.reshape(3, F)
                    Ds.append(float((J[0] - J[1]) / (2 * eps)))
                    base_rets.append(float(J[2].mean()))
                ok, why = reliability_gate(torch.tensor(base_rets), torch.tensor(Ds))
                results["eps_probe"].append({"eps": eps, "D_values": Ds, "gate": ok, "reason": why})
                log("eps_probe", eps=eps, D=Ds, gate=ok, reason=why)
            del env

        n_ok = sum(1 for p in results["eps_probe"] if p["gate"])
        results["verdict"] = {
            "largest_stable_envs": largest_stable,
            # sequential-branch FD design (Phase 1 v2) needs only 3 envs per
            # triplet (one start); 4-env contexts are sufficient
            "deterministic_ok": largest_stable >= 3,
            "eps_gate_passes": f"{n_ok}/{len(results['eps_probe'])}",
            "phase1_feasible": largest_stable >= 3 and n_ok >= 1,
            "budget_multiplier": 1.0 if largest_stable >= 12 else (2.0 if largest_stable >= 3 else 3.0),
        }
        log("verdict", **results["verdict"])

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_prefix.with_suffix(".json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(json.dumps(results["verdict"], indent=1))


if __name__ == "__main__":
    main()
