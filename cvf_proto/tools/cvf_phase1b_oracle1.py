"""CVF-0 Phase 1b Session A: Oracle 1 (Direct Tangent-Space State Perturbation).

16 balanced starts (8 distinct LAFAN1 motion categories x 2 frames/clip),
4 batches x 4 envs (deterministic run_to_run), 5 directions x 2 eps {0.015, 0.03} x +/- + 2 base reps,
T_max = 128 steps (offline export at T in {32, 64, 96, 128}).
Incremental per-batch checkpointing + JSONL budget ledger + wall-clock fuse.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CVF_ROOT = _HERE.parent
_REPO = _CVF_ROOT.parent
sys.path.insert(0, str(_CVF_ROOT))
sys.path.insert(0, str(_REPO / "pada_proto"))

import pada_bridge as bridge  # noqa: E402

bridge.import_stack()
bridge.fix_paths()

import torch  # noqa: E402
from cvf_core import (  # noqa: E402
    critic_tangent_grad,
    foot_ball_clearances,
    joint_limit_slack,
    perturb_state,
    screen_safe_direction,
)
from supermimic.mdp import flatten_state, normalize_xi  # noqa: E402


def pick_16_balanced_indices(starts: list[dict]) -> list[int]:
    """Select 16 indices (8 distinct clips x 2 frames, 12.5% per clip)."""
    by_clip = {}
    for idx, s in enumerate(starts):
        by_clip.setdefault(s["clip_name"], []).append(idx)
    # Keep 8 distinct clips (drop extra walk clips beyond the first walk clip)
    chosen_clips = []
    seen_walk = False
    for cname in sorted(by_clip):
        if cname.startswith("walk"):
            if seen_walk:
                continue
            seen_walk = True
        chosen_clips.append(cname)
        if len(chosen_clips) == 8:
            break
    out = []
    for cname in chosen_clips:
        out.extend(by_clip[cname][:2])
    assert len(out) == 16, f"Expected 16 starts, got {len(out)}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts_pt", type=Path, default=Path("results/cvf/phase0/phase0_starts_and_dirs.pt"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/cvf/phase1b"))
    ap.add_argument("--max_minutes", type=float, default=80.0)
    ap.add_argument("--h_pre", type=int, default=16)
    ap.add_argument("--t_max", type=int, default=128)
    ap.add_argument("--envs", type=int, default=4)
    from isaaclab.app import add_launcher_args
    add_launcher_args(ap)
    args = ap.parse_args()

    dev = getattr(args, "device", None) or "cuda:0"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger = args.out_dir / "oracle1.budget.jsonl"
    t0 = time.time()

    def log(event, **kw):
        with open(ledger, "a") as fh:
            fh.write(json.dumps({"t": round(time.time() - t0, 1), "event": event, **kw}) + "\n")

    p0 = torch.load(args.starts_pt, map_location="cpu", weights_only=False)
    idx16 = pick_16_balanced_indices(p0["starts"])
    starts16 = [p0["starts"][i] for i in idx16]
    f0_16 = p0["f0"][idx16]
    vel_reach_16 = p0["oracle1_dirs"]["vel_reachable"][idx16]

    pair = bridge.load_pair_bundle("online_B0", 2100, device=dev)
    world = bridge.setup_world(device=dev, pair=pair)
    motion, mdp, proto, kin = world["motion"], world["mdp"], world["proto"], world["kin"]
    pi_old = pair["pi_old"]
    critic = pair["critic"]
    B = args.envs
    xi_one = torch.zeros(B, 33, device=dev); xi_one[:, 0] = 1.0
    xi_n = normalize_xi(xi_one)
    gamma = 0.99
    horizons_out = (32, 64, 96, 128)
    eps_grid = (0.015, 0.03)

    from isaaclab.app import launch_simulation
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
    import isaaclab.sim as sim_utils

    def det_cfg():
        return sim_utils.SimulationCfg(
            dt=0.02, device=dev,
            physics=NewtonCfg(
                deterministic_mode="run_to_run",
                solver_cfg=MJWarpSolverCfg(njmax=500, nconmax=128, cone="pyramidal",
                                           integrator="implicitfast", use_mujoco_contacts=True,
                                           disable_sensors=True),
                num_substeps=4,
            ),
        )

    log("start", n_starts=len(starts16), clips=sorted({s["clip_name"] for s in starts16}),
        t_max=args.t_max, eps_grid=list(eps_grid))

    batch_files = []
    with launch_simulation(det_cfg(), args):
        from supermimic.env import TrackingSim
        env = TrackingSim(det_cfg(), args.envs, motion, mdp, gamma=gamma, eval_mode=True, reset_noise=False)

        for bi in range(0, len(starts16), B):
            if (time.time() - t0) > args.max_minutes * 60:
                log("STOP", reason="wall_clock_cap", completed_batches=bi // B)
                break
            b_idx = bi // B
            b_starts = starts16[bi:bi + B]
            f0_b = f0_16[bi:bi + B].to(dev)

            # 1. Prefix H_pre=16 in real sim
            env.reset(torch.arange(B, device=dev), f0_b)
            env.xi[:] = xi_one
            env.refresh()
            S_init = env.state()
            o0 = proto.obs(S_init, torch.zeros(B, 29, device=dev), xi_one, f0_b)
            o_hist = o0.unsqueeze(1).expand(-1, 10, -1).contiguous()
            with torch.no_grad():
                for _ in range(args.h_pre):
                    ag = proto.actor_goal(env.f, env.state())
                    a = torch.clamp(pi_old(o_hist, ag), -20.0, 20.0)
                    env.step(a)
                    o1 = proto.obs(env.state(), a, xi_one, env.f)
                    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
            snap_star = env.snapshot_state()
            S_star = snap_star["S"]
            f_star = snap_star["f"].clone()
            a_prev_star = snap_star["a_prev"].clone()
            oh_prev9_star = o_hist[:, :-1].clone()

            # 2. Screen 5 directions on real S_star_sim
            cg = critic_tangent_grad(critic, mdp, motion, proto, S_star, f_star, xi_n, clamp_vmax=10.0)
            gV = cg["grad_raw"]
            g_rng = torch.Generator().manual_seed(2600 + b_idx * 17)
            u_vel_crit, _ = screen_safe_direction(kin, S_star, gV, eps_max=0.03, subspace="vel")
            u_vel_reach, _ = screen_safe_direction(kin, S_star, vel_reach_16[bi:bi + B].to(dev), eps_max=0.03, subspace="vel")
            u_vel_rand, _ = screen_safe_direction(kin, S_star, torch.randn(B, 70, generator=g_rng).to(dev), eps_max=0.03, subspace="vel")
            u_pose_crit, pose_meta_c = screen_safe_direction(kin, S_star, gV, eps_max=0.03, subspace="pose_safe")
            u_pose_rand, pose_meta_r = screen_safe_direction(kin, S_star, torch.randn(B, 70, generator=g_rng).to(dev), eps_max=0.03, subspace="pose_safe")

            dirs_b = {
                "vel_critic": u_vel_crit,
                "vel_reachable": u_vel_reach,
                "vel_random": u_vel_rand,
                "pose_safe_critic": u_pose_crit,
                "pose_safe_random": u_pose_rand,
            }

            def run_cont(delta_vec: torch.Tensor) -> dict:
                S_inj = perturb_state(S_star, delta_vec)
                snap_inj = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in snap_star.items()}
                snap_inj["S"] = S_inj
                env.restore_state(snap_inj, restore_physics_xi=False)
                o_cur = proto.obs(S_inj, a_prev_star, xi_one, f_star)
                oh = torch.cat((oh_prev9_star, o_cur.unsqueeze(1)), dim=1)
                c_soft = torch.ones(B, device=dev)
                c_hard = torch.ones(B, device=dev)
                J_soft = torch.zeros(B, device=dev)
                J_hard = torch.zeros(B, device=dev)
                out_s, out_h, out_cs, out_ch = {}, {}, {}, {}
                with torch.no_grad():
                    for t in range(args.t_max):
                        ag = proto.actor_goal(env.f, env.state())
                        a = torch.clamp(pi_old(oh, ag), -20.0, 20.0)
                        st = env.step(a)
                        S1 = env.state()
                        f1 = env.f.clone()
                        d_s = mdp.soft_termination(S1, f1)
                        d_h = st["term"].float()
                        J_soft = J_soft + (gamma ** t) * c_soft * st["r"]
                        J_hard = J_hard + (gamma ** t) * c_hard * st["r"]
                        c_soft = c_soft * (1.0 - d_s)
                        c_hard = c_hard * (1.0 - d_h)
                        o1 = proto.obs(S1, a, xi_one, f1)
                        oh = torch.cat((oh[:, 1:], o1.unsqueeze(1)), 1)
                        step = t + 1
                        if step in horizons_out:
                            out_s[step] = J_soft.cpu().clone()
                            out_h[step] = J_hard.cpu().clone()
                            out_cs[step] = c_soft.cpu().clone()
                            out_ch[step] = c_hard.cpu().clone()
                return {"J_soft": out_s, "J_hard": out_h, "c_soft": out_cs, "c_hard": out_ch,
                        "s_end": flatten_state(env.state()).cpu()}

            # 3. 2 base reps + 5 dirs x 2 eps x +/-
            base_res = [run_cont(torch.zeros(B, 70, device=dev)) for _ in range(2)]
            rep_rng = float((base_res[0]["J_soft"][args.t_max] - base_res[1]["J_soft"][args.t_max]).abs().max())
            branches = {}
            for dname, u in dirs_b.items():
                for eps in eps_grid:
                    branches[(dname, eps, "plus")] = run_cont(eps * u)
                    branches[(dname, eps, "minus")] = run_cont(-eps * u)

            b_out = {
                "batch_idx": b_idx,
                "starts": b_starts,
                "f0": f0_b.cpu(),
                "f_star": f_star.cpu(),
                "S_star": tuple(x.cpu() for x in S_star),
                "a_prev_star": a_prev_star.cpu(),
                "oh_prev9_star": oh_prev9_star.cpu(),
                "dirs": {k: v.cpu() for k, v in dirs_b.items()},
                "pose_meta_c": pose_meta_c,
                "pose_meta_r": pose_meta_r,
                "base_res": base_res,
                "base_repeat_range": rep_rng,
                "branches": branches,
            }
            b_path = args.out_dir / f"oracle1_batch_{b_idx}.pt"
            torch.save(b_out, b_path)
            batch_files.append(b_path)
            log("batch_done", batch_idx=b_idx, repeat_range=rep_rng,
                sim_clear_min=float(foot_ball_clearances(kin, S_star).min()),
                sim_slack_min=float(joint_limit_slack(kin, S_star[4]).min()))

    all_batches = [torch.load(p, map_location="cpu", weights_only=False) for p in batch_files]
    torch.save({"starts": starts16, "idx16": idx16, "eps_grid": list(eps_grid),
                "horizons_out": list(horizons_out), "batches": all_batches},
               args.out_dir / "oracle1_sim.pt")
    log("done", n_batches=len(all_batches), elapsed_s=round(time.time() - t0, 1))
    print(json.dumps({"oracle1_batches": len(all_batches), "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
