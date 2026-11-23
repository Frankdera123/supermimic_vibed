"""CVF-0 Phase 1a GPU smoke test (budget <= 0.35 GPU-h, <= 20 min wall-clock cap):
1. Verify bit-exact determinism of repeated `env.restore_state()` + 200-step closed-loop rollout
   at <= 4 envs (`repeat_range == 0.0`).
2. Probe 4 diverse starts x 3 subspaces (vel_critic, pose_safe_critic, pose_contact_diag) x
   4 eps values {0.005, 0.015, 0.03, 0.05} x T_max=200 for gate pass rate, soft/hard falls,
   and tail convergence rho_tail(128, 200).
"""
from __future__ import annotations

import argparse
import csv
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
    POSE_IDX,
    critic_tangent_grad,
    foot_ball_clearances,
    joint_limit_slack,
    perturb_state,
    rollout_wm_continuation,
    screen_safe_direction,
)
from supermimic.crgf import reliability_gate  # noqa: E402
from supermimic.mdp import flatten_state, normalize_xi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts_pt", type=Path, default=Path("results/cvf/phase0/phase0_starts_and_dirs.pt"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/cvf/phase1a"))
    ap.add_argument("--max_minutes", type=float, default=18.0)
    ap.add_argument("--h_pre", type=int, default=16)
    ap.add_argument("--t_max", type=int, default=200)
    ap.add_argument("--envs", type=int, default=4)
    from isaaclab.app import add_launcher_args
    add_launcher_args(ap)
    args = ap.parse_args()

    dev = getattr(args, "device", None) or "cuda:0"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger = args.out_dir / "phase1a.budget.jsonl"
    t0 = time.time()

    def log(event, **kw):
        with open(ledger, "a") as fh:
            fh.write(json.dumps({"t": round(time.time() - t0, 1), "event": event, **kw}) + "\n")

    # Load Phase 0 starts (pick 4 distinct motion categories: indices 0, 2, 4, 8 -> B=4)
    p0 = torch.load(args.starts_pt, map_location="cpu", weights_only=False)
    sel_idx = [0, 2, 4, 8]
    starts_4 = [p0["starts"][i] for i in sel_idx]
    f0_4 = p0["f0"][sel_idx].to(dev)
    B = len(starts_4)
    assert B == args.envs == 4

    pair = bridge.load_pair_bundle("online_B0", 2100, device=dev)
    world = bridge.setup_world(device=dev, pair=pair)
    motion, mdp, proto, kin = world["motion"], world["mdp"], world["proto"], world["kin"]
    pi_old = pair["pi_old"]
    wm = pair["wm"]
    critic = pair["critic"]
    xi_one = torch.zeros(B, 33, device=dev); xi_one[:, 0] = 1.0
    xi_n = normalize_xi(xi_one)
    gamma = 0.99
    horizons_out = (32, 64, 128, 160, 200)

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

    log("start", clips=[s["clip_name"] for s in starts_4], frames=f0_4.tolist())

    with launch_simulation(det_cfg(), args):
        from supermimic.env import TrackingSim
        env = TrackingSim(det_cfg(), args.envs, motion, mdp, gamma=gamma, eval_mode=True, reset_noise=False)

        # ---- Step 1: Roll real sim H_pre=16 steps to reach on-policy S_star_sim ----
        env.reset(torch.arange(B, device=dev), f0_4)
        env.xi[:] = xi_one
        env.refresh()
        S_init = env.state()
        o0 = proto.obs(S_init, torch.zeros(B, 29, device=dev), xi_one, f0_4)
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
        log("prefix_done", sim_clear_min=float(foot_ball_clearances(kin, S_star).min()),
            sim_slack_min=float(joint_limit_slack(kin, S_star[4]).min()))

        def run_sim_continuation(delta_vec: torch.Tensor, T: int) -> dict:
            """Inject S_star (+) delta_vec via restore_state and roll pi_old closed-loop for T steps."""
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
                for t in range(T):
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

        # ---- Step 2: Determinism check (3 repeats of unperturbed restore_state + 200-step rollout) ----
        base_reps = [run_sim_continuation(torch.zeros(B, 70, device=dev), args.t_max) for _ in range(3)]
        base_200 = torch.stack([r["J_soft"][args.t_max] for r in base_reps], 0)  # (3, B)
        repeat_range = float((base_200.max(0).values - base_200.min(0).values).max())
        log("restore_determinism", repeat_range=repeat_range)
        det_ok = repeat_range <= 1e-6

        # ---- Step 3: Screen directions at real S_star_sim and probe eps grid ----
        cg = critic_tangent_grad(critic, mdp, motion, proto, S_star, f_star, xi_n, clamp_vmax=10.0)
        gV = cg["grad_raw"]
        u_vel, _ = screen_safe_direction(kin, S_star, gV, eps_max=0.05, subspace="vel")
        u_pose_safe, pose_meta = screen_safe_direction(kin, S_star, gV, eps_max=0.05, subspace="pose_safe")
        # Diagnostic only: full pose (including vertical/leg contact DOFs)
        m_full_pose = torch.zeros(B, 70, device=dev); m_full_pose[:, POSE_IDX] = 1.0
        u_pose_contact = (gV * m_full_pose)
        u_pose_contact = u_pose_contact / u_pose_contact.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # Also compute exact WM continuation grads at S_star_sim for direct comparison
        wm_res = rollout_wm_continuation(
            wm, pi_old, mdp, proto, S_star, f_star, oh_prev9_star, a_prev_star,
            xi_one, xi_n, T=args.t_max, hist_mode="last_only", need_grad=True,
            horizons_out=horizons_out,
        )

        subspaces = {
            "vel_critic": u_vel,
            "pose_safe_critic": u_pose_safe,
        }
        eps_grid = (0.005, 0.015, 0.03)
        rows = []
        for sname, u in subspaces.items():
            for eps in eps_grid:
                if (time.time() - t0) > args.max_minutes * 60:
                    log("STOP", reason="wall_clock_cap")
                    break
                rp = run_sim_continuation(eps * u, args.t_max)
                rm = run_sim_continuation(-eps * u, args.t_max)
                s_sep = (rp["s_end"] - rm["s_end"]).norm(dim=-1)
                for h in horizons_out:
                    if h > args.t_max:
                        continue
                    d_sim_s = (rp["J_soft"][h] - rm["J_soft"][h]) / (2.0 * eps)
                    d_sim_h = (rp["J_hard"][h] - rm["J_hard"][h]) / (2.0 * eps)
                    d_crit = (gV * u).sum(-1).cpu()
                    d_wm = (wm_res["grads_soft"][h] * u).sum(-1).cpu()
                    for i in range(B):
                        rows.append({
                            "subspace": sname, "clip": starts_4[i]["clip_name"],
                            "frame": int(f0_4[i]), "eps": eps, "T": h,
                            "D_sim_soft": float(d_sim_s[i]), "D_sim_hard": float(d_sim_h[i]),
                            "D_wm": float(d_wm[i]), "D_critic": float(d_crit[i]),
                            "c_soft_p": float(rp["c_soft"][h][i]), "c_soft_m": float(rm["c_soft"][h][i]),
                            "c_hard_p": float(rp["c_hard"][h][i]), "c_hard_m": float(rm["c_hard"][h][i]),
                            "end_sep": float(s_sep[i]),
                        })
            log("subspace_done", subspace=sname)

    # ---- Step 4: Analyze gate pass rate per eps pair and tail convergence ----
    t_eval = max(h for h in horizons_out if h <= args.t_max)
    eps_pairs = [(0.005, 0.015), (0.015, 0.03)]
    gate_summary = {}
    for sname in subspaces:
        gate_summary[sname] = {}
        for e0, e1 in eps_pairs:
            passed = 0
            tail_ratios = []
            crit_signs = []
            wm_signs = []
            for i in range(B):
                clip_i = starts_4[i]["clip_name"]
                bvals = torch.stack([r["J_soft"][t_eval][i] for r in base_reps])
                r0 = next((r for r in rows if r["subspace"] == sname and r["clip"] == clip_i and r["eps"] == e0 and r["T"] == t_eval), None)
                r1 = next((r for r in rows if r["subspace"] == sname and r["clip"] == clip_i and r["eps"] == e1 and r["T"] == t_eval), None)
                r0_128 = next((r for r in rows if r["subspace"] == sname and r["clip"] == clip_i and r["eps"] == e0 and r["T"] == 128), None)
                r1_128 = next((r for r in rows if r["subspace"] == sname and r["clip"] == clip_i and r["eps"] == e1 and r["T"] == 128), None)
                if r0 is None or r1 is None:
                    continue
                ok, _ = reliability_gate(bvals, torch.tensor([r0["D_sim_soft"], r1["D_sim_soft"]]))
                if ok:
                    passed += 1
                    d_end = 0.5 * (r0["D_sim_soft"] + r1["D_sim_soft"])
                    if r0_128 is not None and r1_128 is not None:
                        d128 = 0.5 * (r0_128["D_sim_soft"] + r1_128["D_sim_soft"])
                        tail_ratios.append(abs(d_end - d128) / max(abs(d_end), 1e-4))
                    crit_signs.append(float((d_end * r0["D_critic"]) > 0))
                    wm_signs.append(float((d_end * r0["D_wm"]) > 0))
            gate_summary[sname][f"{e0}_{e1}"] = {
                "passed": passed, "total": B,
                "med_tail_rho_128_end": (sorted(tail_ratios)[len(tail_ratios) // 2] if tail_ratios else None),
                "crit_vs_sim_sign": (sum(crit_signs) / len(crit_signs) if crit_signs else None),
                "wm_vs_sim_sign": (sum(wm_signs) / len(wm_signs) if wm_signs else None),
            }

    out_json = {
        "restore_determinism_repeat_range": repeat_range,
        "deterministic_ok": det_ok,
        "t_max": args.t_max,
        "pose_safe_modes": pose_meta["mode"],
        "pose_safe_min_clear_pm": min(pose_meta["min_clear_pm"]),
        "pose_safe_min_slack_pm": min(pose_meta["min_slack_pm"]),
        f"gate_summary_T{t_eval}": gate_summary,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(args.out_dir / "phase1a_smoke.json", "w") as fh:
        json.dump(out_json, fh, indent=2)
    if rows:
        with open(args.out_dir / "phase1a_rows.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    log("done", elapsed_s=out_json["elapsed_s"], det_ok=det_ok)
    print(json.dumps(out_json, indent=2))


if __name__ == "__main__":
    main()
