"""CVF-0 Phase 1b Session B: Oracle 2 (Action-Induced Reachable-Manifold Continuation).

Same 16 balanced starts (8 distinct clips x 2 frames), 4 batches x 4 envs (deterministic run_to_run).
For each branch (4 action dirs x 2 eps {0.015, 0.03} x +/- + 2 base reps):
  - Phase H=32: open-loop prescribed action sequence A_base +/- eps * d_A in real simulator,
    recording per-step r_state, r_direct, soft_d, hard_term, and terminal S_H^sim (at H=8 and H=32);
  - Phase T_cont=96: seamless closed-loop pi_old continuation in real simulator from S_32^sim,
    recording discounted continuation return V_sim^pi(S_32^sim; T_cont) at T_cont in {32, 64, 96}
    (both soft and hard termination).

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
from cvf_phase1b_oracle1 import pick_16_balanced_indices  # noqa: E402
from supermimic.crgf import reward_split, soft_term  # noqa: E402
from supermimic.mdp import flatten_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts_pt", type=Path, default=Path("results/cvf/phase0/phase0_starts_and_dirs.pt"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/cvf/phase1b"))
    ap.add_argument("--max_minutes", type=float, default=68.0)
    ap.add_argument("--h_open", type=int, default=32)
    ap.add_argument("--t_cont", type=int, default=96)
    ap.add_argument("--envs", type=int, default=4)
    from isaaclab.app import add_launcher_args
    add_launcher_args(ap)
    args = ap.parse_args()

    dev = getattr(args, "device", None) or "cuda:0"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger = args.out_dir / "oracle2.budget.jsonl"
    t0 = time.time()

    def log(event, **kw):
        with open(ledger, "a") as fh:
            fh.write(json.dumps({"t": round(time.time() - t0, 1), "event": event, **kw}) + "\n")

    p0 = torch.load(args.starts_pt, map_location="cpu", weights_only=False)
    idx16 = pick_16_balanced_indices(p0["starts"])
    starts16 = [p0["starts"][i] for i in idx16]
    f0_16 = p0["f0"][idx16]
    base_A_16 = p0["oracle2_base_A"][idx16]
    dirs16 = {k: v[idx16] for k, v in p0["oracle2_dirs"].items()}

    pair = bridge.load_pair_bundle("online_B0", 2100, device=dev)
    world = bridge.setup_world(device=dev, pair=pair)
    motion, mdp, proto = world["motion"], world["mdp"], world["proto"]
    pi_old = pair["pi_old"]
    B = args.envs
    xi_one = torch.zeros(B, 33, device=dev); xi_one[:, 0] = 1.0
    gamma = 0.99
    eps_grid = (0.015, 0.03)
    cont_out = (32, 64, 96)

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

    log("start", n_starts=len(starts16), h_open=args.h_open, t_cont=args.t_cont, eps_grid=list(eps_grid))
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
            A_b = base_A_16[bi:bi + B].to(dev)

            def run_o2_branch(A_seq: torch.Tensor) -> dict:
                env.reset(torch.arange(B, device=dev), f0_b)
                env.xi[:] = xi_one
                env.refresh()
                S0_b = env.state()
                ap = torch.zeros(B, 29, device=dev)
                o0 = proto.obs(S0_b, ap, xi_one, f0_b)
                o_hist = o0.unsqueeze(1).expand(-1, 10, -1).contiguous()

                r_states, r_directs, d_softs, d_hards = [], [], [], []
                S_at_8 = S_at_32 = None
                f_at_8 = f_at_32 = None
                with torch.no_grad():
                    # Open-loop prefix H=32
                    for t in range(args.h_open):
                        a = torch.clamp(A_seq[:, t], -20.0, 20.0)
                        st = env.step(a)
                        S1 = env.state()
                        f1 = env.f.clone()
                        rs, rd = reward_split(mdp, S1, f1, a, ap)
                        ds = soft_term(mdp, S1, f1)
                        r_states.append(rs.cpu())
                        r_directs.append(rd.cpu())
                        d_softs.append(ds.cpu())
                        d_hards.append(st["term"].float().cpu())
                        ap = a
                        o_hist = torch.cat((o_hist[:, 1:], proto.obs(S1, a, xi_one, f1).unsqueeze(1)), 1)
                        if t + 1 == 8:
                            S_at_8 = flatten_state(S1).cpu()
                            f_at_8 = f1.cpu()
                        if t + 1 == args.h_open:
                            S_at_32 = flatten_state(S1).cpu()
                            f_at_32 = f1.cpu()

                    # Closed-loop continuation T_cont=96 from S_32^sim
                    c_soft = torch.ones(B, device=dev)
                    c_hard = torch.ones(B, device=dev)
                    V_soft = torch.zeros(B, device=dev)
                    V_hard = torch.zeros(B, device=dev)
                    cont_s, cont_h, cont_cs, cont_ch = {}, {}, {}, {}
                    for k in range(args.t_cont):
                        ag = proto.actor_goal(env.f, env.state())
                        a = torch.clamp(pi_old(o_hist, ag), -20.0, 20.0)
                        st = env.step(a)
                        S1 = env.state()
                        f1 = env.f.clone()
                        ds = mdp.soft_termination(S1, f1)
                        dh = st["term"].float()
                        V_soft = V_soft + (gamma ** k) * c_soft * st["r"]
                        V_hard = V_hard + (gamma ** k) * c_hard * st["r"]
                        c_soft = c_soft * (1.0 - ds)
                        c_hard = c_hard * (1.0 - dh)
                        o_hist = torch.cat((o_hist[:, 1:], proto.obs(S1, a, xi_one, f1).unsqueeze(1)), 1)
                        step = k + 1
                        if step in cont_out:
                            cont_s[step] = V_soft.cpu().clone()
                            cont_h[step] = V_hard.cpu().clone()
                            cont_cs[step] = c_soft.cpu().clone()
                            cont_ch[step] = c_hard.cpu().clone()

                return {
                    "r_state": torch.stack(r_states, 1),   # (B, 32)
                    "r_direct": torch.stack(r_directs, 1),
                    "d_soft": torch.stack(d_softs, 1),
                    "d_hard": torch.stack(d_hards, 1),
                    "S_at_8": S_at_8, "f_at_8": f_at_8,
                    "S_at_32": S_at_32, "f_at_32": f_at_32,
                    "V_cont_soft": cont_s, "V_cont_hard": cont_h,
                    "c_cont_soft": cont_cs, "c_cont_hard": cont_ch,
                }

            base_res = [run_o2_branch(A_b) for _ in range(2)]
            rep_rng = float((base_res[0]["V_cont_soft"][args.t_cont] - base_res[1]["V_cont_soft"][args.t_cont]).abs().max())
            branches = {}
            for dname, d_all in dirs16.items():
                d_b = d_all[bi:bi + B].to(dev)
                for eps in eps_grid:
                    branches[(dname, eps, "plus")] = run_o2_branch(A_b + eps * d_b)
                    branches[(dname, eps, "minus")] = run_o2_branch(A_b - eps * d_b)

            b_path = args.out_dir / f"oracle2_batch_{b_idx}.pt"
            torch.save({
                "batch_idx": b_idx, "starts": b_starts, "f0": f0_b.cpu(),
                "base_res": base_res, "base_repeat_range": rep_rng,
                "branches": branches,
            }, b_path)
            batch_files.append(b_path)
            log("batch_done", batch_idx=b_idx, repeat_range=rep_rng)

    all_batches = [torch.load(p, map_location="cpu", weights_only=False) for p in batch_files]
    torch.save({"starts": starts16, "idx16": idx16, "eps_grid": list(eps_grid),
                "h_open": args.h_open, "t_cont": args.t_cont, "batches": all_batches},
               args.out_dir / "oracle2_sim.pt")
    log("done", n_batches=len(all_batches), elapsed_s=round(time.time() - t0, 1))
    print(json.dumps({"oracle2_batches": len(all_batches), "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
