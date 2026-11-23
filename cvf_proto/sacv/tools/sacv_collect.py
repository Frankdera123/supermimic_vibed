"""CVF-1a Phase B: real-simulator value data collection (GPU; requires APPROVED=true).

Modeled directly on cvf_proto/tools/cvf_phase1b_oracle2.py (CVF-0 Oracle 2) with the
CVF1A data-contract extensions:
  * starts come from the frozen CVF1A_SPLIT_MANIFEST.json (new clips, never CVF-0's);
  * per-branch full step records (r_state/r_direct/r/d_soft/d_hard, H+T steps) and
    extended states (flattened s, o_hist, a_prev, f) at prefix horizons {8,32} and
    continuation horizons {32,64,96};
  * direction families regenerated per new start with the CVF-0 phase0 recipe:
    delta_2100/delta_2300 = per-step clamp(pi_new)-clamp(pi_old) along the pi_old
    closed-loop WM path; ascent via open_loop_paths with pair["critic"] (== critic_pre,
    identical to cvf_phase0_audit.py); random via crgf.random_dir_seqs;
  * explicit env_transitions vs simulator_batch_steps counters (budget contract §8);
  * smoke mode: base-repeat determinism, simulator restore determinism,
    cross-environment contamination check (companion permutation),
    reward reconstruction (r == r_state + r_direct), G-recalculation cross-check,
    and throughput projection for the pre-registered R0/R2/R3 ladder.

Usage (after approval):
  python cvf_proto/sacv/tools/sacv_collect.py --mode smoke --envs 4 --max_minutes 15
  python cvf_proto/sacv/tools/sacv_collect.py --mode collect --split all --envs <from smoke> \
      --ladder R0|R2|R3 --max_minutes <budget fuse>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SACV = _HERE.parent
_CVF = _SACV.parent
_REPO = _CVF.parent
for p in (str(_SACV), str(_CVF), str(_REPO / "pada_proto"), str(_REPO / "pada_proto" / "tools"),
          str(_REPO / "crgf_proto")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pada_bridge as bridge  # noqa: E402

bridge.import_stack()
bridge.fix_paths()

import torch  # noqa: E402
from supermimic.crgf import normalize_dir_seqs, open_loop_paths, path_directional_derivatives, random_dir_seqs  # noqa: E402
from supermimic.crgf import reward_split, soft_term  # noqa: E402
from supermimic.mdp import flatten_state, normalize_xi  # noqa: E402

sys.path.insert(0, str(_SACV))
from sacv.losses import targets as vt  # noqa: E402  (G cross-check)

SEED = 2701
GAMMA = 0.99
H_OPEN = 32
T_CONT = 96
EPS_GRID = (0.015, 0.03)
PREFIX_SNAP_AT = (8, 32)
CONT_OUT = (32, 64, 96)
FAMILIES = ("delta_2100", "delta_2300", "ascent", "random")

_RAW_BUNDLES: dict[str, dict] = {}


def _raw_bundle(run: str, iteration: int) -> dict:
    key = f"{run}@{iteration}"
    if key not in _RAW_BUNDLES:
        _RAW_BUNDLES[key] = torch.load(
            bridge.pairs_root() / run / "diag_pairs" / f"pair_it_{iteration:06d}.pt",
            map_location="cpu", weights_only=False)
    return _RAW_BUNDLES[key]


def make_dirs_and_base(wm, mdp, proto, pi_old, critic_ascent, S0, f0, xi,
                       seed_offset: int) -> dict:
    """CVF-0 phase0 direction recipe for a batch of starts -> base A0 (B,32,29) + dirs.

    xi is the raw nominal xi (one-hot); WM/goal inputs use normalize_xi(xi) exactly
    as cvf_phase0_audit.py (xi_n convention)."""
    from supermimic.networks import Policy

    dev = S0[0].device
    B = S0[0].shape[0]
    xi_n = normalize_xi(xi)
    raw21 = _raw_bundle("online_B0", 2100)
    raw23 = _raw_bundle("online_B0", 2300)

    def load_policy(sd: dict) -> Policy:
        p = Policy(93, proto.K * 73, hist=10)
        p.load_state_dict(sd, strict=True)
        return p.eval().to(dev)

    pi2100_new = load_policy(raw21["pi_new"])
    pi2300_old = load_policy(raw23["pi_old"])
    pi2300_new = load_policy(raw23["pi_new"])

    S = tuple(x.detach() for x in S0)
    fr = f0.clone()
    oh = proto.obs(S, torch.zeros(B, 29, device=dev), xi, fr).unsqueeze(1).expand(-1, 10, -1).contiguous()
    a0_old, d2100, d2300 = [], [], []
    with torch.no_grad():
        for _ in range(H_OPEN):
            ag = proto.actor_goal(fr, S)
            ao = torch.clamp(pi_old(oh, ag), -20.0, 20.0)
            a0_old.append(ao)
            d2100.append(torch.clamp(pi2100_new(oh, ag), -20.0, 20.0) - ao)
            d2300.append(torch.clamp(pi2300_new(oh, ag), -20.0, 20.0)
                         - torch.clamp(pi2300_old(oh, ag), -20.0, 20.0))
            S = tuple(proto.wm_step(wm, S, ao, xi_n))
            fr = fr + 1
            oh = torch.cat((oh[:, 1:], proto.obs(S, ao, xi, fr).unsqueeze(1)), 1)
    A0 = torch.stack(a0_old, 1)
    dirs = {
        "delta_2100": normalize_dir_seqs(torch.stack(d2100, 1).detach()),
        "delta_2300": normalize_dir_seqs(torch.stack(d2300, 1).detach()),
    }
    A_req = A0.clone().requires_grad_(True)
    paths = open_loop_paths(wm, critic_ascent, mdp, tuple(x.detach() for x in S0), f0,
                            A_req, torch.zeros(B, 29, device=dev), xi_n, gamma=GAMMA,
                            goal_frames=proto.K)
    dirs["ascent"] = normalize_dir_seqs(
        path_directional_derivatives(paths, A_req)["total"].reshape(B, H_OPEN, 29).detach())
    dirs["random"] = random_dir_seqs(B, H_OPEN, 29, seed=SEED + 404 + seed_offset, device=dev)
    return {"A0": A0.detach(), "dirs": dirs}


def run_branch(env, mdp, proto, pi_old, xi_one, f0_b, A_seq, B: int, t_cont: int) -> dict:
    """One branch: open-loop prefix H_OPEN, then closed-loop pi_old for t_cont steps.

    Records (all CPU): per-step r_state/r_direct/r/d_soft/d_hard over H+t_cont;
    extended states (S_flat, f, a_prev, o_hist) at prefix {8,32} and continuation
    {32,64,96}; continuation V_soft/V_hard and survival c at {32,64,96}.
    """
    dev = f0_b.device
    env.reset(torch.arange(B, device=dev), f0_b)
    env.xi[:] = xi_one
    env.refresh()
    S0_b = env.state()
    ap = torch.zeros(B, 29, device=dev)
    o0 = proto.obs(S0_b, ap, xi_one, f0_b)
    o_hist = o0.unsqueeze(1).expand(-1, 10, -1).contiguous()

    r_states, r_directs, r_tots, d_softs, d_hards = [], [], [], [], []
    snaps_prefix: dict[str, dict] = {}
    prefix_horizons = (8, H_OPEN)
    with torch.no_grad():
        for t in range(H_OPEN):
            a = torch.clamp(A_seq[:, t], -20.0, 20.0)
            st = env.step(a)
            S1 = env.state()
            f1 = env.f.clone()
            rs, rd = reward_split(mdp, S1, f1, a, ap)
            ds = soft_term(mdp, S1, f1)
            r_states.append(rs.cpu()); r_directs.append(rd.cpu())
            r_tots.append(st["r"].cpu()); d_softs.append(ds.cpu())
            d_hards.append(st["term"].float().cpu())
            ap = a
            o_hist = torch.cat((o_hist[:, 1:], proto.obs(S1, a, xi_one, f1).unsqueeze(1)), 1)
            if t + 1 in prefix_horizons:
                snaps_prefix[f"prefix_{t+1}"] = {
                    "S_flat": flatten_state(S1).cpu(), "f": f1.cpu(), "a_prev": ap.cpu(),
                    "o_hist": o_hist.cpu()}

        prefix = {"r_state": torch.stack(r_states, 1), "r_direct": torch.stack(r_directs, 1),
                  "r": torch.stack(r_tots, 1), "d_soft": torch.stack(d_softs, 1),
                  "d_hard": torch.stack(d_hards, 1)}
        prefix_n = len(r_states)
        c_soft = torch.ones(B, device=dev)
        c_hard = torch.ones(B, device=dev)
        V_soft = torch.zeros(B, device=dev)
        V_hard = torch.zeros(B, device=dev)
        cont = {"V_soft": {}, "V_hard": {}, "c_soft": {}, "c_hard": {}}
        snaps_cont: dict[str, dict] = {}
        for k in range(t_cont):
            ag = proto.actor_goal(env.f, env.state())
            a = torch.clamp(pi_old(o_hist, ag), -20.0, 20.0)
            st = env.step(a)
            S1 = env.state()
            f1 = env.f.clone()
            rs, rd = reward_split(mdp, S1, f1, a, ap)
            ds = mdp.soft_termination(S1, f1)
            dh = st["term"].float()
            r_states.append(rs.cpu()); r_directs.append(rd.cpu())
            r_tots.append(st["r"].cpu()); d_softs.append(ds.cpu()); d_hards.append(dh.cpu())
            V_soft = V_soft + (GAMMA ** k) * c_soft * st["r"]
            V_hard = V_hard + (GAMMA ** k) * c_hard * st["r"]
            c_soft = c_soft * (1.0 - ds)
            c_hard = c_hard * (1.0 - dh)
            ap = a
            o_hist = torch.cat((o_hist[:, 1:], proto.obs(S1, a, xi_one, f1).unsqueeze(1)), 1)
            step = k + 1
            if step in CONT_OUT:
                cont["V_soft"][step] = V_soft.cpu().clone()
                cont["V_hard"][step] = V_hard.cpu().clone()
                cont["c_soft"][step] = c_soft.cpu().clone()
                cont["c_hard"][step] = c_hard.cpu().clone()
                snaps_cont[f"cont_{step}"] = {
                    "S_flat": flatten_state(S1).cpu(), "f": f1.cpu(), "a_prev": ap.cpu(),
                    "o_hist": o_hist.cpu()}

    full = {"r_state": torch.stack(r_states, 1), "r_direct": torch.stack(r_directs, 1),
            "r": torch.stack(r_tots, 1), "d_soft": torch.stack(d_softs, 1),
            "d_hard": torch.stack(d_hards, 1)}
    # internal consistency: continuation-only G from step records must equal V_soft
    g96 = vt.discounted_return(full["r_state"][:, prefix_n:] + full["r_direct"][:, prefix_n:],
                               full["d_soft"][:, prefix_n:], GAMMA,
                               horizons=(min(CONT_OUT[-1], t_cont),))[min(CONT_OUT[-1], t_cont)]
    return {"prefix": prefix, "cont": cont, "snaps_prefix": snaps_prefix,
            "snaps_cont": snaps_cont, "full": full, "prefix_len": prefix_n,
            "g_recalc_vsoft_absdiff": float(
                (g96 - cont["V_soft"][min(CONT_OUT[-1], t_cont)]).abs().max())}


def _continuation_from_current(env, mdp, proto, pi_old, xi_one, B: int, t_cont: int) -> torch.Tensor:
    """Closed-loop continuation from the CURRENT env state; returns V_soft at t_cont."""
    dev = xi_one.device
    ap = torch.zeros(B, 29, device=dev)
    o0 = proto.obs(env.state(), ap, xi_one, env.f)
    o_hist = o0.unsqueeze(1).expand(-1, 10, -1).contiguous()
    c_soft = torch.ones(B, device=dev)
    V_soft = torch.zeros(B, device=dev)
    with torch.no_grad():
        for k in range(t_cont):
            ag = proto.actor_goal(env.f, env.state())
            a = torch.clamp(pi_old(o_hist, ag), -20.0, 20.0)
            st = env.step(a)
            S1 = env.state()
            ds = mdp.soft_termination(S1, env.f)
            V_soft = V_soft + (GAMMA ** k) * c_soft * st["r"]
            c_soft = c_soft * (1.0 - ds)
            o_hist = torch.cat((o_hist[:, 1:], proto.obs(S1, a, xi_one, env.f).unsqueeze(1)), 1)
    return V_soft.cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "collect"), default="smoke")
    ap.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--max_minutes", type=float, default=15.0)
    ap.add_argument("--ladder", choices=("R0", "R2", "R3"), default="R0",
                    help="R0: all families x2 eps; R2: secondary families single eps; "
                         "R3: R2 family set with T_CONT=64")
    ap.add_argument("--out_root", type=Path, default=_REPO / "results/cvf/sacv/datasets/raw")
    ap.add_argument("--ledger", type=Path, default=_REPO / "results/cvf/sacv/budget/budget_ledger.jsonl")
    from isaaclab.app import add_launcher_args

    add_launcher_args(ap)
    args = ap.parse_args()
    t_cont = T_CONT if args.ladder != "R3" else 64
    dev = getattr(args, "device", None) or "cuda:0"
    t0 = time.time()
    led = args.ledger
    led.parent.mkdir(parents=True, exist_ok=True)
    counters = {"env_transitions": 0, "simulator_batch_steps": 0}

    def log(event, **kw):
        with open(led, "a") as fh:
            fh.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "phase": f"1a_{args.mode}",
                                 "task_id": f"sacv_{args.mode}", "event": event,
                                 "env_transitions": counters["env_transitions"],
                                 "simulator_batch_steps": counters["simulator_batch_steps"],
                                 **kw}) + "\n")

    def count(n_steps: int, B: int):
        counters["env_transitions"] += n_steps * B
        counters["simulator_batch_steps"] += n_steps * B  # synchronous model: equal by construction

    man = json.loads((_REPO / "results/cvf/sacv/manifests/CVF1A_SPLIT_MANIFEST.json").read_text())
    starts = []
    for split in ("train", "val", "test"):
        if args.split != "all" and split != args.split:
            continue
        for clip, cd in man["splits"][split]["per_clip"].items():
            for fr in cd["frames"]:
                starts.append({"clip": clip, "split": split, "frame": fr})
    if args.mode == "smoke":
        by_split = {}
        for s in starts:
            by_split.setdefault(s["split"], s)
        starts = [by_split[k] for k in ("train", "val", "test") if k in by_split]

    pair2100 = bridge.load_pair_bundle("online_B0", 2100, device=dev)
    pair2300 = bridge.load_pair_bundle("online_B0", 2300, device=dev)
    world = bridge.setup_world(device=dev, pair=pair2100)
    mdp, proto = world["mdp"], world["proto"]
    pi_old = pair2100["pi_old"]
    wm = pair2100["wm"]
    B = args.envs
    xi_one = torch.zeros(B, 33, device=dev); xi_one[:, 0] = 1.0

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

    families = list(FAMILIES)
    eps_by_family = {f: list(EPS_GRID) for f in families}
    if args.mode == "smoke":   # keep smoke within its fuse: primary family + one-eps secondaries
        eps_by_family = {"delta_2100": list(EPS_GRID), "delta_2300": [EPS_GRID[0]],
                         "ascent": [EPS_GRID[0]], "random": [EPS_GRID[0]]}
    elif args.ladder == "R2":
        eps_by_family = {"delta_2100": list(EPS_GRID), "delta_2300": list(EPS_GRID),
                         "ascent": [EPS_GRID[0]], "random": [EPS_GRID[0]]}

    log("start", mode=args.mode, split=args.split, n_starts=len(starts), envs=B,
        ladder=args.ladder, t_cont=t_cont, device=dev, eps_map={k: v for k, v in eps_by_family.items()})

    with launch_simulation(det_cfg(), args):
        from supermimic.env import TrackingSim

        env = TrackingSim(det_cfg(), B, world["motion"], mdp, gamma=GAMMA,
                          eval_mode=True, reset_noise=False)
        n_batches = 0
        smoke_summary: dict = {}
        for bi in range(0, len(starts), B):
            if (time.time() - t0) > args.max_minutes * 60:
                log("STOP", reason="wall_clock_cap", completed_batches=n_batches)
                break
            b_starts = starts[bi:bi + B]
            n_valid = len(b_starts)
            while len(b_starts) < B:      # pad with repeats (excluded downstream by n_valid)
                b_starts.append(b_starts[-1])
            f0_b = torch.tensor([s["frame"] for s in b_starts], device=dev)

            env.reset(torch.arange(B, device=dev), f0_b)
            env.xi[:] = xi_one
            env.refresh()
            S0_b = env.state()
            gen = make_dirs_and_base(wm, mdp, proto, pi_old, pair2100["critic"],
                                     S0_b, f0_b, xi_one, bi)
            A0, dirs = gen["A0"], gen["dirs"]

            base_res = [run_branch(env, mdp, proto, pi_old, xi_one, f0_b, A0, B, t_cont)
                        for _ in range(2)]
            count(2 * (H_OPEN + t_cont), B)
            rep_rng = float((base_res[0]["cont"]["V_soft"][min(CONT_OUT[-1], t_cont)]
                             - base_res[1]["cont"]["V_soft"][min(CONT_OUT[-1], t_cont)]).abs().max())

            branches = {}
            for fam in families:
                d_b = dirs[fam].to(dev)
                for eps in eps_by_family[fam]:
                    for sign in ("plus", "minus"):
                        A_seq = A0 + eps * d_b if sign == "plus" else A0 - eps * d_b
                        branches[f"{fam}|eps{eps}|{sign}"] = run_branch(
                            env, mdp, proto, pi_old, xi_one, f0_b, A_seq, B, t_cont)
                        count(H_OPEN + t_cont, B)
            n_batches += 1

            payload = {
                "batch_idx": bi, "ladder": args.ladder, "t_cont": t_cont, "n_valid": n_valid,
                "starts": b_starts, "base_repeat_range": rep_rng,
                "base_res": base_res, "branches": branches,
                "dirs": {k: v.cpu() for k, v in dirs.items()}, "A0": A0.cpu(),
                "eps_by_family": eps_by_family,
            }
            if args.mode == "smoke":
                # (1) simulator restore determinism from a fresh snapshot
                snap0 = env.snapshot_state()
                v1 = _continuation_from_current(env, mdp, proto, pi_old, xi_one, B, t_cont)
                env.restore_state(snap0, restore_physics_xi=False)
                v2 = _continuation_from_current(env, mdp, proto, pi_old, xi_one, B, t_cont)
                restore_rng = float((v1 - v2).abs().max())
                count(2 * t_cont, B)
                # (2) cross-environment contamination: same slot-0 start, permuted companions
                import random as _random

                rest = list(range(1, B))
                _random.Random(SEED + 7 + bi).shuffle(rest)
                comp = [0] + rest               # slot 0 keeps its own start
                f0_perm = torch.tensor([b_starts[comp[i]]["frame"] for i in range(B)], device=dev)
                eps0 = eps_by_family["delta_2100"][0]
                A_perm = A0[comp] + eps0 * dirs["delta_2100"][comp].to(dev)
                b2 = run_branch(env, mdp, proto, pi_old, xi_one, f0_perm, A_perm, B, t_cont)
                count(H_OPEN + t_cont, B)
                slot0_diff = float((b2["cont"]["V_soft"][min(CONT_OUT[-1], t_cont)][0]
                                    - branches[f"delta_2100|eps{eps0}|plus"]["cont"]["V_soft"][min(CONT_OUT[-1], t_cont)][0]).abs())
                # (3) reward reconstruction + G-recalculation cross-check over all branches
                rew_max, g_max = 0.0, 0.0
                for br in [base_res[0], base_res[1], *branches.values()]:
                    rew_max = max(rew_max, float((br["full"]["r"]
                                                  - br["full"]["r_state"] - br["full"]["r_direct"]).abs().max()))
                    g_max = max(g_max, br["g_recalc_vsoft_absdiff"])
                smoke_summary = {"base_repeat_range": rep_rng,
                                 "restore_repeat_range": restore_rng,
                                 "companion_perm_slot0_absdiff": slot0_diff,
                                 "reward_reconstruction_max": rew_max,
                                 "g_recalc_max": g_max}
                payload["smoke"] = smoke_summary
                log("smoke_checks", **smoke_summary)
            out_dir = args.out_root / (args.split if args.split != "all" else "multi")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"batch_{bi:04d}.pt"
            torch.save(payload, out_path)
            sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
            log("batch_done", batch_idx=bi, path=str(out_path), sha256=sha,
                base_repeat_range=rep_rng, elapsed_s=round(time.time() - t0, 1))

        elapsed = time.time() - t0
        if counters["env_transitions"] > 0 and elapsed > 0:
            rate = counters["env_transitions"] / elapsed
            n_all = len(starts)
            rollouts_per_start = {"R0": 18, "R2": 14, "R3": 18}
            branch_steps = {"R0": H_OPEN + T_CONT, "R2": H_OPEN + T_CONT, "R3": H_OPEN + 64}
            proj = {}
            for name in ("R0", "R2", "R3"):
                total = n_all * rollouts_per_start[name] * branch_steps[name]
                proj[name] = {"total_transitions": total,
                              "gpu_hours": round(total / rate / 3600, 3)}
            log("throughput", transitions_per_s=round(rate, 2), projection=proj,
                elapsed_s=round(elapsed, 1))
            print(json.dumps({"elapsed_s": round(elapsed, 1),
                              "env_transitions": counters["env_transitions"],
                              "simulator_batch_steps": counters["simulator_batch_steps"],
                              "rate_per_s": round(rate, 2),
                              "smoke": smoke_summary or None, "projection": proj}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
