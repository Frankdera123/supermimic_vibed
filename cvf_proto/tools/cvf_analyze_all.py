"""CVF-0 Phase 1b Offline CPU Analysis (0 GPU):
Reads `results/cvf/phase1b/{oracle1_sim,oracle2_sim}.pt`, computes WM-side exact autograd
baselines and evaluates all 9 opcl_proto Critic snapshots across Level 1, Level 2 (Oracle 1),
Level 3 (Oracle 2 + Q3 attribution), and Level 4 (Cross-Critic Q5 matrix).
"""
from __future__ import annotations

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
    clip_clustered_bootstrap,
    critic_tangent_grad,
    loco_summary,
    rollout_wm_continuation,
)
from supermimic.crgf import reliability_gate, reward_split, soft_term, spearman  # noqa: E402
from supermimic.mdp import normalize_xi, unflatten_state  # noqa: E402
from supermimic.networks import Critic  # noqa: E402


def _pair_stats(pred: torch.Tensor, ref: torch.Tensor, clips: list[str]) -> dict:
    if len(pred) == 0:
        return {"n": 0}
    dec = (pred.abs() + ref.abs()) > 1e-8
    sign_vec = ((torch.sign(pred) == torch.sign(ref)) & dec).float()
    bs = clip_clustered_bootstrap(sign_vec, clips)
    loco = loco_summary(sign_vec, clips)
    rat = pred.abs() / ref.abs().clamp_min(1e-6)
    rmse = float((pred - ref).square().mean().sqrt())
    return {
        "n": int(len(pred)),
        "n_clips": bs["n_clips"],
        "sign_agree": bs["mean"],
        "sign_ci95": bs["ci95"],
        "loco_range": [loco["min_mean"], loco["max_mean"]],
        "spearman": spearman(pred, ref),
        "med_ratio": float(rat.median()),
        "iqr_ratio": [float(torch.quantile(rat, 0.25)), float(torch.quantile(rat, 0.75))],
        "rmse": rmse,
        "med_abs_pred": float(pred.abs().median()),
        "med_abs_ref": float(ref.abs().median()),
    }


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    if len(a) < 2:
        return float("nan")
    return float(torch.corrcoef(torch.stack((a.float(), b.float())))[0, 1])


def _train_aligned_prefix_J(r_state, r_direct, d_soft, H: int, gamma: float = 0.99):
    """r_state/r_direct/d_soft: (B, 32). Uses c_0=1, c_t = prod_{k<t}(1-d_k)."""
    B = r_state.shape[0]
    c = torch.ones(B)
    Js = torch.zeros(B)
    Jd = torch.zeros(B)
    for t in range(H):
        Js = Js + (gamma ** t) * c * r_state[:, t]
        Jd = Jd + (gamma ** t) * c * r_direct[:, t]
        c = c * (1.0 - d_soft[:, t])
    return Js, Jd, c  # c is c_H


def main():
    t0 = time.time()
    out_dir = Path("results/cvf/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    o1 = torch.load("results/cvf/phase1b/oracle1_sim.pt", map_location="cpu", weights_only=False)
    o2 = torch.load("results/cvf/phase1b/oracle2_sim.pt", map_location="cpu", weights_only=False)
    p0 = torch.load("results/cvf/phase0/phase0_starts_and_dirs.pt", map_location="cpu", weights_only=False)

    pair_2100 = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
    world = bridge.setup_world(device="cpu", pair=pair_2100)
    motion, mdp, proto = world["motion"], world["mdp"], world["proto"]
    wm = pair_2100["wm"]
    pi_old = pair_2100["pi_old"]

    sa = bridge.pairs_root()
    critic_specs = [
        ("b0@1000", sa / "b0_resume/ckpt_001000.pt", "critic"),
        ("b0@1500", sa / "b0_resume/ckpt_001500.pt", "critic"),
        ("b0@2000", sa / "b0_resume/ckpt_002000.pt", "critic"),
        ("B0@2100_pre", sa / "online_B0/diag_pairs/pair_it_002100.pt", "critic_pre"),
        ("B0@2100_post", sa / "online_B0/diag_pairs/pair_it_002100.pt", "critic_post"),
        ("B0@2300", sa / "online_B0/ckpt_002300.pt", "critic"),
        ("B1@2300", sa / "online_B1/ckpt_002300.pt", "critic"),
        ("B2@2300", sa / "online_B2/ckpt_002300.pt", "critic"),
        ("B3@2300", sa / "online_B3/ckpt_002300.pt", "critic"),
    ]
    critics = {}
    for cname, cpath, ckey in critic_specs:
        ck = torch.load(cpath, map_location="cpu", weights_only=False)
        cin = ck[ckey]["norm.mean"].shape[0]
        cr = Critic(cin).eval()
        cr.load_state_dict(ck[ckey], strict=True)
        critics[cname] = cr

    # =========================================================================
    # PART 1: ORACLE 1 ANALYSIS (S_star_sim, T in {32, 64, 96, 128})
    # =========================================================================
    print("[1/3] Analyzing Oracle 1 (direct state perturbation at real S_star_sim)...")
    starts16 = o1["starts"]
    clips16 = [s["clip_name"] for s in starts16]
    eps_grid = o1["eps_grid"]  # [0.015, 0.03]
    horizons_o1 = o1["horizons_out"]  # [32, 64, 96, 128]
    dir_names_o1 = list(o1["batches"][0]["dirs"].keys())

    # Assemble 16-start tensors from 4 batches
    S_star_all = tuple(torch.cat([b["S_star"][k] for b in o1["batches"]], 0) for k in range(6))
    f_star_all = torch.cat([b["f_star"] for b in o1["batches"]], 0)
    a_prev_star_all = torch.cat([b["a_prev_star"] for b in o1["batches"]], 0)
    oh_prev9_star_all = torch.cat([b["oh_prev9_star"] for b in o1["batches"]], 0)
    dirs_o1_all = {d: torch.cat([b["dirs"][d] for b in o1["batches"]], 0) for d in dir_names_o1}
    xi_16 = torch.zeros(16, 33); xi_16[:, 0] = 1.0
    xi_n_16 = normalize_xi(xi_16)

    # Exact WM continuation gradients at real S_star_sim
    wm_o1 = rollout_wm_continuation(
        wm, pi_old, mdp, proto, S_star_all, f_star_all, oh_prev9_star_all, a_prev_star_all,
        xi_16, xi_n_16, T=128, hist_mode="last_only", need_grad=True, horizons_out=tuple(horizons_o1)
    )

    # Base scalar values at S_star_sim
    V_sim_base_soft = {h: torch.cat([b["base_res"][0]["J_soft"][h] for b in o1["batches"]], 0) for h in horizons_o1}
    V_sim_base_hard = {h: torch.cat([b["base_res"][0]["J_hard"][h] for b in o1["batches"]], 0) for h in horizons_o1}
    base_reps_T = {h: torch.stack([
        torch.cat([b["base_res"][0]["J_soft"][h] for b in o1["batches"]], 0),
        torch.cat([b["base_res"][1]["J_soft"][h] for b in o1["batches"]], 0),
    ], 1) for h in horizons_o1}  # (16, 2)

    # Compute Critic gradients for all 9 critics at S_star_sim
    crit_o1 = {cname: critic_tangent_grad(cr, mdp, motion, proto, S_star_all, f_star_all, xi_n_16)
               for cname, cr in critics.items()}

    # Level 1 scalar calibration
    l1_scalar = {}
    for cname, cg in crit_o1.items():
        vr = cg["v_raw"]
        l1_scalar[cname] = {
            "mean_V_crit": float(vr.mean()),
            "mean_V_wm128": float(wm_o1["J_soft"][128].mean()),
            "mean_V_sim128_soft": float(V_sim_base_soft[128].mean()),
            "mean_V_sim128_hard": float(V_sim_base_hard[128].mean()),
            "pearson_vs_wm128": _pearson(vr, wm_o1["J_soft"][128]),
            "pearson_vs_sim128_soft": _pearson(vr, V_sim_base_soft[128]),
            "pearson_vs_sim128_hard": _pearson(vr, V_sim_base_hard[128]),
            "spearman_vs_sim128_soft": spearman(vr, V_sim_base_soft[128]),
        }

    # Assemble Sim FD & gates per (horizon, dir, start)
    o1_rows = []
    o1_summary = {}
    for h in horizons_o1:
        o1_summary[f"T{h}"] = {}
        for dname in dir_names_o1:
            u = dirs_o1_all[dname]
            d_wm = (wm_o1["grads_soft"][h] * u).sum(-1)
            d_sim_by_eps = {}
            d_sim_hard_by_eps = {}
            for eps in eps_grid:
                jp = torch.cat([b["branches"][(dname, eps, "plus")]["J_soft"][h] for b in o1["batches"]], 0)
                jm = torch.cat([b["branches"][(dname, eps, "minus")]["J_soft"][h] for b in o1["batches"]], 0)
                jph = torch.cat([b["branches"][(dname, eps, "plus")]["J_hard"][h] for b in o1["batches"]], 0)
                jmh = torch.cat([b["branches"][(dname, eps, "minus")]["J_hard"][h] for b in o1["batches"]], 0)
                d_sim_by_eps[eps] = (jp - jm) / (2.0 * eps)
                d_sim_hard_by_eps[eps] = (jph - jmh) / (2.0 * eps)
            d_sim_pool = 0.5 * (d_sim_by_eps[eps_grid[0]] + d_sim_by_eps[eps_grid[1]])
            gates = []
            reasons = []
            for i in range(16):
                ok, why = reliability_gate(base_reps_T[h][i], torch.tensor([d_sim_by_eps[e][i] for e in eps_grid]))
                gates.append(ok); reasons.append(why)
            gate_mask = torch.tensor(gates, dtype=torch.bool)
            gated_clips = [clips16[i] for i in range(16) if gates[i]]

            # Primary critic B0@2100_post + all critics
            crit_block = {}
            for cname, cg in crit_o1.items():
                d_crit = (cg["grad_raw"] * u).sum(-1)
                crit_block[cname] = {
                    "crit_vs_sim_gated": _pair_stats(d_crit[gate_mask], d_sim_pool[gate_mask], gated_clips),
                    "crit_vs_wm_gated": _pair_stats(d_crit[gate_mask], d_wm[gate_mask], gated_clips),
                    "crit_vs_sim_all16_eps015": _pair_stats(d_crit, d_sim_by_eps[0.015], clips16),
                }
            o1_summary[f"T{h}"][dname] = {
                "n_gated": int(gate_mask.sum()),
                "gated_clips": sorted(set(gated_clips)),
                "wm_vs_sim_gated": _pair_stats(d_wm[gate_mask], d_sim_pool[gate_mask], gated_clips),
                "wm_vs_sim_all16_eps015": _pair_stats(d_wm, d_sim_by_eps[0.015], clips16),
                "critics": crit_block,
            }
            if h in (64, 128):
                d_c_main = (crit_o1["B0@2100_post"]["grad_raw"] * u).sum(-1)
                for i in range(16):
                    o1_rows.append({
                        "T": h, "direction": dname, "clip": clips16[i], "frame": starts16[i]["frame"],
                        "gate": int(gates[i]), "gate_reason": reasons[i],
                        "D_sim_pool": float(d_sim_pool[i]), "D_sim_015": float(d_sim_by_eps[0.015][i]),
                        "D_sim_030": float(d_sim_by_eps[0.03][i]), "D_wm": float(d_wm[i]),
                        "D_crit_B0_2100": float(d_c_main[i]),
                    })

    # =========================================================================
    # PART 2: ORACLE 2 ANALYSIS (Reachable Manifold H=32 + T_cont=96 & Q3)
    # =========================================================================
    print("[2/3] Analyzing Oracle 2 (action-induced reachable manifold & Q3 attribution)...")
    idx16 = o2["idx16"]
    f0_16 = p0["f0"][idx16]
    S0_16 = motion.ref_state(f0_16)
    base_A_16 = p0["oracle2_base_A"][idx16]
    dirs_o2_all = {k: v[idx16] for k, v in p0["oracle2_dirs"].items()}
    dir_names_o2 = list(dirs_o2_all.keys())
    gamma = 0.99

    # Base reps for Oracle 2 gating
    base_r_state = torch.cat([b["base_res"][0]["r_state"] for b in o2["batches"]], 0)
    base_r_direct = torch.cat([b["base_res"][0]["r_direct"] for b in o2["batches"]], 0)
    base_d_soft = torch.cat([b["base_res"][0]["d_soft"] for b in o2["batches"]], 0)
    base_Js, base_Jd, base_c32 = _train_aligned_prefix_J(base_r_state, base_r_direct, base_d_soft, 32, gamma)
    base_Vcont0 = torch.cat([b["base_res"][0]["V_cont_soft"][96] for b in o2["batches"]], 0)
    base_Vcont1 = torch.cat([b["base_res"][1]["V_cont_soft"][96] for b in o2["batches"]], 0)
    base_Jtot_true0 = base_Js + base_Jd + (gamma ** 32) * base_c32 * base_Vcont0
    base_Jtot_true1 = base_Js + base_Jd + (gamma ** 32) * base_c32 * base_Vcont1
    base_o2_reps = torch.stack((base_Jtot_true0, base_Jtot_true1), 1)

    # WM-side exact autograd directional derivatives for Oracle 2 (H=32)
    def wm_o2_grads(critic_obj):
        A_seq = base_A_16.clone().requires_grad_(True)
        S = tuple(x.clone() for x in S0_16)
        ap = torch.zeros(16, 29)
        c = torch.ones(16)
        J_state = torch.zeros(16)
        J_direct = torch.zeros(16)
        f = f0_16.clone()
        for t in range(32):
            a = torch.clamp(A_seq[:, t], -20.0, 20.0)
            S = proto.wm_step(wm, S, a, xi_n_16)
            f = f + 1
            rs, rd = reward_split(mdp, S, f, a, ap)
            ds = soft_term(mdp, S, f)
            J_state = J_state + (gamma ** t) * c * rs
            J_direct = J_direct + (gamma ** t) * c * rd
            c = c * (1.0 - ds)
            ap = a
        gw = proto.wm_goal(f, S)
        V = critic_obj(proto.critic_input(S, f, xi_n_16, gw)).clamp(max=10.0)
        J_crit = (gamma ** 32) * c * V
        J_tot = J_state + J_direct + J_crit
        g_s = torch.autograd.grad(J_state.sum(), A_seq, retain_graph=True)[0].detach()
        g_d = torch.autograd.grad(J_direct.sum(), A_seq, retain_graph=True)[0].detach()
        g_c = torch.autograd.grad(J_crit.sum(), A_seq, retain_graph=True)[0].detach()
        g_t = torch.autograd.grad(J_tot.sum(), A_seq, retain_graph=False)[0].detach()
        return {"state": g_s, "direct": g_d, "critic": g_c, "total": g_t}

    wm_g_by_crit = {cname: wm_o2_grads(cr) for cname, cr in critics.items()}

    o2_summary = {}
    o2_rows = []
    for dname in dir_names_o2:
        dA = dirs_o2_all[dname]
        # Sim FD per eps
        sim_paths_by_eps = {}
        for eps in eps_grid:
            res_pm = {}
            for br in ("plus", "minus"):
                rs = torch.cat([b["branches"][(dname, eps, br)]["r_state"] for b in o2["batches"]], 0)
                rd = torch.cat([b["branches"][(dname, eps, br)]["r_direct"] for b in o2["batches"]], 0)
                ds = torch.cat([b["branches"][(dname, eps, br)]["d_soft"] for b in o2["batches"]], 0)
                Js, Jd, c32 = _train_aligned_prefix_J(rs, rd, ds, 32, gamma)
                S32 = unflatten_state(torch.cat([b["branches"][(dname, eps, br)]["S_at_32"] for b in o2["batches"]], 0))
                f32 = torch.cat([b["branches"][(dname, eps, br)]["f_at_32"] for b in o2["batches"]], 0)
                Vcont96 = torch.cat([b["branches"][(dname, eps, br)]["V_cont_soft"][96] for b in o2["batches"]], 0)
                Vcont96_h = torch.cat([b["branches"][(dname, eps, br)]["V_cont_hard"][96] for b in o2["batches"]], 0)
                J_cont = (gamma ** 32) * c32 * Vcont96
                J_cont_h = (gamma ** 32) * c32 * Vcont96_h
                with torch.no_grad():
                    J_crit_by_c = {
                        cname: (gamma ** 32) * c32 * cr(proto.critic_input(S32, f32, xi_n_16, proto.wm_goal(f32, S32))).clamp(max=10.0)
                        for cname, cr in critics.items()
                    }
                res_pm[br] = {"state": Js, "direct": Jd, "cont_true": J_cont, "cont_true_hard": J_cont_h, "crit_learned": J_crit_by_c}
            D_state = (res_pm["plus"]["state"] - res_pm["minus"]["state"]) / (2.0 * eps)
            D_direct = (res_pm["plus"]["direct"] - res_pm["minus"]["direct"]) / (2.0 * eps)
            D_cont = (res_pm["plus"]["cont_true"] - res_pm["minus"]["cont_true"]) / (2.0 * eps)
            D_cont_h = (res_pm["plus"]["cont_true_hard"] - res_pm["minus"]["cont_true_hard"]) / (2.0 * eps)
            D_crit_l = {cname: (res_pm["plus"]["crit_learned"][cname] - res_pm["minus"]["crit_learned"][cname]) / (2.0 * eps)
                        for cname in critics}
            sim_paths_by_eps[eps] = {
                "state": D_state, "direct": D_direct, "cont_true": D_cont, "cont_true_hard": D_cont_h,
                "total_true": D_state + D_direct + D_cont, "crit_learned": D_crit_l,
            }

        # Gate on state + cont_true (total_true)
        gates = []
        for i in range(16):
            ok, _ = reliability_gate(base_o2_reps[i], torch.tensor([sim_paths_by_eps[e]["total_true"][i] for e in eps_grid]))
            gates.append(ok)
        gmask = torch.tensor(gates, dtype=torch.bool)
        gclips = [clips16[i] for i in range(16) if gates[i]]

        D_sim_state = 0.5 * (sim_paths_by_eps[0.015]["state"] + sim_paths_by_eps[0.03]["state"])
        D_sim_cont = 0.5 * (sim_paths_by_eps[0.015]["cont_true"] + sim_paths_by_eps[0.03]["cont_true"])
        D_sim_tot_true = 0.5 * (sim_paths_by_eps[0.015]["total_true"] + sim_paths_by_eps[0.03]["total_true"])

        crit_o2_block = {}
        for cname in critics:
            D_sim_crit_l = 0.5 * (sim_paths_by_eps[0.015]["crit_learned"][cname] + sim_paths_by_eps[0.03]["crit_learned"][cname])
            D_sim_tot_l = D_sim_state + 0.5 * (sim_paths_by_eps[0.015]["direct"] + sim_paths_by_eps[0.03]["direct"]) + D_sim_crit_l
            D_wm_state = (wm_g_by_crit[cname]["state"] * dA).sum((1, 2))
            D_wm_crit_l = (wm_g_by_crit[cname]["critic"] * dA).sum((1, 2))
            D_wm_tot_l = (wm_g_by_crit[cname]["total"] * dA).sum((1, 2))

            e_state = D_wm_state - D_sim_state
            e_critic_true = D_wm_crit_l - D_sim_cont
            crit_o2_block[cname] = {
                "Q4_sim_learned_vs_sim_cont_true": _pair_stats(D_sim_crit_l[gmask], D_sim_cont[gmask], gclips),
                "WM_critic_vs_sim_cont_true": _pair_stats(D_wm_crit_l[gmask], D_sim_cont[gmask], gclips),
                "shared_illusion_WM_tot_vs_sim_tot_learned": _pair_stats(D_wm_tot_l[gmask], D_sim_tot_l[gmask], gclips),
                "true_physical_WM_tot_vs_sim_tot_true": _pair_stats(D_wm_tot_l[gmask], D_sim_tot_true[gmask], gclips),
                "Q3_err_corr_pearson_gated": _pearson(e_state[gmask], e_critic_true[gmask]),
                "Q3_err_spearman_gated": spearman(e_state[gmask], e_critic_true[gmask]),
            }

        o2_summary[dname] = {
            "n_gated": int(gmask.sum()),
            "gated_clips": sorted(set(gclips)),
            "WM_state_vs_sim_state": _pair_stats(
                (wm_g_by_crit["B0@2100_post"]["state"] * dA).sum((1, 2))[gmask],
                D_sim_state[gmask], gclips),
            "critics": crit_o2_block,
        }
        # Save per-start rows for primary critic B0@2100_post
        D_wm_s_main = (wm_g_by_crit["B0@2100_post"]["state"] * dA).sum((1, 2))
        D_wm_c_main = (wm_g_by_crit["B0@2100_post"]["critic"] * dA).sum((1, 2))
        D_wm_t_main = (wm_g_by_crit["B0@2100_post"]["total"] * dA).sum((1, 2))
        D_sim_cl_main = 0.5 * (sim_paths_by_eps[0.015]["crit_learned"]["B0@2100_post"] + sim_paths_by_eps[0.03]["crit_learned"]["B0@2100_post"])
        for i in range(16):
            o2_rows.append({
                "direction": dname, "clip": clips16[i], "frame": starts16[i]["frame"], "gate": int(gates[i]),
                "D_sim_state": float(D_sim_state[i]), "D_wm_state": float(D_wm_s_main[i]),
                "D_sim_cont_true": float(D_sim_cont[i]), "D_sim_crit_learned": float(D_sim_cl_main[i]),
                "D_wm_crit_learned": float(D_wm_c_main[i]), "D_sim_tot_true": float(D_sim_tot_true[i]),
                "D_wm_tot_learned": float(D_wm_t_main[i]),
            })

    summary = {
        "elapsed_s": round(time.time() - t0, 1),
        "level1_scalar": l1_scalar,
        "level2_oracle1": o1_summary,
        "level3_oracle2": o2_summary,
    }
    with open(out_dir / "cvf_full_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(out_dir / "oracle1_per_start.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(o1_rows[0].keys()))
        w.writeheader(); w.writerows(o1_rows)
    with open(out_dir / "oracle2_per_start.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(o2_rows[0].keys()))
        w.writeheader(); w.writerows(o2_rows)

    # Print concise executive readout
    print("\n=== ORACLE 1 (B0@2100_post, T=64 & T=128) ===")
    for h in (64, 128):
        print(f"  -- T={h} --")
        for dname in dir_names_o1:
            blk = o1_summary[f"T{h}"][dname]
            cs = blk["critics"]["B0@2100_post"]["crit_vs_sim_gated"]
            cw = blk["critics"]["B0@2100_post"]["crit_vs_wm_gated"]
            ws = blk["wm_vs_sim_gated"]
            print(f"    {dname:17s} (ng={blk['n_gated']:2d}): CvsSim sign={cs.get('sign_agree', float('nan')):.2f} sp={cs.get('spearman', float('nan')):+.2f} rat={cs.get('med_ratio', float('nan')):.2f} | CvsWM sign={cw.get('sign_agree', float('nan')):.2f} sp={cw.get('spearman', float('nan')):+.2f} | WMvsSim sign={ws.get('sign_agree', float('nan')):.2f} sp={ws.get('spearman', float('nan')):+.2f}")

    print("\n=== ORACLE 2 (B0@2100_post, H=32 + T_cont=96) ===")
    for dname in dir_names_o2:
        blk = o2_summary[dname]
        cb = blk["critics"]["B0@2100_post"]
        q4 = cb["Q4_sim_learned_vs_sim_cont_true"]
        il = cb["shared_illusion_WM_tot_vs_sim_tot_learned"]
        tr = cb["true_physical_WM_tot_vs_sim_tot_true"]
        ws = blk["WM_state_vs_sim_state"]
        print(f"  {dname:11s} (ng={blk['n_gated']:2d}): WMst vs SimSt sign={ws.get('sign_agree', float('nan')):.2f} | Q4(SimCrit vs TrueCont) sign={q4.get('sign_agree', float('nan')):.2f} sp={q4.get('spearman', float('nan')):+.2f} rat={q4.get('med_ratio', float('nan')):.2f} | IllusTot sign={il.get('sign_agree', float('nan')):.2f} vs TrueTot sign={tr.get('sign_agree', float('nan')):.2f} | Q3_err_r={cb['Q3_err_corr_pearson_gated']:+.3f}")

    print(f"\nDone in {summary['elapsed_s']}s -> {out_dir}")


if __name__ == "__main__":
    main()
