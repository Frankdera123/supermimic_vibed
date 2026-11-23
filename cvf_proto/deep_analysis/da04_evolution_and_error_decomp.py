"""Phase 7 & 9: Nine-Critic Longitudinal Evolution & Error A vs Error B Decomposition (CPU-only)."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent.parent
sys.path.insert(0, str(REPO / "pada_proto"))
sys.path.insert(0, str(REPO / "cvf_proto"))

import pada_bridge as bridge  # noqa: E402

bridge.import_stack()
bridge.fix_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import cvf_core  # noqa: E402
from supermimic.crgf import reliability_gate, reward_split, soft_term, spearman  # noqa: E402
from supermimic.mdp import normalize_xi, unflatten_state  # noqa: E402
from supermimic.networks import Critic  # noqa: E402

assert not torch.cuda.is_available(), "Must run CPU-only"
torch.set_num_threads(min(16, os.cpu_count() or 4))

OUT = REPO / "results" / "cvf" / "deep_analysis"
OUT.mkdir(parents=True, exist_ok=True)
SA = bridge.pairs_root()
GAMMA = 0.99
V_MAX = 10.0

CRITIC_SPECS = [
    ("b0@1000", SA / "b0_resume/ckpt_001000.pt", "critic", 1000, "b0_resume"),
    ("b0@1500", SA / "b0_resume/ckpt_001500.pt", "critic", 1500, "b0_resume"),
    ("b0@2000", SA / "b0_resume/ckpt_002000.pt", "critic", 2000, "b0_resume"),
    ("B0@2100_pre", SA / "online_B0/diag_pairs/pair_it_002100.pt", "critic_pre", 2100, "online_B0"),
    ("B0@2100_post", SA / "online_B0/diag_pairs/pair_it_002100.pt", "critic_post", 2100, "online_B0"),
    ("B0@2300", SA / "online_B0/ckpt_002300.pt", "critic", 2300, "online_B0"),
    ("B1@2300", SA / "online_B1/ckpt_002300.pt", "critic", 2300, "online_B1"),
    ("B2@2300", SA / "online_B2/ckpt_002300.pt", "critic", 2300, "online_B2"),
    ("B3@2300", SA / "online_B3/ckpt_002300.pt", "critic", 2300, "online_B3"),
]


def err_decomp_stats(d_c: torch.Tensor, d_m: torch.Tensor, d_s: torch.Tensor) -> dict:
    e_A = d_c - d_m
    e_B = d_m - d_s
    e_tot = d_c - d_s
    mse_A = float(e_A.square().mean())
    mse_B = float(e_B.square().mean())
    mse_tot = float(e_tot.square().mean())
    cross2 = float(2.0 * (e_A * e_B).mean())
    denom = max(mse_A + mse_B, 1e-12)
    pearson_AB = float(torch.corrcoef(torch.stack((e_A, e_B)))[0, 1]) if len(e_A) >= 2 and e_A.std() > 0 and e_B.std() > 0 else float("nan")
    return {
        "n": int(len(e_A)),
        "mean_eA": float(e_A.mean()),
        "mean_eB": float(e_B.mean()),
        "mae_eA": float(e_A.abs().mean()),
        "mae_eB": float(e_B.abs().mean()),
        "rmse_eA": float(mse_A ** 0.5),
        "rmse_eB": float(mse_B ** 0.5),
        "rmse_tot": float(mse_tot ** 0.5),
        "mse_eA": mse_A,
        "mse_eB": mse_B,
        "cross_2E_eAeB": cross2,
        "mse_tot": mse_tot,
        "mse_identity_err": abs(mse_tot - (mse_A + mse_B + cross2)),
        "share_A_of_sum": mse_A / denom,
        "interaction_ratio": cross2 / denom,  # <0 cancellation, >0 amplification
        "pearson_eA_eB": pearson_AB,
    }


def main():
    o1 = torch.load(REPO / "results/cvf/phase1b/oracle1_sim.pt", map_location="cpu", weights_only=False)
    o2 = torch.load(REPO / "results/cvf/phase1b/oracle2_sim.pt", map_location="cpu", weights_only=False)
    p0 = torch.load(REPO / "results/cvf/phase0/phase0_starts_and_dirs.pt", map_location="cpu", weights_only=False)

    pair = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
    world = bridge.setup_world(device="cpu", pair=pair)
    motion, mdp, proto = world["motion"], world["mdp"], world["proto"]
    wm, pi_old = pair["wm"], pair["pi_old"]

    S_star = tuple(torch.cat([b["S_star"][k] for b in o1["batches"]], 0) for k in range(6))
    f_star = torch.cat([b["f_star"] for b in o1["batches"]], 0)
    ap_star = torch.cat([b["a_prev_star"] for b in o1["batches"]], 0)
    oh9_star = torch.cat([b["oh_prev9_star"] for b in o1["batches"]], 0)
    xi_16 = torch.zeros(16, 33); xi_16[:, 0] = 1.0
    xi_n_16 = normalize_xi(xi_16)
    dir_names = list(o1["batches"][0]["dirs"].keys())
    dirs_all = {d: torch.cat([b["dirs"][d] for b in o1["batches"]], 0) for d in dir_names}
    clips16 = [s["clip_name"] for s in o1["starts"]]

    critics = {}
    for cname, cpath, ckey, _, _ in CRITIC_SPECS:
        ck = torch.load(cpath, map_location="cpu", weights_only=False)
        cr = Critic(ck[ckey]["norm.mean"].shape[0]).eval()
        cr.load_state_dict(ck[ckey], strict=True)
        for p in cr.parameters():
            p.requires_grad_(False)
        critics[cname] = cr

    # WM O1 rollout
    wm_o1 = cvf_core.rollout_wm_continuation(
        wm, pi_old, mdp, proto, S_star, f_star, oh9_star, ap_star,
        xi_16, xi_n_16, T=128, hist_mode="last_only", need_grad=True, horizons_out=(32, 64, 96, 128)
    )
    V_sim128 = torch.cat([b["base_res"][0]["J_soft"][128] for b in o1["batches"]], 0)

    # Precompute Sim O1 slopes (eps=0.015 & gated pool)
    base_reps_T = {h: torch.stack([
        torch.cat([b["base_res"][0]["J_soft"][h] for b in o1["batches"]], 0),
        torch.cat([b["base_res"][1]["J_soft"][h] for b in o1["batches"]], 0),
    ], 1) for h in (32, 64, 96, 128)}
    sim_o1 = {}
    sim_o1_pool = {}
    sim_o1_gates = {}
    for h in (32, 64, 96, 128):
        sim_o1[h], sim_o1_pool[h], sim_o1_gates[h] = {}, {}, {}
        for dname in dir_names:
            d015 = (
                torch.cat([b["branches"][(dname, 0.015, "plus")]["J_soft"][h] for b in o1["batches"]], 0)
                - torch.cat([b["branches"][(dname, 0.015, "minus")]["J_soft"][h] for b in o1["batches"]], 0)
            ) / 0.030
            d030 = (
                torch.cat([b["branches"][(dname, 0.03, "plus")]["J_soft"][h] for b in o1["batches"]], 0)
                - torch.cat([b["branches"][(dname, 0.03, "minus")]["J_soft"][h] for b in o1["batches"]], 0)
            ) / 0.060
            sim_o1[h][dname] = d015
            sim_o1_pool[h][dname] = 0.5 * (d015 + d030)
            sim_o1_gates[h][dname] = torch.tensor([reliability_gate(base_reps_T[h][i], torch.tensor([d015[i], d030[i]]))[0] for i in range(16)], dtype=torch.bool)

    # Precompute Oracle 2 Sim references (ALL_POLICY n=48)
    idx16 = o2["idx16"]
    f0_16 = p0["f0"][idx16]
    S0_16 = motion.ref_state(f0_16)
    base_A_16 = p0["oracle2_base_A"][idx16]
    dirs_o2 = {k: v[idx16] for k, v in p0["oracle2_dirs"].items()}
    policy_dirs = ["delta_2100", "delta_2300", "ascent"]

    # Rollout WM state/direct once for Oracle 2
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
        J_state = J_state + (GAMMA ** t) * c * rs
        J_direct = J_direct + (GAMMA ** t) * c * rd
        c = c * (1.0 - ds)
        ap = a
    g_wm_state_A = torch.autograd.grad(J_state.sum(), A_seq, retain_graph=True)[0].detach()
    g_wm_direct_A = torch.autograd.grad(J_direct.sum(), A_seq, retain_graph=True)[0].detach()
    S32_wm_det = tuple(x.detach() for x in S)
    f32_wm = f.detach()
    c32_wm_det = c.detach()

    # Sim O2 terms per policy dir
    sim_o2_cache = {}
    for dname in policy_dirs:
        by_eps = {}
        for eps in o2["eps_grid"]:
            pm = {}
            for br in ("plus", "minus"):
                rs = torch.cat([b["branches"][(dname, eps, br)]["r_state"] for b in o2["batches"]], 0)
                rd = torch.cat([b["branches"][(dname, eps, br)]["r_direct"] for b in o2["batches"]], 0)
                ds = torch.cat([b["branches"][(dname, eps, br)]["d_soft"] for b in o2["batches"]], 0)
                cc = torch.ones(16); Js = torch.zeros(16); Jd = torch.zeros(16)
                for t in range(32):
                    Js = Js + (GAMMA ** t) * cc * rs[:, t]
                    Jd = Jd + (GAMMA ** t) * cc * rd[:, t]
                    cc = cc * (1.0 - ds[:, t])
                S32 = unflatten_state(torch.cat([b["branches"][(dname, eps, br)]["S_at_32"] for b in o2["batches"]], 0))
                f32 = torch.cat([b["branches"][(dname, eps, br)]["f_at_32"] for b in o2["batches"]], 0)
                Vcont = torch.cat([b["branches"][(dname, eps, br)]["V_cont_soft"][96] for b in o2["batches"]], 0)
                pm[br] = {"state": Js, "direct": Jd, "cont": (GAMMA ** 32) * cc * Vcont, "S32": S32, "f32": f32, "c32": cc}
            by_eps[eps] = pm
        sim_o2_cache[dname] = by_eps

    # 1. Nine-Critic Longitudinal Evolution
    print("=== [1] Nine-Critic Longitudinal Evolution (Common-Policy Evaluation) ===")
    evo_rows = []
    S_det = tuple(x.detach() for x in S_star)
    for cname, _, _, it_num, run_name in CRITIC_SPECS:
        cr = critics[cname]
        cg = cvf_core.critic_tangent_grad(cr, mdp, motion, proto, S_star, f_star, xi_n_16, clamp_vmax=V_MAX)
        v_raw = cg["v_raw"]
        g_crit = cg["grad_raw"]

        # Directional curvature on pose_safe_critic and vel_reachable
        kappas = {}
        for dname in ("pose_safe_critic", "vel_reachable"):
            u = dirs_all[dname]
            dlt = torch.zeros(16, 70, requires_grad=True)
            Sp = cvf_core.perturb_state(S_det, dlt)
            vv = cr(proto.critic_input(Sp, f_star, xi_n_16, proto.wm_goal(f_star, Sp)))
            g1 = torch.autograd.grad(vv.sum(), dlt, create_graph=True)[0]
            g2 = torch.autograd.grad((g1 * u).sum(), dlt)[0].detach()
            kappas[dname] = float((g2 * u).sum(-1).abs().median())

        # O1 metrics at T=64 (eps=0.015)
        o1_m = {}
        for dname in ("pose_safe_critic", "vel_critic", "vel_reachable", "vel_random"):
            u = dirs_all[dname]
            dc = (g_crit * u).sum(-1)
            ds = sim_o1[64][dname]
            dw = (wm_o1["grads_soft"][64] * u).sum(-1)
            o1_m[dname] = {
                "sign_sim64": float(((dc * ds) > 0).float().mean()),
                "sp_sim64": spearman(dc, ds),
                "ratio_sim64": float((dc.abs() / ds.abs().clamp_min(1e-6)).median()),
                "sign_wm64": float(((dc * dw) > 0).float().mean()),
            }

        # O2 ALL_POLICY (n=48) for this critic
        # Need d(J_crit_wm)/dA for this critic:
        # Since S32_wm depends on A_seq, recompute S32_wm with grad for this critic
        A_seq_c = base_A_16.clone().requires_grad_(True)
        Sc = tuple(x.clone() for x in S0_16)
        cc_w = torch.ones(16); fc_w = f0_16.clone()
        for t in range(32):
            ac = torch.clamp(A_seq_c[:, t], -20.0, 20.0)
            Sc = proto.wm_step(wm, Sc, ac, xi_n_16)
            fc_w = fc_w + 1
            cc_w = cc_w * (1.0 - soft_term(mdp, Sc, fc_w))
        Vc_w = cr(proto.critic_input(Sc, fc_w, xi_n_16, proto.wm_goal(fc_w, Sc))).clamp(max=V_MAX)
        Jc_w = (GAMMA ** 32) * cc_w * Vc_w
        g_wm_crit_A = torch.autograd.grad(Jc_w.sum(), A_seq_c)[0].detach()

        v_sim_s, v_sim_c, v_sim_cl, v_sim_tot, v_wm_s, v_wm_c, v_wm_t = [], [], [], [], [], [], []
        for dname in policy_dirs:
            dA = dirs_o2[dname]
            ds_e, dd_e, dc_e, dcl_e = [], [], [], []
            for eps in o2["eps_grid"]:
                pm = sim_o2_cache[dname][eps]
                ds_e.append((pm["plus"]["state"] - pm["minus"]["state"]) / (2 * eps))
                dd_e.append((pm["plus"]["direct"] - pm["minus"]["direct"]) / (2 * eps))
                dc_e.append((pm["plus"]["cont"] - pm["minus"]["cont"]) / (2 * eps))
                with torch.no_grad():
                    vp = (GAMMA ** 32) * pm["plus"]["c32"] * cr(proto.critic_input(pm["plus"]["S32"], pm["plus"]["f32"], xi_n_16, proto.wm_goal(pm["plus"]["f32"], pm["plus"]["S32"]))).clamp(max=V_MAX)
                    vm = (GAMMA ** 32) * pm["minus"]["c32"] * cr(proto.critic_input(pm["minus"]["S32"], pm["minus"]["f32"], xi_n_16, proto.wm_goal(pm["minus"]["f32"], pm["minus"]["S32"]))).clamp(max=V_MAX)
                dcl_e.append((vp - vm) / (2 * eps))
            Dss = 0.5 * (ds_e[0] + ds_e[1])
            Dsd = 0.5 * (dd_e[0] + dd_e[1])
            Dsc = 0.5 * (dc_e[0] + dc_e[1])
            Dscl = 0.5 * (dcl_e[0] + dcl_e[1])
            v_sim_s.append(Dss); v_sim_c.append(Dsc); v_sim_cl.append(Dscl); v_sim_tot.append(Dss + Dsd + Dsc)
            v_wm_s.append((g_wm_state_A * dA).sum((1, 2)))
            v_wm_c.append((g_wm_crit_A * dA).sum((1, 2)))
            v_wm_t.append(((g_wm_state_A + g_wm_direct_A + g_wm_crit_A) * dA).sum((1, 2)))

        v_sim_s = torch.cat(v_sim_s); v_sim_c = torch.cat(v_sim_c); v_sim_cl = torch.cat(v_sim_cl); v_sim_tot = torch.cat(v_sim_tot)
        v_wm_s = torch.cat(v_wm_s); v_wm_c = torch.cat(v_wm_c); v_wm_t = torch.cat(v_wm_t)

        row = {
            "critic": cname, "run": run_name, "iter": it_num,
            "val_pearson_sim128": float(torch.corrcoef(torch.stack((v_raw, V_sim128)))[0, 1]),
            "val_bias_sim128": float((v_raw - V_sim128).mean()),
            "grad_norm_mean": float(g_crit.norm(dim=-1).mean()),
            "norm_std_mean": float(cr.norm.std().mean()),
            "pose_c_sign64": o1_m["pose_safe_critic"]["sign_sim64"],
            "pose_c_sp64": o1_m["pose_safe_critic"]["sp_sim64"],
            "pose_c_ratio64": o1_m["pose_safe_critic"]["ratio_sim64"],
            "vel_c_sign64": o1_m["vel_critic"]["sign_sim64"],
            "vel_c_sp64": o1_m["vel_critic"]["sp_sim64"],
            "vel_r_sign64": o1_m["vel_reachable"]["sign_sim64"],
            "vel_r_sp64": o1_m["vel_reachable"]["sp_sim64"],
            "vel_r_ratio64": o1_m["vel_reachable"]["ratio_sim64"],
            "kappa_pose_c": kappas["pose_safe_critic"],
            "kappa_vel_r": kappas["vel_reachable"],
            "o2_Q4_sign": float(((v_sim_cl * v_sim_c) > 0).float().mean()),
            "o2_Q4_sp": spearman(v_sim_cl, v_sim_c),
            "o2_TrueTot_sign": float(((v_wm_t * v_sim_tot) > 0).float().mean()),
            "o2_TrueTot_sp": spearman(v_wm_t, v_sim_tot),
            "o2_Q3_err_r": float(torch.corrcoef(torch.stack((v_wm_s - v_sim_s, v_wm_c - v_sim_c)))[0, 1]),
        }
        evo_rows.append(row)
        print(f"  {cname:13s} | r_val={row['val_pearson_sim128']:.3f} | pose_c={row['pose_c_sign64']:.2f} | vel_c={row['vel_c_sign64']:.2f} | vel_r={row['vel_r_sign64']:.2f} (sp={row['vel_r_sp64']:+.2f}) | O2_Q4={row['o2_Q4_sign']:.2f} | TrueTot={row['o2_TrueTot_sign']:.2f} (sp={row['o2_TrueTot_sp']:+.2f})")

    # 2. Error A vs Error B Decomposition (B0@2100_post)
    print("\n=== [2] Error A (D_c - D_m) vs Error B (D_m - D_s) Decomposition (B0@2100_post) ===")
    cr_main = critics["B0@2100_post"]
    cg_main = cvf_core.critic_tangent_grad(cr_main, mdp, motion, proto, S_star, f_star, xi_n_16, clamp_vmax=V_MAX)
    decomp = {}
    for h in (32, 64, 96, 128):
        decomp[f"T{h}"] = {}
        for dname in dir_names:
            u = dirs_all[dname]
            dc = (cg_main["grad_raw"] * u).sum(-1)
            dm = (wm_o1["grads_soft"][h] * u).sum(-1)
            ds_all = sim_o1[h][dname]
            ds_pool = sim_o1_pool[h][dname]
            gm = sim_o1_gates[h][dname]
            decomp[f"T{h}"][dname] = {
                "all16_eps015": err_decomp_stats(dc, dm, ds_all),
                "gated_pool": err_decomp_stats(dc[gm], dm[gm], ds_pool[gm]) if gm.sum() >= 2 else {"n": int(gm.sum())},
            }
            if h in (64, 128):
                st = decomp[f"T{h}"][dname]["all16_eps015"]
                print(f"  T{h:3d} {dname:17s} | rmseA={st['rmse_eA']:.3f} rmseB={st['rmse_eB']:.3f} | shareA={st['share_A_of_sum']:.2f} | interact={st['interaction_ratio']:+.2f} | r(eA,eB)={st['pearson_eA_eB']:+.2f}")

    with open(OUT / "checkpoint_evolution.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(evo_rows[0].keys()))
        w.writeheader(); w.writerows(evo_rows)
    (OUT / "da04_evolution_and_decomp.json").write_text(json.dumps({
        "checkpoint_evolution": evo_rows,
        "error_decomposition": decomp,
    }, indent=2))
    print(f"\nSaved -> {OUT / 'da04_evolution_and_decomp.json'}")


if __name__ == "__main__":
    main()
