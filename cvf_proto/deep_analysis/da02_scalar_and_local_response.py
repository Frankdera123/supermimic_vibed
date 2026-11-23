"""Phase 4 & 12: Scalar Value Calibration, Within-Start Local Response, Tail-Bootstrap Sensitivity, and Statistical Reliability (CPU-only)."""
from __future__ import annotations

import csv
import json
import math
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
from scipy import stats as sp_stats  # noqa: E402
import cvf_core  # noqa: E402
from supermimic.crgf import reliability_gate, spearman  # noqa: E402
from supermimic.mdp import normalize_xi  # noqa: E402
from supermimic.networks import Critic  # noqa: E402

assert not torch.cuda.is_available(), "Must run CPU-only"
torch.set_num_threads(min(16, os.cpu_count() or 4))

OUT = REPO / "results" / "cvf" / "deep_analysis"
OUT.mkdir(parents=True, exist_ok=True)
SA = bridge.pairs_root()
GAMMA = 0.99
V_MAX = 10.0

CRITIC_SPECS = [
    ("b0@1000", SA / "b0_resume/ckpt_001000.pt", "critic"),
    ("b0@1500", SA / "b0_resume/ckpt_001500.pt", "critic"),
    ("b0@2000", SA / "b0_resume/ckpt_002000.pt", "critic"),
    ("B0@2100_pre", SA / "online_B0/diag_pairs/pair_it_002100.pt", "critic_pre"),
    ("B0@2100_post", SA / "online_B0/diag_pairs/pair_it_002100.pt", "critic_post"),
    ("B0@2300", SA / "online_B0/ckpt_002300.pt", "critic"),
    ("B1@2300", SA / "online_B1/ckpt_002300.pt", "critic"),
    ("B2@2300", SA / "online_B2/ckpt_002300.pt", "critic"),
    ("B3@2300", SA / "online_B3/ckpt_002300.pt", "critic"),
]


def scalar_metrics(pred: torch.Tensor, ref: torch.Tensor) -> dict:
    p = pred.double().numpy()
    r = ref.double().numpy()
    err = p - r
    slope, intercept, r_val, p_val, _ = sp_stats.linregress(p, r)
    rho, _ = sp_stats.spearmanr(p, r)
    return {
        "n": int(len(p)),
        "mean_pred": float(p.mean()),
        "mean_ref": float(r.mean()),
        "pearson": float(r_val),
        "spearman": float(rho),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(err.mean()),
        "cal_slope": float(slope),
        "cal_intercept": float(intercept),
    }


def pair_stats_full(pred: torch.Tensor, ref: torch.Tensor, clips: list[str]) -> dict:
    n = len(pred)
    if n == 0:
        return {"n": 0}
    dec = (pred.abs() + ref.abs()) > 1e-8
    sign_vec = ((torch.sign(pred) == torch.sign(ref)) & dec).float()
    k_pos = int(sign_vec.sum().item())
    n_dec = int(dec.sum().item())
    exact_p = float(sp_stats.binomtest(k_pos, n_dec, 0.5).pvalue) if n_dec > 0 else 1.0
    bs_clip = cvf_core.clip_clustered_bootstrap(sign_vec, clips, n_boot=10000, seed=2700)
    loco = cvf_core.loco_summary(sign_vec, clips)
    # Leave-one-start-out
    loso_means = [float(torch.cat([sign_vec[:i], sign_vec[i+1:]]).mean()) for i in range(n)] if n > 1 else [float("nan")]
    rat = pred.abs() / ref.abs().clamp_min(1e-6)
    p_np = pred.double().numpy()
    r_np = ref.double().numpy()
    pearson = float(np.corrcoef(p_np, r_np)[0, 1]) if n >= 2 and np.std(p_np) > 0 and np.std(r_np) > 0 else float("nan")
    return {
        "n": n,
        "n_clips": bs_clip["n_clips"],
        "sign_agree": bs_clip["mean"],
        "sign_ci95_clip": bs_clip["ci95"],
        "sign_exact_p": exact_p,
        "loco_range": [loco["min_mean"], loco["max_mean"]],
        "loso_range": [min(loso_means), max(loso_means)],
        "pearson": pearson,
        "spearman": spearman(pred, ref),
        "med_ratio": float(rat.median()),
        "iqr_ratio": [float(torch.quantile(rat, 0.25)), float(torch.quantile(rat, 0.75))],
        "rmse": float((pred - ref).square().mean().sqrt()),
        "mae": float((pred - ref).abs().mean()),
    }


def main():
    o1 = torch.load(REPO / "results/cvf/phase1b/oracle1_sim.pt", map_location="cpu", weights_only=False)
    pair = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
    world = bridge.setup_world(device="cpu", pair=pair)
    motion, mdp, proto = world["motion"], world["mdp"], world["proto"]
    wm, pi_old = pair["wm"], pair["pi_old"]

    starts16 = o1["starts"]
    clips16 = [s["clip_name"] for s in starts16]
    S_star = tuple(torch.cat([b["S_star"][k] for b in o1["batches"]], 0) for k in range(6))
    f_star = torch.cat([b["f_star"] for b in o1["batches"]], 0)
    ap_star = torch.cat([b["a_prev_star"] for b in o1["batches"]], 0)
    oh9_star = torch.cat([b["oh_prev9_star"] for b in o1["batches"]], 0)
    xi_16 = torch.zeros(16, 33); xi_16[:, 0] = 1.0
    xi_n_16 = normalize_xi(xi_16)
    dir_names = list(o1["batches"][0]["dirs"].keys())
    dirs_all = {d: torch.cat([b["dirs"][d] for b in o1["batches"]], 0) for d in dir_names}

    critics = {}
    for cname, cpath, ckey in CRITIC_SPECS:
        ck = torch.load(cpath, map_location="cpu", weights_only=False)
        cr = Critic(ck[ckey]["norm.mean"].shape[0]).eval()
        cr.load_state_dict(ck[ckey], strict=True)
        for p in cr.parameters():
            p.requires_grad_(False)
        critics[cname] = cr

    # WM rollout with S_by_h and bootstrapped gradients for primary critic
    S_det = tuple(x.detach() for x in S_star)
    delta = torch.zeros(16, 70, requires_grad=True)
    S = cvf_core.perturb_state(S_det, delta)
    o0 = proto.obs(S, ap_star.detach(), xi_16, f_star)
    oh = torch.cat((oh9_star.detach(), o0.unsqueeze(1)), dim=1)
    ap = ap_star.detach()
    f = f_star.clone()
    c_s = torch.ones(16)
    J_s = torch.zeros(16)
    J_wm_unboot, J_wm_boot = {}, {}
    g_wm_unboot, g_wm_boot = {}, {}
    cr_main = critics["B0@2100_post"]
    horizons = (32, 64, 96, 128)
    for t in range(128):
        ag = proto.actor_goal(f, S)
        a = torch.clamp(pi_old(oh, ag), -20.0, 20.0)
        S = proto.wm_step(wm, S, a, xi_n_16)
        f = f + 1
        r = mdp.reward(S, f, a, ap)
        ds = mdp.soft_termination(S, f, tz=0.02, tt=0.04)
        J_s = J_s + (GAMMA ** t) * c_s * r
        c_s = c_s * (1.0 - ds)
        ap = a
        o1_obs = proto.obs(S, a, xi_16, f)
        oh = torch.cat((oh[:, 1:], o1_obs.unsqueeze(1)), dim=1)
        step = t + 1
        if step in horizons:
            J_wm_unboot[step] = J_s
            vh = cr_main(proto.critic_input(S, f, xi_n_16, proto.wm_goal(f, S))).clamp(max=V_MAX)
            J_wm_boot[step] = J_s + (GAMMA ** step) * c_s * vh

    for h in horizons:
        g_wm_unboot[h] = torch.autograd.grad(J_wm_unboot[h].sum(), delta, retain_graph=True)[0].detach()
        g_wm_boot[h] = torch.autograd.grad(J_wm_boot[h].sum(), delta, retain_graph=(h != 128))[0].detach()

    V_sim_soft = {h: torch.cat([b["base_res"][0]["J_soft"][h] for b in o1["batches"]], 0) for h in horizons}
    V_sim_hard = {h: torch.cat([b["base_res"][0]["J_hard"][h] for b in o1["batches"]], 0) for h in horizons}
    c_sim_soft = {h: torch.cat([b["base_res"][0]["c_soft"][h] for b in o1["batches"]], 0) for h in horizons}

    # 1. Scalar Value Calibration across all 9 Critics
    print("=== [1] Scalar Value Calibration (Level 1) ===")
    scalar_calib = {}
    per_start_scalar_rows = []
    for cname, cr in critics.items():
        with torch.no_grad():
            vc = cr(proto.critic_input(S_star, f_star, xi_n_16, proto.wm_goal(f_star, S_star)))
        scalar_calib[cname] = {
            "vs_sim128_soft": scalar_metrics(vc, V_sim_soft[128]),
            "vs_sim128_hard": scalar_metrics(vc, V_sim_hard[128]),
            "vs_wm128_unboot": scalar_metrics(vc, J_wm_unboot[128].detach()),
            "vs_wm128_boot_post": scalar_metrics(vc, J_wm_boot[128].detach()),
            "vs_wm32_boot_post": scalar_metrics(vc, J_wm_boot[32].detach()),
        }
        if cname in ("b0@1000", "b0@2000", "B0@2100_post", "B0@2300", "B2@2300", "B3@2300"):
            for i in range(16):
                per_start_scalar_rows.append({
                    "critic": cname,
                    "start_idx": i,
                    "clip": clips16[i],
                    "frame": int(starts16[i]["frame"]),
                    "V_crit": float(vc[i]),
                    "V_sim128_soft": float(V_sim_soft[128][i]),
                    "V_sim128_hard": float(V_sim_hard[128][i]),
                    "V_wm128_unboot": float(J_wm_unboot[128][i].detach()),
                    "V_wm128_boot_post": float(J_wm_boot[128][i].detach()),
                    "c_sim128_soft": float(c_sim_soft[128][i]),
                })
    print("B0@2100_post vs sim128_soft:", json.dumps(scalar_calib["B0@2100_post"]["vs_sim128_soft"], indent=2))
    print("B0@2100_post vs wm128_boot_post:", json.dumps(scalar_calib["B0@2100_post"]["vs_wm128_boot_post"], indent=2))

    # 2. Within-start local response vs Cross-start global correlation
    print("\n=== [2] Within-Start Local Response (Delta V_c vs Delta G_s) ===")
    cg_main = cvf_core.critic_tangent_grad(cr_main, mdp, motion, proto, S_star, f_star, xi_n_16)
    within_start = {}
    for eps in (0.015, 0.03):
        # Compute Delta V_c = (V(s+) - V(s-)) / (2*eps) directly on Critic
        D_crit_fd_mat = torch.zeros(16, len(dir_names))
        D_crit_ag_mat = torch.zeros(16, len(dir_names))
        D_sim_128_mat = torch.zeros(16, len(dir_names))
        D_sim_64_mat = torch.zeros(16, len(dir_names))
        D_sim_32_mat = torch.zeros(16, len(dir_names))
        D_wm_64_mat = torch.zeros(16, len(dir_names))
        D_wm_boot64_mat = torch.zeros(16, len(dir_names))
        for j, dname in enumerate(dir_names):
            u = dirs_all[dname]
            Sp = cvf_core.perturb_state(S_star, eps * u)
            Sm = cvf_core.perturb_state(S_star, -eps * u)
            with torch.no_grad():
                vp = cr_main(proto.critic_input(Sp, f_star, xi_n_16, proto.wm_goal(f_star, Sp)))
                vm = cr_main(proto.critic_input(Sm, f_star, xi_n_16, proto.wm_goal(f_star, Sm)))
            D_crit_fd_mat[:, j] = (vp - vm) / (2.0 * eps)
            D_crit_ag_mat[:, j] = (cg_main["grad_raw"] * u).sum(-1)
            D_wm_64_mat[:, j] = (g_wm_unboot[64] * u).sum(-1)
            D_wm_boot64_mat[:, j] = (g_wm_boot[64] * u).sum(-1)
            for h, mat in ((32, D_sim_32_mat), (64, D_sim_64_mat), (128, D_sim_128_mat)):
                jp = torch.cat([b["branches"][(dname, eps, "plus")]["J_soft"][h] for b in o1["batches"]], 0)
                jm = torch.cat([b["branches"][(dname, eps, "minus")]["J_soft"][h] for b in o1["batches"]], 0)
                mat[:, j] = (jp - jm) / (2.0 * eps)

        # Within-start direction ranking (5 dirs per start)
        rank_sp_64, rank_sp_128, rank_sp_wm64 = [], [], []
        for i in range(16):
            rank_sp_64.append(spearman(D_crit_fd_mat[i], D_sim_64_mat[i]))
            rank_sp_128.append(spearman(D_crit_fd_mat[i], D_sim_128_mat[i]))
            rank_sp_wm64.append(spearman(D_wm_64_mat[i], D_sim_64_mat[i]))

        # Demeaned across directions per start (all 16*5 = 80 points)
        dc_dem = (D_crit_fd_mat - D_crit_fd_mat.mean(dim=1, keepdim=True)).flatten()
        ds64_dem = (D_sim_64_mat - D_sim_64_mat.mean(dim=1, keepdim=True)).flatten()
        ds128_dem = (D_sim_128_mat - D_sim_128_mat.mean(dim=1, keepdim=True)).flatten()
        dw64_dem = (D_wm_64_mat - D_wm_64_mat.mean(dim=1, keepdim=True)).flatten()

        within_start[f"eps_{eps:.3f}"] = {
            "critic_fd_vs_ag_max_diff": float((D_crit_fd_mat - D_crit_ag_mat).abs().max()),
            "critic_fd_vs_ag_rmse": float((D_crit_fd_mat - D_crit_ag_mat).square().mean().sqrt()),
            "within_start_5dir_spearman_crit_vs_sim64_mean": float(np.mean(rank_sp_64)),
            "within_start_5dir_spearman_crit_vs_sim64_median": float(np.median(rank_sp_64)),
            "within_start_5dir_spearman_crit_vs_sim128_mean": float(np.mean(rank_sp_128)),
            "within_start_5dir_spearman_wm64_vs_sim64_mean": float(np.mean(rank_sp_wm64)),
            "demeaned_80_pearson_crit_vs_sim64": float(np.corrcoef(dc_dem.numpy(), ds64_dem.numpy())[0, 1]),
            "demeaned_80_pearson_crit_vs_sim128": float(np.corrcoef(dc_dem.numpy(), ds128_dem.numpy())[0, 1]),
            "demeaned_80_pearson_wm64_vs_sim64": float(np.corrcoef(dw64_dem.numpy(), ds64_dem.numpy())[0, 1]),
            "pooled_80_sign_crit_vs_sim64": float(((D_crit_fd_mat * D_sim_64_mat) > 0).float().mean()),
            "pooled_80_sign_crit_vs_sim128": float(((D_crit_fd_mat * D_sim_128_mat) > 0).float().mean()),
            "pooled_80_sign_wm64_vs_sim64": float(((D_wm_64_mat * D_sim_64_mat) > 0).float().mean()),
        }
    print("Within-start summary:", json.dumps(within_start, indent=2))

    # 3. Bootstrapped WM vs Unbootstrapped WM vs Critic vs Sim across T
    print("\n=== [3] Horizon & Tail-Bootstrap Sensitivity (B0@2100_post) ===")
    horizon_sens = {}
    base_reps_T = {h: torch.stack([
        torch.cat([b["base_res"][0]["J_soft"][h] for b in o1["batches"]], 0),
        torch.cat([b["base_res"][1]["J_soft"][h] for b in o1["batches"]], 0),
    ], 1) for h in horizons}

    for h in horizons:
        horizon_sens[f"T{h}"] = {}
        for dname in dir_names:
            u = dirs_all[dname]
            d_crit = (cg_main["grad_raw"] * u).sum(-1)
            d_wm_u = (g_wm_unboot[h] * u).sum(-1)
            d_wm_b = (g_wm_boot[h] * u).sum(-1)
            d_sim_015 = (
                torch.cat([b["branches"][(dname, 0.015, "plus")]["J_soft"][h] for b in o1["batches"]], 0)
                - torch.cat([b["branches"][(dname, 0.015, "minus")]["J_soft"][h] for b in o1["batches"]], 0)
            ) / 0.030
            d_sim_030 = (
                torch.cat([b["branches"][(dname, 0.03, "plus")]["J_soft"][h] for b in o1["batches"]], 0)
                - torch.cat([b["branches"][(dname, 0.03, "minus")]["J_soft"][h] for b in o1["batches"]], 0)
            ) / 0.060
            d_sim_hard_015 = (
                torch.cat([b["branches"][(dname, 0.015, "plus")]["J_hard"][h] for b in o1["batches"]], 0)
                - torch.cat([b["branches"][(dname, 0.015, "minus")]["J_hard"][h] for b in o1["batches"]], 0)
            ) / 0.030
            d_sim_pool = 0.5 * (d_sim_015 + d_sim_030)
            gates = [reliability_gate(base_reps_T[h][i], torch.tensor([d_sim_015[i], d_sim_030[i]]))[0] for i in range(16)]
            gmask = torch.tensor(gates, dtype=torch.bool)
            gclips = [clips16[i] for i in range(16) if gates[i]]

            horizon_sens[f"T{h}"][dname] = {
                "crit_vs_sim_all16_015": pair_stats_full(d_crit, d_sim_015, clips16),
                "crit_vs_sim_all16_030": pair_stats_full(d_crit, d_sim_030, clips16),
                "crit_vs_sim_all16_hard015": pair_stats_full(d_crit, d_sim_hard_015, clips16),
                "crit_vs_sim_gated": pair_stats_full(d_crit[gmask], d_sim_pool[gmask], gclips),
                "wm_unboot_vs_sim_all16_015": pair_stats_full(d_wm_u, d_sim_015, clips16),
                "wm_boot_vs_sim_all16_015": pair_stats_full(d_wm_b, d_sim_015, clips16),
                "crit_vs_wm_unboot_all16": pair_stats_full(d_crit, d_wm_u, clips16),
                "crit_vs_wm_boot_all16": pair_stats_full(d_crit, d_wm_b, clips16),
            }

    for dname in ("pose_safe_critic", "vel_critic", "vel_reachable"):
        b64 = horizon_sens["T64"][dname]
        b128 = horizon_sens["T128"][dname]
        print(f"  {dname} T64: CvsSim015 sign={b64['crit_vs_sim_all16_015']['sign_agree']:.2f} (p={b64['crit_vs_sim_all16_015']['sign_exact_p']:.3f}), "
              f"CvsWM_unboot={b64['crit_vs_wm_unboot_all16']['sign_agree']:.2f}, CvsWM_boot={b64['crit_vs_wm_boot_all16']['sign_agree']:.2f}")
        print(f"  {dname} T128: CvsSim015 sign={b128['crit_vs_sim_all16_015']['sign_agree']:.2f} (p={b128['crit_vs_sim_all16_015']['sign_exact_p']:.3f}), "
              f"CvsWM_unboot={b128['crit_vs_wm_unboot_all16']['sign_agree']:.2f}, CvsWM_boot={b128['crit_vs_wm_boot_all16']['sign_agree']:.2f}")

    # Save CSV & JSON
    with open(OUT / "scalar_calibration_per_start.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_start_scalar_rows[0].keys()))
        w.writeheader(); w.writerows(per_start_scalar_rows)

    out_json = {
        "scalar_calibration": scalar_calib,
        "within_start_local_response": within_start,
        "horizon_sensitivity": horizon_sens,
    }
    (OUT / "da02_scalar_and_local.json").write_text(json.dumps(out_json, indent=2))
    print(f"\nSaved -> {OUT / 'da02_scalar_and_local.json'}")


if __name__ == "__main__":
    main()
