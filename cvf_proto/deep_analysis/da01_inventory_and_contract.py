"""Phase 1 & 2: Asset Inventory, Integrity, Contract Audit, and Independent Reproduction (CPU-only)."""
from __future__ import annotations

import hashlib
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
from supermimic.rotations import so3_exp, yaw_mat  # noqa: E402

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


def file_meta(p: Path) -> dict:
    if not p.exists():
        return {"path": str(p), "exists": False}
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    st = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": int(st.st_size),
        "sha256": h.hexdigest(),
    }


def indep_perturb(S, delta):
    p, R, v, w, q, qd = S
    dp, dth, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, 29, 29), -1)
    Hy = yaw_mat(R)
    return (
        p + torch.einsum("bij,bj->bi", Hy, dp),
        R @ so3_exp(dth),
        v + torch.einsum("bij,bj->bi", R, dv),
        w + torch.einsum("bij,bj->bi", R, dw),
        q + dq,
        qd + dqd,
    )


def indep_wm_rollout(wm, pi, mdp, proto, S_base, f0, oh_prev9, a_prev0, xi, xi_n, T, horizons=(32, 64, 96, 128)):
    B = S_base[0].shape[0]
    S_det = tuple(x.detach() for x in S_base)
    delta = torch.zeros(B, 70, dtype=S_base[0].dtype, requires_grad=True)
    S = indep_perturb(S_det, delta)
    o0 = proto.obs(S, a_prev0.detach(), xi, f0)
    oh = torch.cat((oh_prev9.detach(), o0.unsqueeze(1)), dim=1)
    ap = a_prev0.detach()
    f = f0.clone()
    c_s = torch.ones(B)
    J_s = torch.zeros(B)
    by_h = {}
    c_by_h = {}
    S_by_h = {}
    for t in range(T):
        ag = proto.actor_goal(f, S)
        a = torch.clamp(pi(oh, ag), -20.0, 20.0)
        S = proto.wm_step(wm, S, a, xi_n)
        f = f + 1
        r = mdp.reward(S, f, a, ap)
        ds = mdp.soft_termination(S, f, tz=0.02, tt=0.04)
        J_s = J_s + (GAMMA ** t) * c_s * r
        c_s = c_s * (1.0 - ds)
        ap = a
        o1 = proto.obs(S, a, xi, f)
        oh = torch.cat((oh[:, 1:], o1.unsqueeze(1)), dim=1)
        step = t + 1
        if step in horizons:
            by_h[step] = J_s
            c_by_h[step] = c_s
            S_by_h[step] = tuple(x for x in S)
    grads = {}
    for h in sorted(by_h):
        grads[h] = torch.autograd.grad(by_h[h].sum(), delta, retain_graph=(h != max(by_h)))[0].detach()
    return {
        "J_soft": {h: v.detach() for h, v in by_h.items()},
        "c_soft": {h: v.detach() for h, v in c_by_h.items()},
        "grads_soft": grads,
        "S_by_h": {h: tuple(x.detach() for x in sv) for h, sv in S_by_h.items()},
    }


def main():
    print("=== [1] Asset Inventory & Integrity ===")
    cvf_files = sorted(
        list((REPO / "results/cvf/phase0").glob("*"))
        + list((REPO / "results/cvf/phase1a").glob("*"))
        + list((REPO / "results/cvf/phase1b").glob("*"))
        + list((REPO / "results/cvf/analysis").glob("*"))
    )
    inv_files = [file_meta(p) for p in cvf_files]

    ckpt_seen = {}
    ckpt_inventory = []
    for name, cpath, key, it_expected, run_name in CRITIC_SPECS:
        sp = str(cpath)
        if sp not in ckpt_seen:
            ckpt_seen[sp] = file_meta(cpath)
        ck = torch.load(cpath, map_location="cpu", weights_only=False)
        sd = ck[key]
        norm_count = float(sd["norm.count"].item())
        cin = int(sd["norm.mean"].shape[0])
        args_path = SA / run_name / "args.json"
        args_data = json.loads(args_path.read_text()) if args_path.exists() else {}
        ckpt_inventory.append({
            "name": name,
            "run": run_name,
            "iteration_expected": it_expected,
            "iteration_in_ckpt": int(ck.get("iteration", -1)),
            "key": key,
            "file": sp,
            "sha256": ckpt_seen[sp]["sha256"],
            "size_bytes": ckpt_seen[sp]["size_bytes"],
            "critic_in_dim": cin,
            "norm_count": norm_count,
            "critic_updates": int(ck.get("critic_updates", ck.get("critic_updates_after", -1))),
            "pi_updates": int(ck.get("pi_updates", -1)),
            "wm_updates": int(ck.get("wm_updates", -1)),
            "opcl_enabled": bool(args_data.get("opcl_enabled", False)),
            "opcl_lambda_p": float(args_data.get("opcl_lambda_p", 0.0)),
            "opcl_lambda_o": float(args_data.get("opcl_lambda_o", 0.0)),
            "b1_match_opcl": bool(args_data.get("b1_match_opcl", False)),
            "horizon": int(args_data.get("horizon", -1)),
            "gamma": float(args_data.get("gamma", -1)),
            "lam": float(args_data.get("lam", -1)),
        })

    # Inspect Phase 0, Oracle 1, Oracle 2 data shapes
    p0 = torch.load(REPO / "results/cvf/phase0/phase0_starts_and_dirs.pt", map_location="cpu", weights_only=False)
    o1 = torch.load(REPO / "results/cvf/phase1b/oracle1_sim.pt", map_location="cpu", weights_only=False)
    o2 = torch.load(REPO / "results/cvf/phase1b/oracle2_sim.pt", map_location="cpu", weights_only=False)

    starts16 = o1["starts"]
    clips16 = [s["clip_name"] for s in starts16]
    o1_repeat_ranges = [float(b["base_repeat_range"]) for b in o1["batches"]]
    o2_repeat_ranges = [float(b["base_repeat_range"]) for b in o2["batches"]]

    data_shapes = {
        "phase0_n_starts": len(p0["starts"]),
        "phase0_o1_dirs": {k: list(v.shape) for k, v in p0["oracle1_dirs"].items()},
        "phase0_o2_dirs": {k: list(v.shape) for k, v in p0["oracle2_dirs"].items()},
        "oracle1_n_starts": len(starts16),
        "oracle1_unique_clips": sorted(set(clips16)),
        "oracle1_n_clips": len(set(clips16)),
        "oracle1_eps_grid": list(o1["eps_grid"]),
        "oracle1_horizons": list(o1["horizons_out"]),
        "oracle1_dirs": list(o1["batches"][0]["dirs"].keys()),
        "oracle1_base_repeat_ranges": o1_repeat_ranges,
        "oracle2_n_starts": len(o2["starts"]),
        "oracle2_h_open": int(o2["h_open"]),
        "oracle2_t_cont": int(o2["t_cont"]),
        "oracle2_eps_grid": list(o2["eps_grid"]),
        "oracle2_dirs": list(p0["oracle2_dirs"].keys()),
        "oracle2_base_repeat_ranges": o2_repeat_ranges,
    }
    print(f"Oracle1 clips ({len(set(clips16))}): {sorted(set(clips16))}")
    print(f"Oracle1 repeat ranges: {o1_repeat_ranges}")
    print(f"Oracle2 repeat ranges: {o2_repeat_ranges}")

    print("\n=== [2] Independent Implementation & Batching Cross-Check ===")
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

    ref_wm = cvf_core.rollout_wm_continuation(
        wm, pi_old, mdp, proto, S_star, f_star, oh9_star, ap_star,
        xi_16, xi_n_16, T=128, hist_mode="last_only", need_grad=True, horizons_out=(32, 64, 96, 128)
    )
    ind_wm = indep_wm_rollout(wm, pi_old, mdp, proto, S_star, f_star, oh9_star, ap_star, xi_16, xi_n_16, T=128)

    wm_diff = {
        f"T{h}": {
            "J_max_abs_diff": float((ref_wm["J_soft"][h] - ind_wm["J_soft"][h]).abs().max()),
            "grad_max_abs_diff": float((ref_wm["grads_soft"][h] - ind_wm["grads_soft"][h]).abs().max()),
        }
        for h in (32, 64, 96, 128)
    }
    print("WM rollout independent cross-check:", wm_diff)

    # Batching consistency B=1 vs B=16 for critic_tangent_grad & rollout_wm_continuation
    cr_main = Critic(938).eval()
    ck2100 = torch.load(SA / "online_B0/diag_pairs/pair_it_002100.pt", map_location="cpu", weights_only=False)
    cr_main.load_state_dict(ck2100["critic_post"], strict=True)

    cg_b16 = cvf_core.critic_tangent_grad(cr_main, mdp, motion, proto, S_star, f_star, xi_n_16)
    max_b1_v_diff = 0.0
    max_b1_g_diff = 0.0
    max_b1_wm_j_diff = 0.0
    max_b1_wm_g_diff = 0.0
    for i in range(16):
        Si = tuple(x[i:i+1] for x in S_star)
        cgi = cvf_core.critic_tangent_grad(cr_main, mdp, motion, proto, Si, f_star[i:i+1], xi_n_16[i:i+1])
        max_b1_v_diff = max(max_b1_v_diff, float((cgi["v_raw"] - cg_b16["v_raw"][i:i+1]).abs().max()))
        max_b1_g_diff = max(max_b1_g_diff, float((cgi["grad_raw"] - cg_b16["grad_raw"][i:i+1]).abs().max()))
        wmi = cvf_core.rollout_wm_continuation(
            wm, pi_old, mdp, proto, Si, f_star[i:i+1], oh9_star[i:i+1], ap_star[i:i+1],
            xi_16[i:i+1], xi_n_16[i:i+1], T=64, hist_mode="last_only", need_grad=True, horizons_out=(64,)
        )
        max_b1_wm_j_diff = max(max_b1_wm_j_diff, float((wmi["J_soft"][64] - ref_wm["J_soft"][64][i:i+1]).abs().max()))
        max_b1_wm_g_diff = max(max_b1_wm_g_diff, float((wmi["grads_soft"][64] - ref_wm["grads_soft"][64][i:i+1]).abs().max()))

    batch_consistency = {
        "critic_B1_vs_B16_v_max_diff": max_b1_v_diff,
        "critic_B1_vs_B16_grad_max_diff": max_b1_g_diff,
        "wm_B1_vs_B16_J64_max_diff": max_b1_wm_j_diff,
        "wm_B1_vs_B16_grad64_max_diff": max_b1_wm_g_diff,
    }
    print("Batch consistency B=1 vs B=16:", batch_consistency)

    print("\n=== [3] Clamp Activation & Survival Indexing Audit ===")
    critics = {}
    clamp_audit = {}
    for name, cpath, key, _, _ in CRITIC_SPECS:
        ck = torch.load(cpath, map_location="cpu", weights_only=False)
        cr = Critic(ck[key]["norm.mean"].shape[0]).eval()
        cr.load_state_dict(ck[key], strict=True)
        critics[name] = cr
        cg = cvf_core.critic_tangent_grad(cr, mdp, motion, proto, S_star, f_star, xi_n_16, clamp_vmax=10.0)
        n_clamped = int(cg["clamp_active"].sum())
        clamp_audit[name] = {
            "o1_S_star_n_clamped": n_clamped,
            "o1_S_star_v_raw_min": float(cg["v_raw"].min()),
            "o1_S_star_v_raw_max": float(cg["v_raw"].max()),
            "o1_S_star_v_raw_mean": float(cg["v_raw"].mean()),
        }

    # Check Oracle 2 S_at_32 clamp rate
    idx16 = o2["idx16"]
    for cname, cr in critics.items():
        n_tot = 0
        n_cl = 0
        vmax_seen = -1e9
        for dname in p0["oracle2_dirs"]:
            for eps in o2["eps_grid"]:
                for br in ("plus", "minus"):
                    S32 = unflatten_state(torch.cat([b["branches"][(dname, eps, br)]["S_at_32"] for b in o2["batches"]], 0))
                    f32 = torch.cat([b["branches"][(dname, eps, br)]["f_at_32"] for b in o2["batches"]], 0)
                    with torch.no_grad():
                        vr = cr(proto.critic_input(S32, f32, xi_n_16, proto.wm_goal(f32, S32)))
                    n_tot += len(vr)
                    n_cl += int((vr >= 10.0).sum())
                    vmax_seen = max(vmax_seen, float(vr.max()))
        clamp_audit[cname]["o2_S32_n_total"] = n_tot
        clamp_audit[cname]["o2_S32_n_clamped"] = n_cl
        clamp_audit[cname]["o2_S32_clamp_rate"] = n_cl / max(n_tot, 1)
        clamp_audit[cname]["o2_S32_v_raw_max"] = vmax_seen
    print("Clamp audit (primary B0@2100_post):", clamp_audit["B0@2100_post"])
    print("Clamp audit (B0@2300):", clamp_audit["B0@2300"])

    # Check finite-horizon vs bootstrapped WM continuation at T=32, 64, 96, 128
    print("\n=== [4] Finite-Horizon vs Bootstrapped WM Continuation ===")
    tail_audit = {}
    cr_post = critics["B0@2100_post"]
    for h in (32, 64, 96, 128):
        Sh = ind_wm["S_by_h"][h]
        fh = f_star + h
        with torch.no_grad():
            vh = cr_post(proto.critic_input(Sh, fh, xi_n_16, proto.wm_goal(fh, Sh))).clamp(max=V_MAX)
        tail = (GAMMA ** h) * ind_wm["c_soft"][h] * vh
        J_unboot = ind_wm["J_soft"][h]
        J_boot = J_unboot + tail
        with torch.no_grad():
            vc = cr_post(proto.critic_input(S_star, f_star, xi_n_16, proto.wm_goal(f_star, S_star)))
        tail_audit[f"T{h}"] = {
            "mean_J_unboot": float(J_unboot.mean()),
            "mean_tail": float(tail.mean()),
            "mean_J_boot": float(J_boot.mean()),
            "mean_V_crit": float(vc.mean()),
            "rmse_crit_vs_unboot": float((vc - J_unboot).square().mean().sqrt()),
            "rmse_crit_vs_boot": float((vc - J_boot).square().mean().sqrt()),
            "pearson_crit_vs_unboot": float(torch.corrcoef(torch.stack((vc, J_unboot)))[0, 1]),
            "pearson_crit_vs_boot": float(torch.corrcoef(torch.stack((vc, J_boot)))[0, 1]),
        }
    print("Tail audit (B0@2100_post):", json.dumps(tail_audit, indent=2))

    # Reproduce CVF0_FINAL_REPORT Table 3 ALL_POLICY (n=48) and compare with cvf_full_summary.json
    print("\n=== [5] Independent Reproduction of Historical Summary ===")
    hist_sum = json.loads((REPO / "results/cvf/analysis/cvf_full_summary.json").read_text())
    f0_16 = p0["f0"][idx16]
    S0_16 = motion.ref_state(f0_16)
    base_A_16 = p0["oracle2_base_A"][idx16]
    dirs_o2 = {k: v[idx16] for k, v in p0["oracle2_dirs"].items()}

    # Recompute WM O2 grads for B0@2100_post
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
    V32 = cr_post(proto.critic_input(S, f, xi_n_16, proto.wm_goal(f, S))).clamp(max=V_MAX)
    J_crit = (GAMMA ** 32) * c * V32
    J_tot = J_state + J_direct + J_crit
    g_s = torch.autograd.grad(J_state.sum(), A_seq, retain_graph=True)[0].detach()
    g_c = torch.autograd.grad(J_crit.sum(), A_seq, retain_graph=True)[0].detach()
    g_t = torch.autograd.grad(J_tot.sum(), A_seq, retain_graph=False)[0].detach()

    # Pool ALL_POLICY = [delta_2100, delta_2300, ascent] (n=48)
    policy_dirs = ["delta_2100", "delta_2300", "ascent"]
    vecs = {k: [] for k in ("D_wm_state", "D_sim_state", "D_sim_crit_l", "D_sim_cont", "D_wm_tot_l", "D_sim_tot_l", "D_sim_tot_true", "D_wm_crit_l")}
    for dname in policy_dirs:
        dA = dirs_o2[dname]
        by_eps = {}
        for eps in o2["eps_grid"]:
            res_pm = {}
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
                with torch.no_grad():
                    Vc = cr_post(proto.critic_input(S32, f32, xi_n_16, proto.wm_goal(f32, S32))).clamp(max=V_MAX)
                res_pm[br] = {"state": Js, "direct": Jd, "cont": (GAMMA ** 32) * cc * Vcont, "crit": (GAMMA ** 32) * cc * Vc}
            by_eps[eps] = {
                "state": (res_pm["plus"]["state"] - res_pm["minus"]["state"]) / (2.0 * eps),
                "direct": (res_pm["plus"]["direct"] - res_pm["minus"]["direct"]) / (2.0 * eps),
                "cont": (res_pm["plus"]["cont"] - res_pm["minus"]["cont"]) / (2.0 * eps),
                "crit": (res_pm["plus"]["crit"] - res_pm["minus"]["crit"]) / (2.0 * eps),
            }
        D_sim_s = 0.5 * (by_eps[0.015]["state"] + by_eps[0.03]["state"])
        D_sim_d = 0.5 * (by_eps[0.015]["direct"] + by_eps[0.03]["direct"])
        D_sim_c = 0.5 * (by_eps[0.015]["cont"] + by_eps[0.03]["cont"])
        D_sim_cl = 0.5 * (by_eps[0.015]["crit"] + by_eps[0.03]["crit"])
        vecs["D_sim_state"].append(D_sim_s)
        vecs["D_sim_cont"].append(D_sim_c)
        vecs["D_sim_crit_l"].append(D_sim_cl)
        vecs["D_sim_tot_true"].append(D_sim_s + D_sim_d + D_sim_c)
        vecs["D_sim_tot_l"].append(D_sim_s + D_sim_d + D_sim_cl)
        vecs["D_wm_state"].append((g_s * dA).sum((1, 2)))
        vecs["D_wm_crit_l"].append((g_c * dA).sum((1, 2)))
        vecs["D_wm_tot_l"].append((g_t * dA).sum((1, 2)))

    V48 = {k: torch.cat(v, 0) for k, v in vecs.items()}
    e_st = V48["D_wm_state"] - V48["D_sim_state"]
    e_cr = V48["D_wm_crit_l"] - V48["D_sim_cont"]
    all_policy_repro = {
        "n": 48,
        "WM_state_vs_Sim_state_sign": float(((V48["D_wm_state"] * V48["D_sim_state"]) > 0).float().mean()),
        "WM_state_vs_Sim_state_sp": spearman(V48["D_wm_state"], V48["D_sim_state"]),
        "Q4_SimCrit_vs_TrueCont_sign": float(((V48["D_sim_crit_l"] * V48["D_sim_cont"]) > 0).float().mean()),
        "Q4_SimCrit_vs_TrueCont_sp": spearman(V48["D_sim_crit_l"], V48["D_sim_cont"]),
        "Q4_SimCrit_vs_TrueCont_ratio": float((V48["D_sim_crit_l"].abs() / V48["D_sim_cont"].abs().clamp_min(1e-6)).median()),
        "Illusion_WMtot_vs_Simtot_l_sign": float(((V48["D_wm_tot_l"] * V48["D_sim_tot_l"]) > 0).float().mean()),
        "True_WMtot_vs_Simtot_true_sign": float(((V48["D_wm_tot_l"] * V48["D_sim_tot_true"]) > 0).float().mean()),
        "True_WMtot_vs_Simtot_true_sp": spearman(V48["D_wm_tot_l"], V48["D_sim_tot_true"]),
        "True_WMtot_vs_Simtot_true_ratio": float((V48["D_wm_tot_l"].abs() / V48["D_sim_tot_true"].abs().clamp_min(1e-6)).median()),
        "Q3_corr_e_state_e_critic_true": float(torch.corrcoef(torch.stack((e_st, e_cr)))[0, 1]),
    }
    print("ALL_POLICY (n=48) independent repro:", json.dumps(all_policy_repro, indent=2))

    # Save full report JSON
    report = {
        "cvf_files": inv_files,
        "checkpoints": ckpt_inventory,
        "data_shapes": data_shapes,
        "wm_rollout_crosscheck": wm_diff,
        "batch_consistency": batch_consistency,
        "clamp_audit": clamp_audit,
        "tail_audit": tail_audit,
        "all_policy_repro": all_policy_repro,
    }
    (OUT / "da01_inventory_contract.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {OUT / 'da01_inventory_contract.json'}")


if __name__ == "__main__":
    main()
