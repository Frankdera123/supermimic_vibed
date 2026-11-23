"""Phase 10 & 11: Policy Gradient Propagation through WM & Counterfactual Critic Replacement (CPU-only)."""
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
from scipy import stats as sp_stats  # noqa: E402
import cvf_core  # noqa: E402
from supermimic.crgf import reliability_gate, reward_split, soft_term, spearman  # noqa: E402
from supermimic.mdp import normalize_xi  # noqa: E402
from supermimic.networks import Critic  # noqa: E402
from supermimic.rotations import so3_log, yaw_mat  # noqa: E402

assert not torch.cuda.is_available(), "Must run CPU-only"
torch.set_num_threads(min(16, os.cpu_count() or 4))

OUT = REPO / "results" / "cvf" / "deep_analysis"
OUT.mkdir(parents=True, exist_ok=True)
SA = bridge.pairs_root()
GAMMA = 0.99
V_MAX = 10.0

TANGENT_BLOCKS = {
    "root_pos": slice(0, 3),
    "root_ori": slice(3, 6),
    "root_linvel": slice(6, 9),
    "root_angvel": slice(9, 12),
    "joint_pos": slice(12, 41),
    "joint_vel": slice(41, 70),
}


def eval_cf(pred: torch.Tensor, ref: torch.Tensor, clips: list[str]) -> dict:
    dec = (pred.abs() + ref.abs()) > 1e-8
    sv = ((torch.sign(pred) == torch.sign(ref)) & dec).float()
    k = int(sv.sum().item())
    n_dec = int(dec.sum().item())
    bs = cvf_core.clip_clustered_bootstrap(sv, clips, n_boot=10000, seed=2700)
    loco = cvf_core.loco_summary(sv, clips)
    rat = pred.abs() / ref.abs().clamp_min(1e-6)
    p_np, r_np = pred.double().numpy(), ref.double().numpy()
    pearson = float(np.corrcoef(p_np, r_np)[0, 1]) if len(p_np) >= 2 else float("nan")
    return {
        "n": int(len(pred)),
        "sign_agree": bs["mean"],
        "sign_ci95_clip": bs["ci95"],
        "sign_exact_p": float(sp_stats.binomtest(k, n_dec, 0.5).pvalue) if n_dec > 0 else 1.0,
        "loco_range": [loco["min_mean"], loco["max_mean"]],
        "spearman": spearman(pred, ref),
        "pearson": pearson,
        "med_ratio": float(rat.median()),
        "mae": float((pred - ref).abs().mean()),
        "rmse": float((pred - ref).square().mean().sqrt()),
    }


def main():
    o2 = torch.load(REPO / "results/cvf/phase1b/oracle2_sim.pt", map_location="cpu", weights_only=False)
    p0 = torch.load(REPO / "results/cvf/phase0/phase0_starts_and_dirs.pt", map_location="cpu", weights_only=False)

    pair = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
    world = bridge.setup_world(device="cpu", pair=pair)
    motion, mdp, proto = world["motion"], world["mdp"], world["proto"]
    wm, pi_old = pair["wm"], pair["pi_old"]

    ck2100 = torch.load(SA / "online_B0/diag_pairs/pair_it_002100.pt", map_location="cpu", weights_only=False)
    cr_post = Critic(ck2100["critic_post"]["norm.mean"].shape[0]).eval()
    cr_post.load_state_dict(ck2100["critic_post"], strict=True)
    for p in cr_post.parameters():
        p.requires_grad_(False)

    idx16 = o2["idx16"]
    starts16 = o2["starts"]
    clips16 = [s["clip_name"] for s in starts16]
    f0_16 = p0["f0"][idx16]
    S0_16 = motion.ref_state(f0_16)
    base_A_16 = p0["oracle2_base_A"][idx16]
    dirs_o2 = {k: v[idx16] for k, v in p0["oracle2_dirs"].items()}
    dir_names = list(dirs_o2.keys())
    policy_dirs = ["delta_2100", "delta_2300", "ascent"]

    xi_16 = torch.zeros(16, 33); xi_16[:, 0] = 1.0
    xi_n_16 = normalize_xi(xi_16)

    # Full differentiable WM rollout: H_open=32 + T_cont=96 closed-loop pi_old
    print("=== [1] Computing Full WM Prefix (H=32) + WM Continuation (T=96) Gradients ===")
    A_seq = base_A_16.clone().requires_grad_(True)
    S = tuple(x.clone() for x in S0_16)
    ap = torch.zeros(16, 29)
    o0 = proto.obs(S, ap, xi_16, f0_16)
    oh = o0.unsqueeze(1).expand(-1, 10, -1).contiguous()
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
        o1 = proto.obs(S, a, xi_16, f)
        oh = torch.cat((oh[:, 1:], o1.unsqueeze(1)), dim=1)

    S32 = S
    c32 = c
    f32 = f.clone()
    V_crit32 = cr_post(proto.critic_input(S32, f32, xi_n_16, proto.wm_goal(f32, S32))).clamp(max=V_MAX)
    J_crit32 = (GAMMA ** 32) * c32 * V_crit32

    # Continuation T_cont=96 in WM from S32
    V_wm_cont96 = torch.zeros(16)
    c_cont = torch.ones(16)
    S_c = S32
    ap_c = ap
    f_c = f32.clone()
    oh_c = oh
    for k in range(96):
        ag = proto.actor_goal(f_c, S_c)
        a_c = torch.clamp(pi_old(oh_c, ag), -20.0, 20.0)
        S_c = proto.wm_step(wm, S_c, a_c, xi_n_16)
        f_c = f_c + 1
        r_c = mdp.reward(S_c, f_c, a_c, ap_c)
        d_c = mdp.soft_termination(S_c, f_c, tz=0.02, tt=0.04)
        V_wm_cont96 = V_wm_cont96 + (GAMMA ** k) * c_cont * r_c
        c_cont = c_cont * (1.0 - d_c)
        ap_c = a_c
        o1_c = proto.obs(S_c, a_c, xi_16, f_c)
        oh_c = torch.cat((oh_c[:, 1:], o1_c.unsqueeze(1)), dim=1)

    J_wm_cont96 = (GAMMA ** 32) * c32 * V_wm_cont96

    g_state = torch.autograd.grad(J_state.sum(), A_seq, retain_graph=True)[0].detach()
    g_direct = torch.autograd.grad(J_direct.sum(), A_seq, retain_graph=True)[0].detach()
    g_crit = torch.autograd.grad(J_crit32.sum(), A_seq, retain_graph=True)[0].detach()
    g_wm_cont = torch.autograd.grad(J_wm_cont96.sum(), A_seq, retain_graph=False)[0].detach()
    g_prefix = g_state + g_direct

    # 2. Propagated state perturbation dS_32 = J_{WM,A} dA in 70-D tangent space
    print("=== [2] Controllable Terminal State Subspace dS_32 = J_{WM,A} dA ===")
    def _roll32_flat(A_in):
        S_r = tuple(x.detach().clone() for x in S0_16)
        for t in range(32):
            a_r = torch.clamp(A_in[:, t], -20.0, 20.0)
            S_r = proto.wm_step(wm, S_r, a_r, xi_n_16)
        return S_r

    S32_base, _ = torch.autograd.functional.jvp(_roll32_flat, (base_A_16,), (torch.zeros_like(base_A_16),))
    p_b, R_b, v_b, w_b, q_b, qd_b = S32_base
    Hy_b = yaw_mat(R_b)

    prop_subspace = {}
    for dname in dir_names:
        dA = dirs_o2[dname]
        _, dS_raw = torch.autograd.functional.jvp(_roll32_flat, (base_A_16,), (dA,))
        dp_w, dR_w, dv_w, dw_w, dq, dqd = dS_raw
        # Convert to 70-D local tangent coordinates matching perturb_state
        dp_loc = torch.einsum("bji,bj->bi", Hy_b, dp_w)
        # dR_w = R_b @ [dth]_x  => skew(R_b^T @ dR_w)
        skew = R_b.transpose(-1, -2) @ dR_w
        dth_loc = torch.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), -1)
        dv_loc = torch.einsum("bji,bj->bi", R_b, dv_w)
        dw_loc = torch.einsum("bji,bj->bi", R_b, dw_w)
        dS_tan = torch.cat((dp_loc, dth_loc, dv_loc, dw_loc, dq, dqd), -1)  # (16, 70)
        tot_norm = dS_tan.norm(dim=-1).clamp_min(1e-12)
        prop_subspace[dname] = {
            bname: float((dS_tan[:, sl].norm(dim=-1) / tot_norm).square().mean())
            for bname, sl in TANGENT_BLOCKS.items()
        }
        prop_subspace[dname]["dS_tan_norm_mean"] = float(tot_norm.mean())
    print("Energy fraction of dS_32 by tangent block:")
    for dname, st in prop_subspace.items():
        print(f"  {dname:11s} | norm={st['dS_tan_norm_mean']:.2f} | " + ", ".join(f"{k}={st[k]:.3f}" for k in TANGENT_BLOCKS))

    # 3. Counterfactual Critic Replacement against SAME Simulator Total-Return Reference
    print("\n=== [3] Counterfactual Critic Replacement (vs True Sim Total Return) ===")
    base_rs = torch.cat([b["base_res"][0]["r_state"] for b in o2["batches"]], 0)
    base_rd = torch.cat([b["base_res"][0]["r_direct"] for b in o2["batches"]], 0)
    base_ds = torch.cat([b["base_res"][0]["d_soft"] for b in o2["batches"]], 0)
    cc0 = torch.ones(16); bJs = torch.zeros(16); bJd = torch.zeros(16)
    for t in range(32):
        bJs += (GAMMA ** t) * cc0 * base_rs[:, t]
        bJd += (GAMMA ** t) * cc0 * base_rd[:, t]
        cc0 *= (1.0 - base_ds[:, t])
    base_rep0 = bJs + bJd + (GAMMA ** 32) * cc0 * torch.cat([b["base_res"][0]["V_cont_soft"][96] for b in o2["batches"]], 0)
    base_rep1 = bJs + bJd + (GAMMA ** 32) * cc0 * torch.cat([b["base_res"][1]["V_cont_soft"][96] for b in o2["batches"]], 0)
    base_o2_reps = torch.stack((base_rep0, base_rep1), 1)

    cf_results = {}
    cf_rows = []
    pooled = {k: [] for k in ("A", "B", "C", "D", "ref", "pref", "crit_only", "wm_cont_only", "sim_cont_only")}
    pooled_clips = []
    pooled_gates = []

    for dname in dir_names:
        dA = dirs_o2[dname]
        by_eps = {}
        for eps in o2["eps_grid"]:
            pm = {}
            for br in ("plus", "minus"):
                rs = torch.cat([b["branches"][(dname, eps, br)]["r_state"] for b in o2["batches"]], 0)
                rd = torch.cat([b["branches"][(dname, eps, br)]["r_direct"] for b in o2["batches"]], 0)
                ds = torch.cat([b["branches"][(dname, eps, br)]["d_soft"] for b in o2["batches"]], 0)
                cc = torch.ones(16); Js = torch.zeros(16); Jd = torch.zeros(16)
                for t in range(32):
                    Js += (GAMMA ** t) * cc * rs[:, t]
                    Jd += (GAMMA ** t) * cc * rd[:, t]
                    cc *= (1.0 - ds[:, t])
                Vcont = torch.cat([b["branches"][(dname, eps, br)]["V_cont_soft"][96] for b in o2["batches"]], 0)
                pm[br] = Js + Jd + (GAMMA ** 32) * cc * Vcont
                pm[br + "_cont"] = (GAMMA ** 32) * cc * Vcont
            by_eps[eps] = {
                "tot": (pm["plus"] - pm["minus"]) / (2.0 * eps),
                "cont": (pm["plus_cont"] - pm["minus_cont"]) / (2.0 * eps),
            }
        D_sim_tot = 0.5 * (by_eps[0.015]["tot"] + by_eps[0.03]["tot"])
        D_sim_cont = 0.5 * (by_eps[0.015]["cont"] + by_eps[0.03]["cont"])
        gates = [reliability_gate(base_o2_reps[i], torch.tensor([by_eps[e]["tot"][i] for e in o2["eps_grid"]]))[0] for i in range(16)]

        D_pref = (g_prefix * dA).sum((1, 2))
        D_crit = (g_crit * dA).sum((1, 2))
        D_wmc = (g_wm_cont * dA).sum((1, 2))

        D_A = D_pref + D_crit       # Original learned Critic
        D_B = D_pref + D_wmc        # WM continuation replacement
        D_C = D_pref                # No Critic (prefix only)
        D_D = D_pref + D_sim_cont   # Oracle Sim continuation replacement

        cf_results[dname] = {
            "n_gated": int(sum(gates)),
            "A_learned_critic": eval_cf(D_A, D_sim_tot, clips16),
            "B_wm_continuation": eval_cf(D_B, D_sim_tot, clips16),
            "C_prefix_only": eval_cf(D_C, D_sim_tot, clips16),
            "D_sim_oracle_cont": eval_cf(D_D, D_sim_tot, clips16),
            "prefix_vs_critic_sign": float(((D_pref * D_crit) > 0).float().mean()),
            "prefix_vs_sim_cont_sign": float(((D_pref * D_sim_cont) > 0).float().mean()),
        }
        if dname in policy_dirs:
            pooled["A"].append(D_A); pooled["B"].append(D_B); pooled["C"].append(D_C); pooled["D"].append(D_D)
            pooled["ref"].append(D_sim_tot); pooled["pref"].append(D_pref)
            pooled["crit_only"].append(D_crit); pooled["wm_cont_only"].append(D_wmc); pooled["sim_cont_only"].append(D_sim_cont)
            pooled_clips.extend(clips16)
            pooled_gates.extend(gates)

        for i in range(16):
            cf_rows.append({
                "direction": dname, "clip": clips16[i], "frame": int(starts16[i]["frame"]), "gate": int(gates[i]),
                "D_sim_tot_true": float(D_sim_tot[i]), "D_sim_cont": float(D_sim_cont[i]),
                "D_wm_prefix": float(D_pref[i]), "D_wm_critic": float(D_crit[i]), "D_wm_cont96": float(D_wmc[i]),
                "D_A_learned": float(D_A[i]), "D_B_wm_cont": float(D_B[i]), "D_C_no_critic": float(D_C[i]), "D_D_oracle": float(D_D[i]),
            })

    P = {k: torch.cat(v, 0) for k, v in pooled.items()}
    gm = torch.tensor(pooled_gates, dtype=torch.bool)
    gclips = [c for c, g in zip(pooled_clips, pooled_gates) if g]
    cf_results["ALL_POLICY_48"] = {
        "n_gated": int(gm.sum()),
        "A_learned_critic": eval_cf(P["A"], P["ref"], pooled_clips),
        "B_wm_continuation": eval_cf(P["B"], P["ref"], pooled_clips),
        "C_prefix_only": eval_cf(P["C"], P["ref"], pooled_clips),
        "D_sim_oracle_cont": eval_cf(P["D"], P["ref"], pooled_clips),
        "A_gated": eval_cf(P["A"][gm], P["ref"][gm], gclips),
        "B_gated": eval_cf(P["B"][gm], P["ref"][gm], gclips),
        "C_gated": eval_cf(P["C"][gm], P["ref"][gm], gclips),
        "D_gated": eval_cf(P["D"][gm], P["ref"][gm], gclips),
        "wm_cont_vs_sim_cont": eval_cf(P["wm_cont_only"], P["sim_cont_only"], pooled_clips),
        "learned_crit_vs_sim_cont": eval_cf(P["crit_only"], P["sim_cont_only"], pooled_clips),
        "prefix_vs_critic_sign": float(((P["pref"] * P["crit_only"]) > 0).float().mean()),
        "prefix_vs_sim_cont_sign": float(((P["pref"] * P["sim_cont_only"]) > 0).float().mean()),
    }

    print("\nALL_POLICY (n=48) Counterfactual Summary vs True Sim Total Return:")
    for key in ("A_learned_critic", "B_wm_continuation", "C_prefix_only", "D_sim_oracle_cont"):
        st = cf_results["ALL_POLICY_48"][key]
        print(f"  {key:20s} | sign={st['sign_agree']:.3f} CI95={st['sign_ci95_clip']} p={st['sign_exact_p']:.4f} | sp={st['spearman']:+.3f} | r={st['pearson']:+.3f} | ratio={st['med_ratio']:.2f} | rmse={st['rmse']:.2f}")
    print("  Cont-only: WM_cont96 vs Sim_cont96:", cf_results["ALL_POLICY_48"]["wm_cont_vs_sim_cont"]["sign_agree"], "sp=", cf_results["ALL_POLICY_48"]["wm_cont_vs_sim_cont"]["spearman"])
    print("  Cont-only: Learned_Crit vs Sim_cont96:", cf_results["ALL_POLICY_48"]["learned_crit_vs_sim_cont"]["sign_agree"], "sp=", cf_results["ALL_POLICY_48"]["learned_crit_vs_sim_cont"]["spearman"])

    with open(OUT / "counterfactual_per_start.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cf_rows[0].keys()))
        w.writeheader(); w.writerows(cf_rows)
    (OUT / "da05_prop_and_cf.json").write_text(json.dumps({
        "propagated_subspace": prop_subspace,
        "counterfactuals": cf_results,
    }, indent=2))
    print(f"\nSaved -> {OUT / 'da05_prop_and_cf.json'}")


if __name__ == "__main__":
    main()
