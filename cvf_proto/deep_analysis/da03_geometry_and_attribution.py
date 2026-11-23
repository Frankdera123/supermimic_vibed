"""Phase 5 & 6: Critic Local Value-Function Geometry (1st/2nd order) & Feature Attribution / Normalizer Diagnostics (CPU-only)."""
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
from supermimic.mdp import normalize_xi  # noqa: E402
from supermimic.networks import Critic  # noqa: E402

assert not torch.cuda.is_available(), "Must run CPU-only"
torch.set_num_threads(min(16, os.cpu_count() or 4))

OUT = REPO / "results" / "cvf" / "deep_analysis"
OUT.mkdir(parents=True, exist_ok=True)
SA = bridge.pairs_root()

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

TANGENT_BLOCKS = {
    "root_pos": slice(0, 3),
    "root_ori": slice(3, 6),
    "root_linvel": slice(6, 9),
    "root_angvel": slice(9, 12),
    "joint_pos": slice(12, 41),
    "joint_vel": slice(41, 70),
}
BRYSON = {
    "root_pos": 0.01,
    "root_ori": 0.02,
    "root_linvel": 0.1,
    "root_angvel": 0.3,
    "joint_pos": 0.1,
    "joint_vel": 3.14,
}


def critic_feature_groups(goal_frames: int = 10) -> dict[str, list[int]]:
    g = {
        "height": [0],
        "tilt6d": list(range(1, 7)),
        "root_linvel_b": list(range(7, 10)),
        "root_angvel_b": list(range(10, 13)),
        "joint_pos": list(range(13, 42)),
        "joint_vel": list(range(42, 71)),
        "joint_pos_err": list(range(71, 100)),
        "fk_body_pos_err": list(range(100, 142)),
        "root_pos_err": list(range(142, 145)),
        "xi_n": list(range(145, 178)),
    }
    base = 178
    ref_only, rel_ori = [], []
    for k in range(goal_frames):
        o = base + 73 * k
        ref_only += list(range(o, o + 67))
        rel_ori += list(range(o + 67, o + 73))
    g["goal_ref_only"] = ref_only
    g["goal_rel_ori"] = rel_ori
    g["goal_rel_pos"] = list(range(base + 73 * goal_frames, base + 73 * goal_frames + 3 * goal_frames))
    return g


def cosine_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1) / (a.norm(dim=-1).clamp_min(1e-12) * b.norm(dim=-1).clamp_min(1e-12))


def main():
    o1 = torch.load(REPO / "results/cvf/phase1b/oracle1_sim.pt", map_location="cpu", weights_only=False)
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

    critics = {}
    for cname, cpath, ckey in CRITIC_SPECS:
        ck = torch.load(cpath, map_location="cpu", weights_only=False)
        cr = Critic(ck[ckey]["norm.mean"].shape[0]).eval()
        cr.load_state_dict(ck[ckey], strict=True)
        for p in cr.parameters():
            p.requires_grad_(False)
        critics[cname] = cr

    cr_main = critics["B0@2100_post"]

    # 1. First-order geometry: Critic vs WM(T=64, 128) across 6 physical tangent blocks
    print("=== [1] First-Order Geometry (6 Physical Tangent Blocks) ===")
    cg_main = cvf_core.critic_tangent_grad(cr_main, mdp, motion, proto, S_star, f_star, xi_n_16)
    g_crit = cg_main["grad_raw"]  # (16, 70)
    wm_o1 = cvf_core.rollout_wm_continuation(
        wm, pi_old, mdp, proto, S_star, f_star, oh9_star, ap_star,
        xi_16, xi_n_16, T=128, hist_mode="last_only", need_grad=True, horizons_out=(64, 128)
    )
    g_wm64 = wm_o1["grads_soft"][64]
    g_wm128 = wm_o1["grads_soft"][128]

    block_geom = {}
    for bname, sl in TANGENT_BLOCKS.items():
        sc = BRYSON[bname]
        gc_b = g_crit[:, sl]
        gw64_b = g_wm64[:, sl]
        gw128_b = g_wm128[:, sl]
        block_geom[bname] = {
            "dim": sl.stop - sl.start,
            "bryson_scale": sc,
            "raw_norm_mean_crit": float(gc_b.norm(dim=-1).mean()),
            "raw_norm_mean_wm64": float(gw64_b.norm(dim=-1).mean()),
            "raw_norm_mean_wm128": float(gw128_b.norm(dim=-1).mean()),
            "bryson_norm_mean_crit": float((gc_b * sc).norm(dim=-1).mean()),
            "bryson_norm_mean_wm64": float((gw64_b * sc).norm(dim=-1).mean()),
            "bryson_norm_mean_wm128": float((gw128_b * sc).norm(dim=-1).mean()),
            "cos_crit_wm64_mean": float(cosine_batch(gc_b, gw64_b).mean()),
            "cos_crit_wm64_median": float(cosine_batch(gc_b, gw64_b).median()),
            "cos_crit_wm128_mean": float(cosine_batch(gc_b, gw128_b).mean()),
            "cos_crit_wm128_median": float(cosine_batch(gc_b, gw128_b).median()),
        }
    print(json.dumps(block_geom, indent=2))

    # 2. Second-order geometry: Directional Hessian kappa(u) = u^T H u & eps-linearity
    print("\n=== [2] Second-Order Geometry (Directional Curvature & Asymmetry) ===")
    S_det = tuple(x.detach() for x in S_star)
    curvature = {}
    for dname in dir_names:
        u = dirs_all[dname]
        delta = torch.zeros(16, 70, requires_grad=True)
        S_p = cvf_core.perturb_state(S_det, delta)
        v = cr_main(proto.critic_input(S_p, f_star, xi_n_16, proto.wm_goal(f_star, S_p)))
        g1 = torch.autograd.grad(v.sum(), delta, create_graph=True)[0]
        dir_d1 = (g1 * u).sum(-1)
        g2 = torch.autograd.grad(dir_d1.sum(), delta, retain_graph=False)[0].detach()
        kappa = (g2 * u).sum(-1)  # (16,)
        d1_val = dir_d1.detach()

        # Check Taylor approximation & plus/minus asymmetry across eps in {0.005, 0.015, 0.03, 0.06}
        with torch.no_grad():
            v0 = v.detach()
        eps_sweep = {}
        for eps in (0.005, 0.015, 0.03, 0.06):
            Sp = cvf_core.perturb_state(S_det, eps * u)
            Sm = cvf_core.perturb_state(S_det, -eps * u)
            with torch.no_grad():
                vp = cr_main(proto.critic_input(Sp, f_star, xi_n_16, proto.wm_goal(f_star, Sp)))
                vm = cr_main(proto.critic_input(Sm, f_star, xi_n_16, proto.wm_goal(f_star, Sm)))
            dv_p = vp - v0
            dv_m = vm - v0
            asym = (dv_p + dv_m).abs() / (vp - vm).abs().clamp_min(1e-6)
            taylor1 = eps * d1_val
            taylor2 = eps * d1_val + 0.5 * (eps ** 2) * kappa
            eps_sweep[f"eps_{eps:.3f}"] = {
                "med_asym": float(asym.median()),
                "rel_err_taylor1_med": float(((dv_p - taylor1).abs() / dv_p.abs().clamp_min(1e-6)).median()),
                "rel_err_taylor2_med": float(((dv_p - taylor2).abs() / dv_p.abs().clamp_min(1e-6)).median()),
                "fd_slope_mean_abs": float(((vp - vm) / (2 * eps)).abs().mean()),
            }

        # Sim plus/minus asymmetry at eps=0.015, T=64
        v0_sim64 = torch.cat([b["base_res"][0]["J_soft"][64] for b in o1["batches"]], 0)
        vp_sim64 = torch.cat([b["branches"][(dname, 0.015, "plus")]["J_soft"][64] for b in o1["batches"]], 0)
        vm_sim64 = torch.cat([b["branches"][(dname, 0.015, "minus")]["J_soft"][64] for b in o1["batches"]], 0)
        sim_asym = ((vp_sim64 - v0_sim64) + (vm_sim64 - v0_sim64)).abs() / (vp_sim64 - vm_sim64).abs().clamp_min(1e-6)

        curvature[dname] = {
            "d1_abs_median": float(d1_val.abs().median()),
            "kappa_abs_median": float(kappa.abs().median()),
            "kappa_mean": float(kappa.mean()),
            "curvature_to_slope_eps015_median": float((0.5 * 0.015 * kappa.abs() / d1_val.abs().clamp_min(1e-6)).median()),
            "sim64_eps015_asym_median": float(sim_asym.median()),
            "eps_sweep": eps_sweep,
        }
    print("Curvature (vel_reachable vs pose_safe_critic):")
    print("  pose_safe_critic:", json.dumps(curvature["pose_safe_critic"], indent=2))
    print("  vel_reachable:", json.dumps(curvature["vel_reachable"], indent=2))

    # 3. Feature Attribution: D_g(u) = (dV/dx_g)^T (J_{phi_g} u)
    print("\n=== [3] Critic Feature-Group Gradient Attribution ===")
    fgroups = critic_feature_groups(10)
    # Compute dV/dx (raw 938-D input) and dV/dx_tilde (normalized 938-D input)
    x_raw = proto.critic_input(S_det, f_star, xi_n_16, proto.wm_goal(f_star, S_det)).detach().requires_grad_(True)
    v_x = cr_main(x_raw)
    g_x = torch.autograd.grad(v_x.sum(), x_raw)[0].detach()  # (16, 938)

    x_tilde = cr_main.norm(x_raw.detach()).detach().requires_grad_(True)
    v_xt = cr_main.net(x_tilde).squeeze(-1)
    g_xt = torch.autograd.grad(v_xt.sum(), x_tilde)[0].detach()  # (16, 938)

    def _feat_fn(delta_vec):
        Sp = cvf_core.perturb_state(S_det, delta_vec)
        return proto.critic_input(Sp, f_star, xi_n_16, proto.wm_goal(f_star, Sp))

    zero_d = torch.zeros(16, 70)
    attrib = {}
    attrib_rows = []
    for dname in dir_names:
        u = dirs_all[dname]
        _, dx_du = torch.autograd.functional.jvp(_feat_fn, (zero_d,), (u,))
        d_total = (g_x * dx_du).sum(-1)
        d_ag = (g_crit * u).sum(-1)
        max_sum_err = float((d_total - d_ag).abs().max())
        grp_stats = {}
        for gname, idx in fgroups.items():
            dg = (g_x[:, idx] * dx_du[:, idx]).sum(-1)
            grp_stats[gname] = {
                "mean_signed": float(dg.mean()),
                "mean_abs": float(dg.abs().mean()),
                "median_abs": float(dg.abs().median()),
            }
            attrib_rows.append({
                "direction": dname,
                "feature_group": gname,
                "n_dims": len(idx),
                "mean_signed": float(dg.mean()),
                "mean_abs": float(dg.abs().mean()),
                "median_abs": float(dg.abs().median()),
            })
        sum_abs = sum(v["mean_abs"] for v in grp_stats.values())
        cancel_ratio = float(d_total.abs().mean()) / max(sum_abs, 1e-12)
        attrib[dname] = {
            "max_reconstruction_err": max_sum_err,
            "net_mean_abs": float(d_total.abs().mean()),
            "sum_group_mean_abs": sum_abs,
            "cancellation_ratio_net_over_sum": cancel_ratio,
            "groups": grp_stats,
        }
        print(f"  {dname:17s} | recon_err={max_sum_err:.2e} | net/sum={cancel_ratio:.3f} | top groups (mean_abs): "
              + ", ".join(f"{k}={v['mean_abs']:.2f}" for k, v in sorted(grp_stats.items(), key=lambda kv: -kv[1]["mean_abs"])[:4]))

    # 4. Normalizer Diagnostics across all 9 Critics
    print("\n=== [4] Normalizer Diagnostics Across 9 Critics ===")
    norm_diag = {}
    with torch.no_grad():
        x_star = proto.critic_input(S_det, f_star, xi_n_16, proto.wm_goal(f_star, S_det))
    for cname, cr in critics.items():
        std = cr.norm.std()
        z = cr.norm(x_star)
        # Compute g_xt for this critic
        xt = z.detach().requires_grad_(True)
        g_xt_c = torch.autograd.grad(cr.net(xt).sum(), xt)[0].detach()
        g_x_c = g_xt_c / std.unsqueeze(0)
        grp_norm = {}
        for gname, idx in fgroups.items():
            grp_norm[gname] = {
                "std_mean": float(std[idx].mean()),
                "std_min": float(std[idx].min()),
                "std_max": float(std[idx].max()),
                "z_abs_max": float(z[:, idx].abs().max()),
                "z_gt3_frac": float((z[:, idx].abs() > 3.0).float().mean()),
                "mlp_grad_rms": float(g_xt_c[:, idx].square().mean().sqrt()),
                "raw_grad_rms": float(g_x_c[:, idx].square().mean().sqrt()),
            }
        norm_diag[cname] = {
            "norm_count": float(cr.norm.count.item()),
            "overall_z_abs_max": float(z.abs().max()),
            "overall_z_gt3_frac": float((z.abs() > 3.0).float().mean()),
            "groups": grp_norm,
        }
    print("B0@2100_post normalizer std & MLP grad RMS by group:")
    for gname, st in norm_diag["B0@2100_post"]["groups"].items():
        print(f"  {gname:16s} | std_mean={st['std_mean']:7.4f} | z_max={st['z_abs_max']:5.2f} | mlp_grad_rms={st['mlp_grad_rms']:.4f} | raw_grad_rms={st['raw_grad_rms']:.4f}")

    with open(OUT / "feature_attribution.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(attrib_rows[0].keys()))
        w.writeheader(); w.writerows(attrib_rows)

    (OUT / "da03_geometry_attribution.json").write_text(json.dumps({
        "block_geometry_1st_order": block_geom,
        "curvature_2nd_order": curvature,
        "feature_attribution": attrib,
        "normalizer_diagnostics": norm_diag,
    }, indent=2))
    print(f"\nSaved -> {OUT / 'da03_geometry_attribution.json'}")


if __name__ == "__main__":
    main()
