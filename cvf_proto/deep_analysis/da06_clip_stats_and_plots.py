"""Generate all 10 real-data figures and clip-level statistical tables (CPU-only)."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "results" / "cvf" / "deep_analysis"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    da1 = json.loads((OUT / "da01_inventory_contract.json").read_text())
    da2 = json.loads((OUT / "da02_scalar_and_local.json").read_text())
    da3 = json.loads((OUT / "da03_geometry_attribution.json").read_text())
    da4 = json.loads((OUT / "da04_evolution_and_decomp.json").read_text())
    da5 = json.loads((OUT / "da05_prop_and_cf.json").read_text())

    df_scalar = pd.read_csv(OUT / "scalar_calibration_per_start.csv")
    df_o1 = pd.read_csv(REPO / "results/cvf/analysis/oracle1_per_start.csv")
    df_cf = pd.read_csv(OUT / "counterfactual_per_start.csv")
    df_evo = pd.read_csv(OUT / "checkpoint_evolution.csv")
    df_attr = pd.read_csv(OUT / "feature_attribution.csv")

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    # Fig 1: Value prediction vs Simulator return
    sub = df_scalar[df_scalar["critic"] == "B0@2100_post"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].scatter(sub["V_sim128_soft"], sub["V_crit"], c="#1f77b4", s=55, alpha=0.85)
    lims = [min(sub["V_sim128_soft"].min(), sub["V_crit"].min()) - 0.3,
            max(sub["V_sim128_soft"].max(), sub["V_crit"].max()) + 0.3]
    axes[0].plot(lims, lims, "k--", lw=1, label="y = x")
    m = da2["scalar_calibration"]["B0@2100_post"]["vs_sim128_soft"]
    axes[0].set_title(f"V_crit vs V_sim(128, unboot)\nr={m['pearson']:.3f}, bias={m['bias']:+.2f}")
    axes[0].set_xlabel("V_sim(128, soft, unbootstrapped)")
    axes[0].set_ylabel("V_crit (B0@2100_post)")
    axes[0].legend()

    axes[1].scatter(sub["V_wm128_boot_post"], sub["V_crit"], c="#2ca02c", s=55, alpha=0.85)
    lims2 = [min(sub["V_wm128_boot_post"].min(), sub["V_crit"].min()) - 0.3,
             max(sub["V_wm128_boot_post"].max(), sub["V_crit"].max()) + 0.3]
    axes[1].plot(lims2, lims2, "k--", lw=1, label="y = x")
    mb = da2["scalar_calibration"]["B0@2100_post"]["vs_wm128_boot_post"]
    axes[1].set_title(f"V_crit vs V_wm(128, bootstrapped)\nr={mb['pearson']:.3f}, bias={mb['bias']:+.2f}, RMSE={mb['rmse']:.3f}")
    axes[1].set_xlabel("V_wm(128) + gamma^128 c_128 V_crit(S_128)")
    axes[1].set_ylabel("V_crit (B0@2100_post)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig01_value_pred_vs_sim.png", dpi=160)
    plt.close(fig)

    # Fig 2: Global scalar value accuracy vs local directional gradient accuracy
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    labels = ["Global Scalar\n(Pearson r)", "Within-Start\n5-Dir Spearman", "pose_safe_critic\nSign (T=64)",
              "vel_critic\nSign (T=64)", "vel_reachable\nSign (T=64)", "vel_reachable\nSpearman (T=64)"]
    vals_c = [
        m["pearson"],
        da2["within_start_local_response"]["eps_0.015"]["within_start_5dir_spearman_crit_vs_sim64_mean"],
        da2["horizon_sensitivity"]["T64"]["pose_safe_critic"]["crit_vs_sim_all16_015"]["sign_agree"],
        da2["horizon_sensitivity"]["T64"]["vel_critic"]["crit_vs_sim_all16_015"]["sign_agree"],
        da2["horizon_sensitivity"]["T64"]["vel_reachable"]["crit_vs_sim_all16_015"]["sign_agree"],
        da2["horizon_sensitivity"]["T64"]["vel_reachable"]["crit_vs_sim_all16_015"]["spearman"],
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#2ca02c", "#d62728", "#d62728"]
    bars = ax.bar(range(len(labels)), vals_c, color=colors, alpha=0.85)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="Chance (0.50 for sign)")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.2, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Global Scalar Value Accuracy vs Local Directional Gradient Fidelity (B0@2100_post)")
    for b, v in zip(bars, vals_c):
        ax.text(b.get_x() + b.get_width()/2, v + (0.03 if v >= 0 else -0.07), f"{v:.2f}", ha="center", fontsize=8.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig02_global_vs_local_accuracy.png", dpi=160)
    plt.close(fig)

    # Fig 3: Pose / Velocity / Reachable directional response scatter (T=64, eps=0.015)
    o1_64 = df_o1[df_o1["T"] == 64]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for col, dname in enumerate(["pose_safe_critic", "vel_critic", "vel_reachable"]):
        sd = o1_64[o1_64["direction"] == dname]
        # Top: Critic vs Sim
        axes[0, col].scatter(sd["D_sim_015"], sd["D_crit_B0_2100"], c="#d62728" if dname == "vel_reachable" else "#1f77b4", alpha=0.8)
        axes[0, col].axhline(0, color="k", lw=0.6); axes[0, col].axvline(0, color="k", lw=0.6)
        axes[0, col].set_title(f"{dname}: Critic vs Sim(T=64)")
        axes[0, col].set_xlabel("D_sim (eps=0.015)"); axes[0, col].set_ylabel("D_crit")
        # Bottom: WM vs Sim
        axes[1, col].scatter(sd["D_sim_015"], sd["D_wm"], c="#2ca02c", alpha=0.8)
        axes[1, col].axhline(0, color="k", lw=0.6); axes[1, col].axvline(0, color="k", lw=0.6)
        axes[1, col].set_title(f"{dname}: WM vs Sim(T=64)")
        axes[1, col].set_xlabel("D_sim (eps=0.015)"); axes[1, col].set_ylabel("D_wm")
    fig.tight_layout()
    fig.savefig(FIG / "fig03_o1_direction_scatter.png", dpi=160)
    plt.close(fig)

    # Fig 4: Epsilon stability (Critic Taylor/Asym vs Sim Asym)
    eps_vals = [0.005, 0.015, 0.030, 0.060]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for dname, col in [("pose_safe_critic", "#1f77b4"), ("vel_critic", "#2ca02c"), ("vel_reachable", "#d62728"), ("vel_random", "#9467bd")]:
        sw = da3["curvature_2nd_order"][dname]["eps_sweep"]
        asyms = [sw[f"eps_{e:.3f}"]["med_asym"] for e in eps_vals]
        t1err = [sw[f"eps_{e:.3f}"]["rel_err_taylor1_med"] for e in eps_vals]
        axes[0].plot(eps_vals, asyms, "o-", label=dname, color=col)
        axes[1].plot(eps_vals, t1err, "s--", label=dname, color=col)
    axes[0].set_title("Critic Plus/Minus Asymmetry vs eps\n(cf. Sim64 eps=0.015 asym: pose_c=0.27, vel_r=0.95)")
    axes[0].set_xlabel("epsilon"); axes[0].set_ylabel("Median Asymmetry")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Critic 1st-Order Taylor Relative Error vs eps")
    axes[1].set_xlabel("epsilon"); axes[1].set_ylabel("Median Relative Error")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_eps_stability.png", dpi=160)
    plt.close(fig)

    # Fig 5: Feature-group gradient attribution
    piv = df_attr.pivot(index="feature_group", columns="direction", values="mean_abs")
    top_groups = piv.max(axis=1).sort_values(ascending=False).head(8).index
    piv_top = piv.loc[top_groups, ["pose_safe_critic", "pose_safe_random", "vel_critic", "vel_reachable", "vel_random"]]
    ax = piv_top.plot(kind="bar", figsize=(10.5, 4.6), width=0.78)
    ax.set_title("Critic Feature-Group Contribution |D_g(u)| Across Oracle 1 Directions")
    ax.set_ylabel("Mean |D_g(u)|")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIG / "fig05_feature_attribution.png", dpi=160)
    plt.close()

    # Fig 6: Normalizer std & effective physical gradient
    nd = da3["normalizer_diagnostics"]["B0@2100_post"]["groups"]
    grps = list(nd.keys())
    stds = [nd[g]["std_mean"] for g in grps]
    mlp_g = [nd[g]["mlp_grad_rms"] for g in grps]
    raw_g = [nd[g]["raw_grad_rms"] for g in grps]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    x = np.arange(len(grps))
    axes[0].bar(x, stds, color="#1f77b4", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(grps, rotation=35, ha="right", fontsize=8)
    axes[0].set_title("RunningNorm std (mean across group dims)")
    axes[0].set_ylabel("std")
    w = 0.38
    axes[1].bar(x - w/2, mlp_g, w, label="MLP input grad RMS (normalized)", color="#ff7f0e")
    axes[1].bar(x + w/2, raw_g, w, label="Physical feature grad RMS (raw)", color="#2ca02c")
    axes[1].set_xticks(x); axes[1].set_xticklabels(grps, rotation=35, ha="right", fontsize=8)
    axes[1].set_title("MLP-Space vs Physical-Space Gradient RMS")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_normalizer_and_physical_grad.png", dpi=160)
    plt.close(fig)

    # Fig 7: Nine-Critic evolution
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    b0_idx = list(range(6))
    sub_b0 = df_evo.iloc[b0_idx]
    axes[0].plot(range(6), sub_b0["val_pearson_sim128"], "o-", label="Scalar Value Pearson")
    axes[0].plot(range(6), sub_b0["pose_c_sign64"], "s-", label="pose_safe_critic Sign (T=64)")
    axes[0].plot(range(6), sub_b0["vel_c_sign64"], "^-", label="vel_critic Sign (T=64)")
    axes[0].plot(range(6), sub_b0["vel_r_sign64"], "d-", color="#d62728", label="vel_reachable Sign (T=64)")
    axes[0].plot(range(6), sub_b0["o2_TrueTot_sign"], "x--", color="k", label="O2 TrueTotal Sign (n=48)")
    axes[0].axhline(0.5, color="gray", ls=":", lw=1)
    axes[0].set_xticks(range(6)); axes[0].set_xticklabels(sub_b0["critic"], rotation=25, ha="right", fontsize=8.5)
    axes[0].set_ylim(0.25, 1.05)
    axes[0].set_title("Baseline Lineage Evolution (b0@1000 -> B0@2300)")
    axes[0].legend(fontsize=7.5)

    sub_23 = df_evo.iloc[5:9]
    x2 = np.arange(4); w = 0.2
    axes[1].bar(x2 - 1.5*w, sub_23["pose_c_sign64"], w, label="pose_c Sign64")
    axes[1].bar(x2 - 0.5*w, sub_23["vel_c_sign64"], w, label="vel_c Sign64")
    axes[1].bar(x2 + 0.5*w, sub_23["vel_r_sign64"], w, label="vel_r Sign64")
    axes[1].bar(x2 + 1.5*w, sub_23["o2_TrueTot_sign"], w, label="O2 TrueTot Sign")
    axes[1].axhline(0.5, color="gray", ls=":", lw=1)
    axes[1].set_xticks(x2); axes[1].set_xticklabels(sub_23["critic"], fontsize=9)
    axes[1].set_ylim(0.25, 1.05)
    axes[1].set_title("Divergent Branches at Iter 2300 (B0/B1/B2/B3)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_checkpoint_evolution.png", dpi=160)
    plt.close(fig)

    # Fig 8: Error A vs Error B decomposition
    dirs_o1 = ["pose_safe_critic", "pose_safe_random", "vel_critic", "vel_reachable", "vel_random"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, Tkey in zip(axes, ["T64", "T128"]):
        rmseA = [da4["error_decomposition"][Tkey][d]["all16_eps015"]["rmse_eA"] for d in dirs_o1]
        rmseB = [da4["error_decomposition"][Tkey][d]["all16_eps015"]["rmse_eB"] for d in dirs_o1]
        rAB = [da4["error_decomposition"][Tkey][d]["all16_eps015"]["pearson_eA_eB"] for d in dirs_o1]
        x = np.arange(len(dirs_o1)); w = 0.36
        ax.bar(x - w/2, rmseA, w, label="RMSE(e_A = D_c - D_m)", color="#1f77b4")
        ax.bar(x + w/2, rmseB, w, label="RMSE(e_B = D_m - D_s)", color="#ff7f0e")
        for i, rval in enumerate(rAB):
            ax.text(i, max(rmseA[i], rmseB[i]) + 0.03, f"r={rval:+.2f}", ha="center", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(dirs_o1, rotation=25, ha="right", fontsize=8)
        ax.set_title(f"Error A vs Error B RMSE ({Tkey}, ALL16 eps=0.015)")
        ax.set_ylabel("RMSE")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig08_error_A_vs_B.png", dpi=160)
    plt.close(fig)

    # Fig 9: Clip-level fidelity heatmap (8 clips x O1/O2 sign accuracy)
    clips = sorted(o1_64["clip"].unique())
    heat = np.zeros((len(clips), 6))
    col_names = ["O1_pose_c\nCvsS", "O1_vel_c\nCvsS", "O1_vel_r\nCvsS", "O1_vel_r\nWMvsS", "O2_A\nLearned", "O2_B\nWM_Cont"]
    cf_pol = df_cf[df_cf["direction"].isin(["delta_2100", "delta_2300", "ascent"])]
    clip_rows = []
    for i, cl in enumerate(clips):
        for j, dname in enumerate(["pose_safe_critic", "vel_critic", "vel_reachable"]):
            s = o1_64[(o1_64["clip"] == cl) & (o1_64["direction"] == dname)]
            heat[i, j] = float(((s["D_crit_B0_2100"] * s["D_sim_015"]) > 0).mean())
        s_vr = o1_64[(o1_64["clip"] == cl) & (o1_64["direction"] == "vel_reachable")]
        heat[i, 3] = float(((s_vr["D_wm"] * s_vr["D_sim_015"]) > 0).mean())
        sc = cf_pol[cf_pol["clip"] == cl]
        heat[i, 4] = float(((sc["D_A_learned"] * sc["D_sim_tot_true"]) > 0).mean())
        heat[i, 5] = float(((sc["D_B_wm_cont"] * sc["D_sim_tot_true"]) > 0).mean())
        clip_rows.append({"clip": cl, **{col_names[k].replace("\n", "_"): heat[i, k] for k in range(6)}})

    with open(OUT / "clip_level_fidelity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(clip_rows[0].keys()))
        w.writeheader(); w.writerows(clip_rows)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    im = ax.imshow(heat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels(col_names, fontsize=8.5)
    ax.set_yticks(range(len(clips))); ax.set_yticklabels(clips, fontsize=8.5)
    for i in range(len(clips)):
        for j in range(6):
            ax.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8.5)
    fig.colorbar(im, ax=ax, label="Sign Agreement")
    ax.set_title("Clip-Level Fidelity Heatmap (8 LAFAN1 Clips)")
    fig.tight_layout()
    fig.savefig(FIG / "fig09_clip_fidelity_heatmap.png", dpi=160)
    plt.close(fig)

    # Fig 10: Counterfactual Critic Replacement
    cf48 = da5["counterfactuals"]["ALL_POLICY_48"]
    keys = ["A_learned_critic", "B_wm_continuation", "C_prefix_only", "D_sim_oracle_cont"]
    names = ["A: Learned Critic\n(WM_pref + Crit)", "B: WM Cont96\n(WM_pref + WM_cont)", "C: No Critic\n(WM_pref only)", "D: Sim Oracle\n(WM_pref + Sim_cont)"]
    signs = [cf48[k]["sign_agree"] for k in keys]
    ci_lo = [cf48[k]["sign_agree"] - cf48[k]["sign_ci95_clip"][0] for k in keys]
    ci_hi = [cf48[k]["sign_ci95_clip"][1] - cf48[k]["sign_agree"] for k in keys]
    sps = [cf48[k]["spearman"] for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    x = np.arange(4)
    cols = ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"]
    axes[0].bar(x, signs, yerr=[ci_lo, ci_hi], capsize=4, color=cols, alpha=0.85)
    axes[0].axhline(0.5, color="gray", ls="--", lw=1)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Sign Agreement (95% Clip CI)")
    axes[0].set_title("Counterfactual Sign Agreement vs True Sim Total Return (n=48)")
    for i, v in enumerate(signs):
        axes[0].text(i, v + ci_hi[i] + 0.03, f"{v:.3f}", ha="center", fontsize=8.5)

    axes[1].bar(x, sps, color=cols, alpha=0.85)
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=8)
    axes[1].set_ylim(-0.1, 1.05); axes[1].set_ylabel("Spearman Correlation")
    axes[1].set_title("Counterfactual Spearman vs True Sim Total Return (n=48)")
    for i, v in enumerate(sps):
        axes[1].text(i, v + 0.03, f"{v:+.3f}", ha="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG / "fig10_counterfactual_replacement.png", dpi=160)
    plt.close(fig)
    print(f"All 10 figures saved -> {FIG}")


if __name__ == "__main__":
    main()
