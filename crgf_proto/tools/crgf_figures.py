"""Generate the key figures required by Goal §10 from the WM-side CRGF audit CSVs.

CPU-only, no simulator, no torch.  Inputs:
  results/crgf/wm_audit_b0b1b2_shared501.csv   (shared 501 critic)
  results/crgf/wm_audit_b0b1b2_owncritic.csv   (each branch's own critic)
Outputs (results/crgf/figures/, synced to docs/crgf/figures by the caller):
  fig_path_shares.png        three-path |D| composition per model x direction set x H
  fig_horizon_trend.png      critic share and state-path median vs horizon, per direction set
  fig_direction_sets.png     direction-set comparison of |D_state| / |D_total| at each H
  fig_critic_own_vs_shared.png  critic-source cross: |D_critic| and |D_total|
  fig_summary_table.csv      the aggregated medians behind every figure
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results" / "crgf"
FIG = RES / "figures"
MODELS = ["online_B0", "online_B1", "online_B2"]
LABELS = {"online_B0": "B0", "online_B1": "B1", "online_B2": "B2"}
PATHS = ["D_state", "D_direct", "D_critic"]
COLORS = {"D_state": "#2b7bba", "D_direct": "#7fbf7b", "D_critic": "#d16227"}
DIR_ORDER = ["random", "repeated", "delta", "ascent"]


def load(name):
    rows = list(csv.DictReader(open(RES / name)))
    agg = defaultdict(list)  # (model, H, dirset) -> path -> [ |D| ]
    for r in rows:
        key = (r["model"].split("/")[-2], int(r["horizon"]), r["direction_set"])
        for p in PATHS + ["D_total"]:
            agg[key + (p,)].append(abs(float(r[p])))
    med = {k: st.median(v) for k, v in agg.items()}
    return rows, med


def fig_path_shares(med, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=False)
    for ax, H in zip(axes, (8, 32)):
        xs, state_h, direct_h, critic_h, labels = [], [], [], [], []
        for H_ in [H]:
            for m in MODELS:
                for d in DIR_ORDER:
                    xs.append(len(xs))
                    state_h.append(med[(m, H_, d, "D_state")])
                    direct_h.append(med[(m, H_, d, "D_direct")])
                    critic_h.append(med[(m, H_, d, "D_critic")])
                    labels.append(f"{LABELS[m]}\n{d}")
        ax.bar(xs, state_h, color=COLORS["D_state"], label="state path")
        ax.bar(xs, direct_h, bottom=state_h, color=COLORS["D_direct"], label="direct (action-rate)")
        ax.bar(xs, critic_h, bottom=[a + b for a, b in zip(state_h, direct_h)],
               color=COLORS["D_critic"], label="critic bootstrap")
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7, rotation=60, ha="right")
        ax.set_title(f"H={H_}  (median |D| per start, unit direction)")
        ax.set_ylabel("median |D|")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Three-path decomposition of the imagined-policy gradient (shared 501 critic)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_horizon_trend(med, out):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, d in zip(axes, DIR_ORDER):
        for m, c in zip(MODELS, ("#2b7bba", "#7fbf7b", "#d16227")):
            hs, share, state_m = [], [], []
            for H in (8, 32):
                tot = med[(m, H, d, "D_state")] + med[(m, H, d, "D_direct")] + med[(m, H, d, "D_critic")]
                hs.append(H)
                share.append(med[(m, H, d, "D_critic")] / tot if tot > 0 else float("nan"))
                state_m.append(med[(m, H, d, "D_state")])
            ax.plot(hs, share, "o-", color=c, label=f"{LABELS[m]} critic share")
            ax.plot(hs, state_m, "s--", color=c, alpha=0.45, label=f"{LABELS[m]} med|D_state|")
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 32])
        ax.set_xticklabels(["8", "32"])
        ax.set_title(f"direction: {d}", fontsize=10)
        ax.set_xlabel("horizon")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("critic share / med|D_state|")
    axes[0].legend(fontsize=7)
    fig.suptitle("Horizon trend: critic dominance fades from H=8 to H=32 (secondary axis sense: |D_state| in relative units)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_direction_sets(med, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.27
    for ax, H in zip(axes, (8, 32)):
        for i, (m, c) in enumerate(zip(MODELS, ("#2b7bba", "#7fbf7b", "#d16227"))):
            xs = list(range(len(DIR_ORDER)))
            vals_state = [med[(m, H, d, "D_state")] for d in DIR_ORDER]
            vals_tot = [med[(m, H, d, "D_total")] for d in DIR_ORDER]
            off = (i - 1) * width
            ax.bar([x + off - width / 2 for x in xs], vals_state, width,
                   color=c, alpha=0.9, label=f"{LABELS[m]} |D_state|" if H == 8 else None)
            ax.bar([x + off + width / 2 for x in xs], vals_tot, width,
                   color=c, alpha=0.35, hatch="//", label=f"{LABELS[m]} |D_total|" if H == 8 else None)
        ax.set_xticks(range(len(DIR_ORDER)))
        ax.set_xticklabels(DIR_ORDER)
        ax.set_title(f"H={H}")
        ax.set_ylabel("median |D|")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Direction-set comparison: policy-relevant (delta/ascent) vs random (H4 test, WM side)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_critic_own_vs_shared(med_s, med_o, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.35
    for ax, H in zip(axes, (8, 32)):
        xs = list(range(len(MODELS)))
        own_c = [med_o[(m, H, "ascent", "D_critic")] for m in MODELS]
        sh_c = [med_s[(m, H, "ascent", "D_critic")] for m in MODELS]
        own_t = [med_o[(m, H, "ascent", "D_total")] for m in MODELS]
        sh_t = [med_s[(m, H, "ascent", "D_total")] for m in MODELS]
        ax.bar([x - width / 2 for x in xs], own_c, width, color="#d16227", alpha=0.9,
               label="own critic |D_critic|" if H == 8 else None)
        ax.bar([x + width / 2 for x in xs], sh_c, width, color="#d16227", alpha=0.4,
               label="shared-501 |D_critic|" if H == 8 else None)
        ax.plot(xs, own_t, "o--", color="#333333", label="own |D_total| (ascent)" if H == 8 else None)
        ax.plot(xs, sh_t, "s--", color="#777777", label="shared |D_total| (ascent)" if H == 8 else None)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[m] for m in MODELS])
        ax.set_title(f"H={H} (ascent directions)")
        ax.set_ylabel("median |D|")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("Critic-source cross evaluation: the critic choice moves the gradient more than the WM branch does")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    rows_s, med_s = load("wm_audit_b0b1b2_shared501.csv")
    rows_o, med_o = load("wm_audit_b0b1b2_owncritic.csv")

    fig_path_shares(med_s, FIG / "fig_path_shares.png")
    fig_horizon_trend(med_s, FIG / "fig_horizon_trend.png")
    fig_direction_sets(med_s, FIG / "fig_direction_sets.png")
    fig_critic_own_vs_shared(med_s, med_o, FIG / "fig_critic_own_vs_shared.png")

    # traceable aggregate table behind all figures
    keys = sorted(set(list(med_s.keys()) + list(med_o.keys())))
    with open(FIG / "fig_summary_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "H", "direction_set", "critic_mode",
                    "med|D_state|", "med|D_direct|", "med|D_critic|", "med|D_total|",
                    "critic_share"])
        for k in keys:
            m, H, d, p = k
            src = med_s if p == "D_state" or True else None
            for tag, medd in (("shared501", med_s), ("own", med_o)):
                if k in medd:
                    s, di, c = medd[(m, H, d, "D_state")], medd[(m, H, d, "D_direct")], medd[(m, H, d, "D_critic")]
                    tot = s + di + c
                    w.writerow([m, H, d, tag, s, di, c, medd[(m, H, d, "D_total")],
                                c / tot if tot > 0 else ""])
    meta = {"n_rows_shared": len(rows_s), "n_rows_own": len(rows_o),
            "note": "medians over 14 starts; per-start values in the source CSVs; "
                    "statistics unit = motion clip (cluster), not frame x direction"}
    json.dump(meta, open(FIG / "figure_meta.json", "w"), indent=1)
    print(json.dumps({"figures": sorted(p.name for p in FIG.glob("*.png")),
                      "table": str(FIG / 'fig_summary_table.csv'), **meta}, indent=1))


if __name__ == "__main__":
    main()
