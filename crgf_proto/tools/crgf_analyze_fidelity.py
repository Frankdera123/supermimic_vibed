"""CRGF fidelity analysis and H3/H4/H5 verdicts (CPU-only).

Merges the WM-side gradient CSVs with the simulator-side FD CSV (once Phase
1/2 has run) and the historical per-clip single-update pair audit, then
produces:

  * per-(model, H, direction_set, path) fidelity stats on gated samples:
    sign agreement, MAE, magnitude-ratio median, Spearman, clip-level
    bootstrap CI for B2-vs-B1 paired differences;
  * H3 (metric validity): does per-clip gradient fidelity predict the
    historical per-clip DeltaJ agreement (needs --history_csv)?
  * H4 (direction relevance): fidelity on policy-relevant directions
    (delta/ascent) vs random, paired per clip;
  * H5 (critic attribution): D_critic fidelity vs D_state fidelity.

Usage:
  python tools/crgf_analyze_fidelity.py --wm_csv a.csv b.csv c.csv \
      --sim_csv sim.csv [--history_csv opcl_single_update_feedback_audit.csv] \
      --out_prefix results/crgf/fidelity_verdict
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def fidelity_stats(pairs):
    """pairs: list of (d_wm, d_sim) on gated samples; returns stats dict."""
    if not pairs:
        return {"n": 0, "sign_agreement": None, "mae": None, "median_ratio": None,
                "spearman": None}
    w = [p[0] for p in pairs]
    s = [p[1] for p in pairs]
    dec = [(a, b) for a, b in pairs if abs(a) + abs(b) > 0]
    sign = (sum(1 for a, b in dec if (a > 0) == (b > 0)) / len(dec)) if dec else None
    mae = st.mean([abs(a - b) for a, b in pairs])
    ratio = st.median([abs(a) / max(abs(b), 1e-12) for a, b in pairs])

    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        ranks = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            r = 0.5 * (i + j)
            for k in range(i, j + 1):
                ranks[order[k]] = r
            i = j + 1
        return ranks

    if len(pairs) >= 3 and len({round(v, 12) for v in w}) > 1:
        rx, ry = rank(w), rank(s)
        mx, my = st.mean(rx), st.mean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        rho = num / den if den > 0 else None
    else:
        rho = None
    return {"n": len(pairs), "sign_agreement": sign, "mae": mae,
            "median_ratio": ratio, "spearman": rho}


def clip_bootstrap(diffs, seed=0, iters=2000):
    if len(diffs) < 3:
        return None
    rng = random.Random(seed)
    boots = sorted(st.median([rng.choice(diffs) for _ in diffs]) for _ in range(iters))
    return [boots[int(0.025 * iters)], boots[int(0.975 * iters)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_csv", nargs="+", required=True)
    ap.add_argument("--sim_csv", default=None, help="absent -> WM-side-only verdicts")
    ap.add_argument("--history_csv", default=None,
                    help="OPCL per-clip single-update audit (H3 target)")
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

    # ---- load WM side: (model, H, dirset) -> clip -> {path: D} ----
    wm = defaultdict(dict)
    for path in args.wm_csv:
        for r in csv.DictReader(open(path)):
            raw = r.get("model", r.get("wm_ckpt", "?"))
            model = Path(raw).parent.name if "/" in raw else Path(raw).stem
            key = (model, int(r["horizon"]), r["direction_set"], r["clip"], int(r["frame"]))
            wm[key] = {p: float(r[f"D_{p}"]) for p in ("state", "direct", "critic", "total")}

    # ---- load sim side: (H, dirset) -> clip -> {path: D_sim, gate} ----
    sim = {}
    if args.sim_csv:
        for r in csv.DictReader(open(args.sim_csv)):
            key = (int(r["horizon"]), r["direction_set"], r["clip"], int(r["frame"]))
            sim[key] = {"D": {p: r[f"D_sim_{p}"] for p in ("state", "direct", "critic")},
                        "gate": {p: r[f"gate_{p}"] for p in ("state", "direct", "critic")}}

    # ---- fidelity per (model, H, dirset, path) ----
    fid = {}
    for (model, H, dset, clip, frame), dpath in wm.items():
        # sim-side ascent sets are tagged ascent_<model>; map to the WM "ascent"
        s_dset = f"ascent_{model}" if dset == "ascent" else dset
        skey = (H, s_dset, clip, frame)
        if skey not in sim:
            continue
        s = sim[skey]
        for p in ("state", "direct", "critic", "total"):
            if s["D"].get(p) in (None, "", "None"):
                continue
            gate = s["gate"][p] in ("True", "true", True)
            if not gate:
                continue
            fid.setdefault((model, H, dset, p), []).append(
                (clip, dpath[p], float(s["D"][p])))

    stats = {}
    for (model, H, dset, p), pairs in sorted(fid.items()):
        stats[f"{model}|H{H}|{dset}|{p}"] = fidelity_stats([(a, b) for _, a, b in pairs])
    if not stats:
        print("note: no matched (H, direction_set, clip) cells between WM and sim CSVs "
              "-- run after Phase 1/2 with identical protocol tags")

    # ---- H4: policy-relevant vs random, paired per clip ----
    h4 = {}
    for (model, H, dset, p), pairs in fid.items():
        if dset in ("delta",) or dset.startswith("ascent"):
            h4.setdefault((model, H, p, "relevant"), []).extend(pairs)
        if dset == "random":
            h4.setdefault((model, H, p, "random"), []).extend(pairs)
    h4_verdict = {}
    for (model, H, p, tag), pairs in sorted(h4.items()):
        h4_verdict[f"{model}|H{H}|{p}|{tag}"] = fidelity_stats([(a, b) for _, a, b in pairs])

    # ---- H5: critic vs state fidelity ----
    h5_verdict = {k: v for k, v in stats.items() if k.endswith("|critic") or k.endswith("|state")}

    # ---- H3: per-clip fidelity vs historical per-clip DeltaJ agreement ----
    h3 = None
    if args.history_csv and args.sim_csv:
        hist = {}
        for r in csv.DictReader(open(args.history_csv)):
            hist[r["clip_name"]] = {
                "sign_agree": 1.0 if r.get("matched_sign_agree") == "True" else 0.0,
                "abs_err": abs(float(r["matched_delta_wm"]) - float(r["matched_delta_sim"])),
            }
        # fidelity per clip on the delta direction set (policy-update directions)
        for (model, H, dset, p), pairs in fid.items():
            if dset != "delta" or p != "state":
                continue
            both = [(clip, abs(a - b), hist[clip]["sign_agree"])
                    for clip, a, b in pairs if clip in hist]
            if len(both) >= 5:
                errs = [e for _, e, _ in both]
                agree = [a for _, _, a in both]
                # point-biserial: correlation between |D error| and sign agreement
                n1, n0 = sum(agree), len(agree) - sum(agree)
                m1 = st.mean([err for _, err, ag in both if ag == 1]) if n1 >= 1 else None
                m0 = st.mean([err for _, err, ag in both if ag == 0]) if n0 >= 1 else None
                if n1 >= 2 and n0 >= 2:
                    sd = st.stdev(errs)
                    rpb = ((m1 - m0) / sd) * (n1 * n0 / (len(both) ** 2)) ** 0.5 if sd > 0 else None
                else:
                    rpb = None
                    h3_note = f"insufficient variance (agree={int(n1)}, disagree={int(n0)})"
                h3 = {"model": model, "horizon": H, "n_clips": len(both),
                      "point_biserial_err_vs_agree": rpb,
                      "mean_abs_err_agree": m1, "mean_abs_err_disagree": m0,
                      "note": h3_note if rpb is None else "ok"}

    # ---- paired B2 vs B1 on fidelity ----
    paired = {}
    by_clip = defaultdict(dict)
    for (model, H, dset, p), pairs in fid.items():
        if dset != "delta" or H != 32 or p != "total":
            continue
        for clip, a, b in pairs:
            by_clip[clip][model] = abs(a - b)
    for clip, m in by_clip.items():
        if "online_B1" in m and "online_B2" in m:
            paired.setdefault("diffs", []).append(m["online_B2"] - m["online_B1"])
    if "diffs" in paired:
        paired["n"] = len(paired["diffs"])
        paired["median"] = st.median(paired["diffs"])
        paired["bootstrap_ci95"] = clip_bootstrap(paired["diffs"])
        paired.pop("diffs")

    verdict = {
        "fidelity_stats": stats,
        "H4_direction_relevance": h4_verdict,
        "H5_critic_vs_state": h5_verdict,
        "H3_metric_validity": h3,
        "B2_vs_B1_paired_delta_H32": paired,
        "note": "statistics unit = motion clip; gated samples only (fail-closed); "
                "D_sim model-independent, D_WM per model",
    }
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(verdict, fh, indent=1)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "n", "sign_agreement", "mae", "median_ratio", "spearman"])
        for k, v in stats.items():
            w.writerow([k, v["n"], v["sign_agreement"], v["mae"], v["median_ratio"], v["spearman"]])
    print(json.dumps({"out": str(out.with_suffix(".json")),
                      "n_stat_keys": len(stats),
                      "H3": h3, "B2_vs_B1": paired}, indent=1))


if __name__ == "__main__":
    main()
