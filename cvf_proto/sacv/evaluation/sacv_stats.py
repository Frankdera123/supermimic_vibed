"""CVF-1a Phase D (v2) statistics: layered, paired, clip-clustered (CPU).

Consumes a sacv_preds_v2 file (sacv_dump_preds.py) and reports, per critic:

  layers   all     : every pair
           nobif   : no single-sided termination (same soft-survival class and same
                     hard-termination flag on both branches) -- CVF-0's "排除单侧跌倒分叉"
           gated   : nobif AND cross-epsilon consistency of the REFERENCE response
                     D(eps) = Delta G / (2 eps):  |D(e1) - D(e2)| <= max(5e-4, 0.2 |D(e1)|)
  metrics  finite-difference pair response FD = V(x_H+) - V(x_H-)  vs  Delta G (primary,
           no learned bootstrap) and Delta Y (matched training target); the chord-averaged
           autograd derivative vs Delta G; FD-vs-autograd agreement; reachable subset
           (delta_2100 / delta_2300 = policy-update action deltas); within-start Spearman;
           leave-one-clip-out ranges; per family; cross-horizon sensitivity (Delta G_32/64).
  stats    clip-clustered bootstrap CI95 (B=10000) for sign and Spearman; exact binomial
           sign test vs chance; paired critic-vs-critic residual comparisons with clip-
           clustered bootstrap CI and exact paired sign test.
  gate     the pre-registered C2-vs-C1 criteria evaluated on Val layers (D6), with the
           total-gradient item marked NOT EVALUATED (D4 not implemented).

Nothing here treats two epsilons, two branches or two checkpoints of one start as
independent samples: resampling units are clips, pairing units are pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent          # evaluation/
_SACV = _HERE.parent
_CVF = _SACV.parent
_REPO = _CVF.parent

REACHABLE = ("delta_2100", "delta_2300")
N_BOOT = 10000
SEED = 2700
SIGN_CUTOFF = 0.5        # soft-survival class boundary for "terminated"


# ----------------------------------------------------------------- primitives
def _sign(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(np.sign(pred) == np.sign(ref))) if len(ref) else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    # average ties (exact ties are rare in float data but handled)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra, rb = _rank(a) - (len(a) - 1) / 2.0, _rank(b) - (len(b) - 1) / 2.0
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _ratio(pred: np.ndarray, ref: np.ndarray) -> float:
    m = np.abs(ref) > 0
    return float(np.median(np.abs(pred[m]) / np.abs(ref[m]))) if m.any() else float("nan")


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test of k successes out of n at p=0.5."""
    if n == 0:
        return float("nan")
    from math import comb
    pk = comb(n, k) / 2 ** n
    return float(min(1.0, sum(comb(n, i) / 2 ** n for i in range(n + 1)
                              if comb(n, i) / 2 ** n <= pk * (1 + 1e-9))))


def clip_boot(fn, idx: list[int], clips: list[str], n_boot: int = N_BOOT, seed: int = SEED):
    """Clip-clustered bootstrap CI95 of fn(subset_of_idx) -> float (nan samples dropped)."""
    uniq = sorted({clips[i] for i in idx})
    by = {c: np.array([i for i in idx if clips[i] == c], dtype=int) for c in uniq}
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_boot):
        picks = torch.randint(0, len(uniq), (len(uniq),), generator=g).tolist()
        sub = np.concatenate([by[uniq[p]] for p in picks])
        v = fn(sub)
        if v == v:
            vals.append(v)
    if not vals:
        return [float("nan"), float("nan")]
    a = np.array(vals)
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


# ----------------------------------------------------------------- layers
def build_layers(meta: list[dict]) -> dict[str, list[int]]:
    n = len(meta)
    all_idx = list(range(n))
    nobif = [i for i in all_idx
             if (meta[i]["c_plus"] < SIGN_CUTOFF) == (meta[i]["c_minus"] < SIGN_CUTOFF)
             and meta[i]["term_plus"] == meta[i]["term_minus"]]
    by: dict = {}
    for i in nobif:
        m = meta[i]
        by.setdefault((m["clip"], m["frame"], m["family"]), {})[m["eps"]] = i
    gated: list[int] = []
    for key, m in by.items():
        if len(m) < 2:
            continue
        e1, e2 = sorted(m)
        d1 = meta[m[e1]]["Delta_G"] / (2 * e1)
        d2 = meta[m[e2]]["Delta_G"] / (2 * e2)
        if abs(d1 - d2) <= max(5e-4, 0.20 * abs(d1)):
            gated.extend([m[e1], m[e2]])
    return {"all": all_idx, "nobif": nobif, "gated": sorted(gated)}


# ----------------------------------------------------------------- per critic
def critic_block(rec: dict, meta: list[dict], layers: dict[str, list[int]],
                 with_ci: bool, horizon: int) -> dict:
    fd = (rec["v_plus"] - rec["v_minus"]).numpy().astype(np.float64)
    auto = rec["auto"].numpy().astype(np.float64)
    dG = np.array([m["Delta_G"] for m in meta])
    dY = np.array([m["Delta_Y"] for m in meta])
    clips = [m["clip"] for m in meta]
    fams = [m["family"] for m in meta]
    starts = [f"{m['clip']}@{m['frame']}" for m in meta]

    def dir_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
        out = {"n": int(len(ref)), "sign": _sign(pred, ref), "spearman": _spearman(pred, ref),
               "rmse": float(np.sqrt(np.mean((pred - ref) ** 2))) if len(ref) else float("nan"),
               "mag_ratio": _ratio(pred, ref)}
        ok = int(np.sum(np.sign(pred) == np.sign(ref)))
        out["sign_p_vs_chance"] = _binom_two_sided(ok, int(len(ref)))
        return out

    def loco(pred: np.ndarray, ref: np.ndarray, idx: list[int]) -> dict:
        vals = []
        for c in sorted({clips[i] for i in idx}):
            sub = np.array([i for i in idx if clips[i] != c], dtype=int)
            if len(sub) >= 2:
                vals.append(_spearman(pred[sub], ref[sub]))
        return ({"min": float(min(vals)), "max": float(max(vals)), "n_clips": len(vals)}
                if vals else {"min": float("nan"), "max": float("nan"), "n_clips": 0})

    block = {"value_rmse_vsY": rec["value_rmse_vsY"], "value_rmse_vsG": rec["value_rmse_vsG"],
             "pearson_vsY": rec["pearson_vsY"], "bias_vsY": rec["bias_vsY"],
             "fd_std": rec["fd_std"], "auto_std": rec["auto_std"],
             "b1_vs_bn_consistency_maxabs": rec["b1_vs_bn_consistency_maxabs"], "layers": {}}
    for lname, idx in layers.items():
        if not idx:
            block["layers"][lname] = {"n": 0}
            continue
        ia = np.array(idx, dtype=int)
        ir = np.array([i for i in idx if fams[i] in REACHABLE], dtype=int)
        L = {"n": int(len(ia)), "n_reach": int(len(ir)),
             "n_starts": len({starts[i] for i in idx}),
             "n_clips": len({clips[i] for i in idx})}
        for rname, ref in (("dG", dG), ("dY", dY)):
            L[f"fd_vs_{rname}"] = {"all": dir_metrics(fd[ia], ref[ia]),
                                   "reach": dir_metrics(fd[ir], ref[ir]) if len(ir) else {}}
        L["auto_vs_dG"] = {"all": dir_metrics(auto[ia], dG[ia]),
                           "reach": dir_metrics(auto[ir], dG[ir]) if len(ir) else {}}
        L["fd_vs_auto"] = {"sign": _sign(fd[ia], auto[ia]), "spearman": _spearman(fd[ia], auto[ia]),
                           "mag_ratio_auto_over_fd": _ratio(auto[ia], fd[ia])}
        # within-start ranking (a start has several families x eps)
        ws = []
        for s in sorted(set(starts[i] for i in idx)):
            sub = np.array([i for i in idx if starts[i] == s], dtype=int)
            if len(sub) >= 3:
                ws.append(_spearman(fd[sub], dG[sub]))
        L["within_start_spearman_fd_dG"] = {"mean": float(np.mean(ws)) if ws else float("nan"),
                                            "n_starts": len(ws)}
        L["loco_reach_spearman_fd_dG"] = loco(fd, dG, list(ir))
        L["per_family_fd_vs_dG"] = {fam: dir_metrics(fd[[i for i in idx if fams[i] == fam]],
                                                    dG[[i for i in idx if fams[i] == fam]])
                                    for fam in sorted({fams[i] for i in idx})}
        for h in sorted(meta[0]["Delta_G_by_h"]):
            if h == horizon:
                continue
            refh = np.array([m["Delta_G_by_h"][h] for m in meta])
            L[f"fd_vs_dG_h{h}"] = {"all": {"sign": _sign(fd[ia], refh[ia]),
                                           "spearman": _spearman(fd[ia], refh[ia])},
                                   "reach": {"sign": _sign(fd[ir], refh[ir]),
                                             "spearman": _spearman(fd[ir], refh[ir])}}
        if with_ci:
            ci = {}
            for sub_name, sub in (("all", ia), ("reach", ir)):
                if len(sub) < 2:
                    continue
                ci[sub_name] = {
                    "sign_ci95": clip_boot(lambda s: _sign(fd[s], dG[s]), list(sub), clips),
                    "spearman_ci95": clip_boot(lambda s: _spearman(fd[s], dG[s]), list(sub), clips),
                }
            L["clip_bootstrap_ci95_fd_vs_dG"] = ci
        block["layers"][lname] = L
    return block


def paired_compare(recs: dict[str, dict], meta: list[dict], idx: list[int],
                   names: list[str]) -> dict:
    fd = {k: (r["v_plus"] - r["v_minus"]).numpy().astype(np.float64) for k, r in recs.items()}
    dG = np.array([m["Delta_G"] for m in meta])
    clips = [m["clip"] for m in meta]
    fams = [m["family"] for m in meta]
    out = {}
    for scope, sub in (("all", idx), ("reach", [i for i in idx if fams[i] in REACHABLE])):
        if len(sub) < 2:
            continue
        sub = list(sub)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                A, B = names[a], names[b]
                ra = np.abs(fd[A][sub] - dG[sub])
                rb = np.abs(fd[B][sub] - dG[sub])
                diff = rb - ra                      # < 0 => B closer to the reference
                # clip_boot resamples GLOBAL indices: expose the subset difference on a
                # full-length array so the bootstrap statistic never mixes index spaces
                diffg = np.full(len(meta), np.nan)
                diffg[np.array(sub, dtype=int)] = diff
                ci = clip_boot(lambda s: float(np.mean(diffg[s])), list(sub), clips)
                nb = int(np.sum(diff < 0)); na = int(np.sum(diff > 0))
                out[f"{scope}:{A}|{B}"] = {
                    "n": len(sub), "mean_abs_resid_A": float(ra.mean()),
                    "mean_abs_resid_B": float(rb.mean()), "mean_diff_B_minus_A": float(diff.mean()),
                    "clip_boot_ci95": ci, "n_B_closer": nb, "n_A_closer": na,
                    "exact_paired_sign_p": _binom_two_sided(nb, nb + na),
                    "spearman_A": _spearman(fd[A][sub], dG[sub]),
                    "spearman_B": _spearman(fd[B][sub], dG[sub]),
                    "sign_A": _sign(fd[A][sub], dG[sub]), "sign_B": _sign(fd[B][sub], dG[sub])}
    return out


def gate_c2_vs_c1(recs: dict[str, dict], blocks: dict[str, dict], meta: list[dict],
                  c2: str, c1: str, layers: dict[str, list[int]], init_name: str = "C-init") -> dict:
    fams = [m["family"] for m in meta]
    clips = [m["clip"] for m in meta]
    fd2 = (recs[c2]["v_plus"] - recs[c2]["v_minus"]).numpy().astype(np.float64)
    fd1 = (recs[c1]["v_plus"] - recs[c1]["v_minus"]).numpy().astype(np.float64)
    fdi = (recs[init_name]["v_plus"] - recs[init_name]["v_minus"]).numpy().astype(np.float64)
    dG = np.array([m["Delta_G"] for m in meta])
    out = {"c2": c2, "c1": c1, "reference": "Delta G_96 (no learned bootstrap)"}
    for lname in ("nobif", "gated"):
        idx = layers[lname]
        ir = [i for i in idx if fams[i] in REACHABLE]
        if len(ir) < 2:
            out[lname] = {"n_reach": len(ir), "note": "too few reachable pairs"}
            continue
        b1, b2 = blocks[c1]["layers"][lname], blocks[c2]["layers"][lname]
        r1, r2 = b1["fd_vs_dG"]["reach"], b2["fd_vs_dG"]["reach"]
        rm1, rm2 = blocks[c1]["value_rmse_vsY"], blocks[c2]["value_rmse_vsY"]
        # leave-one-clip-out range of the reachable sign gain
        gains = []
        for c in sorted({clips[i] for i in ir}):
            sub = [i for i in ir if clips[i] != c]
            if sub:
                gains.append(_sign(fd2[sub], dG[sub]) - _sign(fd1[sub], dG[sub]))
        crit = {
            "1_value_rmse_regression_pct": (rm2 / max(rm1, 1e-12) - 1.0) * 100.0,
            "2_reach_sign_gain_pp": 100.0 * (r2["sign"] - r1["sign"]),
            "3_reach_rmse_improvement_pct": (1.0 - r2["rmse"] / max(r1["rmse"], 1e-12)) * 100.0,
            "4_reach_mag_ratio_c1": r1["mag_ratio"], "4_reach_mag_ratio_c2": r2["mag_ratio"],
            "5_total_gradient_fidelity": "NOT EVALUATED (D4 not implemented)",
            "6_loco_sign_gain_min_pp": 100.0 * min(gains) if gains else float("nan"),
            "6_loco_sign_gain_max_pp": 100.0 * max(gains) if gains else float("nan"),
        }
        crit["pass"] = {
            "1": crit["1_value_rmse_regression_pct"] <= 10.0,
            "2": crit["2_reach_sign_gain_pp"] >= 10.0,
            "3": crit["3_reach_rmse_improvement_pct"] >= 10.0,
            "4": (r2["mag_ratio"] <= 3.0 and r2["mag_ratio"] <= 1.5 * r1["mag_ratio"]),
            "5": False,
            "6": (crit["6_loco_sign_gain_min_pp"] > 0.0) if gains else False,
        }
        crit["beats_init_reach"] = {
            "sign_gain_vs_init_pp_c2": 100.0 * (r2["sign"] - _sign(fdi[ir], dG[ir])),
            "sign_gain_vs_init_pp_c1": 100.0 * (r1["sign"] - _sign(fdi[ir], dG[ir])),
            "spearman": {"init": _spearman(fdi[ir], dG[ir]), "c1": r1["spearman"], "c2": r2["spearman"]},
            "rmse": {"init": float(np.sqrt(np.mean((fdi[ir] - dG[ir]) ** 2))), "c1": r1["rmse"], "c2": r2["rmse"]},
        }
        out[lname] = {"n_reach": len(ir), "n_clips": len({clips[i] for i in ir}), "criteria": crit}
    return out


def fmt(m: dict) -> str:
    if not m or not m.get("n"):
        return "   n/a    "
    return (f"sign {m['sign']:.3f} sp {m['spearman']:+.3f} rmse {m['rmse']:.4f} "
            f"ratio {m['mag_ratio']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--primary_c2", default="C2gm03@500")
    ap.add_argument("--primary_c1", default="C1@500")
    ap.add_argument("--primary_c0", default="C0@500")
    ap.add_argument("--no_ci", action="store_true", help="skip the 10000-draw bootstraps")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    t0 = time.time()

    raw = torch.load(args.preds, map_location="cpu", weights_only=False)
    if raw.get("schema") != "sacv_preds_v2":
        raise SystemExit(f"schema {raw.get('schema')!r} is not sacv_preds_v2")
    meta, recs = raw["meta"], raw["critics"]
    names = list(recs)
    layers = build_layers(meta)

    blocks = {k: critic_block(recs[k], meta, layers, with_ci=not args.no_ci,
                              horizon=int(raw["horizon"])) for k in names}
    main_names = [n for n in ("C-init", args.primary_c0, args.primary_c1, args.primary_c2) if n in recs]
    paired = {lname: paired_compare(recs, meta, layers[lname], main_names)
              for lname in ("nobif", "gated")}
    gate = gate_c2_vs_c1(recs, blocks, meta, args.primary_c2, args.primary_c1, layers) \
        if args.primary_c2 in recs and args.primary_c1 in recs else {"note": "primary critics absent"}

    report = {"schema": "sacv_stats_v2", "split": raw["split"], "horizon": raw["horizon"],
              "critic_state": raw["critic_state"], "n_pairs": len(meta),
              "layer_sizes": {k: len(v) for k, v in layers.items()},
              "n_clips": len({m["clip"] for m in meta}),
              "layer_definitions": {
                  "nobif": "no single-sided termination (soft-survival class and hard-term flag equal on both branches)",
                  "gated": "nobif AND |D(e1)-D(e2)| <= max(5e-4, 0.2|D(e1)|), D = DeltaG/(2 eps)"},
              "reachable_families": list(REACHABLE),
              "primary_comparison": {"c2": args.primary_c2, "c1": args.primary_c1, "c0": args.primary_c0},
              "critics": blocks, "paired_primary": paired, "gate_c2_vs_c1": gate,
              "bootstrap": {"unit": "clip", "n_boot": N_BOOT, "seed": SEED},
              "elapsed_s": round(time.time() - t0, 1)}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))

    print(f"== {raw['split']} | n_pairs={len(meta)} | layers "
          + " ".join(f"{k}={len(v)}" for k, v in layers.items()) + " ==")
    for lname in layers:
        print(f"-- layer {lname} --")
        for k in names:
            L = blocks[k]["layers"][lname]
            if not L.get("n"):
                continue
            print(f"  {k:12s} FD|dG all   {fmt(L['fd_vs_dG']['all'])}")
            print(f"  {'':12s} FD|dG reach {fmt(L['fd_vs_dG']['reach'])}"
                  + (f"  signCI {L['clip_bootstrap_ci95_fd_vs_dG']['reach']['sign_ci95']}"
                     if "clip_bootstrap_ci95_fd_vs_dG" in L and "reach" in L["clip_bootstrap_ci95_fd_vs_dG"] else ""))
            print(f"  {'':12s} AUTO|dG reach {fmt(L['auto_vs_dG']['reach'])}"
                  f"  FD~auto sign {L['fd_vs_auto']['sign']:.2f} sp {L['fd_vs_auto']['spearman']:+.2f}")
        print(f"  {'':12s} value RMSE vsY / Pearson: "
              + "  ".join(f"{k} {blocks[k]['value_rmse_vsY']:.3f}/{blocks[k]['pearson_vsY']:.3f}" for k in names))
    print("-- gate (pre-registered, reachable subset vs Delta G) --")
    for lname in ("nobif", "gated"):
        g = gate.get(lname, {})
        if "criteria" in g:
            c = g["criteria"]
            print(f"  [{lname} n_reach={g['n_reach']}] valueRMSE {c['1_value_rmse_regression_pct']:+.1f}% "
                  f"sign {c['2_reach_sign_gain_pp']:+.1f}pp rmse {c['3_reach_rmse_improvement_pct']:+.1f}% "
                  f"ratio c1/c2 {c['4_reach_mag_ratio_c1']:.2f}/{c['4_reach_mag_ratio_c2']:.2f} "
                  f"LOCO gain [{c['6_loco_sign_gain_min_pp']:+.0f},{c['6_loco_sign_gain_max_pp']:+.0f}] "
                  f"pass={c['pass']}")
    print("-- paired (B closer than A on |FD - dG|; reach subset) --")
    for lname, d in paired.items():
        for key, v in d.items():
            if not key.startswith("reach:"):
                continue
            A, B = key.split(":", 1)[1].split("|")
            print(f"  [{lname}] {A:11s} -> {B:11s} resid {v['mean_abs_resid_A']:.4f}->{v['mean_abs_resid_B']:.4f} "
                  f"CI {v['clip_boot_ci95']} B<A {v['n_B_closer']} A<B {v['n_A_closer']} p={v['exact_paired_sign_p']:.3f}")
    if args.out:
        print(f"written {args.out} ({report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
