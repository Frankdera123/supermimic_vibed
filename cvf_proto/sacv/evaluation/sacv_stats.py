"""CVF-1a Phase D: paired statistical analysis of C-init vs C0/C1/C2 (CPU).

Reads the val (and test) pair tables with per-critic Delta_V predictions produced
by sacv_eval.py's evaluate_checkpoint (or recomputed here via --recompute), and
produces the pre-registered comparison bundle:
  * C2 vs C1 on the D6 criteria (value RMSE delta, reachable sign delta,
    reachable RMSE delta, magnitude inflation, LOCO stability);
  * same-sample paired differences with clip-clustered bootstrap CI + exact
    paired sign test (per family and pooled);
  * horizon freeze diagnostics (contract §4.4): cross-horizon Spearman of G and
    tail-share |E[gamma^T c_T V_ref]| / std(G_T) from the scalars tables.

Usage:
  python cvf_proto/sacv/evaluation/sacv_stats.py --split val --metrics metrics/eval_val_preds.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # evaluation/
_SACV = _HERE.parent                             # sacv/
_CVF = _SACV.parent                              # cvf_proto/
_REPO = _CVF.parent                              # repo root
for p in (str(_SACV), str(_REPO / "pada_proto"), str(_REPO / "pada_proto" / "tools"),
          str(_REPO / "crgf_proto"), str(_REPO / "cvf_proto")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

sys.path.insert(0, str(_SACV))
from sacv.evaluation import metrics as M  # noqa: E402
from sacv.losses import targets as vt  # noqa: E402

REACHABLE = ("delta_2100", "delta_2300")   # policy-update action deltas = reachable family
GAMMA = 0.99


def criteria_c2_vs_c1(val_pairs: list[dict], preds: dict[str, dict]) -> dict:
    """Pre-registered D6 gate (protocol §3/D6) computed on Validation pairs.

    preds: critic -> {"d_pred": tensor Delta V per pair, "err": tensor (V-Y) per branch? ...}
    Simplified contract: each critic provides d_pred aligned with val_pairs and
    value RMSE over plus/minus branches."""
    ref = torch.tensor([p["Delta_Y"] for p in val_pairs])
    res = {}
    base = {c: M.direction_metrics(preds[c]["d_pred"], ref) for c in preds}
    for c in preds:
        d_pred = preds[c]["d_pred"]
        clip_of = [p["clip"] for p in val_pairs]
        fam = [p["family"] for p in val_pairs]
        reach = [i for i, f in enumerate(fam) if f in REACHABLE]
        res[c] = {
            "all": base[c],
            "reachable": M.direction_metrics(d_pred[reach], ref[reach]) if reach else {},
            "within_start_spearman": M.within_start_spearman(
                d_pred, ref, [f"{p['clip']}@{p['frame']}" for p in val_pairs]),
            "loco": M.loco_range(d_pred, ref, clip_of),
            "value_rmse_branch": preds[c].get("value_rmse", float("nan")),
        }
    gate = {}
    if "C1" in res and "C2" in res:
        c1, c2 = res["C1"], res["C2"]
        gate = {
            "value_rmse_regression_pct": (c2["value_rmse_branch"] / max(c1["value_rmse_branch"], 1e-12) - 1) * 100,
            "reachable_sign_gain_pp": 100 * (c2["reachable"].get("sign", float("nan"))
                                             - c1["reachable"].get("sign", float("nan"))),
            "reachable_rmse_gain_pct": (1 - c2["reachable"].get("rmse", float("nan"))
                                        / max(c1["reachable"].get("rmse", 1e-12), 1e-12)) * 100,
            "c2_mag_ratio": c2["all"].get("mag_ratio_median"),
            "loco_span_vs_effect": (c2["loco"].get("max", 0) - c2["loco"].get("min", 0)),
        }
    return {"per_critic": res, "gate_C2_vs_C1": gate}


def paired_tests(val_pairs: list[dict], preds: dict[str, dict], critics: list[str]) -> dict:
    """For each critic pair (A, B): paired diff of |residual to Delta_Y| and sign tests."""
    ref = torch.tensor([p["Delta_Y"] for p in val_pairs])
    clips = [p["clip"] for p in val_pairs]
    out = {}
    for i, ca in enumerate(critics):
        for cb in critics[i + 1:]:
            ra = preds[ca]["d_pred"] - ref
            rb = preds[cb]["d_pred"] - ref
            diff = rb.abs() - ra.abs()   # negative => B closer to reference than A
            bt = M.paired_bootstrap_ci(diff, clips)
            st = M.exact_paired_sign_test(-diff)  # P(B better) via plus-count of -diff
            out[f"{ca}__vs__{cb}"] = {
                "mean_abs_improvement": float(-diff.mean()),
                "clip_bootstrap_ci95": bt["ci95"], "n": bt["n"], "n_clips": bt["n_clips"],
                "exact_paired_sign_p_B_better": st["p"], "n_plus": st["n_plus"],
            }
    return out


def horizon_freeze_diagnostic(datasets: Path, horizons: (int, int, int) = (32, 64, 96)) -> dict:
    """Contract §4.4 rules on Train+Val scalars (never touches Test)."""
    import numpy as np

    diag = {}
    for split in ("train", "val"):
        path = datasets / f"{split}_scalars.pt"
        if not path.exists():
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        recs = [s for s in d["scalars"] if isinstance(s, dict) and "base" in s]
        if not recs:
            continue
        row = {}
        for h in horizons:
            key = f"G_soft_{h}"
            g = torch.tensor([s["base"][key] for s in recs if key in s.get("base", {})])
            vt_ = torch.tensor([s["base"].get("V_ref_xT", float("nan")) for s in recs])
            c = torch.tensor([s["base"].get(f"c_soft_{h}", float("nan")) for s in recs])
            if len(g) > 1 and torch.isfinite(g).all():
                tail = float(((GAMMA ** h) * c * vt_).mean())
                row[h] = {"n": len(g), "std_G": float(g.std()), "tail_share": abs(tail) / max(float(g.std()), 1e-9)}
        hs = [h for h in horizons if h in row]
        for a, b in ((32, 96), (64, 96)):
            if a in row and b in row:
                ga = torch.tensor([s["base"][f"G_soft_{a}"] for s in recs])
                gb = torch.tensor([s["base"][f"G_soft_{b}"] for s in recs])
                row[f"spearman_G{a}_G{b}"] = M.spearman(ga, gb)
        diag[split] = row
    return diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--datasets", type=Path, default=_REPO / "results/cvf/sacv/datasets")
    ap.add_argument("--preds", type=Path, required=True,
                    help="sacv_dump_preds.py output (sacv_preds_v1)")
    ap.add_argument("--horizons", type=int, nargs="+", default=[32, 64, 96])
    ap.add_argument("--out", type=Path, default=_REPO / "results/cvf/sacv/metrics")
    args = ap.parse_args()
    t0 = time.time()

    raw = torch.load(args.preds, map_location="cpu", weights_only=False)
    hz = int(raw["horizon"])
    meta = raw["meta"]                                  # ordered, un-gated pair meta
    preds = {}
    for name, rec in raw["critics"].items():
        preds[name] = {"d_pred": rec["v_plus"] - rec["v_minus"],
                       "value_rmse": rec.get("value_rmse", float("nan"))}
    critics = list(preds.keys())

    # strict-gated subset (cross-epsilon gate over families with both eps)
    by: dict = {}
    for i, m in enumerate(meta):
        by.setdefault((m["clip"], m["frame"], m["family"]), {})[m["eps"]] = i
    gated_idx: list[int] = []
    for key, m in by.items():
        if len(m) < 2:
            continue
        e1, e2 = sorted(m)
        d1 = meta[m[e1]]["Delta_Y"] / (2 * e1)
        d2 = meta[m[e2]]["Delta_Y"] / (2 * e2)
        if abs(d1 - d2) <= max(5e-4, 0.20 * abs(d1)):
            gated_idx.extend([m[e1], m[e2]])
    meta_g = [meta[i] for i in gated_idx]
    preds_g = {c: {"d_pred": preds[c]["d_pred"][gated_idx],
                   "value_rmse": preds[c]["value_rmse"]} for c in critics}

    report = {
        "schema": "sacv_stats_v1", "split": args.split, "horizon": hz,
        "n_pairs_all": len(meta), "n_pairs_gated": len(meta_g), "critics": critics,
        "criteria_gated": criteria_c2_vs_c1(meta_g, preds_g),
        "criteria_all": criteria_c2_vs_c1(meta, preds),
        "paired_tests_gated": paired_tests(meta_g, preds_g, critics),
        "paired_tests_all": paired_tests(meta, preds, critics),
        "horizon_freeze_diagnostic": horizon_freeze_diagnostic(args.datasets, tuple(args.horizons)),
        "elapsed_s": round(time.time() - t0, 1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"stats_{args.split}_h{hz}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": str(out), "n_gated": len(meta_g), "n_all": len(meta),
                      "critics": critics,
                      "gate": report["criteria_gated"].get("gate_C2_vs_C1", {})}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
