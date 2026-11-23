"""SACV evaluation metrics (Phase D).  CPU-only, pure torch/numpy.

Layered discipline (protocol §3/D5): ALL-sample metrics never called strict
local derivatives; strict-gated metrics computed on the gate-passing subset;
same-sample paired comparisons; clip-clustered bootstrap + LOCO reuse
cvf_core (single implementation shared with CVF-0).
"""
from __future__ import annotations

import math

import numpy as np
import torch

import sys
from pathlib import Path

_CVF = Path(__file__).resolve().parents[2]
if str(_CVF) not in sys.path:
    sys.path.insert(0, str(_CVF))

from cvf_core import clip_clustered_bootstrap, loco_summary  # noqa: E402,F401  (shared impl)


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().double().cpu().numpy()


def sign_agreement(d_pred: torch.Tensor, d_ref: torch.Tensor,
                   dead_zone: float = 0.0) -> float:
    """Fraction of samples where sign(pred) == sign(ref), excluding dead-zone refs."""
    p, r = _np(d_pred), _np(d_ref)
    m = np.abs(r) > dead_zone
    if m.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(p[m]) == np.sign(r[m])))


def directional_rmse(d_pred: torch.Tensor, d_ref: torch.Tensor) -> float:
    p, r = _np(d_pred), _np(d_ref)
    return float(np.sqrt(np.mean((p - r) ** 2)))


def magnitude_ratio(d_pred: torch.Tensor, d_ref: torch.Tensor,
                    dead_zone: float = 0.0) -> float:
    """Median |pred| / |ref| over non-dead-zone samples."""
    p, r = _np(d_pred), _np(d_ref)
    m = np.abs(r) > dead_zone
    if m.sum() == 0:
        return float("nan")
    return float(np.median(np.abs(p[m]) / np.abs(r[m])))


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = _np(a), _np(b)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def within_start_spearman(pred: torch.Tensor, ref: torch.Tensor,
                          start_ids: list) -> float:
    """Mean over starts of Spearman(pred, ref) across that start's directions."""
    vals = []
    for s in sorted(set(start_ids), key=str):
        idx = [i for i, x in enumerate(start_ids) if x == s]
        if len(idx) >= 2:
            vals.append(spearman(pred[idx], ref[idx]))
    return float(np.mean(vals)) if vals else float("nan")


def exact_paired_sign_test(diffs: torch.Tensor) -> dict:
    """Two-sided exact binomial sign test on paired differences (zeros dropped)."""
    d = _np(diffs)
    d = d[d != 0]
    n = len(d)
    n_plus = int((d > 0).sum())
    if n == 0:
        return {"n": 0, "n_plus": 0, "p": float("nan")}
    from math import comb
    k = min(n_plus, n - n_plus)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return {"n": n, "n_plus": n_plus, "p": min(1.0, float(p))}


def paired_bootstrap_ci(diffs: torch.Tensor, cluster_ids: list,
                        n_boot: int = 10000, seed: int = 2700) -> dict:
    """Clip-clustered bootstrap CI95 of the mean paired difference."""
    return clip_clustered_bootstrap(diffs, cluster_ids, n_boot=n_boot, seed=seed)


def calibration(y_pred: torch.Tensor, y_ref: torch.Tensor) -> dict:
    """MAE / RMSE / bias / Pearson / slope(ref->pred OLS) for scalar value quality."""
    p, r = _np(y_pred), _np(y_ref)
    mae = float(np.mean(np.abs(p - r)))
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    bias = float(np.mean(p - r))
    pc = float(np.corrcoef(p, r)[0, 1]) if len(p) > 1 and np.std(p) > 0 and np.std(r) > 0 \
        else float("nan")
    slope = float(np.dot(p - p.mean(), r - r.mean()) / max(np.dot(r - r.mean(), r - r.mean()), 1e-12)) \
        if len(p) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "bias": bias, "pearson": pc, "slope": slope,
            "spearman": spearman(y_pred, y_ref), "n": len(p)}


def direction_metrics(pred: torch.Tensor, ref: torch.Tensor,
                      dead_zone: float = 0.0) -> dict:
    return {"sign": sign_agreement(pred, ref, dead_zone),
            "spearman": spearman(pred, ref),
            "rmse": directional_rmse(pred, ref),
            "mag_ratio_median": magnitude_ratio(pred, ref, dead_zone),
            "n": int(pred.shape[0])}


def loco_range(pred: torch.Tensor, ref: torch.Tensor, clips: list[str],
               stat=directional_rmse) -> dict:
    """Leave-one-clip-out range of a direction metric (stability vs single clip)."""
    vals = []
    uniq = sorted(set(clips))
    for drop in uniq:
        idx = [i for i, c in enumerate(clips) if c != drop]
        if idx:
            vals.append(stat(pred[idx], ref[idx]))
    if not vals:
        return {"min": float("nan"), "max": float("nan")}
    return {"min": min(vals), "max": max(vals), "n_clips": len(uniq)}


def gamma_survival_weighted_prefix_fd(g_plus: float, g_minus: float,
                                      c_H_plus: float, c_H_minus: float,
                                      gamma: float, H: int, eps: float) -> float:
    """Oracle-2 convention: D = (gamma^H c_H+ G_cont+ - gamma^H c_H- G_cont-) / (2 eps).

    Kept explicit so D4 can never 'forget' the prefix survival weighting."""
    w_plus = (gamma ** H) * c_H_plus
    w_minus = (gamma ** H) * c_H_minus
    return (w_plus * g_plus - w_minus * g_minus) / (2.0 * eps)


def math_overflow_guard(x: float) -> bool:
    return math.isfinite(x)
