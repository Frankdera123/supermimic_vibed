"""Check whether ensemble disagreement is confounded by motion-clip difficulty."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.motion import MotionLib
from supermimic.networks import WorldModel


def model_from_state(state, device):
    key = "residual_net.0.weight"
    hidden = int(state[key].shape[0]) if key in state else 0
    return WorldModel(residual_hidden=hidden).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs=2)
    ap.add_argument("--buffer_checkpoint", default=None,
                    help="optional independent replay-buffer checkpoint; models still come from positional checkpoints")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--use_residual", action="store_true")
    args = ap.parse_args()
    cks = [torch.load(p, map_location=args.device, weights_only=False) for p in args.checkpoints]
    ref = torch.load(args.buffer_checkpoint, map_location=args.device, weights_only=False) if args.buffer_checkpoint else cks[0]
    state = ref["buffer"]; T, N = state["data"]["s"].shape[:2]
    buf = ReplayBuffer(T, N, args.device); buf.load_state_dict(state)
    motion = MotionLib.from_dir(ref.get("args", {}).get("motion_dir", "data/lafan1_g1"),
                                G1Kinematics(args.device), args.device)
    wms = []
    for ck in cks:
        wm = model_from_state(ck["wm"], args.device); wm.load_state_dict(ck["wm"]); wm.eval(); wms.append(wm)
    d_all, dmean_all, dmax_all, e_all, c_all = [], [], [], [], []
    with torch.no_grad():
        for _ in range(args.batches):
            w = buf.sample_windows(args.batch, args.window)
            s = unflatten_state(w["s"][:, 0]); preds = []; step_ds = []
            for wm in wms:
                st = s
                for j in range(args.window):
                    st = wm.step(st, w["a"][:, j], normalize_xi(w["xi"][:, j]), use_residual=args.use_residual)
                    if wm is wms[0]: st0 = st
                    else:
                        st1 = st
                        std_step = torch.sqrt(wms[0].out_norm.var + wms[0].out_norm.eps ** 2).clamp_min(1e-4)
                        step_ds.append((wms[0].state_error(st0, st1) / std_step).square().mean(-1))
                preds.append(st)
            target = unflatten_state(w["s1"][:, -1])
            std = torch.sqrt(wms[0].out_norm.var + wms[0].out_norm.eps ** 2).clamp_min(1e-4)
            d = (wms[0].state_error(preds[0], preds[1]) / std).square().mean(-1)
            e = (wms[0].state_error(preds[0], target) / std).square().mean(-1)
            valid = w["mask"][:, -1].bool()
            c = motion.clip_id[w["f"][:, -1]]
            path_d = torch.stack(step_ds, 1)
            d_all.append(d[valid]); dmean_all.append(path_d.mean(1)[valid]); dmax_all.append(path_d.max(1).values[valid])
            e_all.append(e[valid]); c_all.append(c[valid])
    d, dmean, dmax, e, c = (torch.cat(x) for x in (d_all, dmean_all, dmax_all, e_all, c_all))
    rows = []
    for cid in torch.unique(c).tolist():
        sel = c == cid
        if int(sel.sum()) < 20: continue
        rows.append({"clip_id": int(cid), "name": motion.names[int(cid)], "n": int(sel.sum()),
                     "disagreement_mean": float(d[sel].mean()), "error_mean": float(e[sel].mean()),
                     "error_le_2": float((e[sel] <= 2).float().mean())})
    dm = torch.stack([torch.tensor(x["disagreement_mean"]) for x in rows])
    em = torch.stack([torch.tensor(x["error_mean"]) for x in rows])
    centered = lambda x: x - x.mean()
    corr = float((centered(dm) * centered(em)).mean() /
                 (dm.var(unbiased=False).sqrt() * em.var(unbiased=False).sqrt()).clamp_min(1e-8)) if len(rows) > 1 else None
    d_res, e_res = [], []
    for cid in torch.unique(c):
        sel = c == cid
        d_res.append(d[sel] - d[sel].mean()); e_res.append(e[sel] - e[sel].mean())
    d_res, e_res = torch.cat(d_res), torch.cat(e_res)
    within_corr = float((d_res * e_res).mean() /
                        (d_res.square().mean().sqrt() * e_res.square().mean().sqrt()).clamp_min(1e-8))
    raw_corr = float((centered(d) * centered(e)).mean() /
                     (d.var(unbiased=False).sqrt() * e.var(unbiased=False).sqrt()).clamp_min(1e-8))
    rows.sort(key=lambda x: x["disagreement_mean"], reverse=True)
    global_sel = d <= torch.quantile(d, .5)
    clip_sel = torch.zeros_like(global_sel)
    for cid in torch.unique(c):
        sel = c == cid
        clip_sel[sel] = d[sel] <= torch.quantile(d[sel], .5)
    def risk(sel, values=e):
        return {"selection_rate": float(sel.float().mean()),
                "mean_error": float(values[sel].mean()),
                "error_le_2": float((values[sel] <= 2).float().mean())}
    unique_c = torch.unique(c)
    clip_dmeans = torch.stack([d[c == cid].mean() for cid in unique_c])
    qcuts = [torch.quantile(clip_dmeans, q) for q in (.25, .5, .75)]
    difficulty = torch.zeros_like(c, dtype=torch.long)
    for cid, md in zip(unique_c, clip_dmeans):
        difficulty[c == cid] = (md > torch.stack(qcuts)).sum()
    strata = {str(int(k)): risk(global_sel & (difficulty == k)) for k in range(4)}
    # Held-out risk-budget check: fit clip difficulty strata and median thresholds
    # on one deterministic half, then evaluate on the other half.
    perm = torch.arange(d.numel(), device=d.device)
    fit_idx, eval_idx = perm[::2], perm[1::2]
    fit_clip = torch.unique(c[fit_idx])
    fit_means = torch.stack([d[fit_idx][c[fit_idx] == cid].mean() for cid in fit_clip])
    fit_cuts = [torch.quantile(fit_means, q) for q in (.25, .5, .75)]
    fit_diff = {int(cid): int((md > torch.stack(fit_cuts)).sum()) for cid, md in zip(fit_clip, fit_means)}
    thresholds = [torch.quantile(d[fit_idx][torch.tensor([fit_diff.get(int(x), 3) == k for x in c[fit_idx]], device=d.device)], .5)
                  for k in range(4)]
    eval_diff = torch.tensor([fit_diff.get(int(x), 3) for x in c[eval_idx]], device=d.device)
    heldout_sel = torch.stack([d[eval_idx] <= thresholds[k] for k in range(4)], 1)[torch.arange(eval_idx.numel(), device=d.device), eval_diff]
    heldout_global = d[eval_idx] <= torch.quantile(d[fit_idx], .5)
    # A label-calibrated (diagnostic only) risk budget: choose each threshold as
    # the 80th percentile of disagreement among fit windows with WM error <= 2.
    fit_d, fit_e = d[fit_idx], e[fit_idx]
    safe = fit_e <= 2
    safe_thresholds = []
    for k in range(4):
        m = torch.tensor([fit_diff.get(int(x), 3) == k for x in c[fit_idx]], device=d.device)
        vals = fit_d[m & safe]
        safe_thresholds.append(torch.quantile(vals, .8) if vals.numel() else torch.tensor(float("inf"), device=d.device))
    safe_sel = torch.stack([d[eval_idx] <= safe_thresholds[k] for k in range(4)], 1)[torch.arange(eval_idx.numel(), device=d.device), eval_diff]
    # Fit a small held-out predictor for log WM error. Clip difficulty is learned
    # only from the fit half; no eval labels enter the feature construction.
    fit_clip_mean = {int(cid): float(md) for cid, md in zip(fit_clip, fit_means)}
    global_dm = float(d[fit_idx].mean())
    def features(idx):
        cm = torch.tensor([fit_clip_mean.get(int(x), global_dm) for x in c[idx]], device=d.device)
        x = torch.stack((torch.log1p(d[idx]), torch.log1p(dmean[idx]),
                         torch.log1p(dmax[idx]), cm), 1)
        return x
    x_fit, x_eval = features(fit_idx), features(eval_idx)
    x_mu, x_sd = x_fit.mean(0), x_fit.std(0).clamp_min(1e-6)
    x_fit = (x_fit - x_mu) / x_sd; x_eval = (x_eval - x_mu) / x_sd
    x_fit = torch.cat((torch.ones(x_fit.shape[0], 1, device=d.device), x_fit), 1)
    x_eval = torch.cat((torch.ones(x_eval.shape[0], 1, device=d.device), x_eval), 1)
    beta = torch.linalg.lstsq(x_fit, torch.log1p(fit_e)).solution
    pred = torch.expm1(x_eval @ beta).clamp_min(0.0)
    predictor_sel = pred <= torch.quantile(pred, .5)
    pred_corr = float(((pred - pred.mean()) * (e[eval_idx] - e[eval_idx].mean())).mean() /
                      (pred.var(unbiased=False).sqrt() * e[eval_idx].var(unbiased=False).sqrt()).clamp_min(1e-8))
    print(json.dumps({"checkpoints": args.checkpoints, "window": args.window,
                      "samples": int(d.numel()), "raw_sample_corr": raw_corr,
                      "clip_level_corr": corr, "within_clip_corr": within_corr,
                      "median_threshold_risk": {"global": risk(global_sel),
                                                 "per_clip": risk(clip_sel)},
                      "global_threshold_by_clip_difficulty_quartile": strata,
                      "heldout_half_risk": {"global": risk(heldout_global, e[eval_idx]),
                                            "difficulty_budget": risk(heldout_sel, e[eval_idx]),
                                            "safe80_budget": risk(safe_sel, e[eval_idx]),
                                            "linear_log_error_predictor": {
                                                "selection": risk(predictor_sel, e[eval_idx]),
                                                "pred_error_corr": pred_corr}},
                      "clips_by_disagreement": rows}, indent=2))


if __name__ == "__main__": main()
