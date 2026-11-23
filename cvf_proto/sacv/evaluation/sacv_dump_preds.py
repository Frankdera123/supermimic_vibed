"""CVF-1a Phase D (v2): per-pair critic predictions on the prefix-end state x_H (CPU).

v2 (2026-11-23) corrects the v1 state/label misalignment (v1 evaluated the critic at
x_{H+T}; INVALID, see docs/cvf/sacv/CVF1B_ERRATUM.md).  Consumes sacv_pairs_v2 and
writes sacv_preds_v2 aligned 1:1 with the pairs of one horizon:

  critics[name] = {
    v_plus, v_minus        V(x_H+), V(x_H-)   -- finite difference FD = v_plus - v_minus
    auto                   chord-averaged autograd directional derivative, same units as FD:
                           |chord|/2 * ( <grad V(x_H-), u_-> + <grad V(x_H+), u_+> )
                           with u_- the unit tangent x_H- -> x_H+ and u_+ the unit tangent
                           x_H+ -> x_H- (perturb_state coordinates, cvf_core).  The policy
                           consumes gradients, so an FD gain that does not show up here
                           does not reach the policy update.
    value_rmse_vsY, value_rmse_vsG, pearson_vsY, bias_vsY,
    b1_vs_bn_consistency_maxabs   per-sample rows vs one batched forward (must be ~0)
  }
meta[i] carries the labels of pair i at every exported horizon so sacv_stats.py can
report cross-horizon sensitivity without re-reading raw batches.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SACV = _HERE.parent
_CVF = _SACV.parent
_REPO = _CVF.parent
for p in (str(_SACV), str(_CVF), str(_REPO / "pada_proto"), str(_REPO / "pada_proto" / "tools"),
          str(_REPO / "crgf_proto")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402


def chord_delta(Sm, Sp) -> torch.Tensor:
    """Tangent vector (..., 70) carrying state Sm onto Sp in perturb_state(Sm, .) coords."""
    from supermimic.rotations import so3_log, yaw_mat

    pm, Rm, vm, wm, qm, qdm = Sm
    pp, Rp, vp, wp, qp, qdp = Sp
    Rt = Rm.transpose(-1, -2)
    dp = (yaw_mat(Rm).transpose(-1, -2) @ (pp - pm).unsqueeze(-1)).squeeze(-1)
    dth = so3_log(Rt @ Rp)
    dv = (Rt @ (vp - vm).unsqueeze(-1)).squeeze(-1)
    dw = (Rt @ (wp - wm).unsqueeze(-1)).squeeze(-1)
    return torch.cat((dp, dth, dv, dw, qp - qm, qdp - qdm), -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test", "train"), default="val")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--critics", type=Path, nargs="+", default=[],
                    help="sacv_critic_ckpt_v2 files; C-init is always included")
    ap.add_argument("--datasets", type=Path, default=_REPO / "results/cvf/sacv/datasets/v2")
    ap.add_argument("--raw_root", type=Path, default=_REPO / "results/cvf/sacv/datasets/raw")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()

    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()
    from cvf_core import perturb_state
    from supermimic.mdp import normalize_xi, unflatten_state
    from supermimic.networks import Critic

    pair2100 = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
    proto = bridge.setup_world(device="cpu", pair=pair2100)["proto"]
    v_ref_sd = torch.load(bridge.pairs_root() / "online_B0/diag_pairs/pair_it_002100.pt",
                          map_location="cpu", weights_only=False)["critic_post"]

    data = torch.load(args.datasets / f"{args.split}_pairs.pt", map_location="cpu",
                      weights_only=False)
    if data.get("schema") != "sacv_pairs_v2":
        raise SystemExit(f"schema {data.get('schema')!r} is not sacv_pairs_v2 (v1 is INVALID)")
    all_pairs = data["pairs"]
    by_key: dict = {}
    for q in all_pairs:
        by_key.setdefault((q["clip"], q["frame"], q["family"], q["eps"]), {})[q["horizon"]] = q
    pairs = [q for q in all_pairs if q["horizon"] == args.horizon]
    if not pairs:
        print(json.dumps({"error": "no pairs at horizon", "horizon": args.horizon}))
        return 1
    horizons = sorted(data["horizons"])

    raw_cache: dict[str, dict] = {}

    def snap_of(pref: dict) -> dict:
        if pref.get("snap_group") != "snaps_prefix" or pref.get("snap_key") != "prefix_32":
            raise SystemExit(f"pair state is not x_H: {pref}")
        if pref["batch"] not in raw_cache:
            raw_cache[pref["batch"]] = torch.load(
                next(args.raw_root.rglob(pref["batch"])), map_location="cpu", weights_only=False)
        b = raw_cache[pref["batch"]]
        br = b["branches"][pref["branch"]] if pref["branch"] != "base0" else b["base_res"][0]
        return br["snaps_prefix"][pref["snap_key"]]

    xi_n1 = normalize_xi(torch.tensor([[1.0] + [0.0] * 32]))
    S_plus, S_minus, f_plus, f_minus, X_rows = [], [], [], [], []
    for p in pairs:
        for side in ("plus", "minus"):
            pref = p[f"state_{side}"]
            snap = snap_of(pref)
            s = pref["slot"]
            S = unflatten_state(snap["S_flat"][s:s + 1])
            f = snap["f"][s:s + 1]
            (S_plus if side == "plus" else S_minus).append(S)
            (f_plus if side == "plus" else f_minus).append(f)
            X_rows.append(proto.critic_input(S, f, xi_n1, proto.wm_goal(f, S)))
    N = len(pairs)

    critics = {"C-init": v_ref_sd}
    for cp in args.critics:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        if ck.get("schema") != "sacv_critic_ckpt_v2":
            raise SystemExit(f"{cp.name}: schema {ck.get('schema')!r} is not a v2 checkpoint")
        critics[f"{ck.get('name') or ck['group']}@{ck['steps']}"] = ck["critic"]

    # ---- batched states (index order: pairs) + chord-averaged autograd machinery
    xi_nb = normalize_xi(torch.zeros(N, 33) + torch.tensor([[1.0] + [0.0] * 32]))
    stacks = [tuple(torch.cat([x[i] for x in S_list], 0).detach() for i in range(6))
              for S_list in (S_minus, S_plus)]                      # [0]=minus, [1]=plus
    f_b = [torch.cat(f_list, 0) for f_list in (f_minus, f_plus)]
    d_m = chord_delta(stacks[0], stacks[1])       # tangent x_H- -> x_H+, at the minus states
    d_p = chord_delta(stacks[1], stacks[0])       # tangent x_H+ -> x_H-, at the plus states
    chord = 0.5 * (d_m.norm(dim=-1) + d_p.norm(dim=-1))             # (N,)
    u_m = (d_m / d_m.norm(dim=-1, keepdim=True).clamp_min(1e-12)).detach()
    u_p = (d_p / d_p.norm(dim=-1, keepdim=True).clamp_min(1e-12)).detach()

    def dir_deriv(critic, k, u) -> torch.Tensor:
        """<grad_delta V(perturb_state(S, delta))|_0 , u> for every state of stack k."""
        d = u.clone().requires_grad_(True)
        S_p = perturb_state(stacks[k], d)
        v = critic(proto.critic_input(S_p, f_b[k], xi_nb, proto.wm_goal(f_b[k], S_p)))
        return (torch.autograd.grad(v.sum(), d)[0] * u).sum(-1).detach()

    meta = []
    for p in pairs:
        q = by_key[(p["clip"], p["frame"], p["family"], p["eps"])]
        meta.append({
            "clip": p["clip"], "frame": p["frame"], "family": p["family"], "eps": p["eps"],
            "Delta_Y": p["Delta_Y"], "Delta_G": p["Delta_G"],
            "Y_plus": p["Y_plus"], "Y_minus": p["Y_minus"],
            "G_plus": p["G_plus"], "G_minus": p["G_minus"],
            "c_plus": p["c_plus"], "c_minus": p["c_minus"],
            "term_plus": p["term_plus"], "term_minus": p["term_minus"],
            "Delta_G_by_h": {h: q[h]["Delta_G"] for h in horizons if h in q},
            "Delta_Y_by_h": {h: q[h]["Delta_Y"] for h in horizons if h in q},
        })
    t64 = lambda xs: torch.tensor(xs, dtype=torch.float64)  # noqa: E731
    y_p, y_m = t64([m["Y_plus"] for m in meta]), t64([m["Y_minus"] for m in meta])
    g_p, g_m = t64([m["G_plus"] for m in meta]), t64([m["G_minus"] for m in meta])

    out = {"schema": "sacv_preds_v2", "split": args.split, "horizon": args.horizon,
           "critic_state": "x_H (snaps_prefix/prefix_32)", "n_pairs": N,
           "chord_len_median": float(chord.median()), "meta": meta, "critics": {}}
    for name, sd in critics.items():
        c = Critic(sd["norm.mean"].shape[0])
        c.load_state_dict(sd, strict=True)
        c.eval()
        with torch.no_grad():
            v_rows = c(torch.cat(X_rows, 0))                     # per-sample rows, +,- order
            v_minus_rows, v_plus_rows = v_rows[1::2].double(), v_rows[0::2].double()
            v_batch = c(torch.cat([proto.critic_input(stacks[k], f_b[k], xi_nb,
                                                      proto.wm_goal(f_b[k], stacks[k]))
                                   for k in (0, 1)], 0))          # [minus..., plus...]
        b1_bn = float((torch.cat([v_minus_rows, v_plus_rows], 0) - v_batch.double()).abs().max())
        auto = (chord.double() * 0.5 * (dir_deriv(c, 0, u_m) - dir_deriv(c, 1, u_p))).detach()
        fd = v_plus_rows - v_minus_rows
        vv, yy = torch.cat([v_plus_rows, v_minus_rows]), torch.cat([y_p, y_m])
        out["critics"][name] = {
            "v_plus": v_plus_rows, "v_minus": v_minus_rows, "auto": auto,
            "value_rmse_vsY": float(torch.sqrt(0.5 * ((v_plus_rows - y_p) ** 2
                                                      + (v_minus_rows - y_m) ** 2).mean())),
            "value_rmse_vsG": float(torch.sqrt(0.5 * ((v_plus_rows - g_p) ** 2
                                                      + (v_minus_rows - g_m) ** 2).mean())),
            "pearson_vsY": float(torch.corrcoef(torch.stack([vv, yy]))[0, 1]),
            "bias_vsY": float((vv - yy).mean()),
            "b1_vs_bn_consistency_maxabs": b1_bn,
            "fd_std": float(fd.std()), "auto_std": float(auto.std()),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(json.dumps({"out": str(args.out), "n_pairs": N, "critics": list(out["critics"]),
                      "chord_len_median": out["chord_len_median"],
                      "b1_bn_maxabs": max(r["b1_vs_bn_consistency_maxabs"]
                                          for r in out["critics"].values()),
                      "elapsed_s": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
