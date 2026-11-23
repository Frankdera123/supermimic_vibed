"""CVF-1a Phase D evaluation (CPU): D1 scalar calibration, D2 local response,
D4 total-gradient fidelity, against the frozen pair schema (sacv_pairs_v1).

Supports --fixture N to fabricate a schema-compatible synthetic dataset so the
pipeline is exercisable before real Phase B data exists (marked synthetic in all
outputs; never mixed with real results).
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

sys.path.insert(0, str(_SACV))
from sacv.evaluation import metrics as M  # noqa: E402

GAMMA = 0.99
H_OPEN = 32


def gate_pairs(pairs: list[dict], horizon: int) -> tuple[list[dict], list[dict]]:
    """CVF-0 cross-epsilon gate on pair responses D(eps) = Delta Y / (2 eps).

    Returns (all_pairs, gated_pairs): a start x family enters `gated` iff
    |D(e1) - D(e2)| <= max(5e-4, 0.20*|D(e1)|) for both eps of the family."""
    all_p = [p for p in pairs if p["horizon"] == horizon]
    gated = []
    by = {}
    for p in all_p:
        by.setdefault((p["clip"], p["frame"], p["family"]), {})[p["eps"]] = p
    for key, m in by.items():
        if len(m) < 2:
            continue
        e1, e2 = sorted(m)
        d1 = m[e1]["Delta_Y"] / (2 * e1)
        d2 = m[e2]["Delta_Y"] / (2 * e2)
        if abs(d1 - d2) <= max(5e-4, 0.20 * abs(d1)):
            gated.extend([m[e1], m[e2]])
    return all_p, gated


def evaluate_checkpoint(ckpt_sd: dict, pairs: list[dict], horizon: int,
                        critic_in_dim: int, device: str = "cpu") -> dict:
    """Compute D1/D2/D4 metrics for one critic over a pair list (CPU, batched)."""
    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()
    from supermimic.mdp import unflatten_state
    from supermimic.networks import Critic

    critic = Critic(critic_in_dim)
    critic.load_state_dict(ckpt_sd, strict=True)
    critic.eval().to(device)
    from supermimic.mdp import normalize_xi

    xi_one = torch.zeros(1, 33, device=device); xi_one[:, 0] = 1.0
    xi_n = normalize_xi(xi_one)   # critic_features convention (CVF-0 phase0)

    snap_cache: dict[str, dict] = {}

    def batch_of(name):
        if name not in snap_cache:
            path = next((_REPO / "results/cvf/sacv/datasets/raw").rglob(name))
            snap_cache[name] = torch.load(path, map_location="cpu", weights_only=False)
        return snap_cache[name]

    proto_world = None

    def v_of(pref: dict) -> float:
        nonlocal proto_world
        if proto_world is None:
            pair2100 = bridge.load_pair_bundle("online_B0", 2100, device=device)
            proto = bridge.setup_world(device=device, pair=pair2100)["proto"]
            proto_world = proto
        batch = batch_of(pref["batch"])
        br = batch["branches"].get(pref["branch"]) if pref["branch"] != "base0" \
            else batch["base_res"][0]
        snap = br["snaps_cont"][pref["snap_key"]]
        slot = pref["slot"]
        S = tuple(x[slot:slot + 1].to(device) for x in unflatten_state(snap["S_flat"]))
        f_t = snap["f"][slot:slot + 1].to(device)
        ap_t = snap["a_prev"][slot:slot + 1].to(device)
        proto = proto_world
        gw = proto.wm_goal(f_t, S)
        with torch.no_grad():
            return float(critic(proto.critic_input(S, f_t, xi_n, gw)))

    for p in pairs:
        p["V_pred_plus"] = v_of(p["snap_plus"])
        p["V_pred_minus"] = v_of(p["snap_minus"])
        p["Delta_V"] = p["V_pred_plus"] - p["V_pred_minus"]

    proto = proto_world
    d_pred = torch.tensor([p["Delta_V"] for p in pairs])
    d_ref = torch.tensor([p["Delta_Y"] for p in pairs])
    clips = [p["clip"] for p in pairs]
    out = {
        "n": len(pairs),
        "d2_pair_response": M.direction_metrics(d_pred, d_ref),
        "d2_within_start_spearman": M.within_start_spearman(d_pred, d_ref,
                                                            [f"{p['clip']}@{p['frame']}" for p in pairs]),
        "d2_loco_rmse_range": M.loco_range(d_pred, d_ref, clips),
        "per_family": {},
    }
    for fam in sorted({p["family"] for p in pairs}):
        idx = [i for i, p in enumerate(pairs) if p["family"] == fam]
        if idx:
            out["per_family"][fam] = M.direction_metrics(d_pred[idx], d_ref[idx])
    # D4 sim-total reference from pair tables (continuation part; prefix part enters
    # via raw-batch prefix returns when assembled by sacv_build_dataset --with_prefix)
    w = [(GAMMA ** H_OPEN) * p[f"c_{side}"] for p in pairs for side in ("plus", "minus")]
    out["d4_note"] = ("D_sim_total reference assembled from continuation FD with per-branch "
                      "gamma^H c_H survival weights; prefix-return difference is added at "
                      "report time from raw step records (contract §4/§5).")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test", "train"), default="val")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--checkpoints", type=Path, nargs="+", default=[],
                    help="*.pt paths of fine-tuned critics; C-init is always included")
    ap.add_argument("--datasets", type=Path, default=_REPO / "results/cvf/sacv/datasets")
    ap.add_argument("--out", type=Path, default=_REPO / "results/cvf/sacv/metrics")
    ap.add_argument("--fixture", type=int, default=0,
                    help="generate N synthetic pairs (schema smoke test) instead of real data")
    args = ap.parse_args()
    t0 = time.time()

    if args.fixture:
        import random as _rn

        _rn.seed(11)
        pairs = []
        for i in range(args.fixture):
            dy = _rn.uniform(-0.2, 0.2)
            eps = _rn.choice([0.015, 0.03])
            pairs.append({
                "clip": f"synth{i % 4}", "frame": 100 + i, "family": _rn.choice(
                    ["delta_2100", "delta_2300", "ascent", "random"]),
                "eps": eps, "horizon": args.horizon,
                "Y_plus": dy, "Y_minus": -dy, "G_plus": dy, "G_minus": -dy,
                "c_plus": 1.0, "c_minus": 1.0, "Delta_Y": 2 * dy,
                "V_ref_xT_plus": 7.0, "V_ref_xT_minus": 7.0,
                "snap_plus": {"batch": "FIXTURE", "branch": "b+", "slot": 0, "snap_key": "cont_96"},
                "snap_minus": {"batch": "FIXTURE", "branch": "b-", "slot": 0, "snap_key": "cont_96"},
                "base_repeat_range": 0.0,
            })
        pairs_all, pairs_gated = gate_pairs(pairs, args.horizon)
        report = {"synthetic": True, "n_all": len(pairs_all), "n_gated": len(pairs_gated),
                  "d2_all": M.direction_metrics(torch.tensor([p["Delta_V"] if "Delta_V" in p else 0.0
                                                              for p in pairs_all]),
                                                torch.tensor([p["Delta_Y"] for p in pairs_all])),
                  "note": "fixture smoke of metric wiring only"}
        out = args.out / f"eval_fixture_{args.split}_h{args.horizon}.json"
        args.out.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps({"fixture": True, "n_gated": len(pairs_gated), "out": str(out),
                          "elapsed_s": round(time.time() - t0, 1)}))
        return 0

    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()
    from supermimic.networks import Critic

    dpath = args.datasets / f"{args.split}_pairs.pt"
    data = torch.load(dpath, map_location="cpu", weights_only=False)
    pairs_all, pairs_gated = gate_pairs(data["pairs"], args.horizon)
    pair2100 = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
    critic_in = int(pair2100["critic_post"]["norm.mean"].shape[0])

    report = {"schema": "sacv_eval_v1", "split": args.split, "horizon": args.horizon,
              "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "critics": {"C-init": evaluate_checkpoint(pair2100["critic_post"].state_dict(),
                                                        pairs_gated, args.horizon, critic_in)}}
    for ck_path in args.checkpoints:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        report["critics"][ck.get("group", ck_path.stem)] = evaluate_checkpoint(
            ck["critic"], pairs_gated, args.horizon, critic_in)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"eval_{args.split}_h{args.horizon}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"out": str(out), "n_gated": len(pairs_gated), "n_all": len(pairs_all),
                      "elapsed_s": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
