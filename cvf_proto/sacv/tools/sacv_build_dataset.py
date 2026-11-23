"""CVF-1a: build train/val/test pair datasets from raw collection batches (CPU-only).

Reads results/cvf/sacv/datasets/raw/<split>/batch_*.pt, computes per-branch value
targets under the frozen data contract (docs/cvf/sacv/CVF1A_DATA_CONTRACT.md §4):

  G_T        = sum gamma^k c_k r_k over the CONTINUATION segment only (float64)
  Y_T        = G_T + gamma^T c_T V_ref(x_T)      (V_ref = frozen C-init = B0@2100_post)
  Delta Y    = Y+ - Y- per (start, family, eps)

and emits {split}_pairs.pt + {split}_scalars.pt + reliability_report.json +
dataset manifest with sha256 provenance.  Never modifies raw files.
"""
from __future__ import annotations

import argparse
import hashlib
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
from sacv.losses import targets as vt  # noqa: E402

GAMMA = 0.99
HORIZONS = (32, 64, 96)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "test"), required=True)
    ap.add_argument("--raw_root", type=Path, default=_REPO / "results/cvf/sacv/datasets/raw")
    ap.add_argument("--out_root", type=Path, default=_REPO / "results/cvf/sacv/datasets")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    args = ap.parse_args()
    t0 = time.time()
    horizons = tuple(args.horizons)

    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()
    from supermimic.mdp import normalize_xi, unflatten_state
    from supermimic.networks import Critic

    pair2100 = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
    world = bridge.setup_world(device="cpu", pair=pair2100)
    mdp, proto = world["mdp"], world["proto"]
    # C-init = critic_post lives only in the raw bundle (bridge exports bundle["critic"]==pre)
    raw2100 = torch.load(bridge.pairs_root() / "online_B0/diag_pairs/pair_it_002100.pt",
                         map_location="cpu", weights_only=False)
    v_ref_sd = raw2100["critic_post"]
    v_ref = Critic(v_ref_sd["norm.mean"].shape[0])
    v_ref.load_state_dict(v_ref_sd, strict=True)
    v_ref.eval()
    xi_one1 = torch.zeros(1, 33); xi_one1[:, 0] = 1.0
    xi_n1 = normalize_xi(xi_one1)            # WM / critic_features convention (CVF-0)

    def critic_value(S, f, a_prev):
        """Frozen V_ref at extended state (S 5-tuple, f, a_prev), nominal xi."""
        with torch.no_grad():
            gw = proto.wm_goal(f, S)
            vin = proto.critic_input(S, f, xi_n1, gw)
            return v_ref(vin).cpu()

    raw_dir = args.raw_root
    batch_files = sorted(raw_dir.glob(f"*/batch_*.pt"))  # routed by start metadata, not dir
    if not batch_files:
        print(json.dumps({"error": "no raw batches", "dir": str(raw_dir)}))
        return 1

    pairs: list[dict] = []
    scalars: list[dict] = []
    reliability = {"per_start": [], "reward_reconstruction_max": 0.0, "g_recalc_max": 0.0}
    for bf in batch_files:
        batch = torch.load(bf, map_location="cpu", weights_only=False)
        n_valid = batch["n_valid"]
        t_cont = batch["t_cont"]
        hz = tuple(h for h in horizons if h <= t_cont)
        # route by the split recorded on each start (collection may run --split all)
        # ---- per-branch value targets -----------------------------------
        def branch_records(br: dict, tag: str, slot: int) -> dict:
            pl = br["prefix_len"]
            r = br["full"]["r"][slot:slot + 1, pl:]      # continuation rewards (1, T)
            d = br["full"]["d_soft"][slot:slot + 1, pl:]
            dh = br["full"]["d_hard"][slot:slot + 1, pl:]
            G = vt.discounted_return(r, d, GAMMA, horizons=hz)
            Gh = vt.discounted_return(r, dh, GAMMA, horizons=hz)
            out = {"tag": tag, "batch": bf.name, "slot": slot}
            for h in hz:
                out[f"G_soft_{h}"] = float(G[h][0])
                out[f"G_hard_{h}"] = float(Gh[h][0])
                out[f"c_soft_{h}"] = float(vt.survival_at(d, h)[0])
                out[f"c_hard_{h}"] = float(vt.survival_at(dh, h)[0])
            # extended terminal state at the longest horizon
            key = f"cont_{hz[-1]}"
            snap = br["snaps_cont"][key]
            S_flat = snap["S_flat"][slot:slot + 1]
            f_t = snap["f"][slot:slot + 1]
            a_prev_t = snap["a_prev"][slot:slot + 1]
            S = unflatten_state(S_flat)
            out["V_ref_xT"] = float(critic_value(S, f_t, a_prev_t)[0])
            v_ref_t = torch.tensor([out["V_ref_xT"]], dtype=torch.float64)
            out["Y_soft"] = {h: float(vt.matched_target(G[h].view(1), vt.survival_at(d, h).view(1),
                                                        GAMMA, h, v_ref_t)[0])
                             for h in hz}
            out["snap_ref"] = {"batch": bf.name, "branch": tag, "slot": slot,
                               "snap_key": key}   # extended state fetched by reference
            return out

        for slot in range(n_valid):
            st = batch["starts"][slot]
            if st["split"] != args.split:
                continue
            rec = {"clip": st["clip"], "split": st["split"], "frame": st["frame"],
                   "base_repeat_range": batch["base_repeat_range"]}
            for rep, br in enumerate(batch["base_res"]):
                rec[f"base{rep}"] = branch_records(br, f"base{rep}", slot)
            reliability["reward_reconstruction_max"] = max(
                reliability["reward_reconstruction_max"],
                max(float((br["full"]["r"] - br["full"]["r_state"] - br["full"]["r_direct"]).abs().max())
                    for br in [batch["base_res"][0], batch["base_res"][1]]))
            reliability["g_recalc_max"] = max(reliability["g_recalc_max"],
                                              batch["base_res"][0]["g_recalc_vsoft_absdiff"])
            # scalars: base + every branch end state
            scalars.append({"clip": st["clip"], "frame": st["frame"],
                            "base": rec["base0"], **{k: v for k, v in rec.items()
                                                     if k not in ("base0", "base1", "clip", "frame", "split")}})
            for key, br in batch["branches"].items():
                fam, eps, sign = key.split("|")
                eps = float(eps.removeprefix("eps"))
                reliability["reward_reconstruction_max"] = max(
                    reliability["reward_reconstruction_max"],
                    float((br["full"]["r"] - br["full"]["r_state"] - br["full"]["r_direct"]).abs().max()))
                reliability["g_recalc_max"] = max(reliability["g_recalc_max"],
                                                  br["g_recalc_vsoft_absdiff"])
            # pair assembly for this start
            for key in batch["branches"]:
                fam, eps, sign = key.split("|")
                if sign != "plus":
                    continue
                mk = key.replace("|plus", "|minus")
                if mk not in batch["branches"]:
                    continue
                plus = branch_records(batch["branches"][key], key, slot)
                minus = branch_records(batch["branches"][mk], mk, slot)
                for h in hz:
                    dy = vt.pair_difference(torch.tensor([plus[f"G_soft_{h}"]]),
                                            torch.tensor([minus[f"G_soft_{h}"]]))
                    pairs.append({
                        "clip": st["clip"], "split": st["split"], "frame": st["frame"],
                        "family": fam, "eps": float(eps.removeprefix("eps")), "horizon": h,
                        "Y_plus": plus["Y_soft"][h], "Y_minus": minus["Y_soft"][h],
                        "G_plus": plus[f"G_soft_{h}"], "G_minus": minus[f"G_soft_{h}"],
                        "c_plus": plus[f"c_soft_{h}"], "c_minus": minus[f"c_soft_{h}"],
                        "Delta_Y": float(dy[0]),
                        "V_ref_xT_plus": plus["V_ref_xT"], "V_ref_xT_minus": minus["V_ref_xT"],
                        "snap_plus": plus["snap_ref"], "snap_minus": minus["snap_ref"],
                        "base_repeat_range": batch["base_repeat_range"],
                    })
            reliability["per_start"].append(rec)

    out_pairs = args.out_root / f"{args.split}_pairs.pt"
    out_scalars = args.out_root / f"{args.split}_scalars.pt"
    args.out_root.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "sacv_pairs_v1", "gamma": GAMMA, "horizons": list(horizons),
                "pairs": pairs}, out_pairs)
    torch.save({"schema": "sacv_scalars_v1", "gamma": GAMMA, "horizons": list(horizons),
                "scalars": scalars}, out_scalars)
    manifest = {
        "schema": "sacv_dataset_manifest_v1",
        "split": args.split,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": len(pairs), "n_scalars": len(scalars),
        "per_horizon_pairs": {h: sum(1 for p in pairs if p["horizon"] == h) for h in horizons},
        "n_clips": len({p["clip"] for p in pairs}),
        "raw_batches": [{"file": b.name, "sha256": sha256_file(b)} for b in batch_files],
        "outputs": {"pairs": {"file": out_pairs.name, "sha256": sha256_file(out_pairs)},
                    "scalars": {"file": out_scalars.name, "sha256": sha256_file(out_scalars)}},
        "reliability": {"reward_reconstruction_max": reliability["reward_reconstruction_max"],
                        "g_recalc_max": reliability["g_recalc_max"],
                        "base_repeat_range_max": max(
                            (r["base_repeat_range"] for r in reliability["per_start"]),
                            default=float("nan"))},
        "v_ref": "pair_it_002100.pt[critic_post] (frozen C-init)",
        "elapsed_s": round(time.time() - t0, 1),
    }
    mpath = args.out_root / f"{args.split}_dataset_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"split": args.split, "n_pairs": len(pairs), "n_scalars": len(scalars),
                      "reliability": manifest["reliability"], "out": str(mpath)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
