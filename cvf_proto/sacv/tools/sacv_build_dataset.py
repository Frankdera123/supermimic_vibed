"""CVF-1a: build train/val/test pair datasets from raw collection batches (CPU-only).

v2 (2026-11-23) -- corrected state/label contract.  v1 (2026-11-20) was INVALID:
it evaluated the critic at the end-of-continuation state x_{H+T} while every label
is defined for the prefix-end state x_H, and it stored Delta G in the ``Delta_Y``
field.  v1 artifacts are left untouched under ``datasets/`` (see
docs/cvf/sacv/CVF1B_ERRATUM.md); v2 is written to ``datasets/v2/``.

Contract (docs/cvf/sacv/CVF1A_DATA_CONTRACT.md sec. 3-4), H = 32 action prefix:
  x_H       = real-simulator state at prefix end           (snaps_prefix['prefix_32'])
  G_T(x_H)  = sum_{k<T} gamma^k c_k r_k, continuation only (independent target)
  c_T       = prod_{k<T} (1 - d_k)
  Y_T(x_H)  = G_T + gamma^T c_T V_ref(x_{H+T})             (matched training target,
              V_ref = frozen C-init = B0@2100_post evaluated on the TRUE sim state)
  Delta G   = G+ - G-,   Delta Y = Y+ - Y-   (same start / family / epsilon)
The critic input of a pair is ALWAYS x_H^+/-; x_{H+T} only enters the tail.
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
H_PREFIX = 32
SCHEMA = "sacv_pairs_v2"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_split(split: str, raw_root: Path, horizons: tuple[int, ...]) -> dict:
    """Return {'pairs': [...], 'scalars': [...], 'reliability': {...}, 'batches': [...]},
    routed by the split recorded on each start (collection ran with --split all)."""
    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()
    from supermimic.mdp import normalize_xi, unflatten_state
    from supermimic.networks import Critic

    pair2100 = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
    proto = bridge.setup_world(device="cpu", pair=pair2100)["proto"]
    # C-init = critic_post lives only in the raw bundle (bridge exports bundle["critic"] == pre)
    v_ref_sd = torch.load(bridge.pairs_root() / "online_B0/diag_pairs/pair_it_002100.pt",
                          map_location="cpu", weights_only=False)["critic_post"]
    v_ref = Critic(v_ref_sd["norm.mean"].shape[0])
    v_ref.load_state_dict(v_ref_sd, strict=True)
    v_ref.eval()
    xi_n = normalize_xi(torch.zeros(1, 33) + torch.tensor([[1.0] + [0.0] * 32]))

    def v_ref_at(S, f) -> float:
        with torch.no_grad():
            return float(v_ref(proto.critic_input(S, f, xi_n, proto.wm_goal(f, S)))[0])

    def branch_records(br: dict, tag: str, bf: Path, slot: int, hz: tuple[int, ...]) -> dict:
        pl = br["prefix_len"]
        assert pl == H_PREFIX, f"prefix_len {pl} != {H_PREFIX}"
        r = br["full"]["r"][slot:slot + 1, pl:]            # (1, T) continuation rewards
        d = br["full"]["d_soft"][slot:slot + 1, pl:]
        dh = br["full"]["d_hard"][slot:slot + 1, pl:]
        G = vt.discounted_return(r, d, GAMMA, horizons=hz)
        Gh = vt.discounted_return(r, dh, GAMMA, horizons=hz)
        out = {"tag": tag, "batch": bf.name, "slot": slot, "Y_soft": {}, "Y_hard": {},
               "Y_tail": {}}
        for h in hz:
            c_s, c_h = vt.survival_after(d, h), vt.survival_after(dh, h)
            snap_T = br["snaps_cont"][f"cont_{h}"]          # TRUE sim state x_{H+T}
            S_T = unflatten_state(snap_T["S_flat"][slot:slot + 1])
            v_h = v_ref_at(S_T, snap_T["f"][slot:slot + 1])
            tail = (GAMMA ** h) * c_s * torch.tensor([v_h], dtype=torch.float64)
            out[f"G_soft_{h}"] = float(G[h][0])
            out[f"G_hard_{h}"] = float(Gh[h][0])
            out[f"c_soft_{h}"] = float(c_s[0])
            out[f"c_hard_{h}"] = float(c_h[0])
            out[f"hard_term_{h}"] = float(dh[:, :h].max())
            out[f"V_ref_{h}"] = v_h
            out["Y_soft"][h] = float((G[h].double() + tail)[0])
            out["Y_hard"][h] = float((Gh[h].double()
                                      + (GAMMA ** h) * c_h * torch.tensor([v_h], dtype=torch.float64))[0])
            out["Y_tail"][h] = float(tail[0])
        # the state the CRITIC is evaluated on: prefix end x_H (never x_{H+T})
        out["critic_state"] = {"batch": bf.name, "branch": tag, "slot": slot,
                               "snap_group": "snaps_prefix", "snap_key": f"prefix_{H_PREFIX}"}
        return out

    batch_files = sorted(raw_root.glob(f"*/batch_*.pt"))
    pairs, scalars, batches, per_start = [], [], [], []
    rel = {"reward_reconstruction_max": 0.0, "g_recalc_max": 0.0}
    for bf in batch_files:
        batch = torch.load(bf, map_location="cpu", weights_only=False)
        hz = tuple(h for h in horizons if h <= batch["t_cont"])
        branches_all = [*batch["base_res"], *batch["branches"].values()]
        rel["reward_reconstruction_max"] = max(
            rel["reward_reconstruction_max"],
            max(float((br["full"]["r"] - br["full"]["r_state"] - br["full"]["r_direct"]).abs().max())
                for br in branches_all))
        rel["g_recalc_max"] = max(rel["g_recalc_max"],
                                  max(br["g_recalc_vsoft_absdiff"] for br in branches_all))
        used = False
        for slot in range(batch["n_valid"]):
            st = batch["starts"][slot]
            if st["split"] != split:
                continue
            used = True
            base0 = branch_records(batch["base_res"][0], "base0", bf, slot, hz)
            base1 = branch_records(batch["base_res"][1], "base1", bf, slot, hz)
            scalars.append({"clip": st["clip"], "split": st["split"], "frame": st["frame"],
                            "base": base0, "base1": base1})
            for key, br_plus in batch["branches"].items():
                fam, eps_s, sign = key.split("|")
                if sign != "plus":
                    continue
                mk = key.replace("|plus", "|minus")
                if mk not in batch["branches"]:
                    continue
                eps = float(eps_s.removeprefix("eps"))
                plus = branch_records(br_plus, key, bf, slot, hz)
                minus = branch_records(batch["branches"][mk], mk, bf, slot, hz)
                for h in hz:
                    pairs.append({
                        "clip": st["clip"], "split": st["split"], "frame": st["frame"],
                        "family": fam, "eps": eps, "horizon": h,
                        "G_plus": plus[f"G_soft_{h}"], "G_minus": minus[f"G_soft_{h}"],
                        "Y_plus": plus["Y_soft"][h], "Y_minus": minus["Y_soft"][h],
                        "Delta_G": plus[f"G_soft_{h}"] - minus[f"G_soft_{h}"],
                        "Delta_Y": plus["Y_soft"][h] - minus["Y_soft"][h],
                        "c_plus": plus[f"c_soft_{h}"], "c_minus": minus[f"c_soft_{h}"],
                        "term_plus": plus[f"hard_term_{h}"], "term_minus": minus[f"hard_term_{h}"],
                        "V_ref_plus": plus[f"V_ref_{h}"], "V_ref_minus": minus[f"V_ref_{h}"],
                        "state_plus": plus["critic_state"], "state_minus": minus["critic_state"],
                        "base_repeat_range": batch["base_repeat_range"],
                    })
            per_start.append({"clip": st["clip"], "frame": st["frame"]})
        if used:
            batches.append({"file": bf.name, "sha256": sha256_file(bf)})
    return {"pairs": pairs, "scalars": scalars, "reliability": rel,
            "base_repeat_range_max": max((p["base_repeat_range"] for p in pairs), default=float("nan")),
            "raw_batches": batches, "n_starts": len(per_start)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "test"), required=True)
    ap.add_argument("--raw_root", type=Path, default=_REPO / "results/cvf/sacv/datasets/raw")
    ap.add_argument("--out_root", type=Path, default=_REPO / "results/cvf/sacv/datasets/v2")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    args = ap.parse_args()
    t0 = time.time()
    horizons = tuple(args.horizons)

    out = build_split(args.split, args.raw_root, horizons)
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_pairs = args.out_root / f"{args.split}_pairs.pt"
    out_scalars = args.out_root / f"{args.split}_scalars.pt"
    torch.save({"schema": SCHEMA, "gamma": GAMMA, "horizons": list(horizons),
                "critic_state": f"snaps_prefix/prefix_{H_PREFIX} (x_H)",
                "pairs": out["pairs"]}, out_pairs)
    torch.save({"schema": "sacv_scalars_v2", "gamma": GAMMA, "horizons": list(horizons),
                "scalars": out["scalars"]}, out_scalars)

    # label composition at the frozen horizon (largest): real signal vs frozen-critic tail
    import numpy as np
    hz = horizons[-1]
    comp = {}
    for fam in sorted({p["family"] for p in out["pairs"]}):
        fp = [p for p in out["pairs"] if p["horizon"] == hz and p["family"] == fam]
        dG = np.array([p["Delta_G"] for p in fp])
        dY = np.array([p["Delta_Y"] for p in fp])
        comp[fam] = {"n": len(fp), "std_DeltaG": float(dG.std()), "std_DeltaY": float(dY.std()),
                     "tail_var_over_signal_var": float((dY - dG).var() / max(dG.var(), 1e-12))}
    ph = [p for p in out["pairs"] if p["horizon"] == hz]
    manifest = {
        "schema": "sacv_dataset_manifest_v2", "split": args.split,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": len(out["pairs"]), "n_starts": out["n_starts"],
        "per_horizon_pairs": {h: sum(1 for p in out["pairs"] if p["horizon"] == h) for h in horizons},
        "n_clips": len({p["clip"] for p in out["pairs"]}),
        "n_pairs_single_sided_term_h%d" % hz: sum(
            1 for p in ph if (p["c_plus"] < 0.5) != (p["c_minus"] < 0.5)),
        "n_pairs_either_side_term_h%d" % hz: sum(1 for p in ph if p["c_plus"] < 0.5 or p["c_minus"] < 0.5),
        "label_composition_h%d" % hz: comp,
        "raw_batches": out["raw_batches"],
        "outputs": {"pairs": {"file": out_pairs.name, "sha256": sha256_file(out_pairs)},
                    "scalars": {"file": out_scalars.name, "sha256": sha256_file(out_scalars)}},
        "reliability": out["reliability"],
        "base_repeat_range_max": out["base_repeat_range_max"],
        "v_ref": "pair_it_002100.pt[critic_post] (frozen C-init), evaluated on true sim state x_{H+T}",
        "elapsed_s": round(time.time() - t0, 1),
    }
    mpath = args.out_root / f"{args.split}_dataset_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"split": args.split, "n_pairs": len(out["pairs"]), "n_starts": out["n_starts"],
                      "single_sided_term": manifest[f"n_pairs_single_sided_term_h{hz}"],
                      "reliability": out["reliability"], "out": str(mpath)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
