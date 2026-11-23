"""CVF-1a Phase D: dump per-pair critic predictions for stats (CPU, batched).

Produces {critic: {"v_plus": (N,), "v_minus": (N,)}} aligned 1:1 with the
(un-gated, ordered) pair list at one horizon, plus the pair meta needed for
gating/stratification downstream (sacv_stats.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SACV = _HERE.parent
_REPO = _SACV.parents[1]
for p in (str(_SACV), str(_REPO / "pada_proto"), str(_REPO / "pada_proto" / "tools"),
          str(_REPO / "crgf_proto")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "test", "train"), default="val")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--pairs", type=Path, default=None, help="default: datasets/<split>_pairs.pt")
    ap.add_argument("--critics", type=Path, nargs="+", required=True,
                    help="fine-tuned ckpts (group name from ckpt['group']); C-init always added")
    ap.add_argument("--datasets", type=Path, default=_REPO / "results/cvf/sacv/datasets")
    ap.add_argument("--out", type=Path, default=_REPO / "results/cvf/sacv/metrics/preds.pt")
    args = ap.parse_args()
    t0 = time.time()

    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()
    from supermimic.mdp import normalize_xi, unflatten_state
    from supermimic.networks import Critic

    pair2100 = bridge.load_pair_bundle("online_B0", 2100, device="cpu")
    proto = bridge.setup_world(device="cpu", pair=pair2100)["proto"]
    xi_one = torch.zeros(1, 33); xi_one[:, 0] = 1.0
    xi_n = normalize_xi(xi_one)

    ppath = args.pairs or (args.datasets / f"{args.split}_pairs.pt")
    data = torch.load(ppath, map_location="cpu", weights_only=False)
    pairs = [p for p in data["pairs"] if p["horizon"] == args.horizon]
    if not pairs:
        print(json.dumps({"error": "no pairs at horizon", "horizon": args.horizon}))
        return 1

    # C-init = critic_post from the raw bundle (bridge exports "critic"==pre)
    raw2100 = torch.load(bridge.pairs_root() / "online_B0/diag_pairs/pair_it_002100.pt",
                         map_location="cpu", weights_only=False)

    # unique extended-state fetch list: (batch_file, branch, slot, snap_key, side)
    snaps: dict[str, dict] = {}

    def batch_of(name: str) -> dict:
        if name not in snaps:
            path = next((_REPO / "results/cvf/sacv/datasets/raw").rglob(name))
            snaps[name] = torch.load(path, map_location="cpu", weights_only=False)
        return snaps[name]

    def branch_of(batch: dict, pref: dict) -> dict:
        return batch["branches"][pref["branch"]] if pref["branch"] != "base0" \
            else batch["base_res"][0]

    critic_inputs: list = []      # (pair_idx, side)
    X_rows: list[torch.Tensor] = []
    meta = []
    for i, p in enumerate(pairs):
        meta.append({"clip": p["clip"], "frame": p["frame"], "family": p["family"],
                     "eps": p["eps"], "Delta_Y": p["Delta_Y"]})
        for side in ("plus", "minus"):
            pref = p[f"snap_{side}"]
            batch = batch_of(pref["batch"])
            br = branch_of(batch, pref)
            snap = br["snaps_cont"][pref["snap_key"]]
            slot = pref["slot"]
            S = unflatten_state(snap["S_flat"][slot:slot + 1])
            f_t = snap["f"][slot:slot + 1]
            gw = proto.wm_goal(f_t, S)
            X_rows.append(proto.critic_input(S, f_t, xi_n, gw))
            critic_inputs.append((i, side))

    X = torch.cat(X_rows, 0)      # (2N, 938)
    critics = {}
    for cp in args.critics:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        critics[ck.get("group", cp.stem)] = ck["critic"]

    out = {"schema": "sacv_preds_v1", "horizon": args.horizon, "n_pairs": len(pairs),
           "meta": meta, "critics": {}}
    critics["C-init"] = raw2100["critic_post"]
    for name, sd in critics.items():
        c = Critic(sd["norm.mean"].shape[0])
        c.load_state_dict(sd, strict=True)
        c.eval()
        with torch.no_grad():
            v = c(X)
        v_plus = torch.stack([v[j] for j, (_i, s) in enumerate(critic_inputs) if s == "plus"])
        v_minus = torch.stack([v[j] for j, (_i, s) in enumerate(critic_inputs) if s == "minus"])
        out["critics"][name] = {"v_plus": v_plus, "v_minus": v_minus}
        y_p = torch.tensor([p["Y_plus"] for p in pairs])
        y_m = torch.tensor([p["Y_minus"] for p in pairs])
        rmse = float(torch.sqrt(0.5 * ((v_plus - y_p) ** 2 + (v_minus - y_m) ** 2).mean()))
        out["critics"][name]["value_rmse"] = rmse

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(json.dumps({"out": str(args.out), "n_pairs": len(pairs),
                      "critics": list(out["critics"].keys()),
                      "elapsed_s": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
