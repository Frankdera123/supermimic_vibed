"""Audit that a candidate checkpoint is a real shadow-policy artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _is_running_stat(key: str) -> bool:
    """Exclude non-trainable normalizer buffers from policy distance audits."""
    leaf = key.rsplit(".", 1)[-1]
    return leaf in {"count", "mean", "var", "num_batches_tracked"}


def policy_distance(reference: dict, candidate: dict) -> tuple[float, float, float, float]:
    keys = sorted(k for k in (set(reference) & set(candidate))
                  if not _is_running_stat(k))
    if not keys:
        raise ValueError("checkpoints have no overlapping policy parameters")
    sq = 0.0
    ref_sq = 0.0
    for key in keys:
        a = reference[key].detach().float()
        b = candidate[key].detach().float()
        sq += float((a - b).square().sum())
        ref_sq += float(a.square().sum())
    distance = sq ** 0.5
    ref_norm = ref_sq ** 0.5
    candidate_norm = sum(float(candidate[key].detach().float().square().sum()) for key in keys) ** 0.5
    return distance, ref_norm, candidate_norm, distance / max(candidate_norm, 1e-12)


def layerwise_distance(reference: dict, candidate: dict) -> dict[str, float]:
    """Return relative L2 displacement grouped by state-dict key prefix."""
    accum: dict[str, list[float]] = {}
    for key in sorted(set(reference) & set(candidate)):
        if _is_running_stat(key):
            continue
        prefix = key.split(".", 2)[0] if "." in key else key
        delta = float((reference[key].detach().float() - candidate[key].detach().float()).square().sum())
        base = float(reference[key].detach().float().square().sum())
        slot = accum.setdefault(prefix, [0.0, 0.0])
        slot[0] += delta
        slot[1] += base
    return {k: (v[0] ** 0.5) / max(v[1] ** 0.5, 1e-12) for k, v in accum.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("live_checkpoint", type=Path)
    ap.add_argument("shadow_checkpoint", type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    live = torch.load(args.live_checkpoint, map_location="cpu", weights_only=False)
    shadow = torch.load(args.shadow_checkpoint, map_location="cpu", weights_only=False)
    if "pi" not in live or "pi" not in shadow:
        raise SystemExit("both checkpoints must contain pi")
    shadow_args = shadow.get("args", {})
    source = shadow.get("source_checkpoint")
    if not shadow_args.get("shadow_policy", False):
        raise SystemExit("candidate does not declare shadow_policy=true")
    distance, live_norm, shadow_norm, relative_to_shadow = policy_distance(live["pi"], shadow["pi"])
    report = {
        "live_checkpoint": str(args.live_checkpoint),
        "shadow_checkpoint": str(args.shadow_checkpoint),
        "source_checkpoint": source,
        "iteration": shadow.get("iteration"),
        "policy_l2_distance": distance,
        "live_policy_l2_norm": live_norm,
        "shadow_policy_l2_norm": shadow_norm,
        "policy_relative_distance_to_live": distance / max(live_norm, 1e-12),
        "policy_relative_distance_to_shadow": relative_to_shadow,
        "layerwise_relative_distance": layerwise_distance(live["pi"], shadow["pi"]),
        "is_shadow": True,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
