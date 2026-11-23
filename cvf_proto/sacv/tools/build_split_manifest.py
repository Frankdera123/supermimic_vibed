"""CVF-1a Phase A: build the frozen clip/start split manifest (CPU-only).

Rules (docs/cvf/sacv/CVF1A_DATA_CONTRACT.md §5):
  - clip pool: data/lafan1_g1 minus the 10 CVF-0 clips;
  - pilot sizes: Train 12 / Val 4 / Test 6 clips, 2 starts per clip (>=64 frames apart);
  - stratified per-category allocation with fixed seed; singletons (sprint,
    fightAndSports) forced to Test for eval coverage;
  - horizon feasibility: f0 + H(32) + T_max(96) <= clip_end - 1;
  - CVF-0 data stays a historical diagnostic set (never re-used for training).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SACV = _HERE.parent
_CVF = _SACV.parent
_REPO = _CVF.parent
for p in (str(_REPO / "pada_proto"), str(_REPO / "pada_proto" / "tools"), str(_REPO / "crgf_proto")):
    if p not in sys.path:
        sys.path.insert(0, p)

SEED = 2701
H_PREFIX = 32
T_MAX = 96
MIN_FRAME_GAP = 64
CVF0_CLIPS = {
    "dance2_subject2", "fallAndGetUp1_subject1", "fight1_subject5",
    "fightAndSports1_subject4", "jumps1_subject5", "run2_subject4",
    "sprint1_subject2", "walk1_subject2", "walk3_subject3", "walk4_subject1",
}
# per-category clip counts allocated to each split (sums: train 12 / val 4 / test 6).
# Pool constraints after excluding CVF-0's 10 clips: dance1:3 dance2:4 fallAndGetUp1:2
# fallAndGetUp2:2 fallAndGetUp3:1 fight1:2 fightAndSports1:1 jumps1:2 run1:2 run2:1
# sprint1:1 walk1:2 walk2:3 walk3:4 walk4:0.
ALLOCATION = {
    "train": {"run1": 2, "jumps1": 1, "fight1": 1, "fallAndGetUp1": 2, "fallAndGetUp2": 1,
              "walk2": 2, "walk1": 1, "walk3": 2},
    "val": {"jumps1": 1, "run2": 1, "dance1": 1, "walk2": 1},
    "test": {"sprint1": 1, "fightAndSports1": 1, "fight1": 1, "dance1": 1, "walk1": 1,
             "fallAndGetUp3": 1},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def category(clip_name: str) -> str:
    """dance2_subject3 -> dance2; fallAndGetUp1_subject1 -> fallAndGetUp1."""
    base = clip_name.split("_subject")[0]
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_dir", type=Path, default=_REPO / "data/lafan1_g1")
    ap.add_argument("--out", type=Path, default=_REPO / "results/cvf/sacv/manifests/CVF1A_SPLIT_MANIFEST.json")
    ap.add_argument("--docs_copy", type=Path, default=_REPO / "docs/cvf/sacv/CVF1A_SPLIT_MANIFEST.json")
    args = ap.parse_args()
    t0 = time.time()

    import pada_bridge as bridge

    bridge.import_stack()
    bridge.fix_paths()
    import torch  # noqa: E402

    from supermimic.motion import MotionLib

    world = bridge.setup_world(device="cpu")
    motion = world["motion"]
    names = list(motion.names)
    g = torch.Generator().manual_seed(SEED)

    by_cat: dict[str, list[str]] = defaultdict(list)
    for n in names:
        by_cat[category(n)].append(n)

    pool = {}
    skipped = {"cvf0_reuse": [], "too_short": []}
    for n in names:
        cat = category(n)
        if n in CVF0_CLIPS:
            skipped["cvf0_reuse"].append(n)
            continue
        s0, s1 = int(motion.clip_start[names.index(n)]), int(motion.clip_end[names.index(n)])
        lo, hi = s0 + 1, s1 - 1 - (H_PREFIX + T_MAX)
        if hi - lo < MIN_FRAME_GAP + 1:
            skipped["too_short"].append({"clip": n, "len": s1 - s0})
            continue
        pool[n] = {"clip_id": names.index(n), "category": cat, "clip_start": s0, "clip_end": s1,
                   "f0_lo": lo, "f0_hi": hi, "csv_sha256": sha256_file(args.motion_dir / f"{n}.csv")}

    # deterministic stratified allocation
    assignment: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        for cat, k in ALLOCATION[split].items():
            cands = sorted(c for c in by_cat.get(cat, []) if c in pool and c not in
                           assignment["train"] + assignment["val"] + assignment["test"])
            order = torch.randperm(len(cands), generator=g).tolist()
            picked = [cands[i] for i in order[:k]]
            if len(picked) < k:
                print(json.dumps({"error": "insufficient clips", "split": split, "cat": cat,
                                  "want": k, "got": len(picked)}))
                return 1
            assignment[split].extend(picked)

    # start frames: 2 per clip, >= MIN_FRAME_GAP apart, seeded
    starts = {}
    for split, clips in assignment.items():
        for n in sorted(clips):
            info = pool[n]
            lo, hi = info["f0_lo"], info["f0_hi"]
            for _ in range(200):
                f1 = int(torch.randint(lo, hi + 1, (1,), generator=g))
                f2 = int(torch.randint(lo, hi + 1, (1,), generator=g))
                if abs(f1 - f2) >= MIN_FRAME_GAP:
                    break
            else:
                print(json.dumps({"error": "no feasible frame pair", "clip": n}))
                return 1
            starts[n] = {"split": split, "frames": sorted([f1, f2])}

    cats_train = Counter(pool[n]["category"] for n in assignment["train"])
    manifest = {
        "schema": "sacv_split_manifest_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "h_prefix": H_PREFIX,
        "t_max": T_MAX,
        "min_frame_gap": MIN_FRAME_GAP,
        "motion_dir": str(args.motion_dir),
        "cvf0_excluded_clips": sorted(CVF0_CLIPS),
        "skipped": skipped,
        "splits": {
            split: {
                "clips": sorted(assignment[split]),
                "n_clips": len(assignment[split]),
                "n_starts": 2 * len(assignment[split]),
                "category_counts": dict(sorted(cats_train.items())) if split == "train" else
                dict(sorted(Counter(pool[n]["category"] for n in assignment[split]).items())),
                "per_clip": {n: {"category": pool[n]["category"],
                                 "clip_id": pool[n]["clip_id"],
                                 "clip_start": pool[n]["clip_start"],
                                 "clip_end": pool[n]["clip_end"],
                                 "frames": starts[n]["frames"],
                                 "csv_sha256": pool[n]["csv_sha256"]}
                             for n in sorted(assignment[split])},
            } for split in ("train", "val", "test")
        },
        "reserve_unused_clips": sorted(set(pool) - set(assignment["train"])
                                       - set(assignment["val"]) - set(assignment["test"])),
        "rules": [
            "positive/negative branches, epsilons, prefix and continuation never cross splits",
            "overlapping windows of one trajectory never cross splits",
            "CVF-0 data is a historical diagnostic regression set only",
            "Test split never participates in hyperparameter / alpha / horizon selection",
        ],
    }
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2))
    args.docs_copy.write_text(json.dumps(manifest, indent=2))
    summary = {
        "out": str(args.out), "docs_copy": str(args.docs_copy),
        "pool_size": len(pool), "skipped_cvf0": len(skipped["cvf0_reuse"]),
        "skipped_short": len(skipped["too_short"]),
        "train/val/test clips": [manifest["splits"][s]["n_clips"] for s in ("train", "val", "test")],
        "train_cats": manifest["splits"]["train"]["category_counts"],
        "test_cats": manifest["splits"]["test"]["category_counts"],
        "reserve": len(manifest["reserve_unused_clips"]),
        "elapsed_s": manifest["elapsed_s"],
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
