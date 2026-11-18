"""Watch a training checkpoint directory and run the paired probe gate."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def shadow_reference(candidate: Path, configured_reference: Path, shadow_mode: bool) -> Path | None:
    """Resolve the live checkpoint paired with a shadow candidate."""
    if not shadow_mode:
        return configured_reference
    if not candidate.name.startswith("shadow_ckpt_"):
        return None
    live_path = candidate.with_name(candidate.name.replace("shadow_ckpt_", "ckpt_", 1))
    return live_path if live_path.exists() else None


def shadow_control_token(returncode: int) -> str:
    return "shadow_promote\n" if returncode == 0 else "shadow_discard\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True, type=Path)
    ap.add_argument("--checkpoint-pattern", default="ckpt_*.pt",
                    help="candidate checkpoint glob (use shadow_ckpt_*.pt for shadow policy)")
    ap.add_argument("--require-shadow-artifact", action="store_true",
                    help="reject candidates without shadow_policy=true metadata")
    ap.add_argument("--shadow-mode", action="store_true",
                    help="compare shadow_ckpt_N against ckpt_N and promote/discard the branch")
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--motion-dir", action="append", required=True)
    ap.add_argument("--seed", action="append", required=True, type=int)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--control-file", required=True, type=Path)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, action="append", default=None,
                    help="probe horizon; repeat to require every listed horizon")
    ap.add_argument("--min-steps", type=int, default=None,
                    help="minimum horizon accepted by the gate (defaults to 500 for one horizon, or the shortest explicit horizon)")
    ap.add_argument("--max-relative-drop", type=float, default=0.05)
    ap.add_argument("--max-survival-drop", type=float, default=0.05)
    ap.add_argument("--max-first-failure-drop", type=float, default=0.0)
    ap.add_argument("--max-height-failure-drop", type=float, default=0.0)
    ap.add_argument("--max-tilt-failure-drop", type=float, default=0.0)
    ap.add_argument("--failure-z", type=float, default=0.0)
    ap.add_argument("--stop-on-reject", action="store_true")
    ap.add_argument("--once", action="store_true",
                    help="process currently visible checkpoints once, then exit")
    ap.add_argument("--promote-on-accept", action="store_true",
                    help="promote an accepted shadow branch in the training process")
    args = ap.parse_args()
    horizons = args.steps or [500]
    if any(h < 1 for h in horizons):
        ap.error("--steps must be positive")
    min_steps = args.min_steps if args.min_steps is not None else (500 if args.steps is None else min(horizons))
    if min_steps < 0:
        ap.error("--min-steps must be nonnegative")
    if args.poll_seconds <= 0:
        ap.error("--poll-seconds must be positive")
    seen: set[Path] = set()
    runner = Path(__file__).with_name("run_probe_gate.py")
    while True:
        exit_code = 0
        checkpoints = sorted(args.checkpoint_dir.glob(args.checkpoint_pattern))
        for candidate in checkpoints:
            if candidate in seen:
                continue
            # Atomic checkpoint publication means a visible file is complete;
            # still ignore zero-length files from interrupted writes.
            if candidate.stat().st_size == 0:
                continue
            reference_path = shadow_reference(candidate, args.reference, args.shadow_mode)
            if reference_path is None:
                # A non-shadow artifact is permanently irrelevant; a missing
                # same-iteration live checkpoint may simply be a publication
                # race and should be retried on the next poll.
                if args.shadow_mode and not candidate.name.startswith("shadow_ckpt_"):
                    seen.add(candidate)
                continue
            reference = str(reference_path)
            if args.require_shadow_artifact:
                audit = [sys.executable, str(Path(__file__).with_name("shadow_checkpoint_audit.py")),
                         reference, str(candidate)]
                audit_result = subprocess.run(audit, check=False)
                if audit_result.returncode != 0:
                    seen.add(candidate)
                    continue
            previous = sorted(seen)
            seen.add(candidate)
            cmd = [sys.executable, str(runner), "--reference", reference,
                   "--candidate", str(candidate), "--out-dir", str(args.out_dir / candidate.stem),
                   "--control-file", str(args.control_file), "--batch", str(args.batch),
                   *[item for h in horizons for item in ("--steps", str(h))],
                   "--min-steps", str(min_steps),
                   "--max-relative-drop", str(args.max_relative_drop),
                   "--max-survival-drop", str(args.max_survival_drop)]
            cmd += ["--max-first-failure-drop", str(args.max_first_failure_drop)]
            cmd += ["--max-height-failure-drop", str(args.max_height_failure_drop),
                    "--max-tilt-failure-drop", str(args.max_tilt_failure_drop),
                    "--failure-z", str(args.failure_z)]
            if args.promote_on_accept:
                cmd += ["--promote-on-accept"]
            if previous:
                cmd += ["--rollback-checkpoint", str(previous[-1])]
            for motion in args.motion_dir:
                cmd += ["--motion-dir", motion]
            for seed in args.seed:
                cmd += ["--seed", str(seed)]
            result = subprocess.run(cmd, check=False)
            exit_code = max(exit_code, result.returncode)
            if args.shadow_mode:
                args.control_file.write_text(shadow_control_token(result.returncode), encoding="utf-8")
            if result.returncode != 0 and args.stop_on_reject:
                return result.returncode
        if args.once:
            return exit_code
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
