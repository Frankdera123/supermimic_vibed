import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.watch_probe_gate import shadow_control_token, shadow_reference


class WatchProbeGateTest(unittest.TestCase):
    def test_help_is_available(self):
        result = subprocess.run(
            [sys.executable, "tools/watch_probe_gate.py", "--help"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--checkpoint-dir", result.stdout)
        self.assertIn("--checkpoint-pattern", result.stdout)
        self.assertIn("--shadow-mode", result.stdout)
        self.assertIn("--once", result.stdout)

    def test_shadow_reference_tracks_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "ckpt_000100.pt"
            candidate = root / "shadow_ckpt_000100.pt"
            live.write_bytes(b"live"); candidate.write_bytes(b"shadow")
            self.assertEqual(shadow_reference(candidate, root / "unused.pt", True), live)
            self.assertIsNone(shadow_reference(root / "ckpt_000100.pt", root / "unused.pt", True))

    def test_shadow_reference_waits_for_late_live_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "shadow_ckpt_000100.pt"
            live = root / "ckpt_000100.pt"
            candidate.write_bytes(b"shadow")
            self.assertIsNone(shadow_reference(candidate, root / "unused.pt", True))
            live.write_bytes(b"live")
            self.assertEqual(shadow_reference(candidate, root / "unused.pt", True), live)

    def test_shadow_control_token(self):
        self.assertEqual(shadow_control_token(0), "shadow_promote\n")
        self.assertEqual(shadow_control_token(1), "shadow_discard\n")

    def test_once_exits_when_no_checkpoint_is_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = ["watch_probe_gate.py", "--checkpoint-dir", str(root),
                    "--reference", "ref.pt", "--motion-dir", "motion",
                    "--seed", "0", "--out-dir", str(root / "out"),
                    "--control-file", str(root / "control"), "--once"]
            from tools import watch_probe_gate
            with patch.object(sys, "argv", argv):
                self.assertEqual(watch_probe_gate.main(), 0)
            self.assertFalse((root / "control").exists())

    def test_shadow_mode_rejection_discards_without_live_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "ckpt_000100.pt"
            candidate = root / "shadow_ckpt_000100.pt"
            live.write_bytes(b"live"); candidate.write_bytes(b"shadow")
            control = root / "control"
            argv = ["watch_probe_gate.py", "--checkpoint-dir", str(root),
                    "--checkpoint-pattern", "shadow_ckpt_*.pt", "--reference", str(live),
                    "--motion-dir", "motion", "--seed", "123", "--out-dir", str(root / "out"),
                    "--control-file", str(control), "--shadow-mode",
                    "--require-shadow-artifact",
                    "--stop-on-reject", "--poll-seconds", "1",
                    "--steps", "100", "--steps", "500"]
            calls = []

            def fake_run(cmd, check=False):
                calls.append(cmd)
                return subprocess.CompletedProcess(cmd, 1 if len(calls) == 2 else 0)

            from tools import watch_probe_gate
            with patch.object(sys, "argv", argv), patch.object(
                    subprocess, "run", side_effect=fake_run):
                result = watch_probe_gate.main()
            self.assertEqual(result, 1)
            self.assertEqual(control.read_text(encoding="utf-8"), "shadow_discard\n")
            self.assertIn(str(live), calls[0])
            runner_cmd = calls[1]
            self.assertIn(str(live), runner_cmd)
            self.assertNotIn("--rollback-checkpoint", runner_cmd)
            self.assertEqual(runner_cmd.count("--steps"), 2)
            self.assertIn("--max-first-failure-drop", runner_cmd)
