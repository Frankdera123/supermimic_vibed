import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from tools import run_probe_gate


class RunProbeGateTest(unittest.TestCase):
    def test_help_is_available(self):
        result = subprocess.run(
            [sys.executable, "tools/run_probe_gate.py", "--help"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--control-file", result.stdout)

    def test_promote_on_accept_writes_shadow_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = Path(tmp) / "control"
            argv = ["run_probe_gate.py", "--reference", "ref.pt",
                    "--candidate", "cand.pt", "--motion-dir", "motion",
                    "--seed", "0", "--out-dir", tmp, "--control-file",
                    str(control), "--promote-on-accept"]
            with patch.object(sys, "argv", argv), patch.object(
                    run_probe_gate.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
                with self.assertRaises(SystemExit) as exc:
                    run_probe_gate.main()
            self.assertEqual(exc.exception.code, 0)
            self.assertEqual(control.read_text(encoding="utf-8"), "shadow_promote\n")

    def test_multi_horizon_requires_every_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            control = Path(tmp) / "control"
            argv = ["run_probe_gate.py", "--reference", "ref.pt",
                    "--candidate", "cand.pt", "--motion-dir", "motion",
                    "--seed", "0", "--out-dir", tmp, "--control-file",
                    str(control), "--steps", "100", "--steps", "500"]
            calls = []

            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                if "fixed_policy_eval.py" in str(cmd[1]):
                    out = Path(cmd[cmd.index("--json_out") + 1])
                    horizon = int(cmd[cmd.index("--steps") + 1])
                    out.write_text(json.dumps({"steps": horizon, "results": [
                        {"checkpoint": "ref.pt", "return_norm": 1.0,
                         "survival_at_end": 1.0},
                        {"checkpoint": "cand.pt", "return_norm": 1.0,
                         "survival_at_end": 1.0}]}))
                    return subprocess.CompletedProcess(cmd, 0)
                return subprocess.CompletedProcess(cmd, 1,
                    stdout=json.dumps({"passed": False}))

            with patch.object(sys, "argv", argv), patch.object(
                    run_probe_gate.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(SystemExit) as exc:
                    run_probe_gate.main()
            self.assertEqual(exc.exception.code, 1)
            self.assertEqual(control.read_text(encoding="utf-8"), "disable\n")
            gate_calls = [c for c in calls if "policy_probe_gate.py" in str(c[1])]
            self.assertEqual([c[c.index("--min-steps") + 1] for c in gate_calls], ["100", "500"])
            self.assertTrue(all("--max-first-failure-drop" in c for c in gate_calls))
