import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PolicyProbeGateTest(unittest.TestCase):
    def test_confidence_gate_rejects_large_drop(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe.json"
            p.write_text(json.dumps({"results": [
                {"checkpoint": "ref.pt", "return_norm": 1.0},
                {"checkpoint": "cand.pt", "return_norm": 0.5},
            ]}))
            r = subprocess.run([
                sys.executable, "tools/policy_probe_gate.py", str(p),
                "--reference", "ref.pt", "--candidate", "cand.pt",
            ], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('"passed": false', r.stdout)

    def test_confidence_gate_accepts_small_drop(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe.json"
            p.write_text(json.dumps({"results": [
                {"checkpoint": "ref.pt", "return_norm": 1.0},
                {"checkpoint": "cand.pt", "return_norm": 0.98},
            ]}))
            r = subprocess.run([
                sys.executable, "tools/policy_probe_gate.py", str(p),
                "--reference", "ref.pt", "--candidate", "cand.pt",
            ], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0)
            self.assertIn('"passed": true', r.stdout)

    def test_control_file_follows_decision(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe.json"
            control = Path(td) / "target.control"
            p.write_text(json.dumps({"results": [
                {"checkpoint": "ref.pt", "return_norm": 1.0},
                {"checkpoint": "cand.pt", "return_norm": 0.5},
            ]}))
            r = subprocess.run([
                sys.executable, "tools/policy_probe_gate.py", str(p),
                "--reference", "ref.pt", "--candidate", "cand.pt",
                "--control-file", str(control),
            ], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual(control.read_text(), "disable\n")

    def test_min_steps_rejects_short_probe(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe.json"
            p.write_text(json.dumps({"steps": 100, "results": [
                {"checkpoint": "ref.pt", "return_norm": 1.0},
                {"checkpoint": "cand.pt", "return_norm": 1.0},
            ]}))
            r = subprocess.run([
                sys.executable, "tools/policy_probe_gate.py", str(p),
                "--reference", "ref.pt", "--candidate", "cand.pt",
                "--min-steps", "500",
            ], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("minimum required is 500", r.stderr + r.stdout)

    def test_rejection_can_emit_rollback_token(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe.json"
            control = Path(td) / "control"
            rollback = Path(td) / "accepted.pt"
            p.write_text(json.dumps({"steps": 500, "results": [
                {"checkpoint": "ref.pt", "return_norm": 1.0},
                {"checkpoint": "cand.pt", "return_norm": 0.5},
            ]}))
            r = subprocess.run([
                sys.executable, "tools/policy_probe_gate.py", str(p),
                "--reference", "ref.pt", "--candidate", "cand.pt",
                "--control-file", str(control), "--rollback-checkpoint", str(rollback),
            ], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual(control.read_text(), f"rollback:{rollback}\n")

    def test_survival_drop_can_reject_return_positive_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe.json"
            p.write_text(json.dumps({"steps": 500, "results": [
                {"checkpoint": "ref.pt", "return_norm": 1.0, "survival_at_end": 0.8},
                {"checkpoint": "cand.pt", "return_norm": 1.1, "survival_at_end": 0.0},
            ]}))
            r = subprocess.run([
                sys.executable, "tools/policy_probe_gate.py", str(p),
                "--reference", "ref.pt", "--candidate", "cand.pt",
                "--max-survival-drop", "0.05",
            ], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('"passed": false', r.stdout)

    def test_first_failure_drop_can_reject_when_survival_is_equal(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe.json"
            p.write_text(json.dumps({"steps": 500, "results": [
                {"checkpoint": "ref.pt", "return_norm": 1.0,
                 "survival_at_end": 0.0, "first_failure_step_mean": 50.0},
                {"checkpoint": "cand.pt", "return_norm": 1.1,
                 "survival_at_end": 0.0, "first_failure_step_mean": 40.0},
            ]}))
            r = subprocess.run([
                sys.executable, "tools/policy_probe_gate.py", str(p),
                "--reference", "ref.pt", "--candidate", "cand.pt",
                "--max-first-failure-drop", "5",
            ], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('"passed": false', r.stdout)

    def test_tilt_failure_drop_can_reject_return_positive_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe.json"
            p.write_text(json.dumps({"steps": 500, "results": [
                {"checkpoint": "ref.pt", "return_norm": 1.0,
                 "tilt_failure_rate": 0.1},
                {"checkpoint": "cand.pt", "return_norm": 1.1,
                 "tilt_failure_rate": 0.3},
            ]}))
            r = subprocess.run([
                sys.executable, "tools/policy_probe_gate.py", str(p),
                "--reference", "ref.pt", "--candidate", "cand.pt",
                "--max-tilt-failure-drop", "0.05",
            ], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('"passed": false', r.stdout)

    def test_failure_z_uses_subtype_upper_bound(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "probe.json"
            p.write_text(json.dumps({"steps": 500, "batch": 16, "results": [
                {"checkpoint": "ref.pt", "return_norm": 1.0,
                 "height_failure_rate": 0.10},
                {"checkpoint": "cand.pt", "return_norm": 1.0,
                 "height_failure_rate": 0.12},
            ]}))
            r = subprocess.run([
                sys.executable, "tools/policy_probe_gate.py", str(p),
                "--reference", "ref.pt", "--candidate", "cand.pt",
                "--max-height-failure-drop", "0.03", "--failure-z", "1.96",
            ], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn('"passed": false', r.stdout)
