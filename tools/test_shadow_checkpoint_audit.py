import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


class ShadowCheckpointAuditTest(unittest.TestCase):
    def test_reports_shadow_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live.pt"
            shadow = root / "shadow.pt"
            torch.save({"pi": {"weight": torch.zeros(2, 2)}}, live)
            torch.save({"pi": {"weight": torch.ones(2, 2)},
                        "args": {"shadow_policy": True},
                        "source_checkpoint": str(live), "iteration": 5}, shadow)
            result = subprocess.run(
                [sys.executable, "tools/shadow_checkpoint_audit.py", str(live), str(shadow)],
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["is_shadow"])
            self.assertGreater(report["policy_l2_distance"], 0.0)

    def test_rejects_non_shadow_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live.pt"
            candidate = root / "candidate.pt"
            payload = {"pi": {"weight": torch.zeros(1)}}
            torch.save(payload, live)
            torch.save(payload, candidate)
            result = subprocess.run(
                [sys.executable, "tools/shadow_checkpoint_audit.py", str(live), str(candidate)],
                capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("shadow_policy=true", result.stderr + result.stdout)

    def test_ignores_running_normalizer_buffers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "live.pt"
            shadow = root / "shadow.pt"
            torch.save({"pi": {
                "weight": torch.ones(2, 2),
                "obs_norm.count": torch.tensor(1_000_000.),
                "obs_norm.var": torch.ones(2),
            }}, live)
            torch.save({"pi": {
                "weight": torch.zeros(2, 2),
                "obs_norm.count": torch.tensor(1.),
                "obs_norm.var": torch.zeros(2),
            }, "args": {"shadow_policy": True}}, shadow)
            result = subprocess.run(
                [sys.executable, "tools/shadow_checkpoint_audit.py", str(live), str(shadow)],
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertAlmostEqual(report["live_policy_l2_norm"], 2.0)
            self.assertAlmostEqual(report["shadow_policy_l2_norm"], 0.0)
            self.assertAlmostEqual(report["policy_l2_distance"], 2.0)
            self.assertEqual(report["layerwise_relative_distance"], {"weight": 1.0})
