"""Unit tests for deployment-style residual gate calibration."""

import unittest

import torch

try:
    from residual_gate_audit import summarize_threshold
except ModuleNotFoundError as exc:  # local host may not have Isaac Lab
    summarize_threshold = None
    _IMPORT_ERROR = exc


class ResidualGateCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if summarize_threshold is None:
            raise unittest.SkipTest(f"project runtime unavailable: {_IMPORT_ERROR}")

    def test_fixed_threshold_is_reused_without_recomputing_quantile(self):
        train_u = torch.tensor([0.1, 0.2, 0.3, 0.4])
        train_nom = torch.tensor([1.0, 1.0, 2.0, 2.0])
        train_res = torch.tensor([1.0, 0.8, 1.8, 1.9])
        audit_u = torch.tensor([0.05, 0.25, 0.35, 0.8])
        audit_nom = torch.tensor([1.0, 1.0, 2.0, 4.0])
        audit_res = torch.tensor([1.1, 0.9, 1.5, 3.0])
        threshold = torch.quantile(train_u, 0.75)
        out = summarize_threshold(audit_u, audit_nom, audit_res, threshold, 0.75)
        self.assertAlmostEqual(out["threshold"], float(threshold))
        self.assertEqual(out["selected_rate"], 0.5)
        self.assertLess(out["error_delta_residual_minus_nominal"], 0.0)

    def test_threshold_can_select_no_audit_samples(self):
        out = summarize_threshold(
            torch.tensor([0.1, 0.2]), torch.ones(2), torch.ones(2), torch.tensor(1.0), 0.9
        )
        self.assertEqual(out["selected_rate"], 0.0)
        self.assertEqual(out["residual_improves_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
