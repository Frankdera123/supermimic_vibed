"""Dependency-light tests for SuperMimic dispersion gating."""

import unittest

import torch
import torch.nn as nn

from supermimic.dispersion import (ema_update_, gate_weight, gated_objective,
                                   quantile_gate_weight, relative_disagreement)


class DispersionTests(unittest.TestCase):
    def test_relative_disagreement_and_detach(self):
        online = torch.tensor([1.0, 2.0], requires_grad=True)
        target = torch.tensor([1.0, 1.0], requires_grad=True)
        d = relative_disagreement(online, target)
        self.assertTrue(torch.equal(d, torch.tensor([0.0, 2.0 / 3.0])))
        self.assertFalse(d.requires_grad)

    def test_gated_objective_selects_per_start(self):
        robust = torch.tensor([10.0, 20.0], requires_grad=True)
        nominal = torch.tensor([1.0, 2.0], requires_grad=True)
        out, mask = gated_objective(robust, nominal, torch.tensor([1.0, 2.0]),
                                    torch.tensor([1.0, 1.0]), threshold=0.1)
        self.assertEqual(mask.tolist(), [True, False])
        self.assertEqual(out.tolist(), [10.0, 2.0])
        out.sum().backward()
        self.assertEqual(robust.grad.tolist(), [1.0, 0.0])
        self.assertEqual(nominal.grad.tolist(), [0.0, 1.0])

    def test_soft_gate_is_bounded_and_temperature_zero_is_hard(self):
        online = torch.tensor([1.0, 2.0, 10.0])
        target = torch.tensor([1.0, 1.0, 1.0])
        hard = gate_weight(online, target, threshold=0.1, temperature=0.0)
        soft = gate_weight(online, target, threshold=0.1, temperature=0.2)
        self.assertEqual(hard.tolist(), [1.0, 0.0, 0.0])
        self.assertTrue(((soft >= 0.0) & (soft <= 1.0)).all())
        self.assertGreater(float(soft[0]), float(soft[1]))
        self.assertGreater(float(soft[1]), float(soft[2]))

    def test_quantile_gate_trusts_low_disagreement(self):
        online = torch.arange(1.0, 9.0)
        target = torch.ones(8)
        weight = quantile_gate_weight(online, target, trusted_fraction=0.5)
        self.assertEqual(int(weight.sum()), 4)
        self.assertTrue(torch.equal(weight[:4], torch.ones(4)))
        self.assertTrue(torch.equal(weight[4:], torch.zeros(4)))

    def test_ema_updates_parameters_and_buffers(self):
        source = nn.BatchNorm1d(2)
        target = nn.BatchNorm1d(2)
        with torch.no_grad():
            source.weight.fill_(3.0)
            target.weight.fill_(1.0)
            source.running_mean.fill_(2.0)
            target.running_mean.zero_()
        ema_update_(target, source, beta=0.5)
        self.assertTrue(torch.allclose(target.weight, torch.full((2,), 2.0)))
        self.assertTrue(torch.allclose(target.running_mean, torch.ones(2)))


if __name__ == "__main__":
    unittest.main()
