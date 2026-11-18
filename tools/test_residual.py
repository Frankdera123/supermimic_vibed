"""Dependency-light tests for the optional residual world-model branch."""

import unittest

import torch

try:
    from supermimic.networks import WorldModel, run_mlp
except ModuleNotFoundError as exc:  # local host may not have Isaac Lab installed
    WorldModel = run_mlp = None
    _IMPORT_ERROR = exc


class ResidualWorldModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if WorldModel is None:
            raise unittest.SkipTest(f"project runtime unavailable: {_IMPORT_ERROR}")

    def test_zero_initialized_residual_is_nominally_identity(self):
        torch.manual_seed(0)
        wm = WorldModel(hidden=(32, 32, 32), residual_hidden=8)
        x = torch.randn(7, wm.in_dim)
        base = run_mlp(wm.net, x)
        with_residual = base + run_mlp(wm.residual_net, x)
        self.assertTrue(torch.allclose(base, with_residual))

    def test_residual_gradient_does_not_require_nominal_branch(self):
        wm = WorldModel(hidden=(32, 32, 32), residual_hidden=8)
        for p in wm.net.parameters():
            p.requires_grad_(False)
        x = torch.randn(5, wm.in_dim)
        loss = run_mlp(wm.residual_net, x).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in wm.net.parameters()))
        self.assertTrue(any(p.grad is not None for p in wm.residual_net.parameters()))

    def test_zero_residual_trust_region_has_finite_gradient(self):
        """The output cap must not freeze the zero-initialized residual head."""
        wm = WorldModel(hidden=(32, 32, 32), residual_hidden=8,
                        residual_max_norm=0.1)
        x = torch.randn(5, wm.in_dim)
        r = run_mlp(wm.residual_net, x)
        rn = (r.square().sum(-1, keepdim=True) + 1e-8).sqrt()
        capped = r * (wm.residual_max_norm / rn).clamp_max(1.0)
        capped.square().sum().backward()
        grads = [p.grad for p in wm.residual_net.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main()
