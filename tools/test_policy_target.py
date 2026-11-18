import unittest

import torch
import torch.nn as nn

from supermimic.policy_target import PolicyTargetWorldModel


class _TinyWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("running", torch.tensor(1.0))
        self.in_dim = 3
        self.out_norm = _TinyNorm()

    def inputs(self, S, a, xi_n):
        return a

    def forward(self, x):
        return self.linear(x) + self.running


class _TinyNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(2))
        self.register_buffer("var", torch.ones(2))
        self.eps = 1e-4

    def std(self):
        return torch.sqrt(self.var + self.eps ** 2)

class PolicyTargetTests(unittest.TestCase):
    def test_deepcopy_freeze_and_hard_sync(self):
        nominal = _TinyWM()
        target = PolicyTargetWorldModel(nominal, ema_beta=0.5)
        self.assertFalse(target.enabled)
        self.assertTrue(all(not p.requires_grad for p in target.model.parameters()))
        with torch.no_grad():
            nominal.linear.weight.add_(1.0)
            nominal.running.fill_(3.0)
        target.sync_from(nominal)
        self.assertTrue(torch.equal(target.model.linear.weight, nominal.linear.weight))
        self.assertTrue(torch.equal(target.model.running, nominal.running))

    def test_ema_and_audit_gate(self):
        nominal = _TinyWM()
        target = PolicyTargetWorldModel(nominal, ema_beta=0.5)
        old = target.model.linear.weight.detach().clone()
        with torch.no_grad():
            nominal.linear.weight.add_(2.0)
        target.ema_from(nominal)
        self.assertTrue(torch.allclose(target.model.linear.weight, 0.5 * old + 0.5 * nominal.linear.weight))
        self.assertFalse(target.audit_gate(torch.tensor([0.3, 0.5]), max_error=0.2))
        self.assertTrue(target.audit_gate(torch.tensor([0.1, 0.2]), max_error=0.2))
        self.assertAlmostEqual(float(target.audit_error), 0.15, places=6)

    def test_zero_initialized_adapter_is_nominal_and_is_not_ema_overwritten(self):
        nominal = _TinyWM()
        target = PolicyTargetWorldModel(nominal, ema_beta=0.5, adapter_hidden=4)
        x = torch.randn(5, 3)
        self.assertTrue(torch.allclose(target(x), nominal(x)))
        self.assertTrue(target.adapter_parameters())
        self.assertTrue(all(p.requires_grad for p in target.adapter_parameters()))
        self.assertTrue(all(not p.requires_grad for p in target.model.parameters()))
        with torch.no_grad():
            target.adapter[-1].bias.fill_(1.0)
            before = target.adapter[-1].bias.clone()
            nominal.linear.weight.add_(2.0)
        target.ema_from(nominal)
        self.assertTrue(torch.equal(target.adapter[-1].bias, before))
        self.assertGreater(float(target.adapter_norm().detach()), 0.0)

    def test_adapter_snapshot_restore_is_transactional(self):
        nominal = _TinyWM()
        target = PolicyTargetWorldModel(nominal, adapter_hidden=4)
        token = target.snapshot_adapter()
        with torch.no_grad():
            target.adapter[-1].bias.fill_(2.0)
        changed = {k: v.detach().clone() for k, v in target.adapter.state_dict().items()}
        self.assertFalse(all(torch.equal(changed[k], token[k]) for k in token))
        target.restore_adapter(token)
        self.assertTrue(all(torch.equal(target.adapter.state_dict()[k], token[k]) for k in token))
        with self.assertRaises(ValueError):
            target.restore_adapter({"wrong": torch.zeros(1)})


if __name__ == "__main__":
    unittest.main()
