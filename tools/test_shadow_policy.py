import unittest

import torch
from torch import nn

from supermimic.shadow_policy import ShadowPolicyBranch, cap_parameter_step


class ShadowPolicyBranchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.policy = nn.Linear(4, 2)
        self.critic = nn.Linear(4, 1)
        self.branch = ShadowPolicyBranch(self.policy, self.critic)

    def test_shadow_update_does_not_touch_live(self):
        live_before = self.policy.weight.detach().clone()
        with torch.no_grad():
            self.branch.shadow_policy.weight.add_(1.0)
        self.assertTrue(torch.equal(self.policy.weight, live_before))

    def test_discard_restores_shadow_from_live(self):
        with torch.no_grad():
            self.branch.shadow_policy.weight.add_(1.0)
        self.branch.discard()
        self.assertTrue(torch.equal(self.branch.shadow_policy.weight, self.policy.weight))

    def test_promote_copies_shadow_to_live(self):
        with torch.no_grad():
            self.branch.shadow_policy.weight.add_(1.0)
            self.branch.shadow_critic.bias.add_(1.0)
        self.branch.promote()
        self.assertTrue(torch.equal(self.policy.weight, self.branch.shadow_policy.weight))
        self.assertTrue(torch.equal(self.critic.bias, self.branch.shadow_critic.bias))

    def test_state_round_trip_and_independent_optimizers(self):
        with torch.no_grad():
            self.branch.shadow_policy.bias.add_(2.0)
        state = self.branch.state_dict()
        other = ShadowPolicyBranch(nn.Linear(4, 2), nn.Linear(4, 1))
        other.load_state_dict(state)
        self.assertTrue(torch.equal(other.shadow_policy.bias, self.branch.shadow_policy.bias))
        opt_pi, opt_v = other.make_optimizers(1e-3)
        self.assertIsNot(opt_pi, opt_v)
        self.assertEqual(len(opt_pi.param_groups[0]["params"]), 2)

    def test_parameter_step_cap(self):
        before = [p.detach().clone() for p in self.branch.shadow_policy.parameters()]
        with torch.no_grad():
            self.branch.shadow_policy.weight.add_(10.0)
        applied = cap_parameter_step(self.branch.shadow_policy, before, 0.5)
        self.assertLessEqual(applied, 0.5 + 1e-6)
        delta = torch.sqrt(sum((p - b).square().sum() for p, b in
                               zip(self.branch.shadow_policy.parameters(), before)))
        self.assertLessEqual(float(delta), 0.5 + 1e-6)
