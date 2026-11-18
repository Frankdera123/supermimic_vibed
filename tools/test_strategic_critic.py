import unittest

import torch

from supermimic.strategic_critic import StrategicErrorCritic


class StrategicErrorCriticTest(unittest.TestCase):
    def test_shapes_and_gradients(self):
        model = StrategicErrorCritic(hidden=16)
        s = torch.randn(5, 71)
        a = torch.randn(5, 29)
        xi = torch.randn(5, 33)
        x = model.features(s, a, xi)
        self.assertEqual(tuple(x.shape), (5, 133))
        y = model(x)
        self.assertEqual(tuple(y.shape), (5,))
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_normalization_is_finite(self):
        model = StrategicErrorCritic(hidden=8)
        x = model.features(torch.zeros(2, 71), torch.zeros(2, 29),
                           torch.zeros(2, 33), torch.ones(133), torch.zeros(133))
        self.assertTrue(torch.isfinite(x).all())


if __name__ == "__main__":
    unittest.main()
