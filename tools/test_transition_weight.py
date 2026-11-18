import unittest

import torch

from supermimic.transition_weight import TransitionOdds


class TransitionWeightTests(unittest.TestCase):
    def _checkpoint(self, dim=5):
        model = torch.nn.Sequential(torch.nn.Linear(dim, 4), torch.nn.SiLU(),
                                    torch.nn.Linear(4, 1))
        return {"model": model.state_dict(), "input_dim": dim, "hidden": 4,
                "mean": torch.zeros(dim), "std": torch.ones(dim),
                "temperature": 1.0}

    def test_weights_are_detached_and_bounded(self):
        odds = TransitionOdds(self._checkpoint())
        x = torch.randn(17, 5, requires_grad=True)
        w = odds.weights(x, scale=3.0)
        self.assertEqual(tuple(w.shape), (17,))
        self.assertFalse(w.requires_grad)
        self.assertGreaterEqual(float(w.min()), 0.25)
        self.assertLessEqual(float(w.max()), 4.0)
        self.assertTrue(torch.isfinite(w).all())

    def test_dimension_and_temperature_validation(self):
        odds = TransitionOdds(self._checkpoint())
        with self.assertRaises(ValueError):
            odds.weights(torch.zeros(2, 4))
        ck = self._checkpoint(); ck["temperature"] = 0.0
        with self.assertRaises(ValueError):
            TransitionOdds(ck)


if __name__ == "__main__":
    unittest.main()
