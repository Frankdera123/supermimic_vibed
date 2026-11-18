import unittest

import torch

from supermimic.strategic_probe import StrategicProbeBuffer


class StrategicProbeBufferTest(unittest.TestCase):
    def test_ring_storage_and_metadata(self):
        buf = StrategicProbeBuffer(2, "cpu")
        buf.add({"s": torch.tensor([[1.], [2.]]), "a": torch.tensor([[3.], [4.]])},
                torch.tensor([0.1, 0.2]), source=7)
        buf.add({"s": torch.tensor([[5.]]), "a": torch.tensor([[6.]])},
                torch.tensor([0.9]), source=torch.tensor([3]))
        self.assertEqual(buf.size, 2)
        self.assertEqual(buf.ptr, 1)
        sample = buf.sample(32)
        self.assertEqual(sample["s"].shape, (32, 1))
        self.assertTrue(torch.isin(sample["probe_source"], torch.tensor([3, 7])).all())
        self.assertTrue(torch.isin(sample["probe_score"], torch.tensor([0.2, 0.9])).all())

    def test_state_round_trip(self):
        buf = StrategicProbeBuffer(4, "cpu")
        tr = {"s": torch.arange(6, dtype=torch.float32).reshape(3, 2)}
        buf.add(tr, torch.tensor([1., 2., 3.]), source=torch.tensor([1, 2, 3]))
        restored = StrategicProbeBuffer(4, "cpu")
        restored.load_state_dict(buf.state_dict())
        self.assertEqual(restored.ptr, buf.ptr)
        self.assertEqual(restored.size, buf.size)
        self.assertTrue(torch.equal(restored.data["s"], buf.data["s"]))
        self.assertTrue(torch.equal(restored.score, buf.score))
        self.assertTrue(torch.equal(restored.source, buf.source))


if __name__ == "__main__":
    unittest.main()
