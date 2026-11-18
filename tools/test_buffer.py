import unittest

import torch

from supermimic.buffer import FIELDS, ReplayBuffer


class ReplayBufferTest(unittest.TestCase):
    def test_short_buffer_has_clear_error(self):
        buffer = ReplayBuffer(T=16, N=2, device="cpu")
        with self.assertRaisesRegex(ValueError, "8-step windows from 0 buffered steps"):
            buffer.sample_windows(B=1, k=8)

    def test_exact_window_length_is_valid(self):
        buffer = ReplayBuffer(T=8, N=2, device="cpu")
        transition = {}
        for key, (shape, dtype) in FIELDS.items():
            full_shape = (2,) + ((shape,) if isinstance(shape, int) else shape)
            transition[key] = torch.zeros(full_shape, dtype=dtype)
        for _ in range(8):
            buffer.add(transition)

        window = buffer.sample_windows(B=3, k=8)

        self.assertEqual(window["s"].shape, (3, 8, 71))
        self.assertEqual(window["mask"].shape, (3, 8))
        self.assertTrue(window["mask"].all())
        self.assertNotIn("window_age", window)
        self.assertNotIn("window_env", window)

    def test_env_restricted_sampling(self):
        buffer = ReplayBuffer(T=8, N=4, device="cpu")
        transition = {}
        for key, (shape, dtype) in FIELDS.items():
            full_shape = (4,) + ((shape,) if isinstance(shape, int) else shape)
            transition[key] = torch.zeros(full_shape, dtype=dtype)
        transition["f"] = torch.arange(4, dtype=torch.long)
        for _ in range(8):
            buffer.add(transition)
        window = buffer.sample_windows(B=256, k=2, env_ids=torch.tensor([2, 3]), return_indices=True)
        self.assertTrue(torch.isin(window["window_env"], torch.tensor([2, 3])).all())
        self.assertTrue(torch.isin(window["f"].flatten(), torch.tensor([2, 3])).all())

    def test_prioritized_sampling_returns_indices_and_updates(self):
        buffer = ReplayBuffer(T=8, N=2, device="cpu")
        transition = {}
        for key, (shape, dtype) in FIELDS.items():
            full_shape = (2,) + ((shape,) if isinstance(shape, int) else shape)
            transition[key] = torch.zeros(full_shape, dtype=dtype)
        for _ in range(8):
            buffer.add(transition)
        age = torch.arange(8)
        buffer.update_priorities(age, torch.arange(1, 9, dtype=torch.float32))
        window = buffer.sample_windows(B=10000, k=1, prioritized=True, return_indices=True)
        self.assertEqual(window["window_age"].shape, (10000,))
        self.assertEqual(window["window_env"].shape, (10000,))
        self.assertTrue((window["window_age"] >= 0).all())
        self.assertTrue((window["window_age"] < 8).all())
        # Priorities 1..8 should move the sampled mean above uniform's 3.5.
        self.assertGreater(float(window["window_age"].float().mean()), 4.5)

    def test_priorities_follow_circular_age_mapping(self):
        buffer = ReplayBuffer(T=4, N=1, device="cpu")
        transition = {}
        for key, (shape, dtype) in FIELDS.items():
            full_shape = (1,) + ((shape,) if isinstance(shape, int) else shape)
            transition[key] = torch.zeros(full_shape, dtype=dtype)
        for _ in range(6):
            buffer.add(transition)
        # The live age range is 0..3 after wrap-around; each age must map to
        # the corresponding current storage row.
        ages = torch.arange(4)
        values = torch.tensor([11.0, 22.0, 33.0, 44.0])
        buffer.update_priorities(ages, values)
        rows = buffer._row(ages)
        self.assertTrue(torch.equal(buffer.priority[rows], values))

    def test_priority_values_are_bounded(self):
        buffer = ReplayBuffer(T=2, N=1, device="cpu")
        transition = {}
        for key, (shape, dtype) in FIELDS.items():
            full_shape = (1,) + ((shape,) if isinstance(shape, int) else shape)
            transition[key] = torch.zeros(full_shape, dtype=dtype)
        buffer.add(transition), buffer.add(transition)
        buffer.update_priorities(torch.tensor([0, 1]), torch.tensor([-3.0, 2e9]))
        self.assertAlmostEqual(float(buffer.priority[0]), 1e-6, places=12)
        self.assertAlmostEqual(float(buffer.priority[1]), 1e6, places=3)

    def test_state_round_trip_preserves_ring_and_data(self):
        buffer = ReplayBuffer(T=4, N=1, device="cpu")
        for i in range(6):
            transition = {}
            for key, (shape, dtype) in FIELDS.items():
                full_shape = (1,) + ((shape,) if isinstance(shape, int) else shape)
                transition[key] = torch.full(full_shape, i, dtype=dtype)
            buffer.add(transition)
        buffer.update_priorities(torch.arange(4), torch.tensor([2., 3., 5., 7.]))
        restored = ReplayBuffer(T=4, N=1, device="cpu")
        restored.load_state_dict(buffer.state_dict())
        self.assertEqual(restored.ptr, buffer.ptr)
        self.assertEqual(restored.size, buffer.size)
        self.assertTrue(torch.equal(restored.priority, buffer.priority))
        for key in FIELDS:
            self.assertTrue(torch.equal(restored.data[key], buffer.data[key]))


if __name__ == "__main__":
    unittest.main()
