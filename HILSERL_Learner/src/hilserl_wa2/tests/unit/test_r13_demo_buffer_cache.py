"""Demo buffer disk cache roundtrip. No ROS / JAX."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from r13_learner_train import save_demo_buffer, try_load_demo_buffer  # noqa: E402


class _FakeBuffer:
    def __init__(self, *, filled: bool):
        cap = 8
        n = 3 if filled else 0
        self._capacity = cap
        self._size = n
        self._insert_index = n
        self._first = True
        self._num_stack = 1
        self.pixel_keys = ("head",)
        self._is_correct_index = np.zeros(cap, dtype=bool)
        if filled:
            self._is_correct_index[:n] = True
            self._is_correct_index[0] = False
        self.dataset_dict = {
            "actions": np.arange(cap * 7, dtype=np.float32).reshape(cap, 7),
            "rewards": np.arange(cap, dtype=np.float32),
            "observations": {
                "head": np.arange(cap * 4, dtype=np.uint8).reshape(cap, 2, 2),
                "state": np.arange(cap * 2, dtype=np.float32).reshape(cap, 2),
            },
        }

    def __len__(self):
        return self._size


class DemoBufferCacheTests(unittest.TestCase):
    def test_save_load_restores_prefix_and_flags(self):
        src = _FakeBuffer(filled=True)
        dst = _FakeBuffer(filled=False)
        dst.dataset_dict["actions"][:] = 0
        dst.dataset_dict["rewards"][:] = 0
        dst.dataset_dict["observations"]["head"][:] = 0
        dst.dataset_dict["observations"]["state"][:] = 0
        with tempfile.TemporaryDirectory() as tmp:
            cache = pathlib.Path(tmp) / "demo_buffer_cache"
            save_demo_buffer(
                cache,
                src,
                demo_n=2,
                n_grasp=4,
                demo_pkl_sha256="abc",
            )
            miss = try_load_demo_buffer(cache, dst, demo_pkl_sha256="nope")
            self.assertIsNone(miss)
            loaded = try_load_demo_buffer(cache, dst, demo_pkl_sha256="abc")
            self.assertEqual(loaded, (2, 4))
        self.assertEqual(int(dst._size), 3)
        self.assertEqual(int(dst._insert_index), 3)
        self.assertTrue(bool(dst._first))
        np.testing.assert_array_equal(dst.dataset_dict["actions"][:3], src.dataset_dict["actions"][:3])
        np.testing.assert_array_equal(
            dst.dataset_dict["observations"]["head"][:3],
            src.dataset_dict["observations"]["head"][:3],
        )
        np.testing.assert_array_equal(dst._is_correct_index[:3], src._is_correct_index[:3])
        self.assertFalse(bool(dst._is_correct_index[3]))


if __name__ == "__main__":
    unittest.main()
