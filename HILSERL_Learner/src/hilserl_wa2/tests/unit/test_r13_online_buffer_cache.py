"""Online replay buffer disk cache roundtrip. No ROS / JAX."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from r13_learner_train import (  # noqa: E402
    save_demo_buffer,
    save_replay_buffer,
    try_load_demo_buffer,
    try_load_replay_buffer,
)


class _FakeBuffer:
    def __init__(self, *, filled: bool, n: int = 3):
        cap = 8
        self._capacity = cap
        self._size = n if filled else 0
        self._insert_index = self._size
        self._first = True
        self._num_stack = 1
        self.pixel_keys = ("head",)
        self._is_correct_index = np.zeros(cap, dtype=bool)
        if filled:
            self._is_correct_index[: self._size] = True
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


class OnlineBufferCacheTests(unittest.TestCase):
    def test_online_save_load(self):
        src = _FakeBuffer(filled=True, n=4)
        dst = _FakeBuffer(filled=False)
        dst.dataset_dict["actions"][:] = 0
        with tempfile.TemporaryDirectory() as tmp:
            cache = pathlib.Path(tmp) / "online_buffer_cache"
            save_replay_buffer(
                cache,
                src,
                kind="online",
                extra={"learner_step": 21000, "task_id": "bottle_pick"},
                log_prefix="ONLINE_BUFFER_CACHE",
            )
            miss = try_load_replay_buffer(
                cache, dst, kind="demo", log_prefix="ONLINE_BUFFER_CACHE"
            )
            self.assertIsNone(miss)
            meta = try_load_replay_buffer(
                cache, dst, kind="online", log_prefix="ONLINE_BUFFER_CACHE"
            )
            self.assertIsNotNone(meta)
            self.assertEqual(int(meta["learner_step"]), 21000)
        self.assertEqual(int(dst._size), 4)
        np.testing.assert_array_equal(
            dst.dataset_dict["actions"][:4], src.dataset_dict["actions"][:4]
        )

    def test_expanded_demo_keeps_demo_n_baseline(self):
        """After intvn append, demo_n stays file baseline for INTVN_N."""

        src = _FakeBuffer(filled=True, n=5)
        dst = _FakeBuffer(filled=False)
        with tempfile.TemporaryDirectory() as tmp:
            cache = pathlib.Path(tmp) / "demo_buffer_cache"
            save_demo_buffer(
                cache,
                src,
                demo_n=2,
                n_grasp=4,
                demo_pkl_sha256="abc",
            )
            loaded = try_load_demo_buffer(cache, dst, demo_pkl_sha256="abc")
            self.assertEqual(loaded, (2, 4))
        self.assertEqual(int(dst._size), 5)
        self.assertEqual(max(0, int(dst._size) - int(loaded[0])), 3)


if __name__ == "__main__":
    unittest.main()
