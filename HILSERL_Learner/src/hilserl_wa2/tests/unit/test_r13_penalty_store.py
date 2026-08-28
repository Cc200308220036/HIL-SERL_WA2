"""PenaltyStore must add grasp_penalty on agentlace batch_insert. No ROS."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

import numpy as np

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from r13_learner_train import (  # noqa: E402
    CONTINUOUS_TARGET_ENTROPY,
    DEFAULT_MIN_TEMPERATURE,
    DEFAULT_RESUME_TEMPERATURE,
    GRASP_ACTION_PENALTY,
    SAC_TEMPERATURE_INIT,
    PenaltyStore,
    _find_temperature_lagrange,
    _replace_mapping_path,
    _refresh_buffer_grasp_penalties,
    _jsonable,
    _with_penalty,
    softplus_inv,
)


class _Inner:
    def __init__(self):
        self.rows = []

    def insert(self, data):
        self.rows.append(data)

    def batch_insert(self, batch_data):
        raise AssertionError("inner.batch_insert must not be used")


class PenaltyStoreTests(unittest.TestCase):
    def test_batch_insert_adds_grasp_penalty(self):
        inner = _Inner()
        store = PenaltyStore(inner, validate=lambda _: None)
        row = {
            "observations": {"state": np.zeros(1)},
            "actions": np.zeros(7, np.float32),
            "next_observations": {"state": np.zeros(1)},
            "rewards": np.float32(0.0),
            "masks": np.float32(1.0),
            "dones": np.bool_(False),
        }
        store.batch_insert([row, dict(row)])
        self.assertEqual(len(inner.rows), 2)
        for item in inner.rows:
            self.assertIn("grasp_penalty", item)
            self.assertEqual(float(item["grasp_penalty"]), 0.0)

    def test_nonzero_executed_grasp_gets_penalty(self):
        for command in (-1.0, 1.0):
            action = np.zeros(7, np.float32)
            action[-1] = command
            row = {"actions": action, "grasp_penalty": np.float32(0.0)}
            self.assertAlmostEqual(
                float(_with_penalty(row)["grasp_penalty"]),
                float(GRASP_ACTION_PENALTY),
            )

    def test_existing_penalty_is_recomputed(self):
        row = {
            "actions": np.zeros(7, np.float32),
            "grasp_penalty": np.float32(-1.0),
        }
        self.assertEqual(float(_with_penalty(row)["grasp_penalty"]), 0.0)

    def test_continuous_target_entropy_uses_six_dimensions(self):
        self.assertEqual(CONTINUOUS_TARGET_ENTROPY, -3.0)

    def test_default_grasp_penalty_is_soft(self):
        self.assertAlmostEqual(float(GRASP_ACTION_PENALTY), -0.002)

    def test_softplus_inv_roundtrip(self):
        for alpha in (1e-4, 1e-2, 5e-2, 1.0):
            raw = softplus_inv(alpha)
            # softplus(x) = log(1+exp(x))
            recovered = float(np.log1p(np.exp(raw))) if raw < 20 else raw
            # For positive alpha, softplus(softplus_inv(a)) ≈ a
            recovered = float(np.logaddexp(0.0, raw))
            self.assertAlmostEqual(recovered, alpha, places=5)

    def test_replace_mapping_path_preserves_frozendict(self):
        try:
            from flax.core.frozen_dict import FrozenDict
        except ImportError:
            self.skipTest("flax not installed")
        tree = FrozenDict(
            {
                "modules_temperature": FrozenDict({"lagrange": np.float32(-4.6)}),
                "modules_actor": FrozenDict({"w": np.float32(1.0)}),
            }
        )
        out = _replace_mapping_path(
            tree, ("modules_temperature", "lagrange"), np.float32(-2.97)
        )
        self.assertIsInstance(out, FrozenDict)
        self.assertIsInstance(out["modules_temperature"], FrozenDict)
        self.assertEqual(float(out["modules_actor"]["w"]), 1.0)
        self.assertAlmostEqual(float(out["modules_temperature"]["lagrange"]), -2.97, places=2)

    def test_find_temperature_lagrange_legacy_path(self):
        path = _find_temperature_lagrange({"temperature": {"lagrange": 0.0}})
        self.assertEqual(path, ("temperature", "lagrange"))

    def test_resume_reset_logic_targets(self):
        self.assertAlmostEqual(DEFAULT_MIN_TEMPERATURE, 0.05)
        self.assertGreaterEqual(DEFAULT_RESUME_TEMPERATURE, DEFAULT_MIN_TEMPERATURE)
        self.assertLess(SAC_TEMPERATURE_INIT, DEFAULT_MIN_TEMPERATURE)

    def test_cached_buffer_penalties_are_migrated(self):
        class _Buffer:
            dataset_dict = {
                "actions": np.asarray(
                    [
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, -1],
                    ],
                    dtype=np.float32,
                ),
                "grasp_penalty": np.zeros(3, dtype=np.float32),
            }

            def __len__(self):
                return 3

        buffer = _Buffer()
        _refresh_buffer_grasp_penalties(buffer)
        np.testing.assert_allclose(
            buffer.dataset_dict["grasp_penalty"],
            np.asarray([0.0, float(GRASP_ACTION_PENALTY), float(GRASP_ACTION_PENALTY)], dtype=np.float32),
        )

    def test_jsonable_nested_dict(self):
        payload = _jsonable({"critic": {"loss": np.float32(1.5)}, "ok": True})
        self.assertEqual(payload["critic"]["loss"], 1.5)
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
