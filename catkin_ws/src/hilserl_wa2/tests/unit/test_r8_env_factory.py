"""Unit tests for R8 env factory (needs serl_launcher; no ROS)."""

from __future__ import annotations

import pathlib
import sys
import unittest
import warnings

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT / "hil-serl-main" / "serl_launcher"))
sys.path.insert(0, str(SRC_ROOT / "hil-serl-main" / "examples"))

try:
    import serl_launcher  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"serl_launcher not available: {exc}") from exc

from hilserl_wa2.experiments.env_factory import (  # noqa: E402
    assert_fake_env_isolated,
    build_space_signature,
    make_wa2_environment,
    wrapper_names,
)
from hilserl_wa2.experiments.task_config import load_task  # noqa: E402


class EnvFactoryTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task("bottle_pick")

    def test_fake_env_shapes_and_isolation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = make_wa2_environment(
                self.task, fake_env=True, save_video=True, classifier=False
            )
        try:
            report = assert_fake_env_isolated(env)
            self.assertFalse(report["hardware_touched"])
            self.assertNotIn("WA2SpacemouseIntervention", wrapper_names(env))
            obs, _ = env.reset(seed=0)
            self.assertEqual(set(obs), {"state", "head", "wrist"})
            self.assertEqual(obs["state"].shape, (1, 27))
            self.assertEqual(obs["head"].shape, (1, 128, 128, 3))
            self.assertEqual(obs["wrist"].shape, (1, 128, 128, 3))
            self.assertEqual(env.action_space.shape, (6,))
            self.assertTrue(env.observation_space.contains(obs))
            obs2, r, term, trunc, info = env.step(np.zeros(6, np.float32))
            self.assertEqual(float(r), 0.0)
            self.assertTrue(env.observation_space.contains(obs2))
        finally:
            env.close()

    def test_actor_learner_space_match(self):
        actor = build_space_signature(self.task, "actor")
        learner = build_space_signature(self.task, "learner")
        self.assertEqual(actor["space_hash"], learner["space_hash"])
        self.assertTrue(actor["intervention_wrapped"])
        self.assertFalse(learner["intervention_wrapped"])

    def test_alt_task_same_space_hash(self):
        alt = load_task("r8_mock_alt")
        a = build_space_signature(self.task, "learner")
        b = build_space_signature(alt, "learner")
        self.assertEqual(a["space_hash"], b["space_hash"])

    def test_space_hash_matches_r12_freeze(self):
        from hilserl_wa2.experiments.classifier_io import FROZEN_SPACE_HASH

        learner = build_space_signature(self.task, "learner")
        self.assertEqual(learner["space_hash"], FROZEN_SPACE_HASH)

    def test_classifier_true_without_ckpt_fails(self):
        with self.assertRaises(ValueError):
            make_wa2_environment(self.task, fake_env=True, classifier=True)


if __name__ == "__main__":
    unittest.main()
