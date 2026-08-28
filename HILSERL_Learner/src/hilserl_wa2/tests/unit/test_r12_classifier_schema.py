"""R12 classifier schema tests. Must not import rospy, recorder, or JAX."""

from __future__ import annotations

import inspect
import pathlib
import sys
import tempfile
import unittest

import numpy as np

import gymnasium as gym
from gymnasium import spaces

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.classifier_io import (  # noqa: E402
    FROZEN_SPACE_HASH,
    ClassifierIOError,
    binary_metrics,
    load_classifier_bundle,
    make_sample,
    split_by_episode,
    validate_no_episode_leakage,
    validate_sample,
    write_classifier_bundle,
)
from hilserl_wa2.experiments.task_config import load_task   # noqa: E402
from hilserl_wa2.wrappers.reward_classifier import (  # noqa: E402
    WA2RewardClassifierWrapper,
)


def _obs(fill: int = 7):
    return {
        "state": np.full((1, 27), fill, dtype=np.float32),
        "head": np.full((1, 128, 128, 3), fill, dtype=np.uint8),
        "wrist": np.full((1, 128, 128, 3), fill + 1, dtype=np.uint8),
    }


def _sample(episode_id: str, label: int, index: int = 0):
    return make_sample(
        episode_id=episode_id,
        label=label,
        index=index,
        created_at="2026-08-18T00:00:00Z",
        observations=_obs(index + 1),
    )


class _DummyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, (1, 27), np.float32),
                "head": spaces.Box(0, 255, (1, 128, 128, 3), np.uint8),
                "wrist": spaces.Box(0, 255, (1, 128, 128, 3), np.uint8),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.reset_calls = 0
        self.obs = _obs()

    def reset(self, **kwargs):
        self.reset_calls += 1
        return self.obs, {}

    def step(self, action):
        return self.obs, 0.0, False, False, {"servo": "ok"}


class R12ClassifierSchemaTests(unittest.TestCase):
    def test_module_stays_offline(self):
        import hilserl_wa2.experiments.classifier_io as classifier_io

        source = inspect.getsource(classifier_io)
        self.assertNotIn("rospy", source)
        self.assertNotIn("record_r12", source)
        self.assertNotIn("jax", source)
        self.assertNotIn("naviai_controller", source)

    def test_bottle_pick_keys_and_frozen_hash_constant(self):
        cfg = load_task("bottle_pick")
        self.assertEqual(cfg.classifier_keys, ("head", "wrist"))
        self.assertEqual(len(FROZEN_SPACE_HASH), 64)

    def test_sample_keys_are_r12_v1(self):
        sample = _sample("ep-a", 1)
        validate_sample(sample)
        self.assertEqual(
            set(sample.keys()),
            {"episode_id", "label", "index", "created_at", "observations"},
        )
        self.assertEqual(set(sample["observations"].keys()), {"head", "wrist"})

    def test_reject_r11_transition(self):
        row = {
            "observations": _obs(),
            "actions": np.zeros(6, dtype=np.float32),
            "next_observations": _obs(2),
            "rewards": np.float32(0.0),
            "masks": np.float32(1.0),
            "dones": np.bool_(False),
        }
        with self.assertRaises(ClassifierIOError):
            validate_sample(row)

    def test_refuse_demo_pkl_bundle(self):
        success = [_sample("ep-a", 1, 0), _sample("ep-b", 1, 0)]
        failure = [_sample("ep-a", 0, 1), _sample("ep-b", 0, 1), _sample("ep-c", 0, 0)]
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            bundle = root / "clf"
            write_classifier_bundle(
                bundle,
                bundle_name="unit",
                success=success,
                failure=failure,
                manifest_extra={
                    "task_id": "bottle_pick",
                    "exp_name": "wa2_bottle_pick",
                    "space_hash": FROZEN_SPACE_HASH,
                    "config_bundle_hash": "abc",
                    "operator": "unit",
                    "mode": "fake",
                },
                episode_sidecars=[
                    {"episode_id": "ep-a", "n_success": 1, "n_failure": 1},
                    {"episode_id": "ep-b", "n_success": 1, "n_failure": 1},
                    {"episode_id": "ep-c", "n_success": 0, "n_failure": 1},
                ],
            )
            packed = load_classifier_bundle(bundle)
            self.assertEqual(packed["manifest"]["n_success"], 2)
            self.assertTrue((bundle / "SHA256SUMS").is_file())
            (bundle / "demo.pkl").write_bytes(b"nope")
            with self.assertRaises(ClassifierIOError):
                load_classifier_bundle(bundle)

    def test_split_no_episode_leakage(self):
        samples = []
        for ep_i in range(6):
            samples.append(_sample(f"ep-{ep_i}", 1, 0))
            samples.append(_sample(f"ep-{ep_i}", 0, 1))
        packed = split_by_episode(samples, seed=12)
        validate_no_episode_leakage(packed["splits"])
        seen = {}
        for name, rows in packed["splits"].items():
            for row in rows:
                prev = seen.get(row["episode_id"])
                self.assertTrue(prev is None or prev == name)
                seen[row["episode_id"]] = name
        self.assertGreaterEqual(packed["counts"]["train"]["n_episodes"], 1)
        self.assertGreaterEqual(packed["counts"]["val"]["n_episodes"], 1)
        self.assertGreaterEqual(packed["counts"]["test"]["n_episodes"], 1)

    def test_metrics_confusion(self):
        metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_wrapper_never_sets_done_or_reset(self):
        env = _DummyEnv()

        def predict(_obs):
            return 0.99

        wrapped = WA2RewardClassifierWrapper(
            env, predict, threshold=0.5, consecutive_n=3, end_episode=False
        )
        obs, info = wrapped.reset()
        self.assertFalse(info["succeed"])
        terminated = False
        for _ in range(5):
            obs, reward, terminated, truncated, info = wrapped.step(np.zeros(6))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
        self.assertTrue(info["succeed"])
        self.assertEqual(float(reward), 1.0)
        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(wrapped.reset_calls, 1)
        # R13 may construct end_episode=True; R12 eval still passes False.

    def test_merge_retags_episode_ids(self):
        from hilserl_wa2.experiments.classifier_io import merge_classifier_bundles

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            a = root / "src_a"
            b = root / "src_b"
            out = root / "merged_out"
            for path, tag in ((a, "a"), (b, "b")):
                write_classifier_bundle(
                    path,
                    bundle_name=tag,
                    success=[_sample("live-ep-000", 1, 0)],
                    failure=[_sample("live-ep-000", 0, 1), _sample("live-ep-001", 0, 0)],
                    manifest_extra={
                        "task_id": "bottle_pick",
                        "exp_name": "wa2_bottle_pick",
                        "space_hash": FROZEN_SPACE_HASH,
                        "config_bundle_hash": "abc",
                        "operator": "unit",
                        "mode": "fake",
                    },
                    episode_sidecars=[
                        {"episode_id": "live-ep-000", "n_success": 1, "n_failure": 1},
                        {"episode_id": "live-ep-001", "n_success": 0, "n_failure": 1},
                    ],
                )
            merge_classifier_bundles(
                [a, b],
                out_dir=out,
                bundle_name="merged",
                operator="unit",
                mode="fake",
            )
            packed = load_classifier_bundle(out)
            self.assertEqual(len(packed["success"]), 2)
            self.assertEqual(len(packed["failure"]), 4)
            ids = {row["episode_id"] for row in packed["samples"]}
            self.assertIn("src_a__live-ep-000", ids)
            self.assertIn("src_b__live-ep-000", ids)
            self.assertEqual(packed["manifest"]["n_episodes"], 4)
            self.assertEqual(
                packed["manifest"]["merged_from"],
                ["src_a", "src_b"],
            )

    def test_select_threshold_prefers_stricter_when_precision_ok(self):
        from hilserl_wa2.experiments.classifier_io import select_threshold

        # Loose 0.5 catches a mid-score negative (FP). Mid 0.6 / 0.7 reject it.
        # Fβ=0.5 should prefer a stricter eligible threshold over plain 0.5.
        y = [1, 0, 1, 0]
        p = [0.95, 0.55, 0.92, 0.10]
        result = select_threshold(y, p, candidates=(0.5, 0.6, 0.7, 0.9))
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["chosen"]["threshold"], 0.6)
        self.assertGreaterEqual(result["chosen"]["precision"], 0.85)

    def test_select_threshold_fbeta_avoids_extreme_high_thr(self):
        from hilserl_wa2.experiments.classifier_io import select_threshold

        # Synthetic: 0.5 ok-ish; 0.6 best Fβ; 0.7 too few true positives.
        y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        # scores crafted so eligible set matches the user's val sweep shape
        p = [0.95, 0.90, 0.88, 0.85, 0.80, 0.75, 0.72, 0.65, 0.62, 0.55, 0.52, 0.40, 0.30, 0.20, 0.10]
        result = select_threshold(y, p)
        self.assertTrue(result["ok"])
        # Should not jump to the highest eligible threshold with collapsed recall.
        self.assertLessEqual(result["chosen"]["threshold"], 0.7)
        self.assertGreaterEqual(result["chosen"]["precision"], 0.85)


if __name__ == "__main__":
    unittest.main()
