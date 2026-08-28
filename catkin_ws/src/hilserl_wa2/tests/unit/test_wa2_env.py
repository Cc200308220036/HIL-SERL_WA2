"""Unit tests for Mock WA2Env (R2)."""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402

CONTRACT_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "configs" / "wa2_env_contract.yaml"
)


class WA2EnvTests(unittest.TestCase):
    def setUp(self):
        self.env = WA2Env(fake_env=True, contract_path=CONTRACT_PATH, seed=0)

    def tearDown(self):
        self.env.close()

    def test_reset_obs_in_space(self):
        obs, info = self.env.reset(seed=0)
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertIn("step_count", info)

    def test_step_five_tuple(self):
        self.env.reset(seed=0)
        out = self.env.step(np.zeros(6, dtype=np.float32))
        self.assertEqual(len(out), 5)
        obs, reward, terminated, truncated, info = out
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(float(reward), 0.0)
        self.assertIsInstance(terminated, (bool, np.bool_))
        self.assertIsInstance(truncated, (bool, np.bool_))
        self.assertIsInstance(info, dict)

    def test_action_clip_physical_scale(self):
        self.env.reset(seed=0)
        obs0, _ = self.env.reset(seed=0)
        p0 = obs0["state"]["tcp_pose"][:3].copy()
        # One high-level step repeats the clipped action for five Servo ticks.
        obs, _, _, _, info = self.env.step(
            np.asarray([1.5, 0, 0, 0, 0, 0], dtype=np.float32)
        )
        delta = float(np.linalg.norm(obs["state"]["tcp_pose"][:3] - p0))
        self.assertAlmostEqual(delta, 0.005, places=6)
        self.assertAlmostEqual(info["delta_pos_m"], 0.005, places=6)
        self.assertEqual(info["servo_ticks_requested"], 5)
        self.assertEqual(info["servo_ticks_executed"], 5)

    def test_rotation_physical_limit(self):
        self.env.reset(seed=0)
        _, _, _, _, info = self.env.step(
            np.asarray([0, 0, 0, 1, 0, 0], dtype=np.float32)
        )
        self.assertLessEqual(
            info["delta_rot_rad"], math.radians(1.25) + 1e-9
        )

    def test_action_window_can_be_interrupted_before_third_tick(self):
        self.env.reset(seed=0)
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] >= 3

        self.env.set_action_interrupt_callback(cancel)
        obs, _, _, _, info = self.env.step(
            np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32)
        )
        self.assertEqual(info["servo_ticks_requested"], 5)
        self.assertEqual(info["servo_ticks_executed"], 2)
        self.assertEqual(info["interrupted_by"], "intervention")
        self.assertAlmostEqual(float(obs["state"]["tcp_pose"][0]), 0.002, places=5)

    def test_reject_nan_and_bad_shape(self):
        self.env.reset(seed=0)
        with self.assertRaises(ValueError):
            self.env.step(np.asarray([np.nan] * 6, dtype=np.float32))
        with self.assertRaises(ValueError):
            self.env.step(np.asarray([1, 2, 3], dtype=np.float32))
        with self.assertRaises(ValueError):
            self.env.step(np.asarray([np.inf] + [0] * 5, dtype=np.float32))

    def test_max_steps_truncated(self):
        self.env.max_steps = 3
        self.env.reset(seed=0)
        truncated = False
        for _ in range(3):
            _, _, _, truncated, _ = self.env.step(np.zeros(6, dtype=np.float32))
        self.assertTrue(truncated)

    def test_inject_terminated(self):
        self.env.reset(seed=0)
        self.env.inject_success()
        _, _, terminated, _, _ = self.env.step(np.zeros(6, dtype=np.float32))
        self.assertTrue(terminated)

    def test_close_idempotent(self):
        self.env.reset(seed=0)
        self.env.close()
        self.env.close()

    def test_request_hand_toggle_on_fake_env(self):
        self.env.reset(seed=0)
        grasped = self.env.request_hand("toggle")
        self.assertTrue(grasped["ok"])
        self.assertEqual(grasped["command"], "grasp")
        np.testing.assert_allclose(
            self.env._robot.hand_joints,
            np.asarray(self.env.grasp_target, dtype=np.float32),
        )
        released = self.env.request_hand("toggle")
        self.assertEqual(released["command"], "release")
        np.testing.assert_allclose(
            self.env._robot.hand_joints,
            np.asarray(self.env.release_target, dtype=np.float32),
        )
        with self.assertRaises(ValueError):
            self.env.request_hand("pinch")

    def test_images_dual_keys_and_wrist_zero(self):
        obs, _ = self.env.reset(seed=0)
        self.assertEqual(set(obs["images"].keys()), {"head", "wrist"})
        self.assertEqual(obs["images"]["head"].shape, (128, 128, 3))
        self.assertEqual(obs["images"]["wrist"].dtype, np.uint8)
        self.assertTrue(np.all(obs["images"]["wrist"] == 0))
        _, _, _, truncated, _ = self.env.step(np.zeros(6, dtype=np.float32))
        self.assertFalse(truncated)

    def test_seed_reproducible(self):
        def rollout(seed):
            env = WA2Env(fake_env=True, contract_path=CONTRACT_PATH, seed=seed)
            obs, _ = env.reset(seed=seed)
            poses = [obs["state"]["tcp_pose"].copy()]
            rng = np.random.default_rng(seed)
            for _ in range(15):
                action = rng.uniform(-1, 1, size=(6,)).astype(np.float32)
                obs, *_ = env.step(action)
                poses.append(obs["state"]["tcp_pose"].copy())
            env.close()
            return poses

        a = rollout(99)
        b = rollout(99)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y)

    def test_fake_env_true_required_for_now(self):
        # Motion backend is implemented in R4; this name kept for history.
        env = WA2Env(fake_env=False, read_only=False, dry_run=True, contract_path=CONTRACT_PATH)
        # Construction without live ROS uses injected monitor in other tests;
        # here just ensure dry_run motion path object exists when provided monitor.
        env.close()


if __name__ == "__main__":
    unittest.main()
