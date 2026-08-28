"""Unit tests for scene YAML + reset tolerance helpers."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.envs.scene_config import WA2SceneConfig, load_scene  # noqa: E402
from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.ros_adapters.reset_executor import (  # noqa: E402
    WA2ResetExecutor,
    check_tolerances,
    tcp_errors,
)

SCENE_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "configs"
    / "scenes"
    / "bottle_desktop.yaml"
)


class SceneConfigTests(unittest.TestCase):
    def test_load_bottle_desktop(self):
        scene = WA2SceneConfig.from_yaml(SCENE_PATH)
        self.assertEqual(scene.scene_id, "bottle_desktop")
        self.assertEqual(scene.home_joints_left.shape, (8,))
        self.assertEqual(scene.home_joints_neck.shape, (2,))
        self.assertEqual(scene.home_joints_waist.shape, (4,))
        self.assertEqual(scene.hand_reset.shape, (6,))
        self.assertEqual(scene.task_reset_tcp.shape, (7,))
        self.assertEqual(scene.waist_policy, "auto_movej")
        self.assertEqual(scene.neck_policy, "auto_movej")
        self.assertEqual(scene.max_steps, 600)
        self.assertIsNone(scene.episode_trans_limit_m)
        self.assertIsNone(scene.episode_rot_limit_deg)
        self.assertFalse(scene.do_movel_to_tcp)
        qn = float(np.linalg.norm(scene.task_reset_tcp[3:]))
        self.assertAlmostEqual(qn, 1.0, places=5)
        np.testing.assert_allclose(
            scene.home_joints_left[0],
            -0.30768299878218386,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scene.home_joints_neck,
            [-0.01114473619365148, 0.25312487271017603],
            rtol=0,
            atol=1e-12,
        )

    def test_load_by_name(self):
        scene = load_scene(scene_name="bottle_desktop")
        assert scene is not None
        self.assertEqual(scene.scene_id, "bottle_desktop")

    def test_tcp_errors_zero(self):
        scene = WA2SceneConfig.from_yaml(SCENE_PATH)
        e = tcp_errors(scene.task_reset_tcp, scene.task_reset_tcp)
        self.assertLess(e["tcp_pos_err_m"], 1e-9)
        self.assertLess(e["tcp_rot_err_rad"], 1e-9)

    def test_check_tolerances_pass_fail(self):
        scene = WA2SceneConfig.from_yaml(SCENE_PATH)
        ok = check_tolerances(
            scene,
            joint_pos=scene.home_joints_left,
            waist_joints=scene.home_joints_waist,
            neck_joints=scene.home_joints_neck,
            hand_joints=scene.hand_reset,
            tcp_pose=scene.task_reset_tcp,
            is_singular=False,
        )
        self.assertIsNone(ok)
        bad = check_tolerances(
            scene,
            joint_pos=scene.home_joints_left + 0.2,
            waist_joints=scene.home_joints_waist,
            neck_joints=scene.home_joints_neck,
            hand_joints=scene.hand_reset,
            tcp_pose=scene.task_reset_tcp,
            is_singular=False,
        )
        self.assertIsNotNone(bad)

    def test_fake_env_uses_scene_max_steps(self):
        env = WA2Env(fake_env=True, scene_name="bottle_desktop", seed=0)
        self.assertEqual(env.max_steps, 600)
        self.assertIsNone(env._episode_trans_limit_m)
        self.assertIsNone(env._episode_rot_limit_deg)
        obs, info = env.reset()
        self.assertTrue(info.get("reset_ok"))
        self.assertEqual(info.get("scene_id"), "bottle_desktop")
        self.assertEqual(obs["state"]["joint_pos"].shape, (8,))
        # Truncate at scene max_steps
        for _ in range(3):
            env.step(np.zeros(6, dtype=np.float32))
        self.assertEqual(env._step_count, 3)
        env.close()

    def test_dry_run_reset_executor(self):
        scene = WA2SceneConfig.from_yaml(SCENE_PATH)
        # Bypass human confirm via env in executor path used by unit test:
        ex = WA2ResetExecutor(
            scene=scene,
            dry_run=True,
            confirm_fn=lambda: True,
        )
        result = ex.run()
        self.assertTrue(result.ok)
        self.assertIn("hand_open", result.stages)
        self.assertIn("waist_movej", result.stages)
        self.assertIn("neck_movej", result.stages)
        self.assertIn("arm_movej", result.stages)
        self.assertIn("done", result.stages)


if __name__ == "__main__":
    unittest.main()
