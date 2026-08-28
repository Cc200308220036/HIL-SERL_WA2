"""Unit tests for StateCache / WA2StateMonitor (ROS-free)."""

from __future__ import annotations

import pathlib
import sys
import time
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.ros_adapters.state_monitor import (  # noqa: E402
    LEFT_JOINT_NAMES,
    StateCache,
    WA2StateMonitor,
)


class StateCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = StateCache(state_max_age_s=0.2)

    def _fill(self):
        self.cache.update_tcp_pose([0.1, 0.2, 0.3, 0, 0, 0, 1])
        self.cache.update_tcp_vel([0, 0, 0, 0, 0, 0])
        self.cache.update_joint_pos(np.zeros(8))
        self.cache.update_hand_joints(np.zeros(6))
        self.cache.update_uplimb_state(
            is_singular=False, cmd_num=1, cmd_name="IDLE", iddp_status=True
        )

    def test_shapes(self):
        self._fill()
        st = self.cache.get_state()
        self.assertEqual(st["tcp_pose"].shape, (7,))
        self.assertEqual(st["tcp_vel"].shape, (6,))
        self.assertEqual(st["joint_pos"].shape, (8,))
        self.assertEqual(st["hand_joints"].shape, (6,))

    def test_get_state_returns_copy(self):
        self._fill()
        st = self.cache.get_state()
        st["tcp_pose"][0] += 10.0
        st2 = self.cache.get_state()
        self.assertAlmostEqual(float(st2["tcp_pose"][0]), 0.1, places=5)

    def test_ages_increase_when_frozen(self):
        now = time.monotonic()
        self.cache.update_tcp_pose([0, 0, 0, 0, 0, 0, 1], stamp=now - 0.01)
        self.cache.update_tcp_vel(np.zeros(6), stamp=now - 0.01)
        self.cache.update_joint_pos(np.zeros(8), stamp=now - 0.01)
        self.cache.update_hand_joints(np.zeros(6), stamp=now - 0.01)
        self.cache.update_uplimb_state(
            is_singular=False,
            cmd_num=0,
            cmd_name="X",
            iddp_status=True,
            stamp=now - 0.01,
        )
        age0 = self.cache.get_ages()["tcp_pose"]
        time.sleep(0.05)
        age1 = self.cache.get_ages()["tcp_pose"]
        self.assertGreater(age1, age0)

    def test_stale_over_threshold(self):
        self._fill()
        self.cache.inject_stale_for_test(fields=["tcp_pose"], age_s=1.0)
        self.assertFalse(self.cache.is_fresh())
        self.assertIn("tcp_pose", self.cache.stale_fields())

    def test_monitor_joint_names_left_only(self):
        mon = WA2StateMonitor(arm="left")
        names = mon.left_joint_names()
        self.assertEqual(len(names), 8)
        self.assertEqual(names, LEFT_JOINT_NAMES)
        self.assertTrue(all(n.endswith("_L") for n in names))


class ReadOnlyEnvWithInjectedMonitorTests(unittest.TestCase):
    def test_readonly_step_marks_stale(self):
        from hilserl_wa2.envs.wa2_env import WA2Env

        cache = StateCache(state_max_age_s=0.2)
        cache.update_tcp_pose([0, 0.2, 0.3, 0, 0, 0, 1])
        cache.update_tcp_vel(np.zeros(6))
        cache.update_joint_pos(np.zeros(8))
        cache.update_hand_joints(np.zeros(6))
        cache.update_uplimb_state(
            is_singular=False, cmd_num=0, cmd_name="STOPPED", iddp_status=True
        )
        mon = WA2StateMonitor(arm="left", cache=cache)
        mon._started = True  # skip ROS start

        env = WA2Env(fake_env=False, read_only=True, state_monitor=mon, seed=0)
        obs, info = env.reset(options={"ready_timeout_s": 1.0})
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(info["read_only"])
        self.assertFalse(info["fake_env"])

        mon.inject_stale_for_test(fields=["tcp_pose"], age_s=1.0)
        obs, reward, terminated, truncated, info = env.step(
            np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32)
        )
        self.assertTrue(truncated)
        self.assertTrue(info["stale"])
        self.assertIn("tcp_pose", info["stale_fields"])
        self.assertTrue(info.get("action_ignored_for_motion"))
        env.close()


if __name__ == "__main__":
    unittest.main()
