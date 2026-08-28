"""Read-only WA2Env unit tests (no live ROS required)."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.ros_adapters.state_monitor import StateCache, WA2StateMonitor  # noqa: E402

CONTRACT = (
    pathlib.Path(__file__).resolve().parents[2] / "configs" / "wa2_env_contract.yaml"
)


def _ready_monitor(singular: bool = True) -> WA2StateMonitor:
    cache = StateCache(state_max_age_s=0.2)
    cache.update_tcp_pose([0.0, 0.26, 0.38, 0, 0, 0, 1])
    cache.update_tcp_vel(np.zeros(6))
    cache.update_joint_pos(np.linspace(0.01, 0.08, 8))
    cache.update_hand_joints(np.linspace(0.1, 0.6, 6))
    cache.update_uplimb_state(
        is_singular=singular, cmd_num=0, cmd_name="STOPPED", iddp_status=True
    )
    mon = WA2StateMonitor(arm="left", cache=cache)
    mon._started = True
    return mon


class WA2EnvReadOnlyTests(unittest.TestCase):
    def test_motion_backend_dry_run_constructs(self):
        from hilserl_wa2.ros_adapters.servo_session import WA2ServoSession

        mon = _ready_monitor(singular=False)
        servo = WA2ServoSession(
            contract=__import__(
                "hilserl_wa2.envs.contracts", fromlist=["WA2EnvContract"]
            ).WA2EnvContract.from_yaml(CONTRACT),
            state_monitor=mon,
            dry_run=True,
        )
        env = WA2Env(
            fake_env=False,
            read_only=False,
            dry_run=True,
            contract_path=CONTRACT,
            state_monitor=mon,
            servo_session=servo,
            seed=0,
        )
        obs, info = env.reset()
        self.assertFalse(info["read_only"])
        self.assertTrue(info["dry_run"])
        obs, _, _, truncated, info = env.step(
            np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32)
        )
        self.assertFalse(truncated)
        self.assertAlmostEqual(info["delta_pos_m"], 0.005, places=6)
        self.assertEqual(info["servo_ticks_executed"], 5)
        self.assertFalse(info.get("published", True))
        env.close()

    def test_readonly_obs_and_info(self):
        env = WA2Env(
            fake_env=False,
            read_only=True,
            contract_path=CONTRACT,
            state_monitor=_ready_monitor(),
            seed=0,
        )
        obs, info = env.reset()
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(obs["state"]["joint_pos"].shape, (8,))
        self.assertTrue(np.all(obs["images"]["wrist"] == 0))
        self.assertTrue(info["read_only"])
        self.assertFalse(info["fake_env"])
        self.assertIn("is_singular", info)
        env.close()

    def test_nonzero_action_does_not_change_state(self):
        mon = _ready_monitor()
        env = WA2Env(
            fake_env=False,
            read_only=True,
            contract_path=CONTRACT,
            state_monitor=mon,
            seed=0,
        )
        obs0, _ = env.reset()
        p0 = obs0["state"]["tcp_pose"].copy()
        obs1, _, _, truncated, info = env.step(
            np.asarray([1, 1, 1, 1, 1, 1], dtype=np.float32)
        )
        self.assertFalse(truncated)
        np.testing.assert_allclose(obs1["state"]["tcp_pose"], p0)
        self.assertEqual(info["delta_pos_m"], 0.0)
        self.assertTrue(info["action_ignored_for_motion"])
        env.close()

    def test_fake_env_still_works(self):
        env = WA2Env(fake_env=True, contract_path=CONTRACT, seed=0)
        obs, info = env.reset(seed=0)
        self.assertTrue(info["fake_env"])
        self.assertTrue(env.observation_space.contains(obs))
        env.close()


if __name__ == "__main__":
    unittest.main()
