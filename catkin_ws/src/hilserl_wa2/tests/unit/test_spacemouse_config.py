"""Unit tests for spacemouse YAML config loader."""

from __future__ import annotations

import pathlib
import sys
import unittest

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.interventions.spacemouse_config import (  # noqa: E402
    DEFAULT_SPACEMOUSE_YAML,
    load_spacemouse_config,
)
from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.interventions.joy_watchdog import JoyWatchdog  # noqa: E402
from hilserl_wa2.interventions.wa2_spacemouse_intervention import (  # noqa: E402
    WA2SpacemouseIntervention,
)


class SpaceMouseConfigTests(unittest.TestCase):
    def test_default_yaml_loads(self):
        cfg = load_spacemouse_config()
        self.assertTrue(DEFAULT_SPACEMOUSE_YAML.is_file())
        self.assertEqual(cfg.joy_topic, "/spacenav/joy")
        self.assertEqual(cfg.deadman_button, 1)
        self.assertEqual(cfg.session_mode, "toggle")
        self.assertEqual(cfg.input_config.axis_sign[0], -1.0)
        self.assertEqual(cfg.action_gain, 1.5)
        self.assertEqual(cfg.input_config.axis_switch_hysteresis, 0.25)
        self.assertEqual(cfg.input_config.secondary_axis_ratio, 0.90)
        self.assertIn("linear_scale", cfg.teleop_ros_params())
        self.assertEqual(cfg.teleop_ros_params()["secondary_axis_ratio"], 0.90)
        self.assertEqual(
            list(cfg.teleop_ros_params()["grasp_target"]),
            [0.1, 0.9, 0.7, 0.7, 0.4, 0.4],
        )

    def test_collect_yaml_faster_teleop(self):
        cfg = load_spacemouse_config("collect")
        self.assertEqual(cfg.path.name, "collect.yaml")
        self.assertEqual(cfg.action_gain, 1.5)
        params = cfg.teleop_ros_params()
        self.assertEqual(params["max_step_m"], 0.001)
        self.assertEqual(params["linear_scale"], 0.020)
        self.assertEqual(
            list(params["grasp_target"]),
            [0.1, 0.9, 0.7, 0.7, 0.4, 0.4],
        )
        self.assertEqual(
            list(params["release_target"]),
            [0.1, 0.9, 0.3, 0.3, 0.3, 0.3],
        )
        self.assertEqual(params["initial_hand_state"], "released")

    def test_stem_resolve(self):
        cfg = load_spacemouse_config("default")
        self.assertEqual(cfg.path.name, "default.yaml")

    def test_wrapper_loads_yaml(self):
        joy = JoyWatchdog(max_age_s=0.2)
        env = WA2SpacemouseIntervention(
            WA2Env(fake_env=True, seed=0),
            config_path="default",
            joy_watchdog=joy,
            auto_start_ros=False,
            input_config=None,  # force YAML input
        )
        # Override filters for deterministic step still ok via kwargs elsewhere
        self.assertEqual(env.deadman_button, 1)
        self.assertEqual(env.session_mode, "toggle")
        self.assertEqual(env.joy.topic, "/spacenav/joy")
        self.assertEqual(env.action_gain, 1.5)
        env.close()


if __name__ == "__main__":
    unittest.main()
