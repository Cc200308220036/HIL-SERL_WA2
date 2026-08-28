"""Unit tests for WA2SpacemouseIntervention (synthetic Joy, no robot)."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.interventions.joy_watchdog import JoyWatchdog  # noqa: E402
from hilserl_wa2.interventions.spacemouse_input import SpaceMouseInputConfig  # noqa: E402
from hilserl_wa2.interventions.wa2_spacemouse_intervention import (  # noqa: E402
    WA2SpacemouseIntervention,
)
from hilserl_wa2.tests.unit.test_spacemouse_input import SAMPLES  # noqa: E402


def _wrap(env=None):
    joy = JoyWatchdog(max_age_s=0.2)
    base = env or WA2Env(fake_env=True, seed=0)
    cfg = SpaceMouseInputConfig(
        translation_filter_tau=0.0,
        rotation_filter_tau=0.0,
    )
    wrapped = WA2SpacemouseIntervention(
        base,
        joy_watchdog=joy,
        auto_start_ros=False,
        input_config=cfg,
        intervene_eps=1e-3,
    )
    return wrapped, joy


class InterventionTests(unittest.TestCase):
    def test_no_import_franka_expert(self):
        import hilserl_wa2.interventions.wa2_spacemouse_intervention as mod
        src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from franka_env", src)
        self.assertNotIn("import franka_env", src)
        self.assertNotIn("naviai_controller", src)
        self.assertNotIn("SpaceMouseExpert()", src)

    def test_passthrough_without_deadman(self):
        env, joy = _wrap()
        env.reset()
        policy = np.asarray([0.5, 0, 0, 0, 0, 0], dtype=np.float32)
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 0])
        obs, r, term, trunc, info = env.step(policy)
        self.assertFalse(info.get("intervened"))
        self.assertNotIn("intervene_action", info)
        env.close()

    def test_intervene_with_deadman(self):
        env, joy = _wrap()
        env.reset()
        policy = np.zeros(6, dtype=np.float32)
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
        obs, r, term, trunc, info = env.step(policy)
        self.assertTrue(info.get("intervened"))
        self.assertIn("intervene_action", info)
        ia = np.asarray(info["intervene_action"], dtype=np.float32)
        self.assertEqual(ia.shape, (6,))
        self.assertGreater(float(ia[0]), 0.0)
        self.assertEqual(int(info["intervention_steps"]), 1)
        self.assertEqual(int(info["intervention_count"]), 1)
        env.close()

    def test_twelve_directions_with_deadman(self):
        cases = [
            ("forward_translation", 0, +1),
            ("backward_translation", 0, -1),
            ("left_translation", 1, +1),
            ("right_translation", 1, -1),
            ("up_translation", 2, +1),
            ("down_translation", 2, -1),
            ("left_tilt", 3, +1),
            ("right_tilt", 3, -1),
            ("forward_tilt", 4, +1),
            ("backward_tilt", 4, -1),
            ("clockwise_twist", 5, +1),
            ("counterclockwise_twist", 5, -1),
        ]
        env, joy = _wrap()
        env.reset()
        policy = np.zeros(6, dtype=np.float32)
        for name, axis, sign in cases:
            joy.inject(SAMPLES[name], buttons=[0, 1])
            # fresh processor state between samples
            env.processor.reset()
            _, _, _, _, info = env.step(policy)
            self.assertTrue(info.get("intervened"), name)
            ia = np.asarray(info["intervene_action"])
            self.assertGreater(sign * float(ia[axis]), 0.0, name)
            self.assertEqual(np.count_nonzero(np.abs(ia) > 1e-12), 1, name)
        env.close()

    def test_stale_joy_no_intervene(self):
        env, joy = _wrap()
        env.reset()
        policy = np.asarray([0.2, 0, 0, 0, 0, 0], dtype=np.float32)
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
        joy.inject_stale_for_test(age_s=1.0)
        _, _, _, _, info = env.step(policy)
        self.assertFalse(info.get("intervened"))
        self.assertNotIn("intervene_action", info)
        env.close()

    def test_intervene_action_equals_executed(self):
        env, joy = _wrap()
        env.reset()
        policy = np.ones(6, dtype=np.float32) * 0.3
        joy.inject(SAMPLES["up_translation"], buttons=[0, 1])
        env.processor.reset()
        exec_a, intervened = env.action(policy)
        self.assertTrue(intervened)
        _, _, _, _, info = env.step(policy)
        np.testing.assert_allclose(info["intervene_action"], exec_a, atol=1e-6)
        env.close()

    def test_buffer_harness_counts(self):
        env, joy = _wrap()
        env.reset()
        policy = np.zeros(6, dtype=np.float32)
        env_steps = 0
        intvn_steps = 0
        # 2 passthrough
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 0])
        for _ in range(2):
            _, _, _, _, info = env.step(policy)
            if "intervene_action" in info:
                intvn_steps += 1
            else:
                env_steps += 1
        # 3 intervene (one segment)
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
        env.processor.reset()
        for _ in range(3):
            _, _, _, _, info = env.step(policy)
            if "intervene_action" in info:
                intvn_steps += 1
            else:
                env_steps += 1
        self.assertEqual(env_steps, 2)
        self.assertEqual(intvn_steps, 3)
        self.assertEqual(info["intervention_steps"], 3)
        self.assertEqual(info["intervention_count"], 1)
        env.close()

    def test_toggle_stays_on_after_left_release(self):
        env, joy = _wrap()
        env.reset()
        policy = np.zeros(6, dtype=np.float32)
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
        env.processor.reset()
        _, _, _, _, info = env.step(policy)
        self.assertTrue(info.get("sm_session_enter"))
        self.assertTrue(info.get("sm_session"))
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 0])
        env.processor.reset()
        _, _, _, _, info = env.step(policy)
        self.assertTrue(info.get("sm_session"))
        self.assertTrue(info.get("intervened"))
        self.assertFalse(info.get("sm_session_exit"))
        env.close()

    def test_toggle_second_left_exits(self):
        env, joy = _wrap()
        env.reset()
        policy = np.zeros(6, dtype=np.float32)
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
        env.step(policy)
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 0])
        env.step(policy)
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
        _, _, _, _, info = env.step(policy)
        self.assertTrue(info.get("sm_session_exit"))
        self.assertFalse(info.get("sm_session"))
        self.assertFalse(info.get("intervened"))
        env.close()


if __name__ == "__main__":
    unittest.main()
