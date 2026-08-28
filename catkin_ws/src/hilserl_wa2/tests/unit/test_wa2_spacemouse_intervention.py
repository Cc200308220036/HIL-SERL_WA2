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
        # No prior toggle session: stale Joy must not invent an intervention.
        self.assertFalse(info.get("intervened"))
        self.assertNotIn("intervene_action", info)
        env.close()

    def test_stale_during_toggle_session_holds_not_policy(self):
        env, joy = _wrap()
        env.reset()
        policy = np.ones(6, dtype=np.float32) * 0.8
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
        env.step(policy)
        joy.inject(np.zeros(6), buttons=[0, 0])
        env.step(policy)
        self.assertTrue(env._session_active)
        joy.inject_stale_for_test(age_s=1.0)
        exec_a, intervened = env.action(policy)
        self.assertTrue(intervened)
        self.assertTrue(env._session_active)
        np.testing.assert_allclose(exec_a, 0.0, atol=1e-6)
        _, _, _, _, info = env.step(policy)
        self.assertTrue(info.get("intervened"))
        self.assertTrue(info.get("sm_session"))
        self.assertTrue(info.get("sm_session_dropped_stale"))
        self.assertEqual(info.get("sm_intent"), "hold_stale")
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
        np.testing.assert_allclose(info["intervene_action"], exec_a, atol=1e-5)
        self.assertEqual(int(info.get("window_action_samples") or 0), 5)
        env.close()

    def test_human_provider_aggregates_changing_stick(self):
        env, joy = _wrap()
        env.reset()
        policy = np.zeros(6, dtype=np.float32)
        # Enter session.
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
        env.processor.reset()
        env.step(policy)
        # Hold session with released left; change stick mid high-level step via provider.
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 0])
        base = env._base_env()
        calls = {"n": 0}
        seq = [
            SAMPLES["forward_translation"],
            SAMPLES["forward_translation"],
            SAMPLES["backward_translation"],
            SAMPLES["backward_translation"],
            SAMPLES["backward_translation"],
        ]

        def inject_each_tick():
            joy.inject(seq[min(calls["n"], len(seq) - 1)], buttons=[0, 0])
            calls["n"] += 1
            return env._human_tick_command()

        # Force provider path through a direct base step while session active.
        env._session_active = True
        base.set_action_provider_callback(inject_each_tick)
        try:
            _, _, _, _, info = base.step(np.zeros(6, np.float32))
        finally:
            base.set_action_provider_callback(None)
        self.assertEqual(int(info["window_action_samples"]), 5)
        self.assertEqual(int(info["servo_ticks_executed"]), 5)
        # Mean x should be near 0 (2 forward + 3 backward with unit samples after processing).
        mean = np.asarray(info["window_action_mean"], dtype=np.float32)
        self.assertLess(abs(float(mean[0])), 1.0)
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

    def test_exit_while_left_held_does_not_immediately_reenter(self):
        """Regression: level-triggered policy interrupt re-armed a still-held exit tap."""

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
        # Same press still held across following high-level steps.
        for _ in range(3):
            joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
            _, _, _, _, info = env.step(policy)
            self.assertFalse(info.get("sm_session"), info)
            self.assertFalse(info.get("intervened"), info)
            self.assertFalse(info.get("sm_session_enter"), info)
        # Must release once before a new enter is accepted.
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 0])
        env.step(policy)
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
        _, _, _, _, info = env.step(policy)
        self.assertTrue(info.get("sm_session_enter"))
        self.assertTrue(info.get("sm_session"))
        env.close()

    def test_policy_interrupt_is_rising_edge_only(self):
        env, joy = _wrap()
        env.reset()
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
        self.assertTrue(env._request_policy_interrupt())
        # Held — must not keep asserting interrupt / pending enter.
        self.assertFalse(env._request_policy_interrupt())
        with env._interrupt_lock:
            self.assertTrue(env._need_left_release or env._pending_session_enter)
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 0])
        self.assertFalse(env._request_policy_interrupt())
        env.close()

    def test_session_idle_stick_holds_not_policy(self):
        env, joy = _wrap()
        env.reset()
        policy = np.ones(6, dtype=np.float32) * 0.8
        joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
        env.step(policy)
        joy.inject(np.zeros(6), buttons=[0, 0])
        exec_a, intervened = env.action(policy)
        self.assertTrue(intervened)
        np.testing.assert_allclose(exec_a, 0.0, atol=1e-3)
        env.close()

    def test_processor_dt_uses_wall_clock(self):
        import time

        env, _joy = _wrap()
        env.reset()
        first = env._processor_dt()
        self.assertAlmostEqual(first, env.control_dt, places=3)
        time.sleep(0.03)
        second = env._processor_dt()
        self.assertGreater(second, 0.02)
        self.assertLessEqual(second, env._processor_max_dt + 1e-6)
        env.close()


if __name__ == "__main__":
    unittest.main()
