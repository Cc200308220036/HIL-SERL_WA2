"""R9 transition schema and dual-store routing (no ROS, no JAX)."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.experiments.transition import (  # noqa: E402
    ListStore,
    TransitionError,
    build_actor_transition,
    dones_and_mask,
    route_transition,
    transition_rows_hash,
)
from hilserl_wa2.interventions.joy_watchdog import JoyWatchdog  # noqa: E402
from hilserl_wa2.interventions.spacemouse_input import SpaceMouseInputConfig  # noqa: E402
from hilserl_wa2.interventions.wa2_spacemouse_intervention import (  # noqa: E402
    WA2SpacemouseIntervention,
)
from hilserl_wa2.tests.unit.test_spacemouse_input import SAMPLES  # noqa: E402


def _obs():
    return {
        "state": np.zeros((1, 27), dtype=np.float32),
        "head": np.zeros((1, 128, 128, 3), dtype=np.uint8),
        "wrist": np.zeros((1, 128, 128, 3), dtype=np.uint8),
    }


def _action():
    return np.zeros(6, dtype=np.float32)


class TransitionSemanticsTests(unittest.TestCase):
    def test_normal(self):
        done, mask, end = dones_and_mask(False, False)
        self.assertFalse(done)
        self.assertEqual(float(mask), 1.0)
        self.assertFalse(end)

    def test_terminated(self):
        done, mask, end = dones_and_mask(True, False)
        self.assertTrue(done)
        self.assertEqual(float(mask), 0.0)
        self.assertTrue(end)

    def test_truncated(self):
        done, mask, end = dones_and_mask(False, True)
        self.assertFalse(done)
        self.assertEqual(float(mask), 1.0)
        self.assertTrue(end)

    def test_both_terminated_priority(self):
        done, mask, end = dones_and_mask(True, True)
        self.assertTrue(done)
        self.assertEqual(float(mask), 0.0)
        self.assertTrue(end)

    def test_build_normal_schema(self):
        tr, meta = build_actor_transition(
            _obs(), _action(), _obs(), 0.0, False, False, {}
        )
        self.assertEqual(set(tr), {
            "observations", "actions", "next_observations",
            "rewards", "masks", "dones",
        })
        self.assertFalse(bool(tr["dones"]))
        self.assertEqual(float(tr["masks"]), 1.0)
        self.assertFalse(meta["episode_end"])
        self.assertFalse(meta["intervened"])

    def test_intervene_action_overrides(self):
        policy = np.asarray([0.5, 0, 0, 0, 0, 0], dtype=np.float32)
        ia = np.asarray([0.0, 1.0, 0, 0, 0, 0], dtype=np.float32)
        tr, meta = build_actor_transition(
            _obs(), policy, _obs(), 0.0, False, False,
            {"intervene_action": ia},
        )
        np.testing.assert_allclose(tr["actions"], ia)
        self.assertTrue(meta["intervened"])

    def test_partial_servo_window_scales_effective_arm_action(self):
        policy = np.ones(7, dtype=np.float32)
        policy[-1] = 0.0
        tr, meta = build_actor_transition(
            _obs(),
            policy,
            _obs(),
            0.0,
            False,
            False,
            {"servo_ticks_requested": 5, "servo_ticks_executed": 2},
        )
        np.testing.assert_allclose(tr["actions"][:6], 0.4, atol=1e-7)
        self.assertEqual(float(tr["actions"][-1]), 0.0)
        self.assertFalse(meta["intervened"])

    def test_partial_human_window_uses_same_effective_scale(self):
        human = np.ones(7, dtype=np.float32)
        human[-1] = 0.0
        tr, meta = build_actor_transition(
            _obs(),
            np.zeros(7, dtype=np.float32),
            _obs(),
            0.0,
            False,
            False,
            {
                "intervene_action": human,
                "servo_ticks_requested": 5,
                "servo_ticks_executed": 2,
            },
        )
        np.testing.assert_allclose(tr["actions"][:6], 0.4, atol=1e-7)
        self.assertTrue(meta["intervened"])

    def test_route_dual_store(self):
        env_store = ListStore()
        intvn_store = ListStore()
        normal, meta_n = build_actor_transition(
            _obs(), _action(), _obs(), 0.0, False, False, {}
        )
        route_transition(normal, meta_n, env_store, intvn_store)
        ia = np.asarray([0.2, 0, 0, 0, 0, 0], dtype=np.float32)
        inter, meta_i = build_actor_transition(
            _obs(), _action(), _obs(), 0.0, False, False,
            {"intervene_action": ia},
        )
        route_transition(inter, meta_i, env_store, intvn_store)
        self.assertEqual(len(env_store), 2)
        self.assertEqual(len(intvn_store), 1)
        np.testing.assert_allclose(intvn_store.rows[0]["actions"], ia)

    def test_reject_nan_action(self):
        bad = _action()
        bad[0] = np.nan
        with self.assertRaises(TransitionError):
            build_actor_transition(_obs(), bad, _obs(), 0.0, False, False, {})

    def test_reject_wrong_shape(self):
        with self.assertRaises(TransitionError):
            build_actor_transition(
                _obs(), np.zeros(8, np.float32), _obs(), 0.0, False, False, {}
            )
        # R13 allows 7D; 6D still accepted for R9/R11 paths.
        build_actor_transition(
            _obs(), np.zeros(7, np.float32), _obs(), 0.0, False, False, {}
        )

    def test_reject_out_of_space(self):
        with self.assertRaises(TransitionError):
            build_actor_transition(
                _obs(), np.asarray([2.0, 0, 0, 0, 0, 0], np.float32),
                _obs(), 0.0, False, False, {},
            )

    def test_reject_non_discrete_grasp_action(self):
        bad = np.zeros(7, dtype=np.float32)
        bad[-1] = 0.25
        with self.assertRaises(TransitionError):
            build_actor_transition(_obs(), bad, _obs(), 0.0, False, False, {})

    def test_accept_discrete_grasp_actions(self):
        for command in (-1.0, 0.0, 1.0):
            action = np.zeros(7, dtype=np.float32)
            action[-1] = command
            transition, _ = build_actor_transition(
                _obs(), action, _obs(), 0.0, False, False, {}
            )
            self.assertEqual(float(transition["actions"][-1]), command)

    def test_reject_nan_state(self):
        obs = _obs()
        obs["state"][0, 0] = np.inf
        with self.assertRaises(TransitionError):
            build_actor_transition(obs, _action(), _obs(), 0.0, False, False, {})

    def test_hash_stable(self):
        rows = []
        for _ in range(3):
            tr, _ = build_actor_transition(
                _obs(), _action(), _obs(), 0.0, False, False, {}
            )
            rows.append(tr)
        self.assertEqual(transition_rows_hash(rows), transition_rows_hash(rows))


class FakeEnvInjectionTests(unittest.TestCase):
    def test_force_terminated_and_truncated(self):
        env = WA2Env(fake_env=True, seed=0)
        try:
            obs, _ = env.reset(seed=0)
            env.inject_success()
            nxt, r, term, trunc, info = env.step(np.zeros(6, np.float32))
            self.assertTrue(term)
            tr, meta = build_actor_transition(
                obs, np.zeros(6, np.float32), nxt, r, term, trunc, info
            )
            self.assertTrue(bool(tr["dones"]))
            self.assertEqual(float(tr["masks"]), 0.0)
            self.assertTrue(meta["episode_end"])

            obs, _ = env.reset(seed=1)
            env.inject_truncate()
            nxt, r, term, trunc, info = env.step(np.zeros(6, np.float32))
            self.assertFalse(term)
            self.assertTrue(trunc)
            tr, meta = build_actor_transition(
                obs, np.zeros(6, np.float32), nxt, r, term, trunc, info
            )
            self.assertFalse(bool(tr["dones"]))
            self.assertEqual(float(tr["masks"]), 1.0)
            self.assertTrue(meta["episode_end"])
        finally:
            env.close()

    def test_synthetic_intervene_action_matches_env(self):
        joy = JoyWatchdog(max_age_s=0.2)
        env = WA2SpacemouseIntervention(
            WA2Env(fake_env=True, seed=0),
            joy_watchdog=joy,
            auto_start_ros=False,
            input_config=SpaceMouseInputConfig(
                translation_filter_tau=0.0, rotation_filter_tau=0.0
            ),
            intervene_eps=1e-3,
        )
        try:
            obs, _ = env.reset()
            policy = np.zeros(6, np.float32)
            joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
            nxt, r, term, trunc, info = env.step(policy)
            self.assertTrue(info.get("intervened"))
            tr, meta = build_actor_transition(
                obs, policy, nxt, r, term, trunc, info
            )
            np.testing.assert_allclose(tr["actions"], info["intervene_action"])
            self.assertTrue(meta["intervened"])
            self.assertGreater(float(tr["actions"][0]), 0.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
