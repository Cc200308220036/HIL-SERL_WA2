"""R13 7D grasp wrapper and demo augmentation tests. No ROS / JAX."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.experiments.demo_grasp import (  # noqa: E402
    augment_action,
    augment_transitions,
    infer_grasp_dim,
)
from hilserl_wa2.experiments.transition import (  # noqa: E402
    build_actor_transition,
    route_transition,
    ListStore,
)
from hilserl_wa2.wrappers.grasp_action import (  # noqa: E402
    WA2GraspActionWrapper,
    discretize_grasp,
)


class GraspWrapperTests(unittest.TestCase):
    def setUp(self):
        self.base = WA2Env(fake_env=True, scene_name="bottle_desktop")
        self.env = WA2GraspActionWrapper(self.base)

    def tearDown(self):
        self.env.close()

    def test_action_space_is_7d(self):
        self.assertEqual(self.env.action_space.shape, (7,))
        self.assertEqual(self.base.action_space.shape, (6,))

    def test_step_strips_to_6d_servo(self):
        obs, _ = self.env.reset()
        nxt, _, _, _, info = self.env.step(np.zeros(7, np.float32))
        self.assertEqual(info.get("grasp_command"), 0)
        self.assertEqual(nxt["state"]["tcp_pose"].shape, (7,))

    def test_grasp_edge_toggles_hand(self):
        self.env.reset()
        action = np.zeros(7, np.float32)
        action[6] = 1.0
        _, _, _, _, info = self.env.step(action)
        self.assertEqual(int(info["grasp_command"]), 1)
        _, _, _, _, info2 = self.env.step(action)
        self.assertEqual(int(info2["grasp_command"]), 0)

    def test_right_click_marks_intervened_7d(self):
        import gymnasium as gym

        class RightClick(gym.Wrapper):
            def step(self, action):
                obs, r, t, tr, info = self.env.step(action)
                info = dict(info)
                info["sm_right"] = True
                return obs, r, t, tr, info

        wrapped = WA2GraspActionWrapper(RightClick(self.base))
        wrapped.reset()
        _, _, _, _, info = wrapped.step(np.zeros(7, np.float32))
        self.assertIn("intervene_action", info)
        self.assertEqual(np.asarray(info["intervene_action"]).shape, (7,))
        self.assertTrue(info.get("hand_ok"))
        self.assertFalse(info.get("hand_exec_failed"))
        wrapped.close()

    def test_hand_ok_false_records_zero_grasp(self):
        """A-02: failed request_hand must not label actions[6] as ±1."""

        calls = {"n": 0}

        def _failing_hand(command):
            calls["n"] += 1
            cmd = "grasp" if command in ("grasp", "toggle") else command
            if command == "toggle":
                cmd = "grasp"
            return {"ok": False, "command": cmd, "state": "released"}

        self.base.request_hand = _failing_hand  # type: ignore[method-assign]
        self.env.reset()
        action = np.zeros(7, np.float32)
        action[6] = 1.0
        _, _, _, _, info = self.env.step(action)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(int(info["grasp_command"]), 0)
        self.assertTrue(info.get("hand_fired"))
        self.assertTrue(info.get("hand_exec_failed"))
        self.assertFalse(info.get("hand_ok"))
        # Retry still attempts grasp (software state unchanged on failure).
        _, _, _, _, info2 = self.env.step(action)
        self.assertEqual(calls["n"], 2)
        self.assertEqual(int(info2["grasp_command"]), 0)


class DemoAugmentTests(unittest.TestCase):
    def _obs(self, hand):
        return {
            "state": np.concatenate(
                [
                    np.zeros(7, np.float32),
                    np.zeros(6, np.float32),
                    np.zeros(8, np.float32),
                    np.asarray(hand, dtype=np.float32),
                ]
            ).reshape(1, 27),
            "head": np.zeros((1, 128, 128, 3), dtype=np.uint8),
            "wrist": np.zeros((1, 128, 128, 3), dtype=np.uint8),
        }

    def test_infer_and_augment(self):
        grasp = np.asarray([0.1, 0.9, 0.7, 0.7, 0.4, 0.4], np.float32)
        release = np.asarray([0.1, 0.9, 0.3, 0.3, 0.3, 0.3], np.float32)
        self.assertEqual(infer_grasp_dim(release, grasp), 1.0)
        self.assertEqual(infer_grasp_dim(grasp, release), -1.0)
        self.assertEqual(float(augment_action(np.zeros(6), 1.0).shape[0]), 7)

        row = {
            "observations": self._obs(release),
            "actions": np.zeros(6, np.float32),
            "next_observations": self._obs(grasp),
            "rewards": np.float32(0.0),
            "masks": np.float32(1.0),
            "dones": np.bool_(False),
        }
        out = augment_transitions([row])
        self.assertEqual(out[0]["actions"].shape, (7,))
        self.assertEqual(float(out[0]["actions"][-1]), 1.0)

    def test_route_keeps_intervention_in_both_stores(self):
        obs = self._obs(np.zeros(6))
        intervention = np.ones(7, np.float32) * 0.1
        intervention[-1] = 1.0
        tr, meta = build_actor_transition(
            obs,
            np.zeros(7, np.float32),
            obs,
            0.0,
            False,
            False,
            {"intervene_action": intervention},
        )
        env_store, intvn_store = ListStore(), ListStore()
        route_transition(tr, meta, env_store, intvn_store)
        self.assertEqual(len(env_store.rows), 1)
        self.assertEqual(len(intvn_store.rows), 1)
        self.assertEqual(tr["actions"].shape, (7,))


class FactoryGraspTests(unittest.TestCase):
    def test_factory_grasp_action_is_7d_default_stays_6d(self):
        try:
            from hilserl_wa2.experiments.env_factory import make_wa2_environment
            from hilserl_wa2.experiments.task_config import load_task
        except Exception as exc:  # pragma: no cover
            self.skipTest(str(exc))
        task = load_task("bottle_pick")
        env6 = make_wa2_environment(task, fake_env=True, classifier=False, grasp_action=False)
        try:
            self.assertEqual(env6.action_space.shape, (6,))
        finally:
            env6.close()
        env7 = make_wa2_environment(task, fake_env=True, classifier=False, grasp_action=True)
        try:
            self.assertEqual(env7.action_space.shape, (7,))
        finally:
            env7.close()


class DiscretizeTests(unittest.TestCase):
    def test_round(self):
        self.assertEqual(discretize_grasp(0.6), 1)
        self.assertEqual(discretize_grasp(-0.7), -1)
        self.assertEqual(discretize_grasp(0.1), 0)


if __name__ == "__main__":
    unittest.main()
