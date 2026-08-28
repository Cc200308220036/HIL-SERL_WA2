"""Offline tests for non-blocking dexterous-hand toggle control."""

from __future__ import annotations

import pathlib
import sys
import threading
import unittest


SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.interventions.end_effector import (  # noqa: E402
    DexterousHandAdapter,
    HandState,
)


GRASP = [0.1, 1.5, 1.2, 1.2, 1.2, 1.2]
HAND = object()


class FakeController:
    def __init__(self, joints=None, succeed=True, gate=None):
        self.joints = joints
        self.succeed = succeed
        self.gate = gate
        self.calls = []

    def get_hand_joints(self, hand):
        return self.joints

    def grasp_hand(self, hand, target):
        self.calls.append(("grasp", hand, list(target)))
        if self.gate is not None:
            self.gate.wait(1.0)
        if self.succeed:
            self.joints = list(target)
        return self.succeed

    def release_hand(self, hand):
        self.calls.append(("release", hand, [0.0] * 6))
        if self.gate is not None:
            self.gate.wait(1.0)
        if self.succeed:
            self.joints = [0.0] * 6
        return self.succeed


class DexterousHandAdapterTests(unittest.TestCase):
    def make_adapter(self, controller, **kwargs):
        return DexterousHandAdapter(
            controller,
            HAND,
            GRASP,
            execute=True,
            **kwargs,
        )

    def test_auto_detects_released_and_grasped(self):
        released = self.make_adapter(FakeController([0.0] * 6))
        self.assertEqual(released.initialize(), HandState.RELEASED)

        grasped = self.make_adapter(FakeController(GRASP))
        self.assertEqual(grasped.initialize(), HandState.GRASPED)

    def test_ambiguous_feedback_inhibits_toggle(self):
        adapter = self.make_adapter(FakeController([0.5] * 6))
        self.assertEqual(adapter.initialize(), HandState.UNKNOWN)
        accepted, reason = adapter.request_toggle()
        self.assertFalse(accepted)
        self.assertIn("unknown", reason)

    def test_successful_toggle_changes_state_only_after_service(self):
        controller = FakeController([0.0] * 6)
        adapter = self.make_adapter(controller)
        adapter.initialize()
        accepted, command = adapter.request_toggle()
        self.assertTrue(accepted)
        self.assertEqual(command, "grasp")
        self.assertTrue(adapter.wait_idle())
        self.assertEqual(adapter.state, HandState.GRASPED)
        self.assertEqual(controller.calls, [("grasp", HAND, GRASP)])

        accepted, command = adapter.request_toggle()
        self.assertTrue(accepted)
        self.assertEqual(command, "release")
        self.assertTrue(adapter.wait_idle())
        self.assertEqual(adapter.state, HandState.RELEASED)
        self.assertEqual(
            controller.calls,
            [("grasp", HAND, GRASP), ("release", HAND, [0.0] * 6)],
        )

    def test_failed_service_keeps_previous_state(self):
        controller = FakeController([0.0] * 6, succeed=False)
        adapter = self.make_adapter(controller)
        adapter.initialize()
        adapter.request_toggle()
        self.assertTrue(adapter.wait_idle())
        self.assertEqual(adapter.state, HandState.RELEASED)
        self.assertFalse(adapter.last_result.success)

    def test_busy_adapter_rejects_second_request(self):
        gate = threading.Event()
        controller = FakeController([0.0] * 6, gate=gate)
        adapter = self.make_adapter(controller)
        adapter.initialize()
        self.assertTrue(adapter.request_toggle()[0])
        self.assertEqual(adapter.state, HandState.BUSY)
        accepted, reason = adapter.request_toggle()
        self.assertFalse(accepted)
        self.assertIn("already running", reason)
        gate.set()
        self.assertTrue(adapter.wait_idle())

    def test_dry_run_does_not_call_service(self):
        controller = FakeController([0.0] * 6)
        adapter = DexterousHandAdapter(
            controller,
            HAND,
            GRASP,
            initial_state="released",
            execute=False,
        )
        adapter.initialize()
        adapter.request_toggle()
        self.assertTrue(adapter.wait_idle())
        self.assertEqual(adapter.state, HandState.GRASPED)
        self.assertEqual(controller.calls, [])
        self.assertTrue(adapter.last_result.dry_run)

    def test_explicit_released_ignores_current_joints(self):
        controller = FakeController([0.4] * 6)
        adapter = self.make_adapter(controller, initial_state="released")
        self.assertEqual(adapter.initialize(), HandState.RELEASED)
        accepted, command = adapter.request_toggle()
        self.assertTrue(accepted)
        self.assertEqual(command, "grasp")

    def test_custom_release_uses_grasp_hand(self):
        release = [0.1, 0.9, 0.3, 0.3, 0.3, 0.3]
        controller = FakeController(release)
        adapter = self.make_adapter(
            controller,
            release_target=release,
            initial_state="released",
        )
        adapter.initialize()
        adapter.request_toggle()
        self.assertTrue(adapter.wait_idle())
        adapter.request_toggle()
        self.assertTrue(adapter.wait_idle())
        self.assertEqual(adapter.state, HandState.RELEASED)
        self.assertEqual(
            controller.calls,
            [("grasp", HAND, GRASP), ("grasp", HAND, release)],
        )

    def test_auto_detects_custom_release_target(self):
        release = [0.1, 0.9, 0.3, 0.3, 0.3, 0.3]
        adapter = self.make_adapter(
            FakeController(release),
            release_target=release,
        )
        self.assertEqual(adapter.initialize(), HandState.RELEASED)


if __name__ == "__main__":
    unittest.main()
