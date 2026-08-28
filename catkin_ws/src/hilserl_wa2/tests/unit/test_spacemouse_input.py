"""Regression tests based on the physical SpaceMouse calibration samples."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np


SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.interventions.spacemouse_input import (  # noqa: E402
    MotionIntent,
    SpaceMouseInputConfig,
    SpaceMouseInputProcessor,
)


SAMPLES = {
    "left_translation": [0.1953125, -0.880859375, -0.66015625, 0.0, 0.029296875, -0.1015625],
    "right_translation": [-0.01953125, 0.9765625, -0.84765625, 0.453125, -0.00390625, 0.0],
    "forward_translation": [-0.9765625, 0.146484375, -0.04296875, -0.1015625, 0.087890625, 0.00390625],
    "backward_translation": [0.9765625, 0.201171875, -0.62890625, -0.06640625, 0.015625, 0.0],
    "up_translation": [0.1328125, 0.0, 0.9765625, 0.0, 0.048828125, -0.23046875],
    "down_translation": [0.0, -0.029296875, -0.9765625, 0.0, 0.037109375, 0.0],
    "left_tilt": [-0.208984375, -0.052734375, -0.1328125, 0.9765625, 0.076171875, 0.3125],
    "right_tilt": [0.076171875, 0.390625, 0.0, -0.9765625, -0.0234375, 0.0],
    "forward_tilt": [0.259765625, 0.037109375, -0.0234375, -0.140625, -0.9765625, 0.0],
    "backward_tilt": [0.0, -0.099609375, -0.0859375, 0.017578125, 0.9765625, -0.3515625],
    "clockwise_twist": [-0.00390625, 0.015625, -0.615234375, 0.048828125, 0.029296875, -0.9765625],
    "counterclockwise_twist": [0.0, 0.0703125, -0.892578125, -0.0546875, -0.0546875, 0.9765625],
}


class SpaceMouseCalibrationTests(unittest.TestCase):
    def setUp(self):
        # Disable filtering so each physical sample can be asserted exactly.
        config = SpaceMouseInputConfig(
            translation_filter_tau=0.0,
            rotation_filter_tau=0.0,
        )
        self.processor = SpaceMouseInputProcessor(config)

    def assert_sample(self, name, intent, active_axis, sign):
        result = self.processor.update(SAMPLES[name], dt=0.02)
        self.assertEqual(result.intent, intent, name)
        self.assertEqual(result.active_axis, active_axis, name)
        self.assertEqual(np.count_nonzero(np.abs(result.command) > 1e-12), 1, name)
        self.assertGreater(sign * result.command[active_axis], 0.0, name)
        self.processor.reset()

    def test_all_translation_samples(self):
        # Output order is X(forward), Y(left), Z(up).
        self.assert_sample("forward_translation", MotionIntent.TRANSLATION, 0, +1)
        self.assert_sample("backward_translation", MotionIntent.TRANSLATION, 0, -1)
        self.assert_sample("left_translation", MotionIntent.TRANSLATION, 1, +1)
        self.assert_sample("right_translation", MotionIntent.TRANSLATION, 1, -1)
        self.assert_sample("up_translation", MotionIntent.TRANSLATION, 2, +1)
        self.assert_sample("down_translation", MotionIntent.TRANSLATION, 2, -1)

    def test_all_rotation_samples(self):
        self.assert_sample("left_tilt", MotionIntent.ROTATION, 3, +1)
        self.assert_sample("right_tilt", MotionIntent.ROTATION, 3, -1)
        self.assert_sample("forward_tilt", MotionIntent.ROTATION, 4, +1)
        self.assert_sample("backward_tilt", MotionIntent.ROTATION, 4, -1)
        self.assert_sample("clockwise_twist", MotionIntent.ROTATION, 5, +1)
        self.assert_sample("counterclockwise_twist", MotionIntent.ROTATION, 5, -1)

    def test_clockwise_twist_suppresses_large_z_crosstalk(self):
        result = self.processor.update(SAMPLES["clockwise_twist"], dt=0.02)
        self.assertEqual(result.intent, MotionIntent.ROTATION)
        self.assertEqual(result.active_axis_name, "yaw")
        np.testing.assert_allclose(result.translation, np.zeros(3), atol=0.0)

    def test_right_translation_suppresses_z_and_roll_crosstalk(self):
        result = self.processor.update(SAMPLES["right_translation"], dt=0.02)
        self.assertEqual(result.intent, MotionIntent.TRANSLATION)
        self.assertEqual(result.active_axis_name, "y")
        self.assertEqual(result.command[2], 0.0)
        np.testing.assert_allclose(result.rotation, np.zeros(3), atol=0.0)

    def test_axis_lock_uses_hysteresis(self):
        first = self.processor.update([0.0, -0.80, 0.0, 0.0, 0.0, 0.0], dt=0.02)
        self.assertEqual(first.active_axis_name, "y")

        # Z is strongest, but not by the configured 0.25 margin.
        held = self.processor.update([0.0, -0.75, 0.82, 0.0, 0.0, 0.0], dt=0.02)
        self.assertEqual(held.active_axis_name, "y")

        switched = self.processor.update([0.0, -0.65, 0.90, 0.0, 0.0, 0.0], dt=0.02)
        self.assertEqual(switched.active_axis_name, "z")

    def test_near_equal_translation_allows_two_axes(self):
        result = self.processor.update([-0.80, -0.80, 0.0, 0.0, 0.0, 0.0], dt=0.02)
        self.assertEqual(result.intent, MotionIntent.TRANSLATION)
        self.assertGreater(abs(float(result.command[0])), 0.0)
        self.assertGreater(abs(float(result.command[1])), 0.0)
        self.assertEqual(float(result.command[2]), 0.0)

    def test_coupled_secondary_axis_stays_suppressed(self):
        # right_translation: |Z|/|Y| ≈ 0.868 < secondary_axis_ratio 0.90
        result = self.processor.update(SAMPLES["right_translation"], dt=0.02)
        self.assertEqual(result.active_axis_name, "y")
        self.assertEqual(float(result.command[2]), 0.0)
        self.assertEqual(np.count_nonzero(np.abs(result.command) > 1e-12), 1)

    def test_axis_switch_does_not_zero_filter(self):
        processor = SpaceMouseInputProcessor(
            SpaceMouseInputConfig(
                translation_filter_tau=0.06,
                rotation_filter_tau=0.10,
            )
        )
        for _ in range(8):
            processor.update([-0.90, 0.0, 0.0, 0.0, 0.0, 0.0], dt=0.02)
        before = float(processor.update([-0.90, 0.0, 0.0, 0.0, 0.0, 0.0], dt=0.02).command[0])
        self.assertGreater(before, 0.2)
        switched = processor.update([0.0, -0.90, 0.0, 0.0, 0.0, 0.0], dt=0.02)
        self.assertEqual(switched.active_axis_name, "y")
        self.assertGreater(float(switched.command[0]), 0.0)
        self.assertGreater(float(switched.command[1]), 0.0)

    def test_direct_rotation_to_right_translation_switches_group(self):
        rotating = self.processor.update(SAMPLES["left_tilt"], dt=0.02)
        self.assertEqual(rotating.intent, MotionIntent.ROTATION)
        translated = self.processor.update(SAMPLES["right_translation"], dt=0.02)
        self.assertEqual(translated.intent, MotionIntent.TRANSLATION)
        self.assertEqual(translated.active_axis_name, "y")
        self.assertLess(translated.command[1], 0.0)
        np.testing.assert_allclose(translated.rotation, np.zeros(3), atol=0.0)

    def test_twist_remains_rotation_despite_large_translation_crosstalk(self):
        first = self.processor.update(SAMPLES["left_tilt"], dt=0.02)
        self.assertEqual(first.intent, MotionIntent.ROTATION)
        twist = self.processor.update(SAMPLES["counterclockwise_twist"], dt=0.02)
        self.assertEqual(twist.intent, MotionIntent.ROTATION)
        self.assertEqual(twist.active_axis_name, "yaw")
        np.testing.assert_allclose(twist.translation, np.zeros(3), atol=0.0)

    def test_disabled_path_clears_filter_and_intent(self):
        active = self.processor.update(SAMPLES["forward_translation"], dt=0.02)
        self.assertNotEqual(active.intent, MotionIntent.IDLE)
        stopped = self.processor.update(SAMPLES["forward_translation"], dt=0.02, enabled=False)
        self.assertEqual(stopped.intent, MotionIntent.IDLE)
        self.assertIsNone(stopped.active_axis)
        np.testing.assert_array_equal(stopped.command, np.zeros(6))

    def test_continuous_deadband(self):
        cfg = SpaceMouseInputConfig(
            translation_enter_threshold=0.10,
            intent_exit_threshold=0.05,
            translation_filter_tau=0.0,
            rotation_filter_tau=0.0,
        )
        processor = SpaceMouseInputProcessor(cfg)
        at_edge = processor.update([-cfg.translation_deadzone, 0, 0, 0, 0, 0], 0.02)
        self.assertEqual(at_edge.command[0], 0.0)
        above = processor.update([-cfg.translation_deadzone - 1e-4, 0, 0, 0, 0, 0], 0.02)
        self.assertGreater(above.command[0], 0.0)
        self.assertLess(above.command[0], 1e-3)

    def test_low_pass_filter_uses_dt(self):
        processor = SpaceMouseInputProcessor()
        result = processor.update(SAMPLES["forward_translation"], dt=0.02)
        self.assertGreater(result.command[0], 0.0)
        self.assertLess(result.command[0], 1.0)

    def test_invalid_axis_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.processor.update([0.0] * 5, dt=0.02)
        with self.assertRaises(ValueError):
            self.processor.update([0.0, 0.0, np.nan, 0.0, 0.0, 0.0], dt=0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
