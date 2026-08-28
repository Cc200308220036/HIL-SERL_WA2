"""Offline tests for bounded Cartesian pose integration."""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.interventions.pose_integrator import (  # noqa: E402
    PoseIntegrator,
    PoseIntegratorConfig,
)


IDENTITY_POSE = np.asarray([0.4, 0.1, 0.8, 0.0, 0.0, 0.0, 1.0])


class PoseIntegratorTests(unittest.TestCase):
    def test_reset_normalizes_quaternion(self):
        integrator = PoseIntegrator()
        pose = integrator.reset([0, 0, 0, 0, 0, 0, 2])
        self.assertAlmostEqual(float(np.linalg.norm(pose[3:])), 1.0)
        with self.assertRaises(ValueError):
            integrator.reset([0, 0, 0, 0, 0, 0, 0])

    def test_step_requires_reset(self):
        with self.assertRaises(RuntimeError):
            PoseIntegrator().step(np.zeros(6), 0.02)

    def test_zero_motion_preserves_pose(self):
        integrator = PoseIntegrator()
        origin = integrator.reset(IDENTITY_POSE)
        target = integrator.step(np.zeros(6), 0.02)
        np.testing.assert_allclose(target, origin, atol=0.0)

    def test_translation_does_not_change_orientation(self):
        cfg = PoseIntegratorConfig(linear_scale=0.01, max_linear_step=0.0005)
        integrator = PoseIntegrator(cfg)
        origin = integrator.reset(IDENTITY_POSE)
        target = integrator.step([1, 0, 0, 0, 0, 0], dt=0.10)
        self.assertAlmostEqual(target[0] - origin[0], 0.0005, places=12)
        np.testing.assert_allclose(target[1:3], origin[1:3], atol=0.0)
        np.testing.assert_allclose(target[3:], origin[3:], atol=0.0)

    def test_translation_workspace_limit(self):
        cfg = PoseIntegratorConfig(
            linear_scale=1.0,
            max_linear_step=1.0,
            translation_limit=0.03,
        )
        integrator = PoseIntegrator(cfg)
        integrator.reset(IDENTITY_POSE)
        for _ in range(10):
            integrator.step([1, 0, 0, 0, 0, 0], dt=0.1)
        self.assertAlmostEqual(integrator.relative_translation, 0.03, places=12)

    def test_rotation_does_not_change_position_and_stays_normalized(self):
        cfg = PoseIntegratorConfig(
            angular_scale=0.08,
            max_angular_step=0.01,
            rotation_limit_rad=math.radians(30),
        )
        integrator = PoseIntegrator(cfg)
        origin = integrator.reset(IDENTITY_POSE)
        target = integrator.step([0, 0, 0, 1, 0, 0], dt=0.02)
        np.testing.assert_allclose(target[:3], origin[:3], atol=0.0)
        self.assertAlmostEqual(float(np.linalg.norm(target[3:])), 1.0, places=12)
        self.assertAlmostEqual(integrator.relative_rotation_rad, 0.0016, places=12)

    def test_default_rotation_is_limited_to_two_degrees(self):
        integrator = PoseIntegrator()
        integrator.reset(IDENTITY_POSE)
        for _ in range(100):
            integrator.step([0, 0, 0, 1, 0, 0], dt=0.02)
        self.assertAlmostEqual(
            integrator.relative_rotation_rad,
            math.radians(2.0),
            places=10,
        )

    def test_tool_and_base_rotation_composition_are_distinct(self):
        initial_quaternion = Rotation.from_euler("y", 90, degrees=True).as_quat()
        pose = np.concatenate(([0.0, 0.0, 0.0], initial_quaternion))
        common = dict(
            angular_scale=0.2,
            max_angular_step=0.2,
            rotation_limit_rad=1.0,
        )
        tool = PoseIntegrator(PoseIntegratorConfig(rotation_frame="tool", **common))
        base = PoseIntegrator(PoseIntegratorConfig(rotation_frame="base", **common))
        tool.reset(pose)
        base.reset(pose)
        tool_target = tool.step([0, 0, 0, 1, 0, 0], dt=0.5)
        base_target = base.step([0, 0, 0, 1, 0, 0], dt=0.5)
        self.assertFalse(np.allclose(tool_target[3:], base_target[3:]))

    def test_mixed_motion_is_rejected_in_first_phase(self):
        integrator = PoseIntegrator()
        integrator.reset(IDENTITY_POSE)
        with self.assertRaises(ValueError):
            integrator.step([1, 0, 0, 1, 0, 0], dt=0.02)

    def test_invalid_motion_and_dt_are_rejected(self):
        integrator = PoseIntegrator()
        integrator.reset(IDENTITY_POSE)
        with self.assertRaises(ValueError):
            integrator.step([0] * 5, dt=0.02)
        with self.assertRaises(ValueError):
            integrator.step([0] * 6, dt=0.0)
        with self.assertRaises(ValueError):
            integrator.step([0, 0, 0, 0, np.nan, 0], dt=0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
