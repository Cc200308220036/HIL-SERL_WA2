"""Actor-only R13 safety helpers tests. No ROS."""

from __future__ import annotations

import os
import pathlib
import sys
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.actor_safety import assert_no_teleop, assert_r13_hardware_confirm  # noqa: E402
from hilserl_wa2.experiments.r13_protocol import scale_arm_action  # noqa: E402
from hilserl_wa2.wrappers.grasp_action import discretize_grasp  # noqa: E402


class R13ActorSafetyTests(unittest.TestCase):
    def test_scale_does_not_touch_grasp(self):
        action = np.array([1, -1, 1, -1, 1, -1, 1], dtype=np.float32)
        out = scale_arm_action(action, 0.2)
        np.testing.assert_allclose(out[:6], action[:6] * 0.2)
        self.assertEqual(float(out[6]), 1.0)
        self.assertEqual(discretize_grasp(out[6]), 1)

    def test_assert_no_teleop_runs(self):
        assert_no_teleop()

    def test_r13_live_requires_r4_confirm(self):
        old = os.environ.pop("R4_CONFIRM", None)
        try:
            with self.assertRaises(Exception):
                assert_r13_hardware_confirm("live")
            os.environ["R4_CONFIRM"] = "YES"
            assert_r13_hardware_confirm("live")
            assert_r13_hardware_confirm("eval")
        finally:
            if old is None:
                os.environ.pop("R4_CONFIRM", None)
            else:
                os.environ["R4_CONFIRM"] = old


if __name__ == "__main__":
    unittest.main()
