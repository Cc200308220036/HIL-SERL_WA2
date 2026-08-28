#!/usr/bin/env python3
"""Gate 2: ServoL message mapping and input validation, no robot motion."""
from __future__ import annotations

import math
import sys

import numpy as np

from naviai_controller.core.arm import ArmController
from naviai_controller.core.enums import ArmGroup


class Capture:
    def __init__(self):
        self.msg = None

    def publish(self, msg):
        self.msg = msg


def _pose_fields(pose):
    return [
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]


def _assert_close(actual, expected, name):
    if not np.allclose(actual, expected, atol=1e-9):
        raise AssertionError("{} mismatch: {} != {}".format(name, actual, expected))


def test_mapping():
    arm = ArmController.__new__(ArmController)
    pubs = {
        ArmGroup.LEFT: Capture(),
        ArmGroup.RIGHT: Capture(),
        ArmGroup.DUAL: Capture(),
    }
    arm._servol_pubs = pubs

    left = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    right = [-0.1, -0.2, 0.4, 0.0, 0.0, 0.0, 1.0]
    dual = left + right

    arm.servol(left, ArmGroup.LEFT)
    _assert_close(_pose_fields(pubs[ArmGroup.LEFT].msg.left_arm_pose), left, "LEFT")

    arm.servol(right, ArmGroup.RIGHT)
    _assert_close(_pose_fields(pubs[ArmGroup.RIGHT].msg.right_arm_pose), right, "RIGHT")

    arm.servol(dual, ArmGroup.DUAL)
    _assert_close(
        _pose_fields(pubs[ArmGroup.DUAL].msg.left_arm_pose), left, "DUAL.left"
    )
    _assert_close(
        _pose_fields(pubs[ArmGroup.DUAL].msg.right_arm_pose), right, "DUAL.right"
    )
    print("mapping: PASS")


def _expect_value_error(fn, label):
    try:
        fn()
    except ValueError:
        print("{}: PASS".format(label))
        return
    raise AssertionError("{} should raise ValueError".format(label))


def test_invalid_inputs():
    arm = ArmController.__new__(ArmController)
    arm._servol_pubs = {ArmGroup.LEFT: Capture()}

    _expect_value_error(
        lambda: arm.servol([0.0] * 6, ArmGroup.LEFT),
        "bad_length",
    )
    _expect_value_error(
        lambda: arm.servol([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, float("nan")], ArmGroup.LEFT),
        "nan_quat",
    )
    _expect_value_error(
        lambda: arm.servol([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0], ArmGroup.LEFT),
        "zero_quat",
    )
    # Norm far from 1.0
    bad = [0.1, 0.2, 0.3, 2.0, 0.0, 0.0, 0.0]
    _expect_value_error(lambda: arm.servol(bad, ArmGroup.LEFT), "unnormalized_quat")


def test_servo_params_guard():
    arm = ArmController.__new__(ArmController)
    _expect_value_error(
        lambda: arm.set_servo_params(0.01, 800),
        "time_guard",
    )
    _expect_value_error(
        lambda: arm.set_servo_params(0.02, 700),
        "gain_guard",
    )
    _expect_value_error(
        lambda: arm.set_servo_params(0.02, 800, arm=ArmGroup.NECK),
        "arm_guard",
    )


def main():
    test_mapping()
    test_invalid_inputs()
    test_servo_params_guard()
    print("Gate2 offline ServoL checks: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("Gate2 FAILED: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
