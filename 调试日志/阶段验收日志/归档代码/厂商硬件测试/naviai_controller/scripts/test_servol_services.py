#!/usr/bin/env python3
"""Gate 3: set_servo_params / stop / clear_servo_params only. No ServoL publish."""
from __future__ import annotations

import sys

import rospy
from naviai_controller import ArmGroup, NaviController, RobotModel


def main():
    rospy.init_node("test_servol_services", anonymous=True)
    ctrl = NaviController(model=RobotModel.WA2, auto_spin=True)
    rospy.sleep(1.0)

    ok_set = ctrl.set_servo_params(0.02, 800, arm=ArmGroup.LEFT)
    print("set_servo_params(0.02, 800, LEFT):", ok_set)
    if not ok_set:
        raise RuntimeError("set_servo_params failed")

    ok_stop = ctrl.stop()
    print("stop():", ok_stop)
    if not ok_stop:
        raise RuntimeError("stop failed")

    ok_clear = ctrl.clear_servo_params()
    print("clear_servo_params():", ok_clear)
    if not ok_clear:
        raise RuntimeError("clear_servo_params failed")

    print("Gate3 servo services: PASS (no ServoL publish)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("Gate3 FAILED: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
