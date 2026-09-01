#!/usr/bin/env python3
"""Gate 4: hold current left-arm TCP pose with ServoL for 1 second."""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from upperlimb.msg import Pose as UplimbPose
from upperlimb.msg import UplimbState

from naviai_controller import ArmGroup, NaviController, RobotModel


def _quat_angle_deg(q0, q1):
    a = np.asarray(q0, dtype=float)
    b = np.asarray(q1, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def _wait_tcp(ctrl, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout and not rospy.is_shutdown():
        pose = ctrl.get_tcp_rt(ArmGroup.LEFT)
        if pose is not None:
            return list(pose)
        rospy.sleep(0.05)
    raise RuntimeError("timeout waiting for left TCP pose")


def main():
    rospy.init_node("test_servol_hold", anonymous=True)
    ctrl = NaviController(model=RobotModel.WA2, auto_spin=True)

    before_pose = _wait_tcp(ctrl)
    before_joints = ctrl.get_joints(ArmGroup.LEFT)
    before_cmd = rospy.wait_for_message(
        "/zj_humanoid/upperlimb/uplimb_state", UplimbState, timeout=5.0
    )
    print("before_tcp:", [round(x, 6) for x in before_pose])
    print("before_joints:", None if before_joints is None else [round(x, 6) for x in before_joints])
    print("before_cmd_num:", before_cmd.cmd_num)

    hold_pose = list(before_pose)
    rate_hz = 50.0
    duration = 1.0
    period = 1.0 / rate_hz

    if not ctrl.set_servo_params(0.02, 800, arm=ArmGroup.LEFT):
        raise RuntimeError("set_servo_params failed")

    try:
        t_end = time.time() + duration
        while time.time() < t_end and not rospy.is_shutdown():
            ctrl.servol(hold_pose, ArmGroup.LEFT)
            time.sleep(period)
    finally:
        stopped = ctrl.stop()
        cleared = ctrl.clear_servo_params()
        print("stop():", stopped, "clear_servo_params():", cleared)
        if not stopped or not cleared:
            raise RuntimeError("cleanup stop/clear failed")

    rospy.sleep(0.3)
    after_pose = _wait_tcp(ctrl)
    after_joints = ctrl.get_joints(ArmGroup.LEFT)
    after_cmd = rospy.wait_for_message(
        "/zj_humanoid/upperlimb/uplimb_state", UplimbState, timeout=5.0
    )

    delta_xyz = np.asarray(after_pose[:3]) - np.asarray(before_pose[:3])
    delta_mm = float(np.linalg.norm(delta_xyz) * 1000.0)
    delta_deg = _quat_angle_deg(before_pose[3:], after_pose[3:])

    print("after_tcp:", [round(x, 6) for x in after_pose])
    print("after_joints:", None if after_joints is None else [round(x, 6) for x in after_joints])
    print("after_cmd_num:", after_cmd.cmd_num)
    print("delta_mm:", round(delta_mm, 3), "delta_deg:", round(delta_deg, 3))

    if delta_mm > 2.0:
        raise RuntimeError("position drift {:.3f} mm exceeds 2 mm".format(delta_mm))
    if delta_deg > 1.0:
        raise RuntimeError("orientation drift {:.3f} deg exceeds 1 deg".format(delta_deg))

    print("Gate4 ServoL hold: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("Gate4 FAILED: {}".format(exc), file=sys.stderr)
        try:
            rospy.wait_for_service("/zj_humanoid/upperlimb/stop", timeout=1.0)
        except Exception:
            pass
        raise SystemExit(1)
