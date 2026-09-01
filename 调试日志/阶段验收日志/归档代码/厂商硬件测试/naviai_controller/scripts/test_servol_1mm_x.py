#!/usr/bin/env python3
"""Gate 5: smooth +1 mm BASE X ServoL move on left arm."""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import rospy
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
    rospy.init_node("test_servol_1mm_x", anonymous=True)
    ctrl = NaviController(model=RobotModel.WA2, auto_spin=True)

    before = _wait_tcp(ctrl)
    before_cmd = rospy.wait_for_message(
        "/zj_humanoid/upperlimb/uplimb_state", UplimbState, timeout=5.0
    )
    print("before_tcp:", [round(x, 6) for x in before])
    print("before_cmd_num:", before_cmd.cmd_num)

    start = list(before)
    goal = list(before)
    goal[0] += 0.001  # +1 mm on BASE X

    steps = 50
    rate_hz = 50.0
    period = 1.0 / rate_hz

    if not ctrl.set_servo_params(0.02, 800, arm=ArmGroup.LEFT):
        raise RuntimeError("set_servo_params failed")

    try:
        for i in range(1, steps + 1):
            if rospy.is_shutdown():
                break
            alpha = float(i) / float(steps)
            target = list(start)
            target[0] = start[0] + alpha * (goal[0] - start[0])
            # keep orientation fixed
            target[3:] = start[3:]
            ctrl.servol(target, ArmGroup.LEFT)
            time.sleep(period)
    finally:
        stopped = ctrl.stop()
        cleared = ctrl.clear_servo_params()
        print("stop():", stopped, "clear_servo_params():", cleared)
        if not stopped or not cleared:
            raise RuntimeError("cleanup stop/clear failed")

    rospy.sleep(0.3)
    after = _wait_tcp(ctrl)
    after_cmd = rospy.wait_for_message(
        "/zj_humanoid/upperlimb/uplimb_state", UplimbState, timeout=5.0
    )
    delta = np.asarray(after[:3]) - np.asarray(before[:3])
    delta_mm = delta * 1000.0
    delta_deg = _quat_angle_deg(before[3:], after[3:])

    print("after_tcp:", [round(x, 6) for x in after])
    print("after_cmd_num:", after_cmd.cmd_num)
    print(
        "delta_xyz_mm:",
        [round(float(x), 3) for x in delta_mm],
        "delta_deg:",
        round(delta_deg, 3),
    )

    if delta_mm[0] <= 0.0:
        raise RuntimeError("X displacement not positive")
    if abs(float(delta_mm[0])) > 2.0:
        raise RuntimeError("X displacement {:.3f} mm exceeds 2 mm".format(delta_mm[0]))
    if abs(float(delta_mm[1])) > 1.0 or abs(float(delta_mm[2])) > 1.0:
        raise RuntimeError("Y/Z crosstalk exceeds 1 mm")
    if delta_deg > 1.0:
        raise RuntimeError("orientation change {:.3f} deg exceeds 1 deg".format(delta_deg))

    # Confirm stop holds: pose should not keep drifting after stop.
    settle = _wait_tcp(ctrl)
    settle_delta_mm = float(np.linalg.norm(np.asarray(settle[:3]) - np.asarray(after[:3])) * 1000.0)
    print("post_stop_drift_mm:", round(settle_delta_mm, 3))
    if settle_delta_mm > 1.0:
        raise RuntimeError("pose continued changing after stop")

    print("Gate5 ServoL +1mm X: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("Gate5 FAILED: {}".format(exc), file=sys.stderr)
        try:
            from std_srvs.srv import Trigger, TriggerRequest

            rospy.wait_for_service("/zj_humanoid/upperlimb/stop", timeout=1.0)
            rospy.ServiceProxy("/zj_humanoid/upperlimb/stop", Trigger)(TriggerRequest())
        except Exception:
            pass
        raise SystemExit(1)
