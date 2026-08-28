#!/usr/bin/env python3
"""SpaceMouse teleoperation for one WA2 arm through ServoL absolute targets."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import Joy

from naviai_controller import ArmGroup, HandType, NaviController, RobotModel


# Support package imports and direct execution/importlib loading used by the
# existing hardware-gate scripts.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pose_integrator import PoseIntegrator, PoseIntegratorConfig  # noqa: E402
from spacemouse_input import SpaceMouseInputConfig, SpaceMouseInputProcessor  # noqa: E402
from end_effector import DexterousHandAdapter, HandState  # noqa: E402


class SpaceMouseWA2Teleop:
    """Deadman-gated, bounded SpaceMouse control for one WA2 TCP."""

    def __init__(self):
        self.joy_topic = rospy.get_param("~joy_topic", "/spacenav/joy")
        self.execute = bool(rospy.get_param("~execute", False))
        self.arm_side = str(rospy.get_param("~arm_side", "left")).strip().lower()
        if self.arm_side not in ("left", "right"):
            raise ValueError("arm_side must be 'left' or 'right'")
        self.arm_group = (
            ArmGroup.LEFT if self.arm_side == "left" else ArmGroup.RIGHT
        )

        self.watchdog = float(rospy.get_param("~watchdog", 0.25))
        self.tcp_watchdog = float(rospy.get_param("~tcp_watchdog", 0.25))
        self.state_watchdog = float(rospy.get_param("~state_watchdog", 0.25))
        self.publish_rate = float(rospy.get_param("~publish_rate", 50.0))
        self.max_dt = float(rospy.get_param("~max_dt", 0.05))
        self.servo_time = float(rospy.get_param("~servo_time", 0.02))
        self.servo_gain = int(rospy.get_param("~servo_gain", 800))

        self.deadman_button = int(rospy.get_param("~deadman_button", 1))
        self.hand_button = int(rospy.get_param("~hand_button", 0))
        self.hand_execute = bool(rospy.get_param("~hand_execute", False))
        self.hand_cooldown = float(rospy.get_param("~hand_cooldown", 1.0))
        self.hand_neutral_threshold = float(
            rospy.get_param("~hand_neutral_threshold", 0.08)
        )
        self.hand_feedback_timeout = float(
            rospy.get_param("~hand_feedback_timeout", 2.0)
        )
        self.hand_feedback_tolerance = float(
            rospy.get_param("~hand_feedback_tolerance", 0.35)
        )
        self.initial_hand_state = str(
            rospy.get_param("~initial_hand_state", "auto")
        ).strip().lower()
        self.grasp_target = list(
            rospy.get_param("~grasp_target", [0.1, 1.5, 1.2, 1.2, 1.2, 1.2])
        )
        self.release_target = list(
            rospy.get_param("~release_target", [0.0] * 6)
        )
        self.hand_type = (
            HandType.LEFT if self.arm_side == "left" else HandType.RIGHT
        )
        self.axis_map = list(rospy.get_param("~axis_map", [0, 1, 2, 3, 4, 5]))
        self.axis_sign = [
            float(value)
            for value in rospy.get_param(
                "~axis_sign", [-1, -1, 1, 1, -1, -1]
            )
        ]
        self.axis_enable = [
            int(value)
            for value in rospy.get_param("~axis_enable", [1, 1, 1, 0, 0, 0])
        ]

        # Keep the original deadzone parameter as the translation default.
        legacy_deadzone = float(rospy.get_param("~deadzone", 0.15))
        translation_deadzone = float(
            rospy.get_param("~translation_deadzone", legacy_deadzone)
        )
        rotation_deadzone = float(rospy.get_param("~rotation_deadzone", 0.18))

        self.linear_scale = float(rospy.get_param("~linear_scale", 0.0))
        self.angular_scale = float(rospy.get_param("~angular_scale", 0.0))
        self.max_step_m = float(rospy.get_param("~max_step_m", 0.0005))
        self.max_angular_step = float(
            rospy.get_param("~max_angular_step", 0.002)
        )
        self.workspace_m = float(rospy.get_param("~workspace_m", 0.02))
        self.rotation_limit_deg = float(
            rospy.get_param("~rotation_limit_deg", 2.0)
        )
        self.rotation_frame = str(
            rospy.get_param("~rotation_frame", "tool")
        ).strip().lower()

        self._validate_config()

        input_config = SpaceMouseInputConfig(
            axis_map=tuple(self.axis_map),
            axis_sign=tuple(self.axis_sign),
            translation_deadzone=translation_deadzone,
            rotation_deadzone=rotation_deadzone,
            translation_curve_mix=float(
                rospy.get_param("~translation_curve_mix", 0.25)
            ),
            rotation_curve_mix=float(rospy.get_param("~rotation_curve_mix", 0.45)),
            translation_filter_tau=float(
                rospy.get_param("~translation_filter_tau", 0.06)
            ),
            rotation_filter_tau=float(
                rospy.get_param("~rotation_filter_tau", 0.10)
            ),
            rotation_enter_threshold=float(
                rospy.get_param("~rotation_enter_threshold", 0.65)
            ),
            translation_enter_threshold=float(
                rospy.get_param("~translation_enter_threshold", 0.35)
            ),
            intent_exit_threshold=float(
                rospy.get_param("~intent_exit_threshold", 0.20)
            ),
            group_switch_hysteresis=float(
                rospy.get_param("~group_switch_hysteresis", 0.15)
            ),
            axis_switch_hysteresis=float(
                rospy.get_param("~axis_switch_hysteresis", 0.25)
            ),
            secondary_axis_ratio=float(
                rospy.get_param("~secondary_axis_ratio", 0.90)
            ),
        )
        pose_config = PoseIntegratorConfig(
            linear_scale=self.linear_scale,
            angular_scale=self.angular_scale,
            max_linear_step=self.max_step_m,
            max_angular_step=self.max_angular_step,
            translation_limit=self.workspace_m,
            rotation_limit_rad=math.radians(self.rotation_limit_deg),
            rotation_frame=self.rotation_frame,
            allow_mixed_motion=False,
        )
        self._input = SpaceMouseInputProcessor(input_config)
        self._integrator = PoseIntegrator(pose_config)

        self._lock = threading.Lock()
        self._axes = None
        self._buttons = None
        self._last_message = None
        self._ctrl = None
        self._servo_ready = False
        self._session_active = False
        self._fault_latched = False
        self._fault_reason = None
        self._target_pose = None
        self._origin_pose = None
        self._was_pressed = False
        self._last_tick = None
        self._hand_adapter = None
        self._hand_button_was_pressed = False
        self._last_hand_request = float("-inf")

        self._subscriber = rospy.Subscriber(
            self.joy_topic,
            Joy,
            self._joy_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.on_shutdown(self.stop_robot)

    def _validate_config(self):
        if not (
            len(self.axis_map) == len(self.axis_sign) == len(self.axis_enable) == 6
        ):
            raise ValueError("axis_map, axis_sign and axis_enable must each have 6 items")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.axis_map):
            raise ValueError("axis_map must contain non-negative integers")
        if len(set(self.axis_map)) != 6:
            raise ValueError("axis_map entries must be unique")
        if any(sign not in (-1.0, 1.0) for sign in self.axis_sign):
            raise ValueError("axis_sign entries must be +1 or -1")
        if any(enabled not in (0, 1) for enabled in self.axis_enable):
            raise ValueError("axis_enable entries must be 0 or 1")
        if not any(self.axis_enable):
            raise ValueError("at least one axis must be enabled")
        if self.deadman_button < 0:
            raise ValueError("deadman_button must be non-negative")
        if self.hand_button < 0 or self.hand_button == self.deadman_button:
            raise ValueError("hand_button must be non-negative and differ from deadman_button")

        values = (
            self.watchdog,
            self.tcp_watchdog,
            self.state_watchdog,
            self.publish_rate,
            self.max_dt,
            self.servo_time,
            self.linear_scale,
            self.angular_scale,
            self.max_step_m,
            self.max_angular_step,
            self.workspace_m,
            self.rotation_limit_deg,
            self.hand_cooldown,
            self.hand_neutral_threshold,
            self.hand_feedback_timeout,
            self.hand_feedback_tolerance,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("all numeric parameters must be finite")
        if min(self.watchdog, self.tcp_watchdog, self.state_watchdog) <= 0.0:
            raise ValueError("all watchdog values must be greater than zero")
        if self.publish_rate <= 0.0 or self.max_dt <= 0.0:
            raise ValueError("publish_rate and max_dt must be greater than zero")
        if self.linear_scale < 0.0 or self.angular_scale < 0.0:
            raise ValueError("linear_scale and angular_scale must be non-negative")
        if self.linear_scale > 0.03:
            raise ValueError("linear_scale safety maximum is 0.03 m/s")
        if self.angular_scale > 0.30:
            raise ValueError("angular_scale safety maximum is 0.30 rad/s")
        if any(self.axis_enable[:3]) and self.linear_scale <= 0.0:
            raise ValueError("enabled translation axes require linear_scale > 0")
        if any(self.axis_enable[3:]) and self.angular_scale <= 0.0:
            raise ValueError("enabled rotation axes require angular_scale > 0")
        if self.max_step_m <= 0.0 or self.max_step_m > 0.002:
            raise ValueError("max_step_m must be in (0, 0.002]")
        if self.max_angular_step <= 0.0 or self.max_angular_step > 0.01:
            raise ValueError("max_angular_step must be in (0, 0.01]")
        if self.workspace_m <= 0.0 or self.workspace_m > 0.10:
            raise ValueError("workspace_m must be in (0, 0.10]")
        if self.rotation_limit_deg <= 0.0 or self.rotation_limit_deg > 90.0:
            raise ValueError("rotation_limit_deg must be in (0, 90]")
        if self.rotation_frame not in ("tool", "base"):
            raise ValueError("rotation_frame must be 'tool' or 'base'")
        if not np.isclose(self.servo_time, 0.02) or self.servo_gain != 800:
            raise ValueError("only verified ServoL parameters time=0.02 gain=800 are allowed")
        if self.hand_cooldown < 0.3:
            raise ValueError("hand_cooldown must be at least 0.3 seconds")
        if not 0.0 <= self.hand_neutral_threshold <= 0.20:
            raise ValueError("hand_neutral_threshold must be in [0, 0.20]")
        if self.hand_feedback_timeout <= 0.0:
            raise ValueError("hand_feedback_timeout must be greater than zero")
        if self.hand_feedback_tolerance <= 0.0:
            raise ValueError("hand_feedback_tolerance must be greater than zero")

    def _joy_callback(self, msg):
        with self._lock:
            self._axes = list(msg.axes)
            self._buttons = list(msg.buttons)
            self._last_message = time.monotonic()

    def _snapshot(self):
        with self._lock:
            axes = None if self._axes is None else self._axes.copy()
            buttons = None if self._buttons is None else self._buttons.copy()
            stamp = self._last_message
        return axes, buttons, stamp

    def _deadman_pressed(self, buttons):
        return (
            buttons is not None
            and 0 <= self.deadman_button < len(buttons)
            and buttons[self.deadman_button] == 1
        )

    @staticmethod
    def _button_pressed(buttons, index):
        return buttons is not None and 0 <= index < len(buttons) and buttons[index] == 1

    def _initialize_hand(self):
        self._hand_adapter = DexterousHandAdapter(
            controller=self._ctrl,
            hand=self.hand_type,
            grasp_target=self.grasp_target,
            release_target=self.release_target,
            initial_state=self.initial_hand_state,
            feedback_tolerance=self.hand_feedback_tolerance,
            execute=self.hand_execute,
        )
        if self.initial_hand_state == "auto":
            deadline = time.monotonic() + self.hand_feedback_timeout
            while (
                self._ctrl.get_hand_joints(self.hand_type) is None
                and time.monotonic() < deadline
                and not rospy.is_shutdown()
            ):
                rospy.sleep(0.02)
        state = self._hand_adapter.initialize()
        rospy.logwarn(
            "%s hand %s: initial_state=%s configured=%s feedback=%s "
            "grasp_target=%s release_target=%s",
            self.arm_side,
            "LIVE" if self.hand_execute else "DRY-RUN",
            state.value,
            self.initial_hand_state,
            self._ctrl.get_hand_joints(self.hand_type),
            self.grasp_target,
            self.release_target,
        )
        if state == HandState.UNKNOWN:
            rospy.logerr(
                "hand button inhibited: feedback is not close to grasp/release; "
                "set teleop.initial_hand_state to released or grasped"
            )

    def _on_hand_result(self, result):
        logger = rospy.loginfo if result.success else rospy.logerr
        logger(
            "hand command=%s success=%s dry_run=%s duration=%.3fs target=%s feedback=%s state=%s",
            result.command,
            result.success,
            result.dry_run,
            result.duration,
            list(result.target),
            None if result.feedback is None else list(result.feedback),
            self._hand_adapter.state.value,
        )

    def _handle_hand_button(
        self,
        rising,
        deadman_pressed,
        joy_fresh,
        axes,
        health_reason,
        now,
    ):
        if not rising:
            return
        reason = None
        if not joy_fresh:
            reason = "Joy watchdog expired"
        elif deadman_pressed:
            reason = "deadman must be released"
        elif self._session_active or self._was_pressed:
            reason = "ServoL session must be stopped"
        elif health_reason is not None:
            reason = health_reason
        elif axes is None or len(axes) <= max(self.axis_map):
            reason = "Joy axes missing"
        else:
            raw_axes = np.asarray(axes, dtype=np.float64)
            if not np.all(np.isfinite(raw_axes)):
                reason = "Joy axes contain non-finite values"
            elif float(np.max(np.abs(raw_axes))) > self.hand_neutral_threshold:
                reason = "SpaceMouse must be centered"
        if reason is None and now - self._last_hand_request < self.hand_cooldown:
            reason = "hand button cooldown active"
        if reason is not None:
            rospy.logwarn("hand button rejected: %s", reason)
            return

        accepted, detail = self._hand_adapter.request_toggle(self._on_hand_result)
        if accepted:
            self._last_hand_request = now
            rospy.logwarn("hand button accepted: %s requested", detail)
        else:
            rospy.logwarn("hand button rejected: %s", detail)

    def _mask_disabled_axes(self, axes):
        if axes is None or max(self.axis_map) >= len(axes):
            return None
        masked = list(axes)
        for output_index, enabled in enumerate(self.axis_enable):
            if not enabled:
                masked[self.axis_map[output_index]] = 0.0
        return masked

    def _health_snapshot(self):
        pose = self._ctrl.get_tcp_rt(self.arm_group)
        tcp_age = self._ctrl.get_tcp_age(self.arm_group)
        state_age = self._ctrl.get_uplimb_state_age()
        singular = self._ctrl.get_is_singular(self.arm_group)

        if pose is None or tcp_age is None:
            return pose, "{} TCP has not arrived".format(self.arm_side)
        if tcp_age > self.tcp_watchdog:
            return pose, "{} TCP stale: {:.3f}s".format(self.arm_side, tcp_age)
        if state_age is None:
            return pose, "UplimbState has not arrived"
        if state_age > self.state_watchdog:
            return pose, "UplimbState stale: {:.3f}s".format(state_age)
        if singular is None:
            return pose, "{} singular flag unavailable".format(self.arm_side)
        if singular:
            return pose, "{} arm is singular".format(self.arm_side)
        return pose, None

    def _wait_robot_ready(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        last_reason = "no state"
        while time.monotonic() < deadline and not rospy.is_shutdown():
            pose, reason = self._health_snapshot()
            if reason is None:
                return list(pose)
            last_reason = reason
            if "is singular" in reason:
                raise RuntimeError(reason)
            rospy.sleep(0.02)
        raise RuntimeError("robot state not ready: {}".format(last_reason))

    def _prepare_servo(self):
        if not self.execute or self._servo_ready:
            return True
        ok = self._ctrl.set_servo_params(
            self.servo_time, self.servo_gain, arm=self.arm_group
        )
        self._servo_ready = bool(ok)
        return self._servo_ready

    def _stop_servo_session(self):
        if self._ctrl is None or not self.execute:
            self._servo_ready = False
            self._session_active = False
            return
        if not (self._servo_ready or self._session_active):
            return
        ok_stop = self._ctrl.stop()
        ok_clear = self._ctrl.clear_servo_params()
        rospy.logwarn("ServoL stop=%s clear=%s", ok_stop, ok_clear)
        self._servo_ready = False
        self._session_active = False
        if not ok_stop or not ok_clear:
            self._fault_latched = True
            self._fault_reason = "ServoL stop/clear failed"

    def stop_robot(self):
        try:
            self._stop_servo_session()
        except Exception as exc:  # noqa: BLE001
            rospy.logerr("ServoL cleanup exception: %s", exc)
        finally:
            self._input.reset()
            self._target_pose = None
            self._origin_pose = None
            self._was_pressed = False
            self._last_tick = None

    def run(self):
        rospy.loginfo("waiting for SpaceMouse topic: %s", self.joy_topic)
        try:
            rospy.wait_for_message(self.joy_topic, Joy, timeout=5.0)
        except rospy.ROSException as exc:
            raise RuntimeError("no SpaceMouse Joy within 5s: {}".format(exc))

        self._ctrl = NaviController(model=RobotModel.WA2, auto_spin=True)
        current = self._wait_robot_ready()
        self._initialize_hand()
        self._integrator.reset(current)
        self._origin_pose = list(current)
        self._target_pose = list(current)

        # Preserve compatibility with existing live-gate helpers that wait for
        # servo readiness before publishing the first deadman sample.
        if self.execute and not self._prepare_servo():
            raise RuntimeError("WA2 set_servo_params failed")

        rospy.logwarn(
            "%s arm %s: axes=%s signs=%s enable=%s rotation_frame=%s limit=%.1fdeg",
            self.arm_side,
            "LIVE" if self.execute else "DRY-RUN",
            self.axis_map,
            self.axis_sign,
            self.axis_enable,
            self.rotation_frame,
            self.rotation_limit_deg,
        )

        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            now = time.monotonic()
            axes, buttons, joy_stamp = self._snapshot()
            joy_fresh = (
                joy_stamp is not None and now - joy_stamp <= self.watchdog
            )
            pressed = self._deadman_pressed(buttons)
            hand_pressed = self._button_pressed(buttons, self.hand_button)
            hand_rising = hand_pressed and not self._hand_button_was_pressed
            masked_axes = self._mask_disabled_axes(axes)
            tcp, health_reason = self._health_snapshot()
            hand_busy = (
                self._hand_adapter is not None
                and self._hand_adapter.state == HandState.BUSY
            )

            if not pressed and self._fault_latched:
                # A release is the explicit acknowledgement required before a
                # new session. Health is checked again on the next rising edge.
                rospy.logwarn("fault acknowledged by deadman release: %s", self._fault_reason)
                self._fault_latched = False
                self._fault_reason = None

            enabled = (
                pressed
                and joy_fresh
                and masked_axes is not None
                and health_reason is None
                and not hand_busy
                and not self._fault_latched
            )

            if pressed and not enabled:
                reason = health_reason
                if not joy_fresh:
                    reason = "Joy watchdog expired"
                elif masked_axes is None:
                    reason = "Joy axes missing"
                elif hand_busy:
                    reason = "hand command is still running"
                if reason and reason != self._fault_reason:
                    rospy.logerr("teleop inhibited: %s", reason)
                self._fault_latched = True
                self._fault_reason = reason or "teleop fault"

            if enabled and not self._was_pressed:
                self._input.reset()
                self._integrator.reset(tcp)
                self._origin_pose = list(tcp)
                self._target_pose = list(tcp)
                self._last_tick = now
                if not self._prepare_servo():
                    self._fault_latched = True
                    self._fault_reason = "set_servo_params failed"
                    enabled = False
                else:
                    self._session_active = self.execute

            dt = self.max_dt if self._last_tick is None else min(
                max(now - self._last_tick, 1e-6), self.max_dt
            )
            self._last_tick = now if enabled else None

            if not enabled:
                self._input.update([0.0] * 6, dt=max(dt, 1e-6), enabled=False)
                if self._was_pressed or (self.execute and self._fault_latched):
                    self._stop_servo_session()
                elif hand_rising and self.execute and self._servo_ready:
                    # Even a configured-but-idle ServoL session is cleared
                    # before an end-effector command is accepted.
                    self._stop_servo_session()
                self._handle_hand_button(
                    hand_rising,
                    pressed,
                    joy_fresh,
                    axes,
                    health_reason,
                    now,
                )
                self._was_pressed = False
                self._hand_button_was_pressed = hand_pressed
                rospy.loginfo_throttle(
                    0.5,
                    "idle joy_fresh=%s deadman=%s fault=%s target=%s",
                    joy_fresh,
                    pressed,
                    self._fault_reason,
                    None if self._target_pose is None else [round(x, 5) for x in self._target_pose],
                )
                rate.sleep()
                continue

            motion = self._input.update(masked_axes, dt=dt, enabled=True)
            command = motion.command.copy()
            command *= np.asarray(self.axis_enable, dtype=np.float64)
            target = self._integrator.step(command, dt=dt)
            self._origin_pose = list(self._integrator.origin_pose)
            self._target_pose = list(target)
            self._was_pressed = True
            self._handle_hand_button(
                hand_rising,
                pressed,
                joy_fresh,
                axes,
                health_reason,
                now,
            )
            self._hand_button_was_pressed = hand_pressed

            if self.execute:
                self._ctrl.servol(self._target_pose, self.arm_group)
            else:
                rospy.loginfo_throttle(
                    0.2,
                    "intent=%s axis=%s cmd=%s dxyz_mm=%.3f drot_deg=%.3f target=%s",
                    motion.intent.value,
                    motion.active_axis_name,
                    [round(float(x), 4) for x in command],
                    self._integrator.relative_translation * 1000.0,
                    math.degrees(self._integrator.relative_rotation_rad),
                    [round(float(x), 5) for x in target],
                )
            rate.sleep()


def main():
    rospy.init_node("spacemouse_wa2_teleop", anonymous=False)
    teleop = SpaceMouseWA2Teleop()
    try:
        teleop.run()
    finally:
        teleop.stop_robot()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TypeError, ValueError) as exc:
        rospy.logfatal("SpaceMouse teleop startup failed: %s", exc)
        raise SystemExit(1)
