"""Non-blocking end-effector adapter used by SpaceMouse teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


class HandState(Enum):
    UNKNOWN = "unknown"
    RELEASED = "released"
    GRASPED = "grasped"
    BUSY = "busy"


@dataclass(frozen=True)
class HandCommandResult:
    command: str
    target: Tuple[float, ...]
    success: bool
    duration: float
    feedback: Optional[Tuple[float, ...]]
    dry_run: bool = False


class DexterousHandAdapter:
    """Toggle a same-side dexterous hand without blocking the servo loop."""

    def __init__(
        self,
        controller,
        hand,
        grasp_target: Sequence[float],
        release_target: Sequence[float] = (0.0,) * 6,
        initial_state: str = "auto",
        feedback_tolerance: float = 0.35,
        execute: bool = False,
    ):
        self._controller = controller
        self._hand = hand
        self.grasp_target = self._validate_target(grasp_target, "grasp_target")
        self.release_target = self._validate_target(release_target, "release_target")
        if initial_state not in ("auto", "released", "grasped"):
            raise ValueError("initial_hand_state must be auto, released or grasped")
        if not math.isfinite(feedback_tolerance) or feedback_tolerance <= 0.0:
            raise ValueError("hand_feedback_tolerance must be greater than zero")

        self.initial_state = initial_state
        self.feedback_tolerance = float(feedback_tolerance)
        self.execute = bool(execute)
        self._lock = threading.Lock()
        self._stable_state = HandState.UNKNOWN
        self._busy = False
        self._thread = None
        self._last_result = None

    @staticmethod
    def _validate_target(values, name):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (6,) or not np.all(np.isfinite(array)):
            raise ValueError("{} must contain six finite values".format(name))
        return tuple(float(value) for value in array)

    @staticmethod
    def _is_zero_target(values):
        return all(abs(float(value)) <= 1e-12 for value in values)

    def _execute_command(self, command, target):
        joints = list(target)
        if command == "grasp":
            return bool(self._controller.grasp_hand(self._hand, joints))
        if self._is_zero_target(target):
            return bool(self._controller.release_hand(self._hand))
        return bool(self._controller.grasp_hand(self._hand, joints))

    @property
    def state(self):
        with self._lock:
            return HandState.BUSY if self._busy else self._stable_state

    @property
    def last_result(self):
        with self._lock:
            return self._last_result

    def _feedback(self):
        joints = self._controller.get_hand_joints(self._hand)
        if joints is None:
            return None
        try:
            return self._validate_target(joints, "hand feedback")
        except (TypeError, ValueError):
            return None

    def initialize(self):
        """Infer the stable state, or apply an explicit startup state.

        ``released`` / ``grasped`` trust the YAML value and do not require the
        current joints to match ``release_target`` or ``grasp_target``.
        """
        if self.initial_state == "released":
            state = HandState.RELEASED
        elif self.initial_state == "grasped":
            state = HandState.GRASPED
        else:
            feedback = self._feedback()
            if feedback is None:
                state = HandState.UNKNOWN
            else:
                joints = np.asarray(feedback)
                release_distance = float(
                    np.linalg.norm(joints - np.asarray(self.release_target))
                )
                grasp_distance = float(
                    np.linalg.norm(joints - np.asarray(self.grasp_target))
                )
                if release_distance <= self.feedback_tolerance and release_distance < grasp_distance:
                    state = HandState.RELEASED
                elif grasp_distance <= self.feedback_tolerance and grasp_distance < release_distance:
                    state = HandState.GRASPED
                else:
                    state = HandState.UNKNOWN
        with self._lock:
            if not self._busy:
                self._stable_state = state
        return state

    def request_toggle(
        self,
        callback: Optional[Callable[[HandCommandResult], None]] = None,
    ):
        """Start one open/close command and return ``(accepted, reason)``."""
        with self._lock:
            if self._busy:
                return False, "hand command is already running"
            if self._stable_state == HandState.UNKNOWN:
                return False, "hand state is unknown"
            previous = self._stable_state
            command = "grasp" if previous == HandState.RELEASED else "release"
            target = self.grasp_target if command == "grasp" else self.release_target
            self._busy = True

        self._thread = threading.Thread(
            target=self._run_command,
            args=(previous, command, target, callback),
            daemon=True,
        )
        self._thread.start()
        return True, command

    def _run_command(self, previous, command, target, callback):
        started = time.monotonic()
        try:
            if self.execute:
                success = self._execute_command(command, target)
            else:
                success = True
            feedback = self._feedback()
        except Exception:  # The result is reported to the ROS-facing caller.
            success = False
            feedback = self._feedback()

        result = HandCommandResult(
            command=command,
            target=target,
            success=success,
            duration=time.monotonic() - started,
            feedback=feedback,
            dry_run=not self.execute,
        )
        with self._lock:
            if success:
                self._stable_state = (
                    HandState.GRASPED if command == "grasp" else HandState.RELEASED
                )
            else:
                self._stable_state = previous
            self._busy = False
            self._last_result = result
        if callback is not None:
            callback(result)

    def wait_idle(self, timeout=2.0):
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
        return self.state != HandState.BUSY
