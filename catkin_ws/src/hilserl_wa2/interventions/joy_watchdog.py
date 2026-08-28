"""Joy age watchdog: ROS /spacenav/joy or synthetic injection (no robot cmds)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class JoySample:
    axes: np.ndarray
    buttons: np.ndarray
    stamp_mono: float

    @property
    def age_s(self) -> float:
        return max(0.0, time.monotonic() - self.stamp_mono)


class JoyWatchdog:
    """Latest Joy cache with max-age freshness.

    - ``inject()`` for offline tests (no ROS).
    - ``start_ros()`` subscribes to ``/spacenav/joy`` for live use.
    """

    def __init__(
        self,
        topic: str = "/spacenav/joy",
        max_age_s: float = 0.2,
    ):
        self.topic = str(topic)
        self.max_age_s = float(max_age_s)
        self._lock = threading.Lock()
        self._sample: Optional[JoySample] = None
        self._forced_age_s: Optional[float] = None
        self._started = False
        self._sub = None
        self._rospy = None

    def inject(
        self,
        axes: Sequence[float],
        buttons: Optional[Sequence[float]] = None,
        *,
        stamp: Optional[float] = None,
    ) -> None:
        ax = np.asarray(axes, dtype=np.float64).reshape(-1)
        if ax.size < 6:
            raise ValueError("axes must have length >= 6")
        if not np.all(np.isfinite(ax)):
            raise ValueError("axes must be finite")
        btn = np.asarray(
            [0, 0] if buttons is None else buttons, dtype=np.float64
        ).reshape(-1)
        with self._lock:
            self._sample = JoySample(
                axes=ax.copy(),
                buttons=btn.copy(),
                stamp_mono=time.monotonic() if stamp is None else float(stamp),
            )
            self._forced_age_s = None

    def inject_stale_for_test(self, age_s: float = 1.0) -> None:
        with self._lock:
            self._forced_age_s = float(age_s)

    def clear_stale_injection(self) -> None:
        with self._lock:
            self._forced_age_s = None

    def get_sample(self) -> Optional[JoySample]:
        with self._lock:
            if self._sample is None:
                return None
            return JoySample(
                axes=self._sample.axes.copy(),
                buttons=self._sample.buttons.copy(),
                stamp_mono=self._sample.stamp_mono,
            )

    def get_age(self) -> Optional[float]:
        with self._lock:
            if self._forced_age_s is not None:
                return float(self._forced_age_s)
            if self._sample is None:
                return None
            return max(0.0, time.monotonic() - self._sample.stamp_mono)

    def is_fresh(self) -> bool:
        age = self.get_age()
        return age is not None and age <= self.max_age_s

    def start_ros(self) -> None:
        if self._started:
            return
        import rospy
        from sensor_msgs.msg import Joy

        self._rospy = rospy
        if not rospy.core.is_initialized():
            rospy.init_node("wa2_joy_watchdog", anonymous=True, disable_signals=True)

        def _cb(msg):
            self.inject(list(msg.axes), list(msg.buttons))

        self._sub = rospy.Subscriber(self.topic, Joy, _cb, queue_size=1)
        self._started = True

    def stop(self) -> None:
        if self._sub is not None:
            try:
                self._sub.unregister()
            except Exception:
                pass
            self._sub = None
        self._started = False

    def wait_ready(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.is_fresh():
                return
            time.sleep(0.01)
        raise TimeoutError(
            f"JoyWatchdog not fresh within {timeout_s}s on {self.topic}; "
            f"age={self.get_age()}"
        )
