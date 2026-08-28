"""Pure-Python SpaceMouse mapping, intent classification, and smoothing.

This module deliberately has no ROS or robot dependency.  It converts the six
raw ``sensor_msgs/Joy.axes`` values into a normalized motion command.

Translation and rotation stay mutually exclusive (device crosstalk).  Inside a
group, near-equal axes may move together so diagonals are not stair-stepped;
weaker coupled axes stay suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Sequence, Tuple

import numpy as np


AXIS_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")


class MotionIntent(Enum):
    """High-level operator intent inferred from the SpaceMouse input."""

    IDLE = "idle"
    TRANSLATION = "translation"
    ROTATION = "rotation"


@dataclass(frozen=True)
class SpaceMouseInputConfig:
    """Configuration for :class:`SpaceMouseInputProcessor`.

    The defaults are derived from the 2026-08 physical calibration:

    * +X: push forward (``-axes[0]``)
    * +Y: push left (``-axes[1]``)
    * +Z: pull up (``+axes[2]``)
    * +Roll: tilt left (``+axes[3]``)
    * +Pitch: tilt forward (``-axes[4]``)
    * +Yaw: twist clockwise (``-axes[5]``)
    """

    axis_map: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    axis_sign: Tuple[float, ...] = (-1.0, -1.0, 1.0, 1.0, -1.0, -1.0)
    translation_deadzone: float = 0.15
    rotation_deadzone: float = 0.18
    translation_curve_mix: float = 0.25
    rotation_curve_mix: float = 0.45
    translation_filter_tau: float = 0.06
    rotation_filter_tau: float = 0.10
    rotation_enter_threshold: float = 0.65
    translation_enter_threshold: float = 0.35
    intent_exit_threshold: float = 0.20
    group_switch_hysteresis: float = 0.15
    axis_switch_hysteresis: float = 0.25
    secondary_axis_ratio: float = 0.90

    def __post_init__(self) -> None:
        if len(self.axis_map) != 6 or len(self.axis_sign) != 6:
            raise ValueError("axis_map and axis_sign must each contain 6 items")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in self.axis_map):
            raise ValueError("axis_map must contain 6 non-negative integers")
        if len(set(self.axis_map)) != 6:
            raise ValueError("axis_map entries must be unique")
        if any(sign not in (-1.0, 1.0) for sign in self.axis_sign):
            raise ValueError("axis_sign entries must be +1 or -1")

        finite_values = (
            self.translation_deadzone,
            self.rotation_deadzone,
            self.translation_curve_mix,
            self.rotation_curve_mix,
            self.translation_filter_tau,
            self.rotation_filter_tau,
            self.rotation_enter_threshold,
            self.translation_enter_threshold,
            self.intent_exit_threshold,
            self.group_switch_hysteresis,
            self.axis_switch_hysteresis,
            self.secondary_axis_ratio,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise ValueError("all SpaceMouse configuration values must be finite")
        if not 0.0 <= self.translation_deadzone < 1.0:
            raise ValueError("translation_deadzone must be in [0, 1)")
        if not 0.0 <= self.rotation_deadzone < 1.0:
            raise ValueError("rotation_deadzone must be in [0, 1)")
        if not 0.0 <= self.translation_curve_mix <= 1.0:
            raise ValueError("translation_curve_mix must be in [0, 1]")
        if not 0.0 <= self.rotation_curve_mix <= 1.0:
            raise ValueError("rotation_curve_mix must be in [0, 1]")
        if self.translation_filter_tau < 0.0 or self.rotation_filter_tau < 0.0:
            raise ValueError("filter time constants must be non-negative")
        if not 0.0 < self.intent_exit_threshold < self.translation_enter_threshold:
            raise ValueError(
                "intent_exit_threshold must be positive and lower than translation_enter_threshold"
            )
        if self.rotation_enter_threshold <= self.intent_exit_threshold:
            raise ValueError(
                "rotation_enter_threshold must be greater than intent_exit_threshold"
            )
        if self.rotation_enter_threshold > 1.0 or self.translation_enter_threshold > 1.0:
            raise ValueError("intent entry thresholds must not exceed 1")
        if self.group_switch_hysteresis < 0.0 or self.axis_switch_hysteresis < 0.0:
            raise ValueError("group and axis switch hysteresis must be non-negative")
        if not 0.0 <= self.secondary_axis_ratio <= 1.0:
            raise ValueError("secondary_axis_ratio must be in [0, 1]")


@dataclass(frozen=True)
class ProcessedMotion:
    """One processed SpaceMouse sample.

    ``command`` is ordered as ``[x, y, z, roll, pitch, yaw]`` and remains
    normalized to approximately ``[-1, 1]``.  Physical velocity scaling belongs
    to the pose integrator or robot-control layer.
    """

    intent: MotionIntent
    active_axis: Optional[int]
    command: np.ndarray
    mapped_axes: np.ndarray

    @property
    def active_axis_name(self) -> Optional[str]:
        return None if self.active_axis is None else AXIS_NAMES[self.active_axis]

    @property
    def translation(self) -> np.ndarray:
        return self.command[:3].copy()

    @property
    def rotation(self) -> np.ndarray:
        return self.command[3:].copy()


class SpaceMouseInputProcessor:
    """Stateful trans/rot classifier with optional in-group multi-axis output."""

    def __init__(self, config: Optional[SpaceMouseInputConfig] = None):
        self.config = config or SpaceMouseInputConfig()
        self._intent = MotionIntent.IDLE
        self._active_axis: Optional[int] = None
        self._filtered = np.zeros(6, dtype=np.float64)

    @property
    def intent(self) -> MotionIntent:
        return self._intent

    @property
    def active_axis(self) -> Optional[int]:
        return self._active_axis

    def reset(self) -> None:
        """Immediately clear intent, axis lock, and all filter history."""

        self._intent = MotionIntent.IDLE
        self._active_axis = None
        self._filtered.fill(0.0)

    def map_axes(self, raw_axes: Sequence[float]) -> np.ndarray:
        """Map raw Joy axes to calibrated ``[X,Y,Z,Roll,Pitch,Yaw]`` order."""

        if raw_axes is None:
            raise ValueError("raw_axes must not be None")
        if max(self.config.axis_map) >= len(raw_axes):
            raise ValueError(
                "raw_axes length {} does not satisfy axis_map {}".format(
                    len(raw_axes), self.config.axis_map
                )
            )
        raw = np.asarray(raw_axes, dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            raise ValueError("raw_axes must contain only finite values")

        mapped = np.asarray(
            [raw[source] * sign for source, sign in zip(self.config.axis_map, self.config.axis_sign)],
            dtype=np.float64,
        )
        return np.clip(mapped, -1.0, 1.0)

    def update(
        self,
        raw_axes: Sequence[float],
        dt: float,
        enabled: bool = True,
    ) -> ProcessedMotion:
        """Process a Joy sample.

        Setting ``enabled=False`` is the deadman/watchdog path.  It clears all
        state immediately rather than allowing a filtered command to decay.
        """

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and greater than zero")
        mapped = self.map_axes(raw_axes)

        if not enabled:
            self.reset()
            return self._result(mapped)

        translation_max = float(np.max(np.abs(mapped[:3])))
        rotation_max = float(np.max(np.abs(mapped[3:])))
        next_intent = self._classify_intent(translation_max, rotation_max)

        if next_intent != self._intent:
            self._intent = next_intent
            self._active_axis = None
            self._filtered.fill(0.0)

        if self._intent == MotionIntent.IDLE:
            self._active_axis = None
            self._filtered.fill(0.0)
            return self._result(mapped)

        group_start = 0 if self._intent == MotionIntent.TRANSLATION else 3
        group_values = mapped[group_start : group_start + 3]
        candidate_axis = group_start + int(np.argmax(np.abs(group_values)))
        self._update_axis_lock(candidate_axis, mapped)
        assert self._active_axis is not None

        deadzone = (
            self.config.translation_deadzone
            if self._intent == MotionIntent.TRANSLATION
            else self.config.rotation_deadzone
        )
        curve_mix = (
            self.config.translation_curve_mix
            if self._intent == MotionIntent.TRANSLATION
            else self.config.rotation_curve_mix
        )
        dominant_mag = abs(float(mapped[self._active_axis]))
        selected = np.zeros(6, dtype=np.float64)
        for axis in range(group_start, group_start + 3):
            magnitude = abs(float(mapped[axis]))
            if magnitude <= deadzone:
                continue
            if axis != self._active_axis:
                if dominant_mag < 1e-9:
                    continue
                if magnitude < dominant_mag * self.config.secondary_axis_ratio:
                    continue
            selected[axis] = self._shape(
                self._apply_deadband(mapped[axis], deadzone), curve_mix
            )

        tau = (
            self.config.translation_filter_tau
            if self._intent == MotionIntent.TRANSLATION
            else self.config.rotation_filter_tau
        )
        alpha = 1.0 if tau == 0.0 else 1.0 - math.exp(-dt / tau)
        self._filtered += alpha * (selected - self._filtered)

        # Keep the other group exactly zero; in-group unused axes decay via the filter.
        other = np.ones(6, dtype=bool)
        other[group_start : group_start + 3] = False
        self._filtered[other] = 0.0
        return self._result(mapped)

    def _classify_intent(
        self, translation_max: float, rotation_max: float
    ) -> MotionIntent:
        cfg = self.config
        if self._intent == MotionIntent.ROTATION:
            if rotation_max >= cfg.rotation_enter_threshold:
                return MotionIntent.ROTATION
            if (
                translation_max >= cfg.translation_enter_threshold
                and translation_max
                >= rotation_max + cfg.group_switch_hysteresis
            ):
                return MotionIntent.TRANSLATION
            if rotation_max > cfg.intent_exit_threshold:
                return MotionIntent.ROTATION
            return MotionIntent.IDLE

        if self._intent == MotionIntent.TRANSLATION:
            if rotation_max >= cfg.rotation_enter_threshold:
                return MotionIntent.ROTATION
            if translation_max > cfg.intent_exit_threshold:
                return MotionIntent.TRANSLATION
            return MotionIntent.IDLE

        if rotation_max >= cfg.rotation_enter_threshold:
            return MotionIntent.ROTATION
        if translation_max >= cfg.translation_enter_threshold:
            return MotionIntent.TRANSLATION
        return MotionIntent.IDLE

    def _update_axis_lock(self, candidate_axis: int, mapped: np.ndarray) -> bool:
        if self._active_axis is None:
            self._active_axis = candidate_axis
            return True
        if candidate_axis == self._active_axis:
            return False

        candidate_magnitude = abs(float(mapped[candidate_axis]))
        current_magnitude = abs(float(mapped[self._active_axis]))
        if candidate_magnitude >= current_magnitude + self.config.axis_switch_hysteresis:
            self._active_axis = candidate_axis
            return True
        return False

    @staticmethod
    def _apply_deadband(value: float, deadzone: float) -> float:
        value = max(-1.0, min(1.0, float(value)))
        if abs(value) <= deadzone:
            return 0.0
        return math.copysign(
            (abs(value) - deadzone) / (1.0 - deadzone), value
        )

    @staticmethod
    def _shape(value: float, mix: float) -> float:
        return (1.0 - mix) * value + mix * value**3

    def _result(self, mapped: np.ndarray) -> ProcessedMotion:
        return ProcessedMotion(
            intent=self._intent,
            active_axis=self._active_axis,
            command=self._filtered.copy(),
            mapped_axes=mapped.copy(),
        )
