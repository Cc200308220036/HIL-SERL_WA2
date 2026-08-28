"""WA2 SpaceMouse Intervention ActionWrapper (ROS Joy → Env action).

Does NOT call ServoL/hand services. Uses ROS /spacenav/joy only (not Franka HID expert).
Defaults load from ``configs/spacemouse/default.yaml`` unless overridden.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Any, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np

from hilserl_wa2.interventions.joy_watchdog import JoyWatchdog
from hilserl_wa2.interventions.spacemouse_config import (
    SpaceMouseRuntimeConfig,
    load_spacemouse_config,
)
from hilserl_wa2.interventions.spacemouse_input import (
    SpaceMouseInputConfig,
    SpaceMouseInputProcessor,
)


class WA2SpacemouseIntervention(gym.ActionWrapper):
    """Override policy actions with SpaceMouse in toggle or hold session.

    ``session_mode=toggle`` (default): tap left to enter, tap left again to exit.
    While the session is on, **policy is discarded** (idle stick = hold / zeros).
    Right button is reported in ``info['sm_right']``; the recorder/HIL loop owns the hand.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        config_path: Optional[Union[str, Path]] = None,
        runtime_config: Optional[SpaceMouseRuntimeConfig] = None,
        joy_watchdog: Optional[JoyWatchdog] = None,
        joy_topic: Optional[str] = None,
        joy_max_age_s: Optional[float] = None,
        deadman_button: Optional[int] = None,
        hand_button: Optional[int] = None,
        intervene_eps: Optional[float] = None,
        control_dt: Optional[float] = None,
        input_config: Optional[SpaceMouseInputConfig] = None,
        auto_start_ros: Optional[bool] = None,
        session_mode: Optional[str] = None,
    ):
        super().__init__(env)
        if self.action_space.shape != (6,):
            raise ValueError(
                f"WA2 intervention expects action_space shape (6,), "
                f"got {self.action_space.shape}"
            )

        cfg = runtime_config or load_spacemouse_config(config_path)
        self.runtime_config = cfg

        self.deadman_button = int(
            cfg.deadman_button if deadman_button is None else deadman_button
        )
        self.hand_button = int(
            cfg.hand_button if hand_button is None else hand_button
        )
        self.session_mode = str(
            session_mode if session_mode is not None else cfg.session_mode
        ).strip().lower()
        if self.session_mode not in ("toggle", "hold"):
            raise ValueError("session_mode must be toggle|hold")
        self.intervene_eps = float(
            cfg.intervene_eps if intervene_eps is None else intervene_eps
        )
        self.control_dt = float(
            cfg.control_dt if control_dt is None else control_dt
        )
        self.action_gain = float(cfg.action_gain)
        resolved_input = input_config or cfg.input_config
        self.processor = SpaceMouseInputProcessor(resolved_input)

        topic = cfg.joy_topic if joy_topic is None else joy_topic
        max_age = cfg.joy_max_age_s if joy_max_age_s is None else joy_max_age_s
        self.joy = joy_watchdog or JoyWatchdog(topic=topic, max_age_s=max_age)

        self._auto_start_ros = bool(
            cfg.auto_start_ros if auto_start_ros is None else auto_start_ros
        )
        self._ros_started = False
        self._last_intervened = False
        self._last_sm_left = False
        self._last_sm_right = False
        self._session_active = False
        self._session_exit = False
        self._session_enter = False
        self._prev_left = False
        self._session_dropped_stale = False
        self._last_intent = "idle"
        self._last_axis = None
        self._intervention_steps = 0
        self._intervention_count = 0
        self._in_intervention_segment = False
        self._last_processor_t: Optional[float] = None
        self._processor_max_dt = 0.05
        self._pending_session_enter = False
        self._need_left_release = False
        self._interrupt_lock = threading.Lock()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._ensure_ros()
        self.processor.reset()
        self._last_intervened = False
        self._in_intervention_segment = False
        self._intervention_steps = 0
        self._intervention_count = 0
        self._session_active = False
        self._session_exit = False
        self._session_enter = False
        self._prev_left = False
        self._session_dropped_stale = False
        self._last_intent = "idle"
        self._last_axis = None
        self._last_processor_t = None
        with self._interrupt_lock:
            self._pending_session_enter = False
            self._need_left_release = False
        return self.env.reset(seed=seed, options=options)

    def action(self, action: Any) -> Tuple[np.ndarray, bool]:
        """Return (exec_action, intervened)."""

        policy = np.asarray(action, dtype=np.float32).reshape(-1)
        if policy.shape != (6,):
            raise ValueError(f"policy action must be shape (6,), got {policy.shape}")
        if not np.all(np.isfinite(policy)):
            raise ValueError("policy action must be finite")

        sample = self.joy.get_sample()
        fresh = self.joy.is_fresh()
        with self._interrupt_lock:
            pending_enter = bool(self._pending_session_enter)
            self._pending_session_enter = False
            need_release = bool(self._need_left_release)
        left = False
        hand_btn = False
        if sample is not None and sample.buttons.size > self.deadman_button:
            left = bool(sample.buttons[self.deadman_button] > 0.5)
        if sample is not None and sample.buttons.size > self.hand_button:
            hand_btn = bool(sample.buttons[self.hand_button] > 0.5)
        self._last_sm_left = left
        self._last_sm_right = hand_btn
        self._session_exit = False
        self._session_enter = False
        self._session_dropped_stale = False

        # Stale Joy while a toggle session is open: hold zeros and KEEP the
        # session. Falling back to SAC here feels like "I pressed left but the
        # arm keeps charging" and has hit workspace limits on the real robot.
        hold_stale_session = False
        if not fresh:
            if self._session_active:
                self._session_dropped_stale = True
                hold_stale_session = True
                enabled = False
            else:
                self._session_active = False
                self._prev_left = False
                enabled = False
        elif self.session_mode == "hold":
            enabled = bool((left or pending_enter) and sample is not None)
            self._session_active = enabled
            self._prev_left = left
            if not left:
                with self._interrupt_lock:
                    self._need_left_release = False
        else:
            # After an exit (or mid-press enter), ignore further enter until the
            # left button has been fully released once. Otherwise a still-held
            # tap re-arms pending_enter on the next policy window and the
            # session looks impossible to leave.
            if need_release:
                pending_enter = False
                if not left:
                    with self._interrupt_lock:
                        self._need_left_release = False
                    need_release = False
            rising = bool(
                (not need_release)
                and (pending_enter or (left and not self._prev_left))
            )
            self._prev_left = left
            if rising:
                if self._session_active:
                    self._session_active = False
                    self._session_exit = True
                    self.processor.reset()
                    with self._interrupt_lock:
                        self._need_left_release = True
                else:
                    self._session_active = True
                    self._session_enter = True
                    self.processor.reset()
                    with self._interrupt_lock:
                        self._need_left_release = True
            enabled = bool(self._session_active and sample is not None)

        if sample is None or hold_stale_session:
            axes = np.zeros(6, dtype=np.float64)
        else:
            axes = sample.axes

        motion = self.processor.update(
            axes, dt=self._processor_dt(), enabled=bool(enabled and not hold_stale_session)
        )
        self._last_intent = "hold_stale" if hold_stale_session else str(motion.intent.value)
        self._last_axis = None if hold_stale_session else motion.active_axis_name
        sm_action = np.asarray(motion.command, dtype=np.float32).reshape(6)
        sm_action = (sm_action * np.float32(self.action_gain)).astype(np.float32)
        sm_action = np.clip(
            sm_action,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)

        # Session on ⇒ exclusive human control. Do not fall back to policy when
        # the stick is inside the deadzone (that feels like fighting the SAC).
        # Stale-during-session also holds zeros (never SAC).
        if hold_stale_session:
            return np.zeros(6, dtype=np.float32), True
        if enabled:
            return sm_action, True
        return policy.astype(np.float32), False

    def step(
        self, action: Any
    ) -> Tuple[Any, SupportsFloat, bool, bool, dict]:
        exec_action, intervened = self.action(action)
        base = self._base_env()
        can_cancel = hasattr(base, "set_action_interrupt_callback")
        can_provider = hasattr(base, "set_action_provider_callback")
        if can_provider:
            # T3-05: human windows re-sample SpaceMouse every Servo tick (~50 Hz).
            # Policy windows keep a constant latch (provider unset).
            base.set_action_provider_callback(
                self._human_tick_command if intervened else None
            )
        if can_cancel:
            if intervened:
                base.set_action_interrupt_callback(self._request_session_exit)
            else:
                base.set_action_interrupt_callback(self._request_policy_interrupt)
        try:
            obs, reward, terminated, truncated, info = self.env.step(exec_action)
        finally:
            if can_provider:
                base.set_action_provider_callback(None)
            if can_cancel:
                base.set_action_interrupt_callback(None)
        info = dict(info)
        info["intervened"] = bool(intervened)
        info["sm_left"] = bool(self._last_sm_left)
        info["sm_right"] = bool(self._last_sm_right)
        info["sm_session"] = bool(self._session_active)
        info["sm_session_enter"] = bool(self._session_enter)
        info["sm_session_exit"] = bool(self._session_exit)
        info["sm_session_dropped_stale"] = bool(self._session_dropped_stale)
        info["sm_intent"] = str(self._last_intent)
        info["sm_axis"] = self._last_axis
        info["session_mode"] = str(self.session_mode)
        info["joy_age"] = self.joy.get_age()
        info["joy_fresh"] = bool(self.joy.is_fresh())
        if intervened:
            # Prefer mean of per-tick human commands over the single sample taken
            # at high-level step entry (matches actual Servo integrations).
            if info.get("window_action_mean") is not None:
                info["intervene_action"] = np.asarray(
                    info["window_action_mean"], dtype=np.float32
                ).reshape(-1).copy()
            else:
                info["intervene_action"] = np.asarray(
                    exec_action, dtype=np.float32
                ).copy()
            self._intervention_steps += 1
            if not self._in_intervention_segment:
                self._intervention_count += 1
                self._in_intervention_segment = True
        else:
            info.pop("intervene_action", None)
            self._in_intervention_segment = False
        info["intervention_steps"] = int(self._intervention_steps)
        info["intervention_count"] = int(self._intervention_count)
        self._last_intervened = intervened
        return obs, reward, terminated, truncated, info

    def _base_env(self) -> Any:
        cur = self.env
        seen = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if type(cur).__name__ == "WA2Env":
                return cur
            inner = getattr(cur, "env", None)
            if inner is None or inner is cur:
                break
            cur = inner
        return self.env

    def _human_tick_command(self) -> np.ndarray:
        """Non-blocking SpaceMouse sample for one 20 ms Servo tick (T3-05).

        Does not toggle the session; session enter/exit is handled at the
        high-level step boundary or via ``_request_session_exit``.
        """

        sample = self.joy.get_sample()
        fresh = self.joy.is_fresh()
        if (not fresh) or sample is None:
            if sample is not None and sample.buttons.size > self.deadman_button:
                self._last_sm_left = bool(
                    sample.buttons[self.deadman_button] > 0.5
                )
            if sample is not None and sample.buttons.size > self.hand_button:
                self._last_sm_right = bool(sample.buttons[self.hand_button] > 0.5)
            self._last_intent = "hold_stale"
            self._last_axis = None
            return np.zeros(6, dtype=np.float32)

        left = False
        hand_btn = False
        if sample.buttons.size > self.deadman_button:
            left = bool(sample.buttons[self.deadman_button] > 0.5)
        if sample.buttons.size > self.hand_button:
            hand_btn = bool(sample.buttons[self.hand_button] > 0.5)
        self._last_sm_left = left
        self._last_sm_right = hand_btn

        motion = self.processor.update(
            sample.axes,
            dt=float(self.control_dt),
            enabled=True,
        )
        self._last_intent = str(motion.intent.value)
        self._last_axis = motion.active_axis_name
        sm_action = np.asarray(motion.command, dtype=np.float32).reshape(6)
        sm_action = (sm_action * np.float32(self.action_gain)).astype(np.float32)
        return np.clip(
            sm_action, self.action_space.low, self.action_space.high
        ).astype(np.float32)

    def _request_policy_interrupt(self) -> bool:
        """Fast, non-motion callback polled before every policy Servo tick."""

        if self._session_active or not self.joy.is_fresh():
            return False
        sample = self.joy.get_sample()
        if sample is None or sample.buttons.size <= self.deadman_button:
            return False
        pressed = bool(sample.buttons[self.deadman_button] > 0.5)
        with self._interrupt_lock:
            if self._need_left_release:
                if not pressed:
                    self._need_left_release = False
                return False
            # Rising edge only — a still-held exit tap must not re-enter.
            rising = bool(pressed and not self._prev_left)
            if rising:
                self._pending_session_enter = True
                self._prev_left = True
            elif not pressed:
                self._prev_left = False
        return rising

    def _request_session_exit(self) -> bool:
        """Cancel a human window early when the operator leaves the session."""

        if not self._session_active or not self.joy.is_fresh():
            return False
        sample = self.joy.get_sample()
        if sample is None or sample.buttons.size <= self.deadman_button:
            return False
        left = bool(sample.buttons[self.deadman_button] > 0.5)
        if self.session_mode == "hold":
            if left:
                self._prev_left = True
                return False
            self._session_active = False
            self._session_exit = True
            self._prev_left = False
            self.processor.reset()
            with self._interrupt_lock:
                self._need_left_release = True
            return True

        rising = bool(left and not self._prev_left)
        self._prev_left = left
        if not rising:
            return False
        self._session_active = False
        self._session_exit = True
        self.processor.reset()
        with self._interrupt_lock:
            self._need_left_release = True
        return True

    def close(self):
        try:
            self.joy.stop()
        except Exception:
            pass
        return self.env.close()

    def _processor_dt(self) -> float:
        now = time.monotonic()
        if self._last_processor_t is None:
            dt = float(self.control_dt)
        else:
            dt = min(
                max(now - self._last_processor_t, 1e-6),
                float(self._processor_max_dt),
            )
        self._last_processor_t = now
        return dt

    def _ensure_ros(self) -> None:
        if not self._auto_start_ros or self._ros_started:
            return
        try:
            self.joy.start_ros()
            self._ros_started = True
        except Exception:
            self._ros_started = False
