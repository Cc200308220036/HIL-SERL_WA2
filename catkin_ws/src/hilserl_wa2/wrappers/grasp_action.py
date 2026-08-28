"""R13 7D grasp wrapper. ServoL stays 6D; last dim is discrete grasp/release."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


GRASP_DIM = 7
ARM_DIM = 6


def discretize_grasp(value: Any) -> int:
    """Map continuous last-dim to {-1, 0, +1}."""

    scalar = float(np.asarray(value).reshape(-1)[-1])
    rounded = int(np.clip(np.round(scalar), -1, 1))
    return rounded


class WA2GraspActionWrapper(gym.ActionWrapper):
    """Expand action to 7D and edge-trigger ``request_hand``.

    Policy / human arm commands occupy ``[:6]``. ``[6]`` is grasp:
    ``+1`` grasp, ``-1`` release, ``0`` hold. Right-click (``sm_right``)
    from the intervention wrapper is treated as a human toggle edge and
    marked as ``intervene_action`` so the step is dual-written to Demo Buffer.

    Only ``request_hand`` results with ``ok=True`` are recorded as executed
    ``±1`` in ``grasp_command`` / ``intervene_action``. Failures keep the
    7th dim at 0 and set ``hand_exec_failed`` so training labels match physics.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        low = np.full((GRASP_DIM,), -1.0, dtype=np.float32)
        high = np.full((GRASP_DIM,), 1.0, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self._prev_sm_right = False
        self._last_nonzero = 0

    def reset(self, **kwargs):
        self._prev_sm_right = False
        self._last_nonzero = 0
        return self.env.reset(**kwargs)

    def action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] == ARM_DIM:
            return np.clip(arr, -1.0, 1.0).astype(np.float32)
        if arr.shape[0] != GRASP_DIM:
            raise ValueError(f"expected action dim {ARM_DIM} or {GRASP_DIM}, got {arr.shape}")
        return np.clip(arr[:ARM_DIM], -1.0, 1.0).astype(np.float32)

    def _base_env(self) -> Any:
        cur = self.env
        seen = set()
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if type(cur).__name__ == "WA2Env":
                return cur
            inner = getattr(cur, "unwrapped", None)
            if inner is not None and inner is not cur:
                cur = inner
                continue
            inner = getattr(cur, "env", None)
            if inner is None or inner is cur:
                return cur
            cur = inner
        return self.env

    def _fire_hand(self, command: str) -> Tuple[bool, str]:
        """Call ``request_hand``; return ``(ok, resolved_command)``."""

        base = self._base_env()
        if not hasattr(base, "request_hand"):
            return True, command
        result = base.request_hand(command)
        resolved = str(result.get("command") or command)
        ok = bool(result.get("ok", False))
        return ok, resolved

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.shape[0] == ARM_DIM:
            raw = np.concatenate([raw, np.zeros((1,), dtype=np.float32)])
        if raw.shape[0] != GRASP_DIM:
            raise ValueError(f"grasp wrapper expected {GRASP_DIM}D action, got {raw.shape}")
        policy_grasp = discretize_grasp(raw[6])
        arm = np.clip(raw[:ARM_DIM], -1.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(arm)
        info = dict(info or {})

        sm_right = bool(info.get("sm_right"))
        right_edge = bool(sm_right and not self._prev_sm_right)
        self._prev_sm_right = sm_right

        executed_g = 0
        policy_interrupted = str(info.get("interrupted_by") or "") == "intervention"
        human = "intervene_action" in info or right_edge
        hand_fired = False
        hand_ok = True
        hand_cmd = ""
        if right_edge:
            hand_fired = True
            hand_ok, hand_cmd = self._fire_hand("toggle")
            if hand_ok:
                executed_g = 1 if hand_cmd == "grasp" else -1
                self._last_nonzero = executed_g
            else:
                executed_g = 0
        elif (
            not human
            and not policy_interrupted
            and policy_grasp != 0
            and policy_grasp != self._last_nonzero
        ):
            hand_cmd = "grasp" if policy_grasp > 0 else "release"
            hand_fired = True
            hand_ok, hand_cmd = self._fire_hand(hand_cmd)
            if hand_ok:
                executed_g = policy_grasp
                self._last_nonzero = executed_g
            else:
                executed_g = 0
        elif not human and not policy_interrupted and policy_grasp == 0:
            self._last_nonzero = 0

        # ``self.env.step(arm)`` sampled ``obs`` before the blocking hand
        # command above. Refresh it after a confirmed grasp/release so the
        # recorded 7-D action and next observation describe the same physical
        # transition, without executing another Servo window or env step.
        post_hand_obs_refreshed = False
        if hand_fired and hand_ok:
            observe = getattr(self._base_env(), "observe", None)
            if callable(observe):
                obs = observe()
                post_hand_obs_refreshed = True

        arm_exec = np.asarray(info.get("intervene_action", arm), dtype=np.float32).reshape(-1)
        if arm_exec.shape[0] > ARM_DIM:
            arm_exec = arm_exec[:ARM_DIM]
        exec_7d = np.concatenate(
            [arm_exec.astype(np.float32), np.asarray([executed_g], dtype=np.float32)]
        )
        if human:
            info["intervene_action"] = exec_7d.copy()
        info["grasp_command"] = int(executed_g)
        info["hand_fired"] = bool(hand_fired)
        info["hand_ok"] = bool(hand_ok) if hand_fired else True
        info["hand_exec_failed"] = bool(hand_fired and not hand_ok)
        info["post_hand_obs_refreshed"] = bool(post_hand_obs_refreshed)
        info["policy_interrupted"] = bool(policy_interrupted)
        if hand_fired and hand_cmd:
            info["hand_command"] = str(hand_cmd)
        return obs, reward, terminated, truncated, info
