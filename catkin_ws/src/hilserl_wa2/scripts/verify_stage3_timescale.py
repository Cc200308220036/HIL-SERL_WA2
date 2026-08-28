#!/usr/bin/env python3
"""Offline stage-three time-scale acceptance check (no ROS/JAX/hardware)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.experiments.r13_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    TRANSITION_SCHEMA_VERSION,
)
from hilserl_wa2.experiments.task_config import load_task  # noqa: E402
from hilserl_wa2.experiments.transition import build_actor_transition  # noqa: E402
from hilserl_wa2.wrappers.grasp_action import WA2GraspActionWrapper  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    task = load_task("bottle_pick")
    env = WA2Env(
        fake_env=True,
        scene_name=task.scene,
        contract_path=task.contract_path,
        spacemouse_path=task.spacemouse_path,
        seed=0,
    )
    try:
        contract = env.contract
        _require(np.isclose(contract.policy_hz, 10.0), "policy_hz is not 10")
        _require(np.isclose(contract.control_hz, 50.0), "servo_hz is not 50")
        _require(contract.servo_ticks_per_action == 5, "ticks/action is not 5")
        _require(np.isclose(task.discount, 0.98), "discount is not 0.98")
        _require(env.max_steps == 600, "episode max_steps is not 600")

        obs0, _ = env.reset(seed=0)
        action = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32)
        obs1, reward, terminated, truncated, info = env.step(action)
        delta_x = float(obs1["state"]["tcp_pose"][0] - obs0["state"]["tcp_pose"][0])
        _require(info["servo_ticks_requested"] == 5, "requested ticks != 5")
        _require(info["servo_ticks_executed"] == 5, "executed ticks != 5")
        _require(np.isclose(delta_x, 0.005, atol=1e-6), "five ticks did not move 5 mm")
        transition, _ = build_actor_transition(
            obs0, action, obs1, reward, terminated, truncated, info
        )
        _require(np.isclose(transition["actions"][0], 1.0), "full action was altered")

        obs0, _ = env.reset(seed=1)
        calls = {"n": 0}

        def cancel_before_third() -> bool:
            calls["n"] += 1
            return calls["n"] >= 3

        env.set_action_interrupt_callback(cancel_before_third)
        obs1, reward, terminated, truncated, info = env.step(action)
        _require(info["servo_ticks_executed"] == 2, "interrupt did not stop before tick 3")
        _require(info["interrupted_by"] == "intervention", "wrong interrupt reason")
        transition, _ = build_actor_transition(
            obs0, action, obs1, reward, terminated, truncated, info
        )
        _require(
            np.isclose(transition["actions"][0], 0.4, atol=1e-6),
            "partial window effective action is not 2/5",
        )
        env.set_action_interrupt_callback(None)

        grasp_env = WA2GraspActionWrapper(env)
        calls_hand = {"n": 0}
        original = env.request_hand

        def counted_hand(command: str):
            calls_hand["n"] += 1
            return original(command)

        env.request_hand = counted_hand  # type: ignore[method-assign]
        grasp_env.reset(seed=2)
        grasp_action = np.zeros(7, dtype=np.float32)
        grasp_action[-1] = 1.0
        _, _, _, _, grasp_info = grasp_env.step(grasp_action)
        _require(calls_hand["n"] == 1, "grasp command was not edge-triggered exactly once")
        _require(grasp_info["grasp_command"] == 1, "executed grasp label is wrong")

        print(
            json.dumps(
                {
                    "result": "PASS",
                    "protocol": PROTOCOL_VERSION,
                    "transition_schema": TRANSITION_SCHEMA_VERSION,
                    "policy_hz": contract.policy_hz,
                    "servo_hz": contract.control_hz,
                    "servo_ticks_per_action": contract.servo_ticks_per_action,
                    "discount": task.discount,
                    "episode_max_steps": env.max_steps,
                    "normal_delta_x_m": delta_x,
                    "interrupt_ticks": info["servo_ticks_executed"],
                    "grasp_service_calls": calls_hand["n"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        print("STAGE3_TIMESCALE_OFFLINE: PASS")
    finally:
        env.close()


if __name__ == "__main__":
    main()
