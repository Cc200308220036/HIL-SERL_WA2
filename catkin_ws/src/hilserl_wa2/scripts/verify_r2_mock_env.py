#!/usr/bin/env python3
"""R2 Gate: Mock WA2Env check_env, isolation, random rollout."""

from __future__ import annotations

import builtins
import math
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]  # hilserl_wa2
CATKIN_SRC = SRC_ROOT.parent  # catkin_ws/src
sys.path.insert(0, str(CATKIN_SRC))

CONTRACT = SRC_ROOT / "configs" / "wa2_env_contract.yaml"


def main() -> None:
    from gymnasium.utils.env_checker import check_env
    from hilserl_wa2.envs.wa2_env import WA2Env

    if not CONTRACT.is_file():
        raise SystemExit(f"FAIL: missing contract {CONTRACT}")

    env = WA2Env(fake_env=True, contract_path=CONTRACT, seed=0)
    check_env(env, skip_render_check=True)
    print("PASS: check_env")

    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), i
        assert float(reward) == 0.0
        assert info["delta_pos_m"] <= 0.005 + 1e-9
        assert info["delta_rot_rad"] <= math.radians(1.25) + 1e-9
        assert info["servo_ticks_requested"] == 5
        assert info["servo_ticks_executed"] == 5
        if terminated or truncated:
            obs, _ = env.reset()
            assert env.observation_space.contains(obs)
    env.close()
    env.close()
    print("PASS: random_1000 + close idempotent")

    # Isolation: constructing in a fresh import guard.
    blocked = {
        "rospy",
        "cv_bridge",
        "sensor_msgs",
        "upperlimb",
        "naviai_controller",
    }
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        root = name.split(".")[0]
        if root in blocked:
            raise ImportError(f"blocked import {name}")
        return real_import(name, *args, **kwargs)

    # Reload modules under guard by constructing another env in-process after
    # verifying current module graph does not require blocked packages at step time.
    for mod in list(sys.modules):
        if mod.split(".")[0] in blocked:
            raise SystemExit(f"FAIL: blocked module already loaded: {mod}")

    builtins.__import__ = guarded
    try:
        env2 = WA2Env(fake_env=True, contract_path=CONTRACT, seed=1)
        env2.reset(seed=1)
        env2.step(env2.action_space.sample())
        env2.close()
    finally:
        builtins.__import__ = real_import
    print("PASS: fake_env isolation")

    # Seed repro
    def rollout(seed: int):
        e = WA2Env(fake_env=True, contract_path=CONTRACT, seed=seed)
        o, _ = e.reset(seed=seed)
        poses = [o["state"]["tcp_pose"].copy()]
        rng = np.random.default_rng(seed)
        for _ in range(20):
            a = rng.uniform(-1, 1, size=(6,)).astype(np.float32)
            o, *_ = e.step(a)
            poses.append(o["state"]["tcp_pose"].copy())
        e.close()
        return poses

    p1, p2 = rollout(7), rollout(7)
    assert all(np.allclose(a, b) for a, b in zip(p1, p2))
    print("PASS: seed_repro")

    # Wrist zeros
    env3 = WA2Env(fake_env=True, contract_path=CONTRACT, seed=0)
    o, _ = env3.reset(seed=0)
    assert np.all(o["images"]["wrist"] == 0)
    env3.close()
    print("PASS: wrist_zero_image")

    print("R2 GATE: PASS")


if __name__ == "__main__":
    main()
