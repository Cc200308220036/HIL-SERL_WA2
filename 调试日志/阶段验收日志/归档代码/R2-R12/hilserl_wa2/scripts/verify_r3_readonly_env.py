#!/usr/bin/env python3
"""R3 Gate: ROS read-only WA2Env (requires live topics)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
CATKIN_SRC = SRC_ROOT.parent
sys.path.insert(0, str(CATKIN_SRC))

CONTRACT = SRC_ROOT / "configs" / "wa2_env_contract.yaml"


def main() -> None:
    from hilserl_wa2.envs.wa2_env import WA2Env
    from hilserl_wa2.ros_adapters.state_monitor import WA2StateMonitor

    # --- lateral isolation on live monitor ---
    mon = WA2StateMonitor(arm="left", state_max_age_s=0.2)
    mon.start()
    mon.wait_ready(timeout_s=8.0)
    names = mon.left_joint_names()
    assert all(n.endswith("_L") for n in names), names
    st = mon.get_state()
    tcp_a = st["tcp_pose"].copy()
    mutated = st["tcp_pose"]
    mutated[0] += 1.0
    st2 = mon.get_state()
    assert abs(float(st2["tcp_pose"][0] - tcp_a[0])) < 0.05, "cache not copied"
    mon.stop()
    print("PASS: lateral_isolation + copy")

    env = WA2Env(fake_env=False, read_only=True, contract_path=CONTRACT, seed=0)
    obs, info = env.reset(options={"ready_timeout_s": 8.0})
    assert env.observation_space.contains(obs)
    assert info.get("read_only") is True
    assert info.get("fake_env") is False
    print(
        "PASS: reset",
        {k: tuple(v.shape) for k, v in obs["state"].items()},
        "singular=",
        info.get("is_singular"),
    )

    tcp0 = obs["state"]["tcp_pose"][:3].copy()
    for i in range(50):
        action = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), i
        assert float(reward) == 0.0
        assert info.get("action_ignored_for_motion") is True
        if truncated and info.get("stale"):
            raise AssertionError(f"unexpected stale during smoke: {info}")
        time.sleep(0.02)
    drift = float(np.linalg.norm(obs["state"]["tcp_pose"][:3] - tcp0))
    print("tcp_drift_m", drift)
    assert drift < 0.005, f"unexpected motion drift={drift}"
    print("PASS: readonly_smoke")

    # stale injection
    env._state_monitor.inject_stale_for_test(fields=["tcp_pose"], age_s=1.0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert truncated is True
    assert info.get("stale") is True
    assert "tcp_pose" in info.get("stale_fields", [])
    print("PASS: stale_gate", info.get("stale_fields"))

    env.close()
    env.close()
    print("R3 GATE: PASS")


if __name__ == "__main__":
    main()
