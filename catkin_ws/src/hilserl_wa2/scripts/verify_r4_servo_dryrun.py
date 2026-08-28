#!/usr/bin/env python3
"""R4 dry-run gate: compute ServoL targets, never publish."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC.parent))
CONTRACT = SRC / "configs" / "wa2_env_contract.yaml"


def main() -> None:
    from hilserl_wa2.envs.contracts import WA2EnvContract
    from hilserl_wa2.envs.wa2_env import WA2Env
    from hilserl_wa2.ros_adapters.servo_session import (
        WA2ServoSession,
        integrate_normalized_action,
    )
    from hilserl_wa2.ros_adapters.mock_cameras import MockCameras
    from hilserl_wa2.ros_adapters.state_monitor import StateCache, WA2StateMonitor

    contract = WA2EnvContract.from_yaml(CONTRACT)
    pose0 = np.asarray([0.3, 0.1, 0.6, 0, 0, 0, 1], dtype=np.float32)
    _, info = integrate_normalized_action(pose0, [1.5, 0, 0, 0, 0, 0], contract)
    assert abs(info["delta_pos_m"] - 0.001) < 1e-9
    _, info_r = integrate_normalized_action(pose0, [0, 0, 0, 1, 0, 0], contract)
    assert info_r["delta_rot_rad"] <= math.radians(0.25) + 1e-9
    print("PASS: integrate limits")

    cache = StateCache(0.2)
    cache.update_tcp_pose(pose0)
    cache.update_tcp_vel(np.zeros(6))
    cache.update_joint_pos(np.zeros(8))
    cache.update_hand_joints(np.zeros(6))
    cache.update_uplimb_state(
        is_singular=False, cmd_num=0, cmd_name="STOPPED", iddp_status=True
    )
    mon = WA2StateMonitor(arm="left", cache=cache)
    mon._started = True
    servo = WA2ServoSession(contract=contract, state_monitor=mon, dry_run=True)
    env = WA2Env(
        fake_env=False,
        read_only=False,
        dry_run=True,
        contract_path=CONTRACT,
        state_monitor=mon,
        servo_session=servo,
        cameras=MockCameras(contract, seed=0),
    )
    obs, info = env.reset()
    assert info["read_only"] is False
    assert info["dry_run"] is True
    for _ in range(5):
        obs, r, term, trunc, info = env.step(
            np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32)
        )
        assert abs(info["delta_pos_m"] - 0.005) < 1e-9
        assert info["servo_ticks_requested"] == 5
        assert info["servo_ticks_executed"] == 5
        assert info.get("published") is False
    health = info["servo_health"]
    assert health["publish_count"] == 0
    env.close()
    assert health["dry_run"] is True
    print("PASS: env dry_run steps")
    print("R4_DRYRUN: PASS")


if __name__ == "__main__":
    main()
