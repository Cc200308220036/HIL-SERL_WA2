#!/usr/bin/env python3
"""R4 visible motion demo: cumulative translation so the arm is easy to see.

Requires:
  export R4_CONFIRM=YES
Physical e-stop is operator-owned.

Default: move +X by 20 mm (20 steps x 1 mm), pause, then return -X 20 mm.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC.parent))


def main() -> None:
    if os.environ.get("R4_CONFIRM") != "YES":
        raise SystemExit("set R4_CONFIRM=YES first (physical e-stop ready)")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--axis",
        default="x",
        choices=["x", "y", "z", "roll", "pitch", "yaw"],
        help="which action dimension to drive",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="number of full-scale steps each direction (1 step ≈ 1mm or 0.25deg)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="sleep between steps (seconds); larger = slower/easier to watch",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="pause at the far point before returning",
    )
    args = parser.parse_args()

    from hilserl_wa2.envs.wa2_env import WA2Env

    idx = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}[args.axis]
    env = WA2Env(
        fake_env=False,
        read_only=False,
        dry_run=False,
        episode_trans_limit_m=0.05,
        episode_rot_limit_deg=10.0,
    )
    obs, info = env.reset(options={"ready_timeout_s": 8.0})
    p0 = obs["state"]["tcp_pose"].copy()
    print("start_tcp", p0.tolist())
    print("singular", info.get("is_singular"), "cmd", info.get("cmd_name"))
    if info.get("is_singular"):
        env.close()
        raise SystemExit("left arm singular; move to non-singular pose first")

    def drive(sign: float, tag: str) -> None:
        action = np.zeros(6, dtype=np.float32)
        action[idx] = float(sign)
        for i in range(args.steps):
            nonlocal_obs, _, _, trunc, step_info = env.step(action)
            if trunc:
                env.close()
                raise SystemExit(f"truncated during {tag}@{i}: {step_info}")
            if i % 5 == 0 or i == args.steps - 1:
                tcp = nonlocal_obs["state"]["tcp_pose"]
                delta = tcp[:3] - p0[:3]
                print(
                    f"{tag} step={i+1}/{args.steps} "
                    f"delta_xyz_mm={(delta*1000).round(2).tolist()} "
                    f"published={step_info.get('published')} "
                    f"cmd_delta_mm={step_info.get('delta_pos_m', 0)*1000:.2f}"
                )
            time.sleep(args.dt)

    try:
        print(f">>> move +{args.axis} for {args.steps} steps (~{args.steps} mm if trans)")
        drive(+1.0, "out")
        print(f">>> pause {args.pause}s at far point")
        time.sleep(args.pause)
        print(f">>> move -{args.axis} for {args.steps} steps (return)")
        drive(-1.0, "back")
        obs, _, _, _, _ = env.step(np.zeros(6, dtype=np.float32))
        p1 = obs["state"]["tcp_pose"]
        net_mm = float(np.linalg.norm(p1[:3] - p0[:3]) * 1000)
        print("end_tcp", p1.tolist())
        print(f"DEMO: DONE net_translation_mm={net_mm:.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
