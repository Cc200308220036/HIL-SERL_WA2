#!/usr/bin/env python3
"""Zero-action ServoL hold. Requires R4_CONFIRM=YES."""

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
        raise SystemExit("set R4_CONFIRM=YES")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    from hilserl_wa2.envs.wa2_env import WA2Env

    env = WA2Env(fake_env=False, read_only=False, dry_run=False)
    obs, _ = env.reset(options={"ready_timeout_s": 8.0})
    p0 = obs["state"]["tcp_pose"][:3].copy()
    z = np.zeros(6, dtype=np.float32)
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.seconds:
        obs, _, _, trunc, info = env.step(z)
        if trunc:
            env.close()
            raise SystemExit(f"truncated: {info}")
        time.sleep(0.02)
    drift = float(np.linalg.norm(obs["state"]["tcp_pose"][:3] - p0))
    env.close()
    print("HOLD: PASS" if drift < 0.002 else "HOLD: FAIL", "drift_m=", drift)
    if drift >= 0.002:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
