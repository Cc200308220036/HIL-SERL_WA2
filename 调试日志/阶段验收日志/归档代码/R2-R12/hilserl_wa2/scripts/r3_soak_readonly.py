#!/usr/bin/env python3
"""R3 soak: continuous read-only steps; no motion commands."""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT.parent))

CONTRACT = SRC_ROOT / "configs" / "wa2_env_contract.yaml"


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    from hilserl_wa2.envs.wa2_env import WA2Env

    seconds = float(os.environ.get("R3_SOAK_SECONDS", "600"))
    env = WA2Env(fake_env=False, read_only=True, contract_path=CONTRACT, seed=0)
    obs, info = env.reset(options={"ready_timeout_s": 10.0})
    assert env.observation_space.contains(obs)

    t0 = time.monotonic()
    rss0 = rss_mb()
    steps = 0
    shape0 = {k: v.shape for k, v in obs["state"].items()}
    zeros = np.zeros(6, dtype=np.float32)

    while time.monotonic() - t0 < seconds:
        obs, reward, terminated, truncated, info = env.step(zeros)
        steps += 1
        if truncated and info.get("stale"):
            raise SystemExit(f"SOAK FAIL stale at step={steps} info={info}")
        if truncated and not info.get("stale") and steps >= env.contract.max_steps:
            obs, info = env.reset(options={"ready_timeout_s": 5.0})
            continue
        for key, arr in obs["state"].items():
            if arr.shape != shape0[key]:
                raise SystemExit(f"SOAK FAIL shape change {key}")
            if not np.all(np.isfinite(arr)):
                raise SystemExit(f"SOAK FAIL non-finite {key}")
        if not env.observation_space.contains(obs):
            raise SystemExit("SOAK FAIL obs out of space")
        # ~50 Hz
        time.sleep(0.02)
        if steps % 500 == 0:
            elapsed = time.monotonic() - t0
            print(
                f"soak progress steps={steps} elapsed={elapsed:.1f}s "
                f"rss_mb={rss_mb():.1f} state_age={info.get('state_age')}"
            )

    rss1 = rss_mb()
    env.close()
    growth = rss1 - rss0
    print(
        f"SOAK: PASS steps={steps} seconds={seconds} "
        f"rss0={rss0:.1f} rss1={rss1:.1f} growth_mb={growth:.1f}"
    )
    if growth > 100.0:
        raise SystemExit(f"SOAK FAIL memory growth {growth:.1f} MB > 100")


if __name__ == "__main__":
    main()
