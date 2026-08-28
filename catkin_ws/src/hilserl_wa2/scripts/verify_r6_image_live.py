#!/usr/bin/env python3
"""R6 live Gate: dual-camera shape/age/FPS via read-only WA2Env."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT.parent))

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--min-fps", type=float, default=15.0)
    parser.add_argument("--max-mean-age", type=float, default=0.2)
    args = parser.parse_args()

    env = WA2Env(fake_env=False, read_only=True)
    obs, info = env.reset(options={"skip_reset_motion": True, "ready_timeout_s": 8.0})
    for key in ("head", "wrist"):
        img = obs["images"][key]
        assert img.shape == (128, 128, 3) and img.dtype == np.uint8, (key, img.shape)
        assert float(img.std()) > 0.5 or float(img.mean()) > 1.0, f"{key} looks empty"

    ages = {"head": [], "wrist": []}
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.seconds:
        obs, r, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
        if trunc and info.get("stale"):
            raise SystemExit(f"R6_LIVE: FAIL unexpected stale {info.get('stale_fields')}")
        ia = info.get("image_ages") or {}
        for k in ("head", "wrist"):
            if ia.get(k) is not None:
                ages[k].append(float(ia[k]))
        n += 1
        time.sleep(0.02)

    elapsed = max(time.time() - t0, 1e-6)
    env_fps = n / elapsed
    print(f"steps={n} env_loop_fps={env_fps:.2f}")
    for k, vals in ages.items():
        mean_age = float(np.mean(vals)) if vals else 1e9
        max_age = float(np.max(vals)) if vals else 1e9
        print(f"{k}: mean_age={mean_age:.4f} max_age={max_age:.4f} n={len(vals)}")
        if mean_age > args.max_mean_age:
            env.close()
            raise SystemExit(f"R6_LIVE: FAIL {k} mean_age {mean_age} > {args.max_mean_age}")

    # Topic-side FPS from camera recv_count if available
    stats = env._cameras.stats() if hasattr(env._cameras, "stats") else {}
    for k, st in stats.items():
        recv = int(st.get("recv_count", 0))
        fps = recv / elapsed
        print(f"{k}: recv_count={recv} approx_fps={fps:.2f}")
        if fps < args.min_fps:
            env.close()
            raise SystemExit(f"R6_LIVE: FAIL {k} fps {fps:.2f} < {args.min_fps}")

    env.close()
    print("R6_LIVE: PASS")


if __name__ == "__main__":
    main()
