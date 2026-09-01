#!/usr/bin/env python3
"""R6 soak: subscribe dual cameras via Env for N minutes; write report."""

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
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--out", type=str, default="/root/catkin_ws/r6_samples/soak_report.txt")
    parser.add_argument("--min-fps", type=float, default=15.0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = WA2Env(fake_env=False, read_only=True)
    # Soak is not an episode horizon test; disable max_steps truncation.
    env.max_steps = 10**9
    env.reset(options={"skip_reset_motion": True, "ready_timeout_s": 8.0})

    duration = float(args.minutes) * 60.0
    t0 = time.time()
    n = 0
    trunc_count = 0
    stale_trunc = 0
    max_ages = {"head": 0.0, "wrist": 0.0}
    sum_ages = {"head": 0.0, "wrist": 0.0}
    age_n = {"head": 0, "wrist": 0}

    while time.time() - t0 < duration:
        obs, r, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
        n += 1
        if trunc:
            trunc_count += 1
            if info.get("stale") or info.get("stale_fields"):
                stale_trunc += 1
        ia = info.get("image_ages") or {}
        for k in ("head", "wrist"):
            if ia.get(k) is not None:
                a = float(ia[k])
                max_ages[k] = max(max_ages[k], a)
                sum_ages[k] += a
                age_n[k] += 1
        time.sleep(0.02)

    elapsed = max(time.time() - t0, 1e-6)
    stats = env._cameras.stats() if hasattr(env._cameras, "stats") else {}
    env.close()

    lines = [
        f"duration_s={elapsed:.1f}",
        f"steps={n}",
        f"trunc_count={trunc_count}",
        f"stale_trunc={stale_trunc}",
        f"env_loop_fps={n / elapsed:.2f}",
    ]
    ok = stale_trunc == 0
    for k in ("head", "wrist"):
        mean_age = (sum_ages[k] / age_n[k]) if age_n[k] else float("nan")
        recv = int((stats.get(k) or {}).get("recv_count", 0))
        fps = recv / elapsed
        lines.append(
            f"{k}: recv={recv} approx_fps={fps:.2f} mean_age={mean_age:.4f} "
            f"max_age={max_ages[k]:.4f}"
        )
        if fps < args.min_fps:
            ok = False
    lines.append("R6_SOAK: PASS" if ok else "R6_SOAK: FAIL")
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    print(text)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
