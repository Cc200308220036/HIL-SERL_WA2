#!/usr/bin/env python3
"""§6 remeasure: warmup + 40 steps + 3s zero-hold servo Hz."""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/catkin_ws/src")

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402

SCALE = 0.3
N = 40
OUT = Path("/root/catkin_ws/r13_stage3_s6_remeasure.json")


def pct(xs, p):
    ys = sorted(xs)
    if not ys:
        return float("nan")
    k = (len(ys) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    return float(ys[f] * (c - k) + ys[c] * (k - f))


def main() -> None:
    env = WA2Env(
        fake_env=False,
        read_only=False,
        dry_run=False,
        scene_name=None,
        auto_reset_motion=False,
        episode_trans_limit_m=0.05,
        episode_rot_limit_deg=10.0,
    )
    try:
        obs, _info = env.reset(
            options={"skip_reset_motion": True, "ready_timeout_s": 8.0}
        )
        print("tcp", [float(x) for x in obs["state"]["tcp_pose"][:3]], flush=True)
        for i in range(5):
            sign = SCALE if i % 2 == 0 else -SCALE
            env.step(np.asarray([sign, 0, 0, 0, 0, 0], np.float32))

        rows = []
        pub0 = env._servo.publish_count
        t0 = time.monotonic()
        for i in range(N):
            sign = SCALE if (i // 5) % 2 == 0 else -SCALE
            t1 = time.monotonic()
            _obs, _r, term, trunc, info = env.step(
                np.asarray([sign, 0, 0, 0, 0, 0], np.float32)
            )
            wall = time.monotonic() - t1
            row = {
                "wall": wall,
                "exec": float(info.get("execution_duration_s") or 0),
                "req": int(info.get("servo_ticks_requested") or -1),
                "exe": int(info.get("servo_ticks_executed") or -1),
                "interrupt": str(info.get("interrupted_by")),
                "dpos": float(info.get("delta_pos_m") or 0),
                "fault": bool(info.get("servo_faulted")),
            }
            rows.append(row)
            print(
                f"re {i:02d} {row['exe']}/{row['req']} "
                f"exec={row['exec']:.4f} wall={row['wall']:.4f}",
                flush=True,
            )
            if trunc or term or row["fault"]:
                raise SystemExit(f"abort {row}")

        elapsed = time.monotonic() - t0
        pub1 = env._servo.publish_count

        time.sleep(0.05)
        hp0 = env._servo.publish_count
        ht0 = time.monotonic()
        while time.monotonic() - ht0 < 3.0:
            env.step(np.zeros(6, np.float32))
        ht1 = time.monotonic()
        hp1 = env._servo.publish_count

        walls = [r["wall"] for r in rows]
        execs = [r["exec"] for r in rows]
        summary = {
            "step_hz_n_over_T": N / elapsed,
            "step_hz_inv_mean_wall": 1.0 / statistics.mean(walls),
            "servo_hz_during_motion": (pub1 - pub0) / elapsed,
            "servo_hz_zero_hold_3s": (hp1 - hp0) / (ht1 - ht0),
            "ticks_all_5_5": all(r["exe"] == 5 and r["req"] == 5 for r in rows),
            "exec_mean": statistics.mean(execs),
            "exec_p95": pct(execs, 95),
            "exec_max": max(execs),
            "wall_mean": statistics.mean(walls),
            "wall_p95": pct(walls, 95),
            "dpos_unique": sorted({round(r["dpos"], 6) for r in rows}),
        }
        checks = {
            "step_hz_9_5_10_5": 9.5 <= summary["step_hz_n_over_T"] <= 10.5,
            "step_hz_inv_mean_9_5_10_5": 9.5
            <= summary["step_hz_inv_mean_wall"]
            <= 10.5,
            "servo_motion_47_5_52_5": 47.5
            <= summary["servo_hz_during_motion"]
            <= 52.5,
            "servo_hold_47_5_52_5": 47.5 <= summary["servo_hz_zero_hold_3s"] <= 52.5,
            "ticks_5_of_5": summary["ticks_all_5_5"],
            "exec_p95_le_0_12": summary["exec_p95"] <= 0.12,
        }
        out = {"summary": summary, "checks": checks}
        OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(out, indent=2), flush=True)
        print("ALL_TIMING_PASS" if all(checks.values()) else "TIMING_PARTIAL", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
