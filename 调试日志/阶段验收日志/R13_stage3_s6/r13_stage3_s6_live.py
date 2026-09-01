#!/usr/bin/env python3
"""Stage-3 §6 live low-scale acceptance (temporary runner).

Requires R4_CONFIRM=YES. Does not run full SAC. Skips scene reset motion
(assumes arm already at reset pose). Scale default 0.3.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path("/root/catkin_ws/src")
sys.path.insert(0, str(SRC))

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.interventions.wa2_spacemouse_intervention import (  # noqa: E402
    WA2SpacemouseIntervention,
)

OUT = Path("/root/catkin_ws/r13_stage3_s6_out.json")
SCALE = float(os.environ.get("S6_SCALE", "0.3"))
NORMAL_STEPS = int(os.environ.get("S6_NORMAL_STEPS", "40"))
INTERRUPT_WAIT_S = float(os.environ.get("S6_INTERRUPT_WAIT_S", "45"))


def _require_confirm() -> None:
    if os.environ.get("R4_CONFIRM") != "YES":
        raise SystemExit("refusing motion: set R4_CONFIRM=YES (e-stop ready)")


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    return float(ys[f] * (c - k) + ys[c] * (k - f))


def _axis_action(sign: float) -> np.ndarray:
    a = np.zeros(6, dtype=np.float32)
    a[0] = float(sign) * SCALE  # base +x / -x oscillation
    return a


def phase_normal(env: WA2Env) -> dict:
    print(
        f"PHASE1_NORMAL start steps={NORMAL_STEPS} scale={SCALE} "
        "(small +/- x oscillation near current pose)",
        flush=True,
    )
    rows: list[dict] = []
    pub0 = int(env._servo.publish_count) if env._servo else 0
    t0 = time.monotonic()
    for i in range(NORMAL_STEPS):
        sign = 1.0 if (i // 5) % 2 == 0 else -1.0
        t_step = time.monotonic()
        obs, _, term, trunc, info = env.step(_axis_action(sign))
        dt = time.monotonic() - t_step
        row = {
            "i": i,
            "dt_s": dt,
            "servo_ticks_requested": info.get("servo_ticks_requested"),
            "servo_ticks_executed": info.get("servo_ticks_executed"),
            "execution_duration_s": info.get("execution_duration_s"),
            "interrupted_by": info.get("interrupted_by"),
            "delta_pos_xyz": [
                float(x) for x in np.asarray(info.get("delta_pos_xyz", [0, 0, 0])).reshape(-1)[:3]
            ],
            "delta_pos_m": float(info.get("delta_pos_m") or 0.0),
            "servo_faulted": bool(info.get("servo_faulted")),
            "stale_fields": list(info.get("stale_fields") or []),
            "term": bool(term),
            "trunc": bool(trunc),
        }
        rows.append(row)
        print(
            f"s6 step={i:02d} ticks={row['servo_ticks_executed']}/"
            f"{row['servo_ticks_requested']} "
            f"exec={float(row['execution_duration_s'] or 0):.4f}s "
            f"wall={dt:.4f}s interrupt={row['interrupted_by']} "
            f"dpos={row['delta_pos_m']:.5f} fault={int(row['servo_faulted'])}",
            flush=True,
        )
        if trunc or term or row["servo_faulted"]:
            print("PHASE1_ABORT", row, flush=True)
            break
    elapsed = time.monotonic() - t0
    pub1 = int(env._servo.publish_count) if env._servo else pub0
    n = len(rows)
    exec_durs = [float(r["execution_duration_s"] or 0.0) for r in rows]
    normal = [
        r
        for r in rows
        if str(r.get("interrupted_by") or "none") == "none" and not r["servo_faulted"]
    ]
    ticks_ok = all(
        int(r["servo_ticks_executed"] or 0) == 5
        and int(r["servo_ticks_requested"] or 0) == 5
        for r in normal
    )
    step_hz = n / max(elapsed, 1e-9)
    servo_hz = (pub1 - pub0) / max(elapsed, 1e-9)
    # After last action, wait briefly and confirm latch does not keep integrating.
    hold_pub0 = int(env._servo.publish_count)
    time.sleep(0.12)
    # Zero action hold windows should publish zeros / not grow non-zero integration.
    for _ in range(3):
        env.step(np.zeros(6, np.float32))
    hold_info = env._last_applied or {}
    summary = {
        "steps": n,
        "elapsed_s": elapsed,
        "step_hz": step_hz,
        "servo_publish_hz": servo_hz,
        "publish_delta": pub1 - pub0,
        "ticks_all_5_of_5": ticks_ok,
        "normal_count": len(normal),
        "exec_duration_mean_s": float(statistics.mean(exec_durs)) if exec_durs else None,
        "exec_duration_p95_s": _percentile(exec_durs, 95),
        "exec_duration_max_s": max(exec_durs) if exec_durs else None,
        "any_fault": any(r["servo_faulted"] for r in rows),
        "hold_after_zero_ticks": [
            int(hold_info.get("servo_ticks_executed") or -1),
            int(hold_info.get("servo_ticks_requested") or -1),
        ],
        "hold_publish_delta_extra": int(env._servo.publish_count) - hold_pub0,
        "rows": rows,
    }
    print(
        f"PHASE1_SUMMARY step_hz={step_hz:.3f} servo_hz={servo_hz:.3f} "
        f"ticks_5of5={ticks_ok} exec_p95={summary['exec_duration_p95_s']:.4f} "
        f"exec_max={summary['exec_duration_max_s']:.4f}",
        flush=True,
    )
    return summary


def phase_interrupt(base: WA2Env) -> dict:
    print(
        "PHASE2_INTERRUPT: wrap SpaceMouse; keep LEFT deadman RELEASED until prompt, "
        f"then PRESS LEFT within {INTERRUPT_WAIT_S:.0f}s to cancel a policy window",
        flush=True,
    )
    env = WA2SpacemouseIntervention(base, auto_start_ros=True)
    env.joy.start_ros()
    env.joy.wait_ready(timeout_s=10.0)
    # Warm: a few zero policy steps so operator can see status.
    for _ in range(3):
        env.step(np.zeros(6, np.float32))

    print(">>> PRESS LEFT BUTTON (deadman) NOW while arm moves slowly in +x <<<", flush=True)
    t0 = time.monotonic()
    saw = None
    next_human = None
    i = 0
    while time.monotonic() - t0 < INTERRUPT_WAIT_S:
        # Non-zero policy motion so cancel is visible.
        act = _axis_action(1.0 if (i // 3) % 2 == 0 else -1.0)
        t_step = time.monotonic()
        obs, _, term, trunc, info = env.step(act)
        dt = time.monotonic() - t_step
        row = {
            "i": i,
            "dt_s": dt,
            "intervened": bool(info.get("intervened")),
            "sm_session": bool(info.get("sm_session")),
            "servo_ticks_requested": info.get("servo_ticks_requested"),
            "servo_ticks_executed": info.get("servo_ticks_executed"),
            "execution_duration_s": info.get("execution_duration_s"),
            "interrupted_by": info.get("interrupted_by"),
            "delta_pos_m": float(info.get("delta_pos_m") or 0.0),
            "servo_faulted": bool(info.get("servo_faulted")),
        }
        print(
            f"s6i step={i:02d} intervened={int(row['intervened'])} "
            f"ticks={row['servo_ticks_executed']}/{row['servo_ticks_requested']} "
            f"interrupt={row['interrupted_by']} exec={float(row['execution_duration_s'] or 0):.4f}s "
            f"sess={int(row['sm_session'])}",
            flush=True,
        )
        if (
            saw is None
            and str(row["interrupted_by"]) == "intervention"
            and not row["intervened"]
        ):
            saw = row
            print("INTERRUPT_BOUNDARY captured; expecting next step may be human", flush=True)
        elif saw is not None and next_human is None:
            next_human = row
            break
        if trunc or term or row["servo_faulted"]:
            break
        i += 1

    # Bound: interrupt should happen within ~2 servo ticks of press; we can only
    # bound by executed ticks < 5 on the cancel step.
    ok_boundary = saw is not None and int(saw["servo_ticks_executed"] or 0) < 5
    ok_next = next_human is not None and (
        bool(next_human.get("intervened")) or str(next_human.get("interrupted_by")) == "none"
    )
    summary = {
        "saw_interrupt_step": saw,
        "next_step": next_human,
        "interrupt_ticks_lt_5": ok_boundary,
        "next_step_recorded": next_human is not None,
        "pass_candidate": bool(ok_boundary and saw is not None),
    }
    print(f"PHASE2_SUMMARY {json.dumps({k: v for k, v in summary.items() if k != 'saw_interrupt_step' and k != 'next_step'})}", flush=True)
    return summary


def main() -> None:
    _require_confirm()
    env = WA2Env(
        fake_env=False,
        read_only=False,
        dry_run=False,
        scene_name=None,
        auto_reset_motion=False,
        episode_trans_limit_m=0.05,
        episode_rot_limit_deg=10.0,
    )
    result: dict = {"scale": SCALE, "normal_steps_requested": NORMAL_STEPS}
    try:
        obs, info = env.reset(
            options={"skip_reset_motion": True, "ready_timeout_s": 8.0}
        )
        print(
            "RESET_OK skip_motion tcp=",
            [float(x) for x in obs["state"]["tcp_pose"][:3]],
            flush=True,
        )
        result["phase1"] = phase_normal(env)
        result["phase2"] = phase_interrupt(env)
    finally:
        try:
            env.close()
        except Exception as exc:
            result["close_error"] = str(exc)

    p1 = result.get("phase1") or {}
    p2 = result.get("phase2") or {}
    checks = {
        "step_hz_9_5_10_5": 9.5 <= float(p1.get("step_hz") or 0) <= 10.5,
        "servo_hz_47_5_52_5": 47.5 <= float(p1.get("servo_publish_hz") or 0) <= 52.5,
        "ticks_5_of_5": bool(p1.get("ticks_all_5_of_5")),
        "exec_p95_le_0_12": float(p1.get("exec_duration_p95_s") or 9) <= 0.12,
        "no_fault_phase1": not bool(p1.get("any_fault")),
        "interrupt_seen": bool(p2.get("pass_candidate")),
    }
    result["checks"] = checks
    result["STAGE3_S6"] = "PASS" if all(checks.values()) else "FAIL"
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "STAGE3_S6": result["STAGE3_S6"]}, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)
    if result["STAGE3_S6"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
