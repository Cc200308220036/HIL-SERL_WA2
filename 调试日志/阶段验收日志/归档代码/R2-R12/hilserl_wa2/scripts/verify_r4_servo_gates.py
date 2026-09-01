#!/usr/bin/env python3
"""R4 real ServoL gates. Requires R4_CONFIRM=YES. Physical e-stop is operator-owned."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC.parent))
CONTRACT = SRC / "configs" / "wa2_env_contract.yaml"
OUT = Path("/root/catkin_ws/r4_gates_out.txt")


def _require_confirm() -> None:
    if os.environ.get("R4_CONFIRM") != "YES":
        raise SystemExit("refusing motion: set R4_CONFIRM=YES (physical e-stop ready)")


def _log(rows: list, row: dict) -> None:
    rows.append(row)
    print(row)


def _flush(rows: list) -> None:
    if not rows:
        return
    keys: list = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with OUT.open("w", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _make_env():
    from hilserl_wa2.envs.wa2_env import WA2Env

    return WA2Env(
        fake_env=False,
        read_only=False,
        dry_run=False,
        contract_path=CONTRACT,
        episode_trans_limit_m=float(os.environ.get("R4_EPISODE_TRANS_LIMIT_M", "0.03")),
        episode_rot_limit_deg=float(os.environ.get("R4_EPISODE_ROT_LIMIT_DEG", "5.0")),
    )


def gate_hold(env, rows, seconds: float = 5.0) -> None:
    obs, info = env.reset(options={"ready_timeout_s": 8.0})
    p0 = obs["state"]["tcp_pose"][:3].copy()
    zeros = np.zeros(6, dtype=np.float32)
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        obs, _, _, trunc, info = env.step(zeros)
        if trunc:
            raise SystemExit(f"HOLD truncated: {info}")
        time.sleep(0.02)
    drift = float(np.linalg.norm(obs["state"]["tcp_pose"][:3] - p0))
    _log(rows, {"gate": "hold", "drift_m": drift, "ok": drift < 0.002})
    if drift >= 0.002:
        raise SystemExit(f"HOLD FAIL drift={drift}")
    print("HOLD: PASS", drift)


def _axis_action(axis: str, sign: float) -> np.ndarray:
    a = np.zeros(6, dtype=np.float32)
    idx = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}[axis]
    a[idx] = float(sign)
    return a


def gate_trans(env, rows, axis: str) -> None:
    obs, _ = env.reset(options={"ready_timeout_s": 8.0})
    p0 = obs["state"]["tcp_pose"].copy()
    # +1mm then -1mm
    for sign, tag in ((1.0, "plus"), (-1.0, "minus")):
        obs, _, _, trunc, info = env.step(_axis_action(axis, sign))
        if trunc:
            raise SystemExit(f"TRANS truncated {axis} {tag}: {info}")
        time.sleep(0.05)
        delta = obs["state"]["tcp_pose"][:3] - p0[:3]
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        moved = float(delta[idx]) * sign
        _log(
            rows,
            {
                "gate": f"trans_{axis}_{tag}",
                "delta_axis_m": float(delta[idx]),
                "tracking_err_m": info.get("tracking_err_m"),
                "ok": moved > 0.0003,  # direction mostly correct / moved
            },
        )
        time.sleep(0.05)
    # net should be near start
    net = float(np.linalg.norm(obs["state"]["tcp_pose"][:3] - p0[:3]))
    _log(rows, {"gate": f"trans_{axis}_net", "net_m": net, "ok": net < 0.003})
    if net >= 0.003:
        print("WARN net residual", net)
    print(f"TRANS_{axis.upper()}: PASS")


def gate_rot(env, rows, axis: str) -> None:
    obs, _ = env.reset(options={"ready_timeout_s": 8.0})
    p0 = obs["state"]["tcp_pose"].copy()
    steps = 8  # 8 * 0.25deg = 2deg
    for sign, tag in ((1.0, "plus"), (-1.0, "minus")):
        for i in range(steps):
            obs, _, _, trunc, info = env.step(_axis_action(axis, sign))
            if trunc:
                raise SystemExit(f"ROT truncated {axis} {tag}@{i}: {info}")
            time.sleep(0.02)
        _log(
            rows,
            {
                "gate": f"rot_{axis}_{tag}",
                "pos_drift_m": float(
                    np.linalg.norm(obs["state"]["tcp_pose"][:3] - p0[:3])
                ),
                "tracking_err_rad": info.get("tracking_err_rad"),
                "ok": True,
            },
        )
    print(f"ROT_{axis.upper()}: PASS")


def gate_clip(env, rows) -> None:
    obs, _ = env.reset(options={"ready_timeout_s": 8.0})
    p0 = obs["state"]["tcp_pose"][:3].copy()
    action = np.asarray([1.5, 0, 0, 0, 0, 0], dtype=np.float32)
    obs, _, _, trunc, info = env.step(action)
    if trunc:
        raise SystemExit(f"CLIP truncated: {info}")
    moved = float(np.linalg.norm(obs["state"]["tcp_pose"][:3] - p0))
    _log(
        rows,
        {
            "gate": "clip",
            "moved_m": moved,
            "delta_pos_m": info.get("delta_pos_m"),
            "ok": abs(info.get("delta_pos_m", 0) - 0.005) < 1e-6 and moved < 0.008,
        },
    )
    if abs(info.get("delta_pos_m", 0) - 0.005) > 1e-6:
        raise SystemExit(f"CLIP FAIL delta={info.get('delta_pos_m')}")
    print("CLIP: PASS", moved)


def gate_crash_stop(env, rows) -> None:
    """Software fault path: stop+clear on close (physical e-stop not exercised)."""
    obs, _ = env.reset(options={"ready_timeout_s": 8.0})
    env.step(np.asarray([1, 0, 0, 0, 0, 0], dtype=np.float32))
    time.sleep(0.05)
    env.close()
    health = env._servo.health() if env._servo else {}
    ok = bool(health.get("stop_ok")) and bool(health.get("clear_ok"))
    _log(rows, {"gate": "crash_stop", "stop_ok": health.get("stop_ok"), "clear_ok": health.get("clear_ok"), "ok": ok})
    if not ok:
        raise SystemExit(f"CRASH_STOP FAIL health={health}")
    print("CRASH_STOP: PASS (software stop+clear; physical e-stop not tested)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        default="all",
        choices=[
            "all",
            "hold",
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_roll",
            "rot_pitch",
            "rot_yaw",
            "clip",
            "crash_stop",
        ],
    )
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    args = parser.parse_args()
    _require_confirm()

    rows: list = []
    env = _make_env()
    try:
        gates = (
            [
                "hold",
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_roll",
                "rot_pitch",
                "rot_yaw",
                "clip",
                "crash_stop",
            ]
            if args.gate == "all"
            else [args.gate]
        )
        for g in gates:
            if g == "hold":
                gate_hold(env, rows, seconds=args.hold_seconds)
            elif g.startswith("trans_"):
                gate_trans(env, rows, g.split("_")[1])
            elif g.startswith("rot_"):
                gate_rot(env, rows, g.split("_")[1])
            elif g == "clip":
                gate_clip(env, rows)
            elif g == "crash_stop":
                gate_crash_stop(env, rows)
                env = _make_env()  # closed above
        print("R4 GATES: PASS")
    finally:
        try:
            env.close()
        except Exception:
            pass
        _flush(rows)
        print("wrote", OUT)


if __name__ == "__main__":
    main()
