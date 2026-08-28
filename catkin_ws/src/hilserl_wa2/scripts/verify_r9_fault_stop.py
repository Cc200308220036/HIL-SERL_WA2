#!/usr/bin/env python3
"""R9 fail-closed Gate: server stop / actor exception / SIGINT → env.close + client.stop."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

CATKIN_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CATKIN_SRC))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "examples"))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "serl_launcher"))

SERVER = Path(__file__).resolve().parent / "r9_dummy_trainer_server.py"
ACTOR = Path(__file__).resolve().parent / "verify_r9_actor_local.py"


def _spawn_server(args, ready_file: Path, stop_after: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(SERVER),
        "--task",
        args.task,
        "--ip",
        args.server_ip,
        "--port",
        str(args.port),
        "--broadcast-port",
        str(args.broadcast_port),
        "--stop-after-env",
        str(stop_after),
        "--expect-env",
        "100000",
        "--ready-file",
        str(ready_file),
    ]
    proc = subprocess.Popen(cmd)
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if ready_file.is_file():
            return proc
        if proc.poll() is not None:
            raise SystemExit("R9_FAULT: FAIL — dummy server exited before READY")
        time.sleep(0.1)
    raise SystemExit("R9_FAULT: FAIL — dummy server READY timeout")


def main() -> None:
    parser = argparse.ArgumentParser(description="R9 Actor fail-closed Gate")
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--mode", choices=("fake", "readonly", "live-zero", "dry-run"), default="fake")
    parser.add_argument("--policy", default="")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5588)
    parser.add_argument("--broadcast-port", type=int, default=5589)
    parser.add_argument("--upload-every-steps", type=int, default=5)
    parser.add_argument("--expect-fault", default="")
    parser.add_argument("--inject-exception-at", type=int, default=0)
    parser.add_argument("--wait-for-sigint", action="store_true")
    parser.add_argument("--stop-after-env", type=int, default=30)
    args = parser.parse_args()

    policy = args.policy or ("zero" if args.mode == "live-zero" else "zero")
    actor_cmd = [
        sys.executable,
        str(ACTOR),
        "--task",
        args.task,
        "--mode",
        args.mode,
        "--policy",
        policy,
        "--steps",
        str(args.steps),
        "--server-ip",
        args.server_ip,
        "--port",
        str(args.port),
        "--broadcast-port",
        str(args.broadcast_port),
        "--upload-every-steps",
        str(args.upload_every_steps),
        "--skip-reset-motion",
    ]

    if args.wait_for_sigint:
        actor_cmd += ["--wait-for-sigint", "--without-server", "--expect-fault", "sigint"]
        print("R9_FAULT: waiting for Ctrl+C / SIGINT on actor ...", flush=True)
        raise SystemExit(subprocess.call(actor_cmd))

    if args.inject_exception_at:
        actor_cmd += [
            "--without-server",
            "--inject-exception-at",
            str(args.inject_exception_at),
            "--expect-fault",
            "actor_exception",
        ]
        rc = subprocess.call(actor_cmd)
        raise SystemExit(rc)

    expect = args.expect_fault or "server_disconnect"
    if expect != "server_disconnect":
        raise SystemExit(f"unsupported expect-fault {expect}")

    if args.mode == "live-zero" and os.environ.get("R4_CONFIRM") != "YES":
        raise SystemExit("R9_FAULT live-zero requires R4_CONFIRM=YES")

    ready = Path("/tmp/r9_fault_server_ready")
    if ready.exists():
        ready.unlink()
    proc = _spawn_server(args, ready, int(args.stop_after_env))
    try:
        actor_cmd += ["--expect-fault", "server_disconnect"]
        if args.mode == "fake":
            actor_cmd += ["--synthetic-intervention-steps", "0"]
        rc = subprocess.call(actor_cmd)
        raise SystemExit(rc)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
