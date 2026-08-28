#!/usr/bin/env python3
"""R12 Actor live eval: load ckpt, print p/succeed, never reset from classifier."""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Orin/Jetson: disable XLA GPU conv autotune before any JAX import path.
# Otherwise first classifier compile can abort with:
#   cudaGetFuncBySymbol: no kernel image is available for execution on the device
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT / "hil-serl-main" / "examples"))
sys.path.insert(0, str(SRC_ROOT / "hil-serl-main" / "serl_launcher"))

from hilserl_wa2.experiments.actor_safety import (  # noqa: E402
    assert_no_teleop,
    unwrap_env,
)
from hilserl_wa2.experiments.env_factory import make_wa2_environment  # noqa: E402
from hilserl_wa2.experiments.task_config import load_task  # noqa: E402
from hilserl_wa2.wrappers.reward_classifier import load_threshold_json  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _poll_cmd() -> Optional[str]:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    token = sys.stdin.readline().strip().lower()
    if token in ("s", "f", "q"):
        return token
    return None


def _parse_image_keys(raw: str):
    keys = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if keys != ("head", "wrist"):
        raise SystemExit(f"--image-keys must be head,wrist, got {keys}")
    return keys


class _ResetGuard:
    def __init__(self, env: Any):
        self.env = env
        self.count = 0
        self._orig = env.reset

        def wrapped(**kwargs):
            self.count += 1
            return self._orig(**kwargs)

        env.reset = wrapped  # type: ignore[method-assign]


class _AsyncHandToggle:
    """Same as record_r12: SpaceMouse RIGHT edge → request_hand(toggle)."""

    def __init__(self, env: Any):
        self._env = env
        self._lock = threading.Lock()
        self._busy = False

    def kick(self) -> None:
        with self._lock:
            if self._busy:
                print("hand busy — ignore extra right press", flush=True)
                return
            self._busy = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            result = self._env.request_hand("toggle")
            if result.get("ok"):
                command = str(result.get("command"))
                nxt = "open/place" if command == "grasp" else "grasp"
                print(
                    f"hand {command} ok — tap RIGHT to {nxt}",
                    flush=True,
                )
            else:
                print(f"hand failed: {result}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"hand exception: {exc}", flush=True)
        finally:
            with self._lock:
                self._busy = False


def _hard_episode_end(info: Dict[str, Any], terminated: bool, truncated: bool) -> bool:
    """True only for real faults / terminated — not soft max_steps truncate.

    Matches record_r12_success_fail: time-cap truncated must not fail Step9.
    """
    if bool(info.get("servo_faulted")) or bool(terminated):
        return True
    if not truncated:
        return False
    step_n = int(info.get("step_count") or 0)
    max_n = int(info.get("max_steps") or 0)
    # Soft time-cap: keep going for live eval (labels + hold).
    if max_n and step_n >= max_n:
        return False
    # Other truncate reasons (singular, stale, etc.) still count as hard end.
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold-json", required=True)
    parser.add_argument("--image-keys", default="head,wrist")
    parser.add_argument("--confirm-live", default="")
    parser.add_argument("--end-episode", default="false")
    parser.add_argument("--n-labels", type=int, default=20)
    parser.add_argument("--hold-seconds", type=float, default=30.0)
    args = parser.parse_args()

    end_episode = str(args.end_episode).strip().lower()
    if end_episode not in ("false", "0", "no"):
        print("END_EPISODE=true")
        print("R12_LIVE_EVAL: FAIL — end_episode must be false")
        return 1
    _parse_image_keys(args.image_keys)
    if str(args.confirm_live) != "YES":
        raise SystemExit("live eval requires --confirm-live YES")
    if os.environ.get("R4_CONFIRM") != "YES" or os.environ.get("R5_CONFIRM") != "YES":
        raise SystemExit("live eval requires R4_CONFIRM=YES and R5_CONFIRM=YES")
    assert_no_teleop()

    payload = load_threshold_json(args.threshold_json)
    threshold = float(payload["threshold"])
    consecutive_n = int(payload.get("consecutive_n", 3))
    task = load_task(args.task)
    env = make_wa2_environment(
        task,
        fake_env=False,
        classifier=True,
        classifier_checkpoint=str(Path(args.checkpoint).expanduser()),
        classifier_threshold=threshold,
        classifier_consecutive_n=consecutive_n,
        end_episode=False,
    )
    guard = _ResetGuard(env)
    base = unwrap_env(env)
    hand = _AsyncHandToggle(base)
    zero = np.zeros(6, dtype=np.float32)
    labels = []
    jsonl_path = Path(args.checkpoint).expanduser().resolve()
    if jsonl_path.is_dir():
        out_log = jsonl_path.parent / "live_eval.jsonl"
    else:
        out_log = jsonl_path.parent / "live_eval.jsonl"
    print(
        f"R12 live eval threshold={threshold} consecutive_n={consecutive_n}\n"
        "zero policy + SpaceMouse: LEFT=session, RIGHT=hand toggle.\n"
        "type s/f to label current frame "
        f"({args.n_labels} needed), then hold {args.hold_seconds}s.\n"
        "Classifier must NOT reset.",
        flush=True,
    )
    obs, info = env.reset()
    prev_right = False
    warned_time_cap = False
    try:
        while len(labels) < int(args.n_labels):
            obs, reward, terminated, truncated, info = env.step(zero)
            info = dict(info)
            right = bool(info.get("sm_right"))
            if right and not prev_right:
                hand.kick()
            prev_right = right
            print(
                f"p={float(info.get('classifier_p', 0)):.3f} "
                f"succeed={int(bool(info.get('succeed')))} "
                f"reward={float(reward):.0f} term={int(bool(terminated))} "
                f"labels={len(labels)}/{args.n_labels}",
                flush=True,
            )
            if _hard_episode_end(info, terminated, truncated):
                print("R12_LIVE_EVAL: FAIL — hard episode end during labeling")
                print(f"RESET_CALLED={str(guard.count > 1).lower()}")
                return 1
            if truncated and not warned_time_cap:
                step_n = int(info.get("step_count") or 0)
                max_n = int(info.get("max_steps") or 0)
                print(
                    f"soft truncated step={step_n}/{max_n} — keep going "
                    "(max_steps does not fail Step9)",
                    flush=True,
                )
                warned_time_cap = True
            cmd = _poll_cmd()
            if cmd in ("s", "f"):
                human = 1 if cmd == "s" else 0
                model = 1 if info.get("succeed") else 0
                row = {
                    "t": _utc_now(),
                    "human": human,
                    "model": model,
                    "p": float(info.get("classifier_p", 0.0)),
                    "reward": float(reward),
                }
                labels.append(row)
                print(f"LABEL human={human} model={model}", flush=True)
            elif cmd == "q":
                break
        hold_s = float(args.hold_seconds)
        bursts = 0
        in_burst = False
        burst_start = None
        max_burst_s = 0.0
        hold_end = time.monotonic() + hold_s
        print(f"hold {hold_s}s — watch false triggers (RIGHT still toggles hand)", flush=True)
        while time.monotonic() < hold_end:
            obs, reward, terminated, truncated, info = env.step(zero)
            info = dict(info)
            right = bool(info.get("sm_right"))
            if right and not prev_right:
                hand.kick()
            prev_right = right
            succeed = bool(info.get("succeed"))
            now = time.monotonic()
            if succeed and not in_burst:
                in_burst = True
                bursts += 1
                burst_start = now
            elif (not succeed) and in_burst:
                max_burst_s = max(max_burst_s, now - (burst_start or now))
                in_burst = False
            if _hard_episode_end(info, terminated, truncated):
                print("R12_LIVE_EVAL: FAIL — hard episode end during hold")
                print("END_EPISODE=false")
                print(f"RESET_CALLED={str(guard.count > 1).lower()}")
                return 1
            if truncated and not warned_time_cap:
                step_n = int(info.get("step_count") or 0)
                max_n = int(info.get("max_steps") or 0)
                print(
                    f"soft truncated step={step_n}/{max_n} — keep hold "
                    "(max_steps does not fail Step9)",
                    flush=True,
                )
                warned_time_cap = True
        if in_burst:
            max_burst_s = max(max_burst_s, time.monotonic() - (burst_start or time.monotonic()))
    finally:
        env.close()

    out_log.parent.mkdir(parents=True, exist_ok=True)
    with out_log.open("w", encoding="utf-8") as handle:
        for row in labels:
            handle.write(json.dumps(row) + "\n")

    reset_called = guard.count > 1
    n_fp = sum(1 for row in labels if row["model"] == 1 and row["human"] == 0)
    n_fn = sum(1 for row in labels if row["model"] == 0 and row["human"] == 1)
    print(f"LABELS={len(labels)}")
    print(f"FP={n_fp} FN={n_fn}")
    print(f"HOLD_SECONDS={hold_s}")
    print(f"SUCCEED_BURSTS={bursts}")
    print(f"MAX_BURST_S={max_burst_s:.3f}")
    print("END_EPISODE=false")
    print(f"RESET_CALLED={str(reset_called).lower()}")
    print(f"LIVE_EVAL_JSONL={out_log}")
    if reset_called or len(labels) < int(args.n_labels):
        print("R12_LIVE_EVAL: FAIL")
        return 1
    print("R12_LIVE_EVAL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
