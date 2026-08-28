#!/usr/bin/env python3
"""R12 Actor recorder: zero policy + SpaceMouse, tty s/f snapshots (not every step)."""

from __future__ import annotations

import argparse
import os
import select
import sys
import threading
import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from hilserl_wa2.experiments.classifier_io import (  # noqa: E402
    FROZEN_SPACE_HASH,
    make_sample,
    write_classifier_bundle,
)
from hilserl_wa2.experiments.env_factory import make_wa2_environment  # noqa: E402
from hilserl_wa2.experiments.task_config import load_task, space_sha256  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--successes-needed", type=int, default=10)
    parser.add_argument("--failures-needed", type=int, default=20)
    parser.add_argument("--mode", choices=("fake", "live"), required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--confirm-live", default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _append_log(run_dir: Path, line: str) -> None:
    path = run_dir / "operator_log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _poll_cmd() -> Optional[str]:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    token = sys.stdin.readline().strip().lower()
    if token in ("s", "f", "e"):
        return token
    return None


class _AsyncHandToggle:
    def __init__(self, env: Any):
        self._env = env
        self._lock = threading.Lock()
        self._busy = False
        self.toggles = 0

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._busy)

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
                with self._lock:
                    self.toggles += 1
                command = str(result.get("command"))
                nxt = "open/place" if command == "grasp" else "grasp"
                print(
                    f"hand {command} ok — tap RIGHT to {nxt}; type e to reset episode",
                    flush=True,
                )
            else:
                print(f"hand failed: {result}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"hand exception: {exc}", flush=True)
        finally:
            with self._lock:
                self._busy = False


def _distribute(total: int, n_eps: int) -> List[int]:
    base, rem = divmod(int(total), int(n_eps))
    return [base + (1 if i < rem else 0) for i in range(n_eps)]


def _episode_sidecars(
    success: List[Dict[str, Any]],
    failure: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n_success": 0, "n_failure": 0}
    )
    for row in success:
        counts[str(row["episode_id"])]["n_success"] += 1
    for row in failure:
        counts[str(row["episode_id"])]["n_failure"] += 1
    sidecars = []
    for index, episode_id in enumerate(sorted(counts)):
        sidecars.append(
            {
                "episode_id": episode_id,
                "n_success": counts[episode_id]["n_success"],
                "n_failure": counts[episode_id]["n_failure"],
                "rel": f"episodes/ep{index:03d}.json",
            }
        )
    return sidecars


def _write_bundle(
    *,
    args: argparse.Namespace,
    task,
    space_hash: str,
    bundle_dir: Path,
    success: List[Dict[str, Any]],
    failure: List[Dict[str, Any]],
) -> None:
    if space_hash != FROZEN_SPACE_HASH:
        raise SystemExit(
            f"space_hash {space_hash} != frozen {FROZEN_SPACE_HASH}"
        )
    write_classifier_bundle(
        bundle_dir,
        bundle_name=str(args.out_name),
        success=success,
        failure=failure,
        manifest_extra={
            "task_id": task.task_id,
            "exp_name": task.exp_name,
            "space_hash": space_hash,
            "config_bundle_hash": task.config_bundle_hash(),
            "operator": args.operator,
            "mode": args.mode,
        },
        episode_sidecars=_episode_sidecars(success, failure),
    )


def _record_fake(args: argparse.Namespace, task, run_dir: Path, bundle_dir: Path) -> None:
    env = make_wa2_environment(task, fake_env=True, classifier=False)
    space_hash = space_sha256(env.observation_space, env.action_space)
    success: List[Dict[str, Any]] = []
    failure: List[Dict[str, Any]] = []
    n_eps = 5 if int(args.successes_needed) <= 10 else 10
    n_eps = max(n_eps, 5)
    succ_plan = _distribute(int(args.successes_needed), n_eps)
    fail_plan = _distribute(int(args.failures_needed), n_eps)
    try:
        for ep_i, (n_s, n_f) in enumerate(zip(succ_plan, fail_plan)):
            obs, _info = env.reset(seed=int(args.seed) + ep_i)
            episode_id = f"fake-ep-{ep_i:03d}"
            index = 0
            for _ in range(max(n_s + n_f, 1)):
                obs, _r, _t, _tr, _i = env.step(np.zeros(6, dtype=np.float32))
            for _ in range(n_s):
                success.append(
                    make_sample(
                        episode_id=episode_id,
                        label=1,
                        index=index,
                        created_at=_utc_now(),
                        observations=obs,
                    )
                )
                index += 1
            for _ in range(n_f):
                failure.append(
                    make_sample(
                        episode_id=episode_id,
                        label=0,
                        index=index,
                        created_at=_utc_now(),
                        observations=obs,
                    )
                )
                index += 1
    finally:
        env.close()
    _write_bundle(
        args=args,
        task=task,
        space_hash=space_hash,
        bundle_dir=bundle_dir,
        success=success,
        failure=failure,
    )
    print("R12_RECORD_FAKE: PASS")
    print(f"SUCCESS_SNAPSHOTS={len(success)}")
    print(f"FAILURE_SNAPSHOTS={len(failure)}")
    print(f"BUNDLE={bundle_dir}")
    _append_log(
        run_dir,
        f"{_utc_now()} fake success={len(success)} failure={len(failure)} "
        f"bundle={bundle_dir.name}",
    )


def _snapshot(
    *,
    obs: Dict[str, Any],
    episode_id: str,
    index: int,
    label: int,
    success: List[Dict[str, Any]],
    failure: List[Dict[str, Any]],
    needed_s: int,
    needed_f: int,
) -> int:
    sample = make_sample(
        episode_id=episode_id,
        label=label,
        index=index,
        created_at=_utc_now(),
        observations=obs,
    )
    if label == 1:
        success.append(sample)
        kind = "success"
    else:
        failure.append(sample)
        kind = "failure"
    print(
        f"SNAP {kind}  success={len(success)}/{needed_s}  "
        f"failure={len(failure)}/{needed_f}  ep={episode_id}",
        flush=True,
    )
    return index + 1


def _record_live(args: argparse.Namespace, task, run_dir: Path, bundle_dir: Path) -> None:
    if str(args.confirm_live) != "YES":
        raise SystemExit("live mode requires --confirm-live YES")
    if os.environ.get("R4_CONFIRM") != "YES" or os.environ.get("R5_CONFIRM") != "YES":
        raise SystemExit("live mode requires R4_CONFIRM=YES and R5_CONFIRM=YES")
    assert_no_teleop()
    env = make_wa2_environment(task, fake_env=False, classifier=False)
    base = unwrap_env(env)
    space_hash = space_sha256(env.observation_space, env.action_space)
    success: List[Dict[str, Any]] = []
    failure: List[Dict[str, Any]] = []
    needed_s = int(args.successes_needed)
    needed_f = int(args.failures_needed)
    zero = np.zeros(6, dtype=np.float32)
    print(
        "R12 classifier capture: TAP left = SpaceMouse session; TAP right = grasp.\n"
        "type s + Enter = success snapshot; f + Enter = failure; e + Enter = reset.\n"
        "Episode does NOT auto-reset on max_steps — only your e (or servo fault).\n"
        "Unlabeled steps are NOT saved. Classifier is NOT loaded.",
        flush=True,
    )
    episode_index = 0
    interrupted = False
    try:
        while len(success) < needed_s or len(failure) < needed_f:
            obs, info = env.reset()
            reset_ok = bool(info.get("reset_ok", False))
            episode_id = f"live-ep-{episode_index:03d}"
            print(
                f"\n=== {episode_id}  success {len(success)}/{needed_s}  "
                f"failure {len(failure)}/{needed_f}  reset_ok={reset_ok} ===",
                flush=True,
            )
            hand = _AsyncHandToggle(base)
            prev_right = False
            snap_index = 0
            last_t = time.monotonic()
            warned_time_cap = False
            while True:
                obs, _reward, terminated, truncated, step_info = env.step(zero)
                now = time.monotonic()
                dt = now - last_t
                last_t = now
                step_info = dict(step_info)
                right = bool(step_info.get("sm_right"))
                if right and not prev_right:
                    hand.kick()
                prev_right = right
                if step_info.get("sm_session_enter"):
                    print("session ON — type s/f to snapshot, e to reset", flush=True)
                if int(step_info.get("step_count") or 0) % 50 == 1:
                    print(
                        f"CTRL dt={dt*1000:.0f}ms step={step_info.get('step_count')} "
                        f"intervened={int(bool(step_info.get('intervened')))} "
                        f"success={len(success)}/{needed_s} "
                        f"failure={len(failure)}/{needed_f}",
                        flush=True,
                    )
                cmd = _poll_cmd()
                if cmd == "s":
                    snap_index = _snapshot(
                        obs=obs,
                        episode_id=episode_id,
                        index=snap_index,
                        label=1,
                        success=success,
                        failure=failure,
                        needed_s=needed_s,
                        needed_f=needed_f,
                    )
                    if len(success) >= needed_s and len(failure) >= needed_f:
                        break
                elif cmd == "f":
                    snap_index = _snapshot(
                        obs=obs,
                        episode_id=episode_id,
                        index=snap_index,
                        label=0,
                        success=success,
                        failure=failure,
                        needed_s=needed_s,
                        needed_f=needed_f,
                    )
                    if len(success) >= needed_s and len(failure) >= needed_f:
                        break
                elif cmd == "e":
                    print(f"end episode {episode_id}", flush=True)
                    break
                # R12: episode ends only on operator `e` (or hard fault).
                # max_steps truncated must NOT auto-reset — operator may still
                # need time to reach the place pose and press s/f.
                if step_info.get("servo_faulted") or bool(terminated):
                    print(
                        "episode fault/terminated — starting a new one "
                        f"(fault={int(bool(step_info.get('servo_faulted')))} "
                        f"term={int(bool(terminated))})",
                        flush=True,
                    )
                    break
                if truncated:
                    step_n = int(step_info.get("step_count") or 0)
                    max_n = int(step_info.get("max_steps") or 0)
                    if max_n and step_n >= max_n:
                        if not warned_time_cap:
                            print(
                                f"time cap step={step_n}/{max_n} — "
                                "keep going; type e when YOU want reset "
                                "(s/f still work)",
                                flush=True,
                            )
                            warned_time_cap = True
                    elif not warned_time_cap:
                        print(
                            "soft truncated (non-fault) — keep episode; "
                            "type e to reset",
                            flush=True,
                        )
                        warned_time_cap = True
            waited = 0.0
            while hand.busy and waited < 6.0:
                time.sleep(0.05)
                waited += 0.05
            _append_log(
                run_dir,
                f"{_utc_now()} {episode_id} snaps={snap_index} "
                f"success={len(success)} failure={len(failure)}",
            )
            episode_index += 1
            if len(success) >= needed_s and len(failure) >= needed_f:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("Ctrl+C — stop+clear, writing collected snapshots", flush=True)
    finally:
        env.close()

    if not success and not failure:
        raise SystemExit("no snapshots collected")
    _write_bundle(
        args=args,
        task=task,
        space_hash=space_hash,
        bundle_dir=bundle_dir,
        success=success,
        failure=failure,
    )
    quota_ok = len(success) >= needed_s and len(failure) >= needed_f
    schema = needed_s <= 10 and needed_f <= 20
    if quota_ok and not interrupted:
        if schema:
            print("R12_RECORD_LIVE_SCHEMA: PASS")
        print("R12_RECORD_LIVE: PASS")
    else:
        print("R12_RECORD_LIVE: FAIL")
    print(f"SUCCESS_SNAPSHOTS={len(success)}")
    print(f"FAILURE_SNAPSHOTS={len(failure)}")
    print(f"EPISODES={episode_index}")
    print("ROBOT_MOTION=true")
    print(f"BUNDLE={bundle_dir}")
    if interrupted or not quota_ok:
        raise SystemExit(1)


def main() -> int:
    args = _parse_args()
    task = load_task(args.task)
    if task.classifier_keys != ("head", "wrist"):
        raise SystemExit(
            f"task.classifier_keys={task.classifier_keys} must be ('head', 'wrist')"
        )
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "classifier").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    bundle_dir = run_dir / "classifier" / f"{args.out_name}_{uuid.uuid4().hex[:8]}"
    if bundle_dir.exists():
        raise SystemExit(f"bundle already exists: {bundle_dir}")
    try:
        if args.mode == "fake":
            _record_fake(args, task, run_dir, bundle_dir)
        else:
            _record_live(args, task, run_dir, bundle_dir)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
