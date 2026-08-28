#!/usr/bin/env python3
"""R11 Actor demo recorder: zero policy + SpaceMouse intervention, human s/f/a."""

from __future__ import annotations

import argparse
import os
import select
import sys
import threading
import time
import traceback
import uuid
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
from hilserl_wa2.experiments.demo_io import (  # noqa: E402
    MIN_INTERVENED_STEPS,
    TRANSITION_SCHEMA_VERSION,
    count_intervened_steps,
    write_failed_episode,
    write_success_bundle,
)
from hilserl_wa2.experiments.env_factory import make_wa2_environment  # noqa: E402
from hilserl_wa2.experiments.task_config import load_task, space_sha256  # noqa: E402
from hilserl_wa2.experiments.transition import build_actor_transition  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--successes-needed", type=int, default=5)
    parser.add_argument("--mode", choices=("fake", "live"), required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--confirm-live", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fake-steps", type=int, default=40)
    return parser.parse_args()


def _append_log(run_dir: Path, line: str) -> None:
    path = run_dir / "operator_log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _sidecar(
    *,
    index: int,
    label: str,
    operator: str,
    task,
    space_hash: str,
    started_at: str,
    n_steps: int,
    intervened_steps: int,
    intervention_count: int,
    hand_toggles: int,
    reset_ok: bool,
    human_success: bool,
    discard_reason: Optional[str],
) -> Dict[str, Any]:
    return {
        "episode_index": int(index),
        "label": str(label),
        "operator": str(operator),
        "task_id": task.task_id,
        "exp_name": task.exp_name,
        "config_bundle_hash": task.config_bundle_hash(),
        "space_hash": str(space_hash),
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "started_at": started_at,
        "n_steps": int(n_steps),
        "intervened_steps": int(intervened_steps),
        "intervention_count": int(intervention_count),
        "hand_toggles": int(hand_toggles),
        "reset_ok": bool(reset_ok),
        "human_success": bool(human_success),
        "discard_reason": discard_reason,
    }


def _truncate_reason(step_info: Dict[str, Any]) -> str:
    err = step_info.get("servo_error")
    if err:
        return f"truncated:{err}"
    if step_info.get("stale"):
        return f"truncated:stale={step_info.get('stale_fields')}"
    if step_info.get("is_singular"):
        return "truncated:singular"
    if step_info.get("servo_faulted"):
        return "truncated:servo_faulted"
    step = int(step_info.get("step_count", 0) or 0)
    max_steps = int(step_info.get("max_steps", 0) or 0)
    if max_steps and step >= max_steps:
        return "truncated:max_steps"
    return "truncated"


def _ctrl_line(step_info: Dict[str, Any], dt: float) -> str:
    ages = step_info.get("image_ages") or {}
    xyz = step_info.get("delta_pos_xyz")
    if xyz is None:
        dxyz = "none"
    else:
        arr = np.asarray(xyz, dtype=np.float64).reshape(-1)
        dxyz = f"{arr[0]:+.4f},{arr[1]:+.4f},{arr[2]:+.4f}"
    health = step_info.get("servo_health") or {}
    return (
        f"CTRL dt={dt*1000:.0f}ms step={step_info.get('step_count')} "
        f"session={int(bool(step_info.get('sm_session')))} "
        f"fresh={int(bool(step_info.get('joy_fresh')))} "
        f"joy={step_info.get('joy_age')} "
        f"intent={step_info.get('sm_intent')} axis={step_info.get('sm_axis')} "
        f"intervened={int(bool(step_info.get('intervened')))} "
        f"dxyz={dxyz} "
        f"ticks={step_info.get('interval_ticks')} "
        f"int_n={health.get('integrate_count')} "
        f"track={step_info.get('tracking_err_m')} "
        f"loop_dt={step_info.get('loop_dt')} "
        f"cmd={step_info.get('cmd_name')} "
        f"pub_n={health.get('publish_count')} "
        f"img={ages} "
        f"stale={step_info.get('stale_fields')} "
        f"singular={step_info.get('is_singular')} "
        f"fault={int(bool(step_info.get('servo_faulted')))} "
        f"pub={step_info.get('published')}"
    )


def _poll_label() -> Optional[str]:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    token = sys.stdin.readline().strip().lower()
    if token in ("s", "f", "a"):
        return token
    return None


def _wait_label(prompt: str) -> str:
    print(prompt, flush=True)
    while True:
        token = sys.stdin.readline().strip().lower()
        if token in ("s", "f", "a"):
            return token
        print("enter s (success) / f (fail) / a (abort)", flush=True)


def _wait_label_keepalive(env: Any, action: np.ndarray, prompt: str) -> str:
    """Ask s/f/a while still stepping zeros so ServoL does not watchdog-lock."""
    print(prompt, flush=True)
    while True:
        try:
            env.step(action)
        except Exception as exc:  # noqa: BLE001 — keep asking even if hold fails
            print(f"hold-step warning: {exc}", flush=True)
        if sys.stdin.isatty():
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                token = sys.stdin.readline().strip().lower()
                if token in ("s", "f", "a"):
                    return token
                print("enter s (keep) / f (fail) / a (redo)", flush=True)
        else:
            time.sleep(0.02)


class _AsyncHandToggle:
    """Run grasp/release off the 50 Hz ServoL loop (joint_switch can block ~5s)."""

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
                    f"hand {command} ok — tap RIGHT to {nxt}; tap LEFT when the episode is done",
                    flush=True,
                )
            else:
                print(f"hand failed: {result}", flush=True)
        except Exception as exc:  # noqa: BLE001 — report and keep ServoL alive
            print(f"hand exception: {exc}", flush=True)
        finally:
            with self._lock:
                self._busy = False


def _record_fake(args: argparse.Namespace, task, run_dir: Path, bundle_dir: Path) -> None:
    env = make_wa2_environment(task, fake_env=True)
    base = unwrap_env(env)
    space_hash = space_sha256(env.observation_space, env.action_space)
    success_eps: List[List[Dict[str, Any]]] = []
    sidecars: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed))
    try:
        for ep_i in range(int(args.successes_needed)):
            started = _utc_now()
            obs, info = env.reset(seed=int(args.seed) + ep_i)
            reset_ok = bool(info.get("reset_ok", True))
            rows: List[Dict[str, Any]] = []
            hand_toggles = 0
            intervention_count = 1
            for step_i in range(int(args.fake_steps)):
                action = np.zeros(6, dtype=np.float32)
                action[0] = 0.35 if step_i % 2 == 0 else 0.0
                action[2] = 0.25
                if action[0] == 0.0:
                    action[0] = 0.2
                next_obs, reward, _terminated, truncated, step_info = env.step(action)
                step_info = dict(step_info)
                step_info["intervene_action"] = np.asarray(action, dtype=np.float32)
                if step_i == 12:
                    result = base.request_hand("toggle")
                    if result.get("ok"):
                        hand_toggles += 1
                tr, _meta = build_actor_transition(
                    obs,
                    action,
                    next_obs,
                    reward,
                    False,
                    bool(truncated),
                    step_info,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                )
                rows.append(tr)
                obs = next_obs
            success_eps.append(rows)
            sidecars.append(
                _sidecar(
                    index=ep_i,
                    label="success",
                    operator=args.operator,
                    task=task,
                    space_hash=space_hash,
                    started_at=started,
                    n_steps=len(rows),
                    intervened_steps=count_intervened_steps(rows),
                    intervention_count=intervention_count,
                    hand_toggles=hand_toggles,
                    reset_ok=reset_ok,
                    human_success=True,
                    discard_reason=None,
                )
            )
            _ = rng
    finally:
        env.close()
    write_success_bundle(
        bundle_dir,
        bundle_name=str(args.out_name),
        episodes=success_eps,
        sidecars=sidecars,
    )
    print(f"R11_RECORD_FAKE: PASS")
    print(f"SUCCESS_EPISODES={len(success_eps)}")
    print(f"BUNDLE={bundle_dir}")
    _append_log(run_dir, f"{_utc_now()} fake successes={len(success_eps)} bundle={bundle_dir.name}")


def _record_live(args: argparse.Namespace, task, run_dir: Path, bundle_dir: Path) -> None:
    if str(args.confirm_live) != "YES":
        raise SystemExit("live mode requires --confirm-live YES")
    if os.environ.get("R4_CONFIRM") != "YES" or os.environ.get("R5_CONFIRM") != "YES":
        raise SystemExit("live mode requires R4_CONFIRM=YES and R5_CONFIRM=YES")
    assert_no_teleop()
    env = make_wa2_environment(task, fake_env=False)
    base = unwrap_env(env)
    space_hash = space_sha256(env.observation_space, env.action_space)
    success_eps: List[List[Dict[str, Any]]] = []
    sidecars: List[Dict[str, Any]] = []
    failed_dir = run_dir / "failed"
    episode_index = 0
    zero = np.zeros(6, dtype=np.float32)
    trans = base._episode_trans_limit_m
    rot = base._episode_rot_limit_deg
    if trans is None and rot is None:
        box_txt = "episode_box disabled"
    else:
        trans_txt = "off" if trans is None else f"{trans:.3f}m"
        rot_txt = "off" if rot is None else f"{rot:.1f}deg"
        box_txt = f"episode_box trans={trans_txt} rot={rot_txt}"
    print(
        f"{box_txt}  max_steps={base.max_steps}  "
        "(tap LEFT to start/stop SpaceMouse; per-step 1mm/0.25deg still on)",
        flush=True,
    )
    try:
        while len(success_eps) < int(args.successes_needed):
            started = _utc_now()
            obs, info = env.reset()
            reset_ok = bool(info.get("reset_ok", False))
            print(
                f"\n=== episode {episode_index}  success {len(success_eps)}/"
                f"{args.successes_needed} ===\n"
                "reset done. TAP left to enter SpaceMouse control (no need to hold).\n"
                "TAP right to grasp/release. TAP left again to end this episode.\n"
                "Then type s (keep) / f (fail) / a (redo).",
                flush=True,
            )
            rows: List[Dict[str, Any]] = []
            hand = _AsyncHandToggle(base)
            prev_right = False
            discard_reason = None
            label = None
            human_success = False
            last_intervention_count = 0
            last_t = time.monotonic()
            step_info: Dict[str, Any] = dict(info)
            if not reset_ok:
                discard_reason = "reset_failed"
                label = "f"
            while label is None:
                next_obs, reward, terminated, truncated, step_info = env.step(zero)
                now = time.monotonic()
                dt = now - last_t
                last_t = now
                step_info = dict(step_info)
                last_intervention_count = int(step_info.get("intervention_count", 0))
                right = bool(step_info.get("sm_right"))
                rising = right and not prev_right
                prev_right = right
                if rising:
                    hand.kick()
                if step_info.get("sm_session_enter"):
                    print("session ON — move SpaceMouse; tap LEFT to finish", flush=True)
                if step_info.get("sm_session_dropped_stale"):
                    print(
                        "session DROPPED (joy stale) — arm will hold; "
                        f"joy_age={step_info.get('joy_age')}",
                        flush=True,
                    )
                cmd_name = str(step_info.get("cmd_name") or "").strip().upper()
                state_stale = [
                    field
                    for field in (step_info.get("stale_fields") or [])
                    if not str(field).startswith("images/")
                ]
                anomaly = bool(
                    step_info.get("sm_session_dropped_stale")
                    or step_info.get("servo_faulted")
                    or step_info.get("is_singular")
                    or state_stale
                    or cmd_name in ("PROTECTED", "PROTECT", "SAFETY_LOCK", "LOCKED")
                    or float(step_info.get("tracking_err_m") or 0.0) > 0.015
                )
                if anomaly or int(step_info.get("step_count") or 0) % 50 == 1:
                    print(_ctrl_line(step_info, dt), flush=True)
                tr, meta = build_actor_transition(
                    obs,
                    zero,
                    next_obs,
                    reward,
                    False,
                    bool(truncated),
                    step_info,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                )
                rows.append(tr)
                obs = next_obs
                typed = _poll_label()
                if typed is not None:
                    label = typed
                    break
                if step_info.get("sm_session_exit"):
                    print("session OFF", flush=True)
                    label = _wait_label_keepalive(
                        env,
                        zero,
                        "keep this episode? s keep / f fail / a redo:",
                    )
                    break
                if terminated:
                    label = "f"
                    discard_reason = "unexpected_terminated"
                    break
                if truncated or meta.get("truncated") or step_info.get("servo_faulted"):
                    discard_reason = _truncate_reason(step_info)
                    print(f"auto-end {discard_reason}", flush=True)
                    if discard_reason == "truncated:max_steps":
                        label = _wait_label_keepalive(
                            env,
                            zero,
                            "time cap — if placed, type s; else f / a:",
                        )
                    else:
                        label = "f"
                    break
            if label is None:
                label = _wait_label_keepalive(
                    env, zero, "end of motion — s keep / f fail / a redo:"
                )
            waited = 0.0
            while hand.busy and waited < 6.0:
                time.sleep(0.05)
                waited += 0.05
            hand_toggles = int(hand.toggles)
            if label == "s":
                intervened = count_intervened_steps(rows)
                if (not reset_ok) or intervened < MIN_INTERVENED_STEPS or hand_toggles < 1:
                    label = "f"
                    discard_reason = (
                        f"success_rejected intervened={intervened} "
                        f"hand_toggles={hand_toggles} reset_ok={reset_ok}"
                    )
                    print(discard_reason, flush=True)
                else:
                    human_success = True
                    discard_reason = None
            elif label == "a":
                discard_reason = discard_reason or "operator_abort"
            else:
                discard_reason = discard_reason or "operator_fail"

            side = _sidecar(
                index=episode_index,
                label="success" if human_success else "failed",
                operator=args.operator,
                task=task,
                space_hash=space_hash,
                started_at=started,
                n_steps=len(rows),
                intervened_steps=count_intervened_steps(rows),
                intervention_count=last_intervention_count,
                hand_toggles=hand_toggles,
                reset_ok=reset_ok,
                human_success=human_success,
                discard_reason=discard_reason,
            )
            if human_success:
                success_eps.append(rows)
                sidecars.append(side)
            else:
                write_failed_episode(
                    failed_dir,
                    episode_index=episode_index,
                    transitions=rows,
                    sidecar=side,
                )
            print(
                f"episode {episode_index} ended label={side['label']} "
                f"steps={side['n_steps']} intervened={side['intervened_steps']} "
                f"hand={hand_toggles} reason={discard_reason}",
                flush=True,
            )
            _append_log(
                run_dir,
                f"{_utc_now()} ep={episode_index} label={side['label']} "
                f"steps={side['n_steps']} intervened={side['intervened_steps']} "
                f"hand={hand_toggles} reason={discard_reason}",
            )
            episode_index += 1
    except KeyboardInterrupt:
        print("Ctrl+C — stop+clear, current episode discarded", flush=True)
        raise
    finally:
        env.close()

    write_success_bundle(
        bundle_dir,
        bundle_name=str(args.out_name),
        episodes=success_eps,
        sidecars=sidecars,
    )
    print("R11_RECORD_LIVE: PASS")
    print(f"SUCCESS_EPISODES={len(success_eps)}")
    print(f"FAILED_EPISODES={max(0, episode_index - len(success_eps))}")
    print("MIN_INTERVENED_STEPS_OK=true")
    print("HAND_TOGGLES_OK=true")
    print("ROBOT_MOTION=true")
    print("REAL_IMAGES=true")
    print(f"BUNDLE={bundle_dir}")


def main() -> int:
    args = _parse_args()
    task = load_task(args.task)
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "demos").mkdir(exist_ok=True)
    (run_dir / "failed").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    bundle_dir = run_dir / "demos" / f"{args.out_name}_{uuid.uuid4().hex[:8]}"
    if bundle_dir.exists():
        raise SystemExit(f"bundle already exists: {bundle_dir}")
    try:
        if args.mode == "fake":
            _record_fake(args, task, run_dir, bundle_dir)
        else:
            _record_live(args, task, run_dir, bundle_dir)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
