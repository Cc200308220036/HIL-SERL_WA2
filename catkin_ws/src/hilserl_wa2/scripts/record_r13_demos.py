#!/usr/bin/env python3
"""R13 Actor demo recorder: 7D grasp + reward classifier (upstream-aligned).

Live recording enables the R12 reward classifier with ``end_episode=True`` so a
true placement can terminate with reward=1 (same idea as upstream
``examples/record_demos.py``). The operator still confirms with keyboard
``s`` / ``f`` / ``a`` whether to keep, discard, or redo the episode.

Orin-only. Do not start this on the Learner tree.
Right-click is handled solely by WA2GraspActionWrapper — do not also request_hand.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
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
    count_intervened_steps,
    grasp_action_counts,
    validate_r13_grasp_edges,
    write_failed_episode,
    write_success_bundle,
)
from hilserl_wa2.experiments.env_factory import make_wa2_environment  # noqa: E402
from hilserl_wa2.experiments.r13_protocol import TRANSITION_SCHEMA_VERSION  # noqa: E402
from hilserl_wa2.experiments.task_config import load_task, space_sha256  # noqa: E402
from hilserl_wa2.experiments.async_transition import TransitionPipeline  # noqa: E402
from hilserl_wa2.experiments.transition import build_actor_transition  # noqa: E402
from hilserl_wa2.envs.contracts import WA2EnvContract  # noqa: E402
from hilserl_wa2.wrappers.grasp_action import GRASP_DIM  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--successes-needed", type=int, default=20)
    parser.add_argument("--mode", choices=("fake", "live"), required=True)
    parser.add_argument("--out-name", default="wa2_bottle_pick_20_success_7d")
    parser.add_argument("--confirm-live", default="")
    parser.add_argument(
        "--classifier-checkpoint",
        default="",
        help="R12 classifier ckpt dir or checkpoint_N (required for live)",
    )
    parser.add_argument(
        "--classifier-threshold",
        type=float,
        default=None,
        help="override threshold.json; default read from ckpt sidecar",
    )
    parser.add_argument(
        "--classifier-consecutive-n",
        type=int,
        default=1,
        help="10 Hz high-level observations with p>=thr before succeed (default 1)",
    )
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
    grasp_counts: Dict[str, int],
    reset_ok: bool,
    human_success: bool,
    discard_reason: Optional[str],
) -> Dict[str, Any]:
    contract = WA2EnvContract.from_yaml(task.contract_path)
    return {
        "episode_index": int(index),
        "label": str(label),
        "operator": str(operator),
        "task_id": task.task_id,
        "exp_name": task.exp_name,
        "config_bundle_hash": task.config_bundle_hash(),
        "space_hash": str(space_hash),
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "policy_hz": float(contract.policy_hz),
        "servo_hz": float(contract.control_hz),
        "servo_ticks_per_action": int(contract.servo_ticks_per_action),
        "discount": float(task.discount),
        "classifier_consecutive_n": int(task.classifier_consecutive_n),
        "action_dim": GRASP_DIM,
        "started_at": started_at,
        "n_steps": int(n_steps),
        "intervened_steps": int(intervened_steps),
        "intervention_count": int(intervention_count),
        "hand_toggles": int(grasp_counts.get("nonzero", 0)),
        "n_grasp_plus": int(grasp_counts.get("plus", 0)),
        "n_grasp_minus": int(grasp_counts.get("minus", 0)),
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
    return (
        f"CTRL dt={dt*1000:.0f}ms step={step_info.get('step_count')} "
        f"session={int(bool(step_info.get('sm_session')))} "
        f"intervened={int(bool(step_info.get('intervened')))} "
        f"grasp={step_info.get('grasp_command')} "
        f"ticks={step_info.get('servo_ticks_executed')}/"
        f"{step_info.get('servo_ticks_requested')} "
        f"interrupt={step_info.get('interrupted_by')} "
        f"dxyz={dxyz} "
        f"cmd={step_info.get('cmd_name')} "
        f"p={float(step_info.get('classifier_p') or 0.0):.3f} "
        f"streak={int(step_info.get('classifier_streak') or 0)} "
        f"img={ages} "
        f"stale={step_info.get('stale_fields')} "
        f"singular={step_info.get('is_singular')} "
        f"fault={int(bool(step_info.get('servo_faulted')))}"
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


def _wait_label_keepalive(env: Any, action: np.ndarray, prompt: str) -> str:
    print(prompt, flush=True)
    while True:
        try:
            env.step(action)
        except Exception as exc:  # noqa: BLE001
            print(f"hold-step warning: {exc}", flush=True)
        if sys.stdin.isatty():
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                token = sys.stdin.readline().strip().lower()
                if token in ("s", "f", "a"):
                    return token
                print("enter s (keep) / f (fail) / a (redo)", flush=True)
        else:
            import time

            time.sleep(0.02)


def _zero7() -> np.ndarray:
    return np.zeros(GRASP_DIM, dtype=np.float32)


def _record_fake(args: argparse.Namespace, task, run_dir: Path, bundle_dir: Path) -> None:
    env = make_wa2_environment(task, fake_env=True, classifier=False, grasp_action=True)
    if tuple(env.action_space.shape) != (GRASP_DIM,):
        raise SystemExit(f"R13_RECORD: FAIL — fake action {env.action_space.shape} is not 7D")
    space_hash = space_sha256(env.observation_space, env.action_space)
    success_eps: List[List[Dict[str, Any]]] = []
    sidecars: List[Dict[str, Any]] = []
    try:
        for ep_i in range(int(args.successes_needed)):
            started = _utc_now()
            obs, info = env.reset(seed=int(args.seed) + ep_i)
            reset_ok = bool(info.get("reset_ok", True))
            rows: List[Dict[str, Any]] = []
            for step_i in range(int(args.fake_steps)):
                action = _zero7()
                action[0] = 0.2
                action[2] = 0.15
                if step_i == 10:
                    action[6] = 1.0
                elif step_i == 24:
                    action[6] = -1.0
                next_obs, reward, _terminated, truncated, step_info = env.step(action)
                step_info = dict(step_info)
                if "intervene_action" not in step_info:
                    step_info["intervene_action"] = np.asarray(action, dtype=np.float32)
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
            counts = validate_r13_grasp_edges(rows)
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
                    intervention_count=1,
                    grasp_counts=counts,
                    reset_ok=reset_ok,
                    human_success=True,
                    discard_reason=None,
                )
            )
    finally:
        env.close()
    write_success_bundle(
        bundle_dir,
        bundle_name=str(args.out_name),
        episodes=success_eps,
        sidecars=sidecars,
    )
    print("R13_RECORD_FAKE: PASS", flush=True)
    print(f"SUCCESS_EPISODES={len(success_eps)}", flush=True)
    print(f"ACTION_DIM={GRASP_DIM}", flush=True)
    print(f"BUNDLE={bundle_dir}", flush=True)
    _append_log(run_dir, f"{_utc_now()} fake successes={len(success_eps)} bundle={bundle_dir.name}")


def _record_live(args: argparse.Namespace, task, run_dir: Path, bundle_dir: Path) -> None:
    import time

    if str(args.confirm_live) != "YES":
        raise SystemExit("live mode requires --confirm-live YES")
    if int(args.classifier_consecutive_n) != int(task.classifier_consecutive_n):
        raise SystemExit(
            "R13_RECORD: FAIL — classifier consecutive count must match task config"
        )
    if os.environ.get("R4_CONFIRM") != "YES" or os.environ.get("R5_CONFIRM") != "YES":
        raise SystemExit("live mode requires R4_CONFIRM=YES and R5_CONFIRM=YES")
    ckpt = str(args.classifier_checkpoint or "").strip() or os.environ.get(
        "WA2_CLASSIFIER_CKPT", ""
    )
    if not ckpt:
        raise SystemExit(
            "live mode requires --classifier-checkpoint or WA2_CLASSIFIER_CKPT "
            "(upstream-aligned classifier demos)"
        )
    assert_no_teleop()
    env = make_wa2_environment(
        task,
        fake_env=False,
        classifier=True,
        grasp_action=True,
        enable_intervention=True,
        classifier_checkpoint=ckpt,
        classifier_threshold=args.classifier_threshold,
        classifier_consecutive_n=int(args.classifier_consecutive_n),
        end_episode=True,
    )
    if tuple(env.action_space.shape) != (GRASP_DIM,):
        raise SystemExit(f"R13_RECORD: FAIL — live action {env.action_space.shape} is not 7D")
    base = unwrap_env(env)
    space_hash = space_sha256(env.observation_space, env.action_space)
    success_eps: List[List[Dict[str, Any]]] = []
    sidecars: List[Dict[str, Any]] = []
    failed_dir = run_dir / "failed"
    episode_index = 0
    zero = _zero7()
    print(
        f"R13 7D record + classifier  max_steps={base.max_steps}  ckpt={ckpt}\n"
        "LEFT=enter/exit SpaceMouse; RIGHT=grasp/release (actions[6]).\n"
        "Classifier succeed → reward=1 + pause for s/f/a (you still decide keep/redo).\n"
        "Or TAP left to end early, then s/f/a. write_success_bundle sets last reward=1.",
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
                "TAP left to enter SpaceMouse. TAP right to grasp, right again to release.\n"
                "Place bottle → wait classifier SUCCEED, or TAP left to end.\n"
                "Then s (keep) / f (fail) / a (redo).",
                flush=True,
            )
            rows: List[Dict[str, Any]] = []
            discard_reason = None
            label = None
            human_success = False
            last_intervention_count = 0
            last_t = time.monotonic()
            step_info: Dict[str, Any] = dict(info)
            classifier_fired = False
            pipe = TransitionPipeline()
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
                if step_info.get("sm_session_enter"):
                    print("session ON — move SpaceMouse; tap LEFT to finish", flush=True)
                if step_info.get("grasp_command"):
                    print(f"grasp_command={step_info.get('grasp_command')}", flush=True)
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

                # Capture loop locals for the worker (transition overlaps next step).
                obs_i, act_i, nxt_i = obs, zero, next_obs
                rew_i = reward
                term_i, trunc_i = bool(terminated), bool(truncated)
                info_i = step_info
                obs_space, act_space = env.observation_space, env.action_space

                def _build(
                    o=obs_i,
                    a=act_i,
                    n=nxt_i,
                    r=rew_i,
                    t=term_i,
                    tr=trunc_i,
                    inf=info_i,
                ):
                    return build_actor_transition(
                        o,
                        a,
                        n,
                        r,
                        t,
                        tr,
                        inf,
                        observation_space=obs_space,
                        action_space=act_space,
                    )

                prev = pipe.push(_build)
                if prev is not None:
                    tr, meta = prev
                    rows.append(tr)
                else:
                    meta = {
                        "truncated": bool(truncated),
                        "episode_end": bool(terminated or truncated),
                    }
                obs = next_obs
                typed = _poll_label()
                if typed is not None:
                    label = typed
                    break
                if step_info.get("sm_session_exit"):
                    print("session OFF", flush=True)
                    label = _wait_label_keepalive(
                        env, zero, "keep this episode? s keep / f fail / a redo:"
                    )
                    break
                if terminated:
                    classifier_fired = bool(step_info.get("succeed"))
                    p_end = float(step_info.get("classifier_p") or 0.0)
                    streak = int(step_info.get("classifier_streak") or 0)
                    print(
                        f"CLASSIFIER_END succeed={classifier_fired} "
                        f"p={p_end:.3f} streak={streak} reward={float(reward):.1f}",
                        flush=True,
                    )
                    label = _wait_label_keepalive(
                        env,
                        zero,
                        "classifier ended episode — s keep / f fail / a redo:",
                    )
                    break
                if truncated or step_info.get("servo_faulted"):
                    discard_reason = _truncate_reason(step_info)
                    print(f"auto-end {discard_reason}", flush=True)
                    if discard_reason == "truncated:max_steps":
                        label = _wait_label_keepalive(
                            env, zero, "time cap — if placed, type s; else f / a:"
                        )
                    else:
                        label = "f"
                    break
            last = pipe.flush()
            if last is not None:
                rows.append(last[0])
            pipe.close()
            if label is None:
                label = _wait_label_keepalive(env, zero, "end of motion — s keep / f fail / a redo:")
            counts = grasp_action_counts(rows)
            if label == "s":
                intervened = count_intervened_steps(rows)
                try:
                    if (not reset_ok) or intervened < MIN_INTERVENED_STEPS:
                        raise ValueError(
                            f"intervened={intervened} < {MIN_INTERVENED_STEPS} or reset_ok={reset_ok}"
                        )
                    validate_r13_grasp_edges(rows)
                    human_success = True
                    discard_reason = None
                except Exception as exc:
                    label = "f"
                    discard_reason = f"success_rejected {exc}"
                    print(discard_reason, flush=True)
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
                grasp_counts=counts,
                reset_ok=reset_ok,
                human_success=human_success,
                discard_reason=discard_reason,
            )
            side["classifier_fired"] = bool(classifier_fired)
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
                f"grasp=+{counts['plus']}/-{counts['minus']} "
                f"clf_fired={classifier_fired} reason={discard_reason}",
                flush=True,
            )
            _append_log(
                run_dir,
                f"{_utc_now()} ep={episode_index} label={side['label']} "
                f"steps={side['n_steps']} grasp=+{counts['plus']}/-{counts['minus']} "
                f"clf_fired={classifier_fired} reason={discard_reason}",
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
    print("R13_RECORD_LIVE: PASS", flush=True)
    print(f"SUCCESS_EPISODES={len(success_eps)}", flush=True)
    print(f"FAILED_EPISODES={max(0, episode_index - len(success_eps))}", flush=True)
    print(f"ACTION_DIM={GRASP_DIM}", flush=True)
    print("GRASP_EDGES_OK=true", flush=True)
    print("TERMINAL_REWARD=1", flush=True)
    print(f"BUNDLE={bundle_dir}", flush=True)


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
