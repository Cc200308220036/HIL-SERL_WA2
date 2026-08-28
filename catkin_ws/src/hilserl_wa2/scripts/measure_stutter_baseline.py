#!/usr/bin/env python3
"""Phase-0 stutter baseline: measure inter-step gap between 5-tick Servo windows.

Does NOT change control logic. Records per high-level step:
  execution_duration_s, inter_step_gap_s, ticks, sm_session, interrupted_by.

Modes:
  auto  — synthetic low-scale motion (no SpaceMouse); quick stack gap check
  human — SpaceMouse session; matches demo/HIL feel (recommended for Phase 0)

Robot may already be at reset pose: default --skip-reset-motion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src") if (ROOT / "src").is_dir() else str(ROOT))


def _pct(xs: List[float], p: float) -> float:
    ys = sorted(float(x) for x in xs)
    if not ys:
        return float("nan")
    k = (len(ys) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    return float(ys[f] * (c - k) + ys[c] * (k - f))


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _summarize(rows: List[Dict[str, Any]], *, label: str) -> Dict[str, Any]:
    if not rows:
        return {"label": label, "n": 0}

    def _col(key: str) -> List[float]:
        out = []
        for r in rows:
            v = r.get(key)
            if v is None:
                continue
            out.append(float(v))
        return out

    gaps = _col("inter_step_gap_s")
    execs = _col("execution_duration_s")
    walls = _col("wall_s")
    post = _col("post_exec_s")
    trans = _col("transition_s")
    dead = _col("arm_idle_s")
    ticks_ok = all(
        int(r["servo_ticks_executed"]) == int(r["servo_ticks_requested"]) == 5 for r in rows
    )

    def _block(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
        return {
            "mean": statistics.mean(xs),
            "p50": _pct(xs, 50),
            "p95": _pct(xs, 95),
            "max": max(xs),
        }

    return {
        "label": label,
        "n": len(rows),
        "inter_step_gap_s": _block(gaps),
        "execution_duration_s": _block(execs),
        "post_exec_s": _block(post),
        "transition_s": _block(trans),
        "arm_idle_s": _block(dead),
        "wall_s": {
            **_block(walls),
            "implied_hz": (1.0 / statistics.mean(walls)) if walls else float("nan"),
        },
        "ticks_all_5_of_5": ticks_ok,
        "phase0_gap_baseline_ok": bool(gaps) and statistics.mean(gaps) > 0.005,
        "phase0_idle_baseline_ok": bool(dead) and statistics.mean(dead) > 0.010,
    }


def _resolve_classifier_sidecars(ckpt: str) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """Find threshold.json near ckpt (incl. run-root sibling used by R12 packs)."""

    from hilserl_wa2.wrappers.reward_classifier import load_threshold_json

    ckpt_path = Path(os.path.expanduser(str(ckpt))).resolve()
    candidates = []
    if ckpt_path.is_dir():
        candidates.append(ckpt_path / "threshold.json")
    candidates.extend(
        [
            ckpt_path.parent / "threshold.json",
            ckpt_path.parent.parent / "threshold.json",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            payload = load_threshold_json(candidate)
            thr = float(payload["threshold"])
            consec = (
                int(payload["consecutive_n"])
                if "consecutive_n" in payload
                else None
            )
            return thr, consec, str(candidate)
    return None, None, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-0 stutter baseline (inter_step_gap)")
    p.add_argument("--mode", choices=("auto", "human"), default="human")
    p.add_argument("--task", default="bottle_pick")
    p.add_argument("--duration-s", type=float, default=45.0, help="measure window length")
    p.add_argument("--warmup-s", type=float, default=2.0)
    p.add_argument("--scale", type=float, default=0.30, help="auto-mode action scale")
    p.add_argument("--skip-reset-motion", action="store_true", default=True)
    p.add_argument("--do-reset-motion", action="store_true", help="force R5 reset motion")
    p.add_argument(
        "--classifier",
        action="store_true",
        help="human mode: enable reward classifier (closer to record_r13_demos stack)",
    )
    p.add_argument(
        "--stack",
        choices=("light", "record"),
        default="light",
        help=(
            "light=no classifier. record=classifier + build_actor_transition "
            "(matches demo stutter path)."
        ),
    )
    p.add_argument(
        "--build-transition",
        action="store_true",
        help="after each step, build_actor_transition like record_r13_demos",
    )
    p.add_argument("--classifier-checkpoint", default="")
    p.add_argument(
        "--classifier-threshold",
        type=float,
        default=None,
        help="override threshold.json",
    )
    p.add_argument(
        "--classifier-consecutive-n",
        type=int,
        default=None,
        help="default: from threshold.json, else 1 for 10 Hz record stack",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="default: /root/catkin_ws/runs/stutter_baseline_<timestamp>",
    )
    p.add_argument("--ready-timeout-s", type=float, default=8.0)
    return p.parse_args()


def _zero_action(dim: int) -> np.ndarray:
    return np.zeros((dim,), dtype=np.float32)


def _auto_action(step_i: int, scale: float, dim: int) -> np.ndarray:
    sign = scale if (step_i // 5) % 2 == 0 else -scale
    a = np.zeros((dim,), dtype=np.float32)
    a[0] = np.float32(sign)
    return a


def main() -> None:
    args = parse_args()
    if args.stack == "record":
        args.classifier = True
        args.build_transition = True
    skip_reset = bool(args.skip_reset_motion) and not bool(args.do_reset_motion)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(
        args.out_dir.strip()
        or f"/root/catkin_ws/runs/stutter_baseline_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "steps.jsonl"
    summary_path = out_dir / "summary.json"

    meta = {
        "phase": 0,
        "mode": args.mode,
        "task": args.task,
        "duration_s": float(args.duration_s),
        "scale": float(args.scale),
        "skip_reset_motion": skip_reset,
        "classifier": bool(args.classifier),
        "stack": str(args.stack),
        "build_transition": bool(args.build_transition),
        "git": _git_rev(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
    }
    print(f"STUTTER_BASELINE_PHASE0 mode={args.mode} out={out_dir}", flush=True)
    print(json.dumps(meta, sort_keys=True), flush=True)

    env: Any = None
    rows: List[Dict[str, Any]] = []

    try:
        if args.mode == "auto":
            from hilserl_wa2.envs.wa2_env import WA2Env

            if os.environ.get("R4_CONFIRM") != "YES":
                raise SystemExit("auto live motion requires R4_CONFIRM=YES")
            env = WA2Env(
                fake_env=False,
                read_only=False,
                dry_run=False,
                scene_name="bottle_desktop",
                auto_reset_motion=False,
                episode_trans_limit_m=0.20,
                episode_rot_limit_deg=25.0,
            )
            act_dim = int(env.action_space.shape[0])
        else:
            from hilserl_wa2.experiments.actor_safety import assert_no_teleop
            from hilserl_wa2.experiments.env_factory import make_wa2_environment
            from hilserl_wa2.experiments.task_config import load_task

            if os.environ.get("R4_CONFIRM") != "YES":
                raise SystemExit("human live mode requires R4_CONFIRM=YES")
            assert_no_teleop()
            task = load_task(args.task)
            ckpt = (
                str(args.classifier_checkpoint or "").strip()
                or os.environ.get("WA2_CLASSIFIER_CKPT", "")
            )
            if args.classifier and not ckpt:
                raise SystemExit(
                    "--classifier needs --classifier-checkpoint or WA2_CLASSIFIER_CKPT"
                )
            thr = args.classifier_threshold
            consec = args.classifier_consecutive_n
            thr_path = None
            if args.classifier:
                found_thr, found_consec, thr_path = _resolve_classifier_sidecars(ckpt)
                if thr is None:
                    thr = found_thr
                if consec is None:
                    # 10 Hz record stack uses n=1 in record_r13_demos default.
                    consec = 1 if args.stack == "record" else found_consec
                if thr is None:
                    raise SystemExit(
                        "no threshold.json near ckpt; pass --classifier-threshold 0.85\n"
                        f"searched beside: {ckpt}"
                    )
                print(
                    f"CLASSIFIER ckpt={ckpt} thr={thr} consecutive_n={consec} "
                    f"threshold_json={thr_path}",
                    flush=True,
                )
            env = make_wa2_environment(
                task,
                fake_env=False,
                classifier=bool(args.classifier),
                grasp_action=True,
                enable_intervention=True,
                classifier_checkpoint=ckpt or None,
                classifier_threshold=thr,
                classifier_consecutive_n=consec,
                end_episode=False,
            )
            act_dim = int(env.action_space.shape[0])
            print(
                "HUMAN: start Joy, TAP left to enter SpaceMouse, push continuously.\n"
                f"Measuring ~{args.duration_s:.0f}s after session is active "
                f"(warmup {args.warmup_s:.0f}s ignored).",
                flush=True,
            )

        reset_opts = {
            "skip_reset_motion": skip_reset,
            "ready_timeout_s": float(args.ready_timeout_s),
        }
        obs, info = env.reset(options=reset_opts)
        print(
            f"RESET skip_motion={skip_reset} reset_ok={info.get('reset_ok')} "
            f"action_dim={act_dim}",
            flush=True,
        )

        # Optional contract print
        base = env
        seen = set()
        while base is not None and id(base) not in seen:
            seen.add(id(base))
            if type(base).__name__ == "WA2Env":
                break
            base = getattr(base, "env", None) or getattr(base, "unwrapped", None)
        if type(base).__name__ == "WA2Env":
            c = base.contract
            print(
                f"TIMEBASE policy_hz={c.policy_hz:g} servo_hz={c.control_hz:g} "
                f"ticks={c.servo_ticks_per_action}",
                flush=True,
            )

        t_loop_end: Optional[float] = None
        measure_t0: Optional[float] = None
        session_seen = args.mode == "auto"
        step_i = 0
        deadline = time.monotonic() + float(args.duration_s) + float(args.warmup_s) + 120.0
        obs = obs
        build_tr = bool(args.build_transition)
        tr_pipe = None
        if build_tr:
            from hilserl_wa2.experiments.async_transition import TransitionPipeline
            from hilserl_wa2.experiments.transition import build_actor_transition

            tr_pipe = TransitionPipeline()
            print(
                "TRANSITION_BUILD=on (pipelined; wait=prev overlaps next step)",
                flush=True,
            )

        with jsonl_path.open("w", encoding="utf-8") as handle:
            while time.monotonic() < deadline:
                if args.mode == "auto":
                    action = _auto_action(step_i, float(args.scale), act_dim)
                else:
                    action = _zero_action(act_dim)

                t_before = time.monotonic()
                gap = (
                    None
                    if t_loop_end is None
                    else float(t_before - t_loop_end)
                )
                next_obs, reward, terminated, truncated, step_info = env.step(action)
                t_step_done = time.monotonic()
                step_info = dict(step_info or {})
                exec_s = float(step_info.get("execution_duration_s") or 0.0)
                step_wall = float(t_step_done - t_before)
                post_exec = max(0.0, step_wall - exec_s)

                transition_s = 0.0
                if tr_pipe is not None:
                    obs_i, act_i, nxt_i = obs, action, next_obs
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

                    t_tr0 = time.monotonic()
                    _prev = tr_pipe.push(_build)
                    transition_s = float(time.monotonic() - t_tr0)
                obs = next_obs
                t_after = time.monotonic()
                t_loop_end = t_after
                wall = float(t_after - t_before)
                # Arm idle after servo window: in-step post_exec + blocked transition wait + gap.
                arm_idle = float(post_exec + transition_s + (gap or 0.0))

                sm = bool(step_info.get("sm_session"))
                if args.mode == "human" and sm:
                    session_seen = True
                    if measure_t0 is None:
                        measure_t0 = t_after
                        print("SM_SESSION active — baseline clock started", flush=True)

                row = {
                    "i": step_i,
                    "wall_s": wall,
                    "inter_step_gap_s": gap,
                    "execution_duration_s": exec_s,
                    "post_exec_s": post_exec,
                    "transition_s": transition_s,
                    "arm_idle_s": arm_idle,
                    "servo_ticks_requested": int(
                        step_info.get("servo_ticks_requested") or -1
                    ),
                    "servo_ticks_executed": int(
                        step_info.get("servo_ticks_executed") or -1
                    ),
                    "interrupted_by": str(step_info.get("interrupted_by") or "none"),
                    "sm_session": sm,
                    "intervened": bool(step_info.get("intervened")),
                    "delta_pos_m": float(step_info.get("delta_pos_m") or 0.0),
                    "servo_faulted": bool(step_info.get("servo_faulted")),
                    "classifier_infer_mode": str(
                        step_info.get("classifier_infer_mode") or ""
                    ),
                    "unix": time.time(),
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                rows.append(row)

                if step_i % 10 == 0:
                    gap_ms = float("nan") if gap is None else gap * 1000.0
                    print(
                        f"i={step_i:04d} ticks={row['servo_ticks_executed']}/"
                        f"{row['servo_ticks_requested']} "
                        f"exec={exec_s*1000:.0f}ms post={post_exec*1000:.0f}ms "
                        f"tr_wait={transition_s*1000:.0f}ms gap={gap_ms:.0f}ms "
                        f"idle={arm_idle*1000:.0f}ms wall={wall*1000:.0f}ms "
                        f"sm={int(sm)} mode={row['classifier_infer_mode'] or '-'}",
                        flush=True,
                    )

                step_i += 1
                if row["servo_faulted"] or terminated or truncated:
                    print(f"STOP fault/term/trunc row={row}", flush=True)
                    break

                if args.mode == "auto":
                    if measure_t0 is None:
                        measure_t0 = t_after
                    if (t_after - measure_t0) >= float(args.duration_s):
                        break
                else:
                    if measure_t0 is not None and (t_after - measure_t0) >= float(
                        args.duration_s
                    ):
                        break
                    if not session_seen and step_i > 0 and step_i % 20 == 0:
                        print(
                            "WAITING sm_session — TAP left on SpaceMouse to start measure",
                            flush=True,
                        )
                    if not session_seen and step_i > 500:
                        raise SystemExit(
                            "no sm_session after many steps — TAP left on SpaceMouse"
                        )

        if tr_pipe is not None:
            tr_pipe.close()

        # Summaries
        all_with_gap = [r for r in rows if r.get("inter_step_gap_s") is not None]
        human_rows = [r for r in all_with_gap if bool(r.get("sm_session"))]
        # Drop first gap after session start (warmup)
        warmup_n = max(1, int(float(args.warmup_s) * 10))
        human_meas = human_rows[warmup_n:] if len(human_rows) > warmup_n else human_rows
        auto_meas = all_with_gap[warmup_n:] if len(all_with_gap) > warmup_n else all_with_gap

        primary = human_meas if args.mode == "human" else auto_meas
        summary = {
            "meta": meta,
            "out_dir": str(out_dir),
            "n_rows_total": len(rows),
            "primary": _summarize(primary, label=("human_session" if args.mode == "human" else "auto")),
            "all_steps_with_gap": _summarize(all_with_gap, label="all_with_gap"),
            "phase0_exit": {
                "report_archived": True,
                "gap_mean_gt_5ms": bool(
                    primary
                    and statistics.mean(
                        float(r["inter_step_gap_s"]) for r in primary
                    )
                    > 0.005
                ),
                "idle_mean_gt_10ms": bool(
                    primary
                    and statistics.mean(float(r["arm_idle_s"]) for r in primary)
                    > 0.010
                ),
                "post_exec_mean_gt_10ms": bool(
                    primary
                    and statistics.mean(float(r["post_exec_s"]) for r in primary)
                    > 0.010
                ),
                "light_stack_smooth": bool(
                    primary
                    and statistics.mean(float(r["arm_idle_s"]) for r in primary)
                    < 0.010
                    and not bool(args.classifier)
                    and not bool(args.build_transition)
                ),
                "note": (
                    "inter_step_gap only measures between loop iterations. "
                    "Classifier cost often sits in post_exec_s (inside env.step after "
                    "servo window). Demo transition cost is transition_s. "
                    "Use arm_idle_s = post_exec + transition + gap as stutter proxy."
                ),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary["primary"], indent=2), flush=True)
        print(f"WROTE {jsonl_path}", flush=True)
        print(f"WROTE {summary_path}", flush=True)
        pe = summary["phase0_exit"]
        if pe["idle_mean_gt_10ms"] or pe["gap_mean_gt_5ms"]:
            print(
                "PHASE0_BASELINE: PASS — measurable arm idle "
                "(post_exec / transition / gap)",
                flush=True,
            )
        elif pe["light_stack_smooth"]:
            print(
                "PHASE0_BASELINE: LIGHT_STACK_SMOOTH — "
                "re-run with --stack record",
                flush=True,
            )
        else:
            print(
                "PHASE0_BASELINE: CHECK — look at post_exec_s / transition_s / arm_idle_s",
                flush=True,
            )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
