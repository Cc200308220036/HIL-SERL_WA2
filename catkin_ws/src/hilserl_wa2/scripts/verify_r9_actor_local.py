#!/usr/bin/env python3
"""R9 finite Actor: fake / readonly / live-zero → dummy TrainerServer. No train_rlpd.py."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

CATKIN_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CATKIN_SRC))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "examples"))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "serl_launcher"))

from agentlace.data.data_store import QueuedDataStore  # noqa: E402
from agentlace.trainer import TrainerClient  # noqa: E402

from hilserl_wa2.experiments.actor_safety import (  # noqa: E402
    ActorSafetyError,
    FailClosedController,
    MotionBudget,
    NetworkWatchdog,
    UploadWatchdog,
    assert_live_policy,
    assert_no_teleop,
    assert_r4_confirm,
    confirm_server_counts,
    find_wrapper,
    host_params_tree,
    make_r9_trainer_config,
    params_tree_signature,
    unwrap_env,
)
from hilserl_wa2.experiments.env_factory import make_wa2_environment  # noqa: E402
from hilserl_wa2.experiments.task_config import load_task  # noqa: E402
from hilserl_wa2.experiments.transition import (  # noqa: E402
    build_actor_transition,
    maybe_note_episode,
    route_transition,
)
from hilserl_wa2.interventions.joy_watchdog import JoyWatchdog  # noqa: E402
from hilserl_wa2.interventions.spacemouse_input import SpaceMouseInputConfig  # noqa: E402
from hilserl_wa2.interventions.wa2_spacemouse_intervention import (  # noqa: E402
    WA2SpacemouseIntervention,
)
from hilserl_wa2.tests.unit.test_spacemouse_input import SAMPLES  # noqa: E402


def _resnet_encoder() -> str:
    pkl = Path.home() / ".serl" / "resnet10_params.pkl"
    if pkl.is_file():
        return "resnet-pretrained"
    raise SystemExit(
        "R9_ACTOR: FAIL — missing ~/.serl/resnet10_params.pkl (resnet-pretrained required)"
    )


def _wrap_synthetic(env):
    joy = JoyWatchdog(max_age_s=0.2)
    wrapped = WA2SpacemouseIntervention(
        env,
        joy_watchdog=joy,
        auto_start_ros=False,
        input_config=SpaceMouseInputConfig(
            translation_filter_tau=0.0, rotation_filter_tau=0.0
        ),
        intervene_eps=1e-3,
    )
    return wrapped, joy


def _make_agent(env, task):
    import jax
    from serl_launcher.utils.launcher import make_sac_pixel_agent

    agent = make_sac_pixel_agent(
        seed=0,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=list(task.image_keys),
        encoder_type=_resnet_encoder(),
        discount=float(task.discount),
    )
    return agent, jax


def _sample_action(policy: str, agent, jax_mod, rng, env, obs) -> Tuple[np.ndarray, Any]:
    if policy == "zero":
        return np.zeros(env.action_space.shape, dtype=np.float32), rng
    if policy == "scripted":
        return np.asarray([0.2, 0, 0, 0, 0, 0], dtype=np.float32), rng
    rng, key = jax_mod.random.split(rng)
    actions = agent.sample_actions(
        observations=jax_mod.device_put(obs),
        seed=key,
        argmax=False,
    )
    return np.asarray(jax_mod.device_get(actions), dtype=np.float32), rng


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="R9 finite Actor Gate")
    p.add_argument("--task", default="bottle_pick")
    p.add_argument("--mode", choices=("fake", "readonly", "live-zero", "dry-run"), default="fake")
    p.add_argument("--policy", choices=("zero", "sac", "scripted"), default="zero")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--synthetic-intervention-steps", type=int, default=0)
    p.add_argument("--server-ip", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5588)
    p.add_argument("--broadcast-port", type=int, default=5589)
    p.add_argument("--upload-every-steps", type=int, default=10)
    p.add_argument("--require-network-update", action="store_true")
    p.add_argument("--network-max-age-s", type=float, default=5.0)
    p.add_argument("--control-hz", type=float, default=0.0)
    p.add_argument("--min-seconds", type=float, default=0.0)
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--require-intervention-steps", type=int, default=0)
    p.add_argument("--max-total-translation-m", type=float, default=0.020)
    p.add_argument("--max-total-rotation-deg", type=float, default=2.0)
    p.add_argument("--skip-reset-motion", action="store_true", default=True)
    p.add_argument("--with-reset", action="store_true")
    p.add_argument("--output", default="")
    p.add_argument("--params-pkl", default="")
    p.add_argument("--without-server", action="store_true")
    p.add_argument("--inject-exception-at", type=int, default=0)
    p.add_argument("--wait-for-sigint", action="store_true")
    p.add_argument("--expect-fault", default="")
    return p.parse_args(argv)


def run_actor(args) -> Dict[str, Any]:
    if args.server_ip not in ("127.0.0.1", "localhost"):
        raise SystemExit("R9_ACTOR: FAIL — Client IP must be 127.0.0.1")
    if int(args.upload_every_steps) > 50 or int(args.upload_every_steps) < 1:
        raise SystemExit("R9_ACTOR: FAIL — upload_every_steps must be 1..50")

    assert_live_policy(args.mode, args.policy)
    if args.mode in ("readonly", "live-zero"):
        assert_no_teleop()
    assert_r4_confirm(args.mode)
    if args.mode == "live-zero" and args.policy != "zero":
        raise SystemExit("R9_ACTOR: FAIL — live-zero veto: policy must be zero")

    task = load_task(args.task)
    human = args.mode in ("readonly", "live-zero")
    control_hz = float(args.control_hz)
    if human and control_hz <= 0:
        control_hz = 50.0
    min_seconds = float(args.min_seconds)
    if human and min_seconds <= 0:
        min_seconds = 20.0

    joy = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.mode in ("fake", "dry-run"):
            env = make_wa2_environment(task, fake_env=True, classifier=False)
            if int(args.synthetic_intervention_steps) > 0:
                env, joy = _wrap_synthetic(env)
        elif args.mode == "readonly":
            env = make_wa2_environment(
                task, fake_env=False, read_only=True, classifier=False
            )
        else:
            env = make_wa2_environment(
                task, fake_env=False, read_only=False, classifier=False
            )

    ctl = FailClosedController()
    upload_wd = UploadWatchdog(max_consecutive_failures=1)
    net_wd = NetworkWatchdog(
        max_age_s=float(args.network_max_age_s),
        enabled=bool(args.require_network_update) and not args.without_server,
    )
    budget = MotionBudget(
        max_translation_m=float(args.max_total_translation_m),
        max_rotation_deg=float(args.max_total_rotation_deg),
    )
    client = None
    agent = None
    jax_mod = None
    rng = None
    local_env = QueuedDataStore(50000)
    local_intvn = QueuedDataStore(50000)
    dumped_rows = []
    intvn_steps = 0
    intvn_count = 0
    readonly_ignored = True
    real_images = False
    state_finite = True
    network_ok = not bool(args.require_network_update)

    stop_flag = {"on": False, "reason": None}

    def _sigint(*_a):
        stop_flag["on"] = True
        stop_flag["reason"] = "sigint"

    signal.signal(signal.SIGINT, _sigint)
    if args.wait_for_sigint:
        signal.signal(signal.SIGTERM, _sigint)
        if control_hz <= 0:
            control_hz = 20.0

    try:
        intervention = find_wrapper(env, "WA2SpacemouseIntervention")
        reset_opts: Dict[str, Any] = {"ready_timeout_s": 8.0, "camera_ready_timeout_s": 8.0}
        if args.with_reset:
            reset_opts["skip_reset_motion"] = False
            os.environ.setdefault("RESET_SCENE_OK", "YES")
            os.environ.setdefault("R5_CONFIRM", "YES")
        else:
            reset_opts["skip_reset_motion"] = True

        if human:
            target = intervention
            if target is None:
                raise SystemExit("R9_ACTOR: FAIL — intervention wrapper missing")
            target.joy.start_ros()
            print("Waiting for /spacenav/joy ...", flush=True)
            target.joy.wait_ready(timeout_s=10.0)

        need_agent = args.policy == "sac" or (
            args.require_network_update and not args.without_server
        )
        if need_agent:
            print("SAC_INIT_START", flush=True)
            agent, jax_mod = _make_agent(env, task)
            rng = jax_mod.random.PRNGKey(0)
            print("SAC_INIT_DONE", flush=True)

        if not args.without_server:
            cfg = make_r9_trainer_config(args.port, args.broadcast_port)
            client = TrainerClient(
                "actor_env",
                args.server_ip,
                cfg,
                data_stores={"actor_env": local_env, "actor_env_intvn": local_intvn},
                wait_for_server=True,
                timeout_ms=3000,
                log_level=__import__("logging").WARNING,
            )

            def update_params(params):
                nonlocal agent, network_ok
                sig = params_tree_signature(params)
                net_wd.note_update(sig)
                if agent is not None:
                    try:
                        import jax.numpy as jnp

                        tree_fn = (
                            jax_mod.tree_util.tree_map
                            if jax_mod is not None
                            else (lambda f, x: x)
                        )
                        tree = tree_fn(lambda x: jnp.asarray(x), params)
                        agent = agent.replace(state=agent.state.replace(params=tree))
                        network_ok = True
                    except Exception as exc:
                        ctl.trigger(f"callback_replace_failed:{exc}")
                        stop_flag["on"] = True
                        stop_flag["reason"] = "callback_replace_failed"
                else:
                    network_ok = True

            client.recv_network_callback(update_params)

            if agent is not None:
                params_path = Path(args.params_pkl) if args.params_pkl else Path(
                    "/tmp/r9_actor_params.pkl"
                )
                params_path.parent.mkdir(parents=True, exist_ok=True)
                host = host_params_tree(agent.state.params)
                with params_path.open("wb") as handle:
                    pickle.dump(host, handle)
                print(f"PARAMS_PKL={params_path}", flush=True)
                print(f"PARAMS_SIGNATURE={params_tree_signature(host)}", flush=True)
                res = client.request("r9-publish-params", {"path": str(params_path)})
                if res is None or (isinstance(res, dict) and res.get("success") is False):
                    raise SystemExit(f"R9_ACTOR: FAIL — publish params failed: {res}")
                deadline = time.monotonic() + 30.0
                while net_wd.update_count < 1 and time.monotonic() < deadline:
                    time.sleep(0.05)
                if net_wd.update_count < 1:
                    raise SystemExit("R9_ACTOR: FAIL — no network callback")
                print(f"NETWORK_UPDATE_COUNT={net_wd.update_count}", flush=True)
                print("PARAM_TREE_COMPATIBLE=PASS", flush=True)

        obs, reset_info = env.reset(seed=0, options=reset_opts)
        print(f"RESET_MODE={reset_info.get('reset_mode', reset_info.get('reset_ok'))}", flush=True)

        if args.policy == "sac":
            print("JIT_START", flush=True)
            _, rng = _sample_action(args.policy, agent, jax_mod, rng, env, obs)
            print("JIT_DONE", flush=True)
        else:
            print("JIT_DONE", flush=True)

        if human:
            print(
                f"HUMAN_WINDOW min_seconds={min_seconds:.0f} "
                f"require_intervention_steps={args.require_intervention_steps} "
                f"HOLD deadman (buttons[1]) and MOVE SpaceMouse",
                flush=True,
            )

        t0 = time.monotonic()
        step = 0
        last_status = t0
        start_intvn = 10
        end_intvn = start_intvn + int(args.synthetic_intervention_steps)

        def should_stop() -> bool:
            if stop_flag["on"] or ctl.result.triggered:
                return True
            if args.wait_for_sigint:
                return False
            elapsed = time.monotonic() - t0
            if elapsed >= float(args.max_seconds) and human:
                return True
            if human:
                enough_time = elapsed >= min_seconds
                enough_intvn = intvn_steps >= int(args.require_intervention_steps)
                return bool(enough_time and enough_intvn)
            return step >= int(args.steps)

        while not should_stop():
            if args.inject_exception_at and step == int(args.inject_exception_at):
                raise RuntimeError("injected actor exception")

            if joy is not None:
                if start_intvn <= step < end_intvn:
                    joy.clear_stale_injection()
                    joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
                else:
                    joy.inject(SAMPLES["forward_translation"], buttons=[0, 0])
                    if hasattr(env, "processor"):
                        env.processor.reset()

            tick = time.monotonic()
            stale = net_wd.check()
            if stale:
                ctl.trigger(stale)
                break

            action, rng = _sample_action(args.policy, agent, jax_mod, rng, env, obs)
            nxt, reward, terminated, truncated, info = env.step(action)
            info = dict(info)
            tr, meta = build_actor_transition(
                obs,
                action,
                nxt,
                reward,
                terminated,
                truncated,
                info,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            route_transition(tr, meta, local_env, local_intvn)
            dumped_rows.append(tr)
            if meta["intervened"]:
                intvn_steps += 1
                if int(info.get("intervention_count") or 0) > intvn_count:
                    intvn_count = int(info["intervention_count"])
            maybe_note_episode(
                info, meta, intervention_count=intvn_count, intervention_steps=intvn_steps
            )

            if args.mode == "readonly":
                if not info.get("action_ignored_for_motion"):
                    readonly_ignored = False
                    ctl.trigger("readonly_motion_not_ignored")
                    break
            if "head" in nxt and "wrist" in nxt:
                real_images = True
            if not np.isfinite(np.asarray(nxt["state"])).all():
                state_finite = False
                ctl.trigger("nan_state")
                break

            fault_budget = budget.note(info, meta["intervened"])
            if args.mode == "live-zero" and fault_budget:
                ctl.trigger(fault_budget)
                break
            if info.get("stale") or info.get("servo_faulted"):
                ctl.trigger("env_stale_or_fault")
                break

            obs = nxt
            step += 1
            ctl.result.steps_executed = step

            if meta["episode_end"]:
                obs, _ = env.reset(seed=None, options=reset_opts)

            do_upload = (
                client is not None
                and step > 0
                and (step % int(args.upload_every_steps) == 0 or meta["episode_end"])
            )
            if do_upload:
                client.update()
                ok, report = confirm_server_counts(
                    client,
                    local_env=len(local_env),
                    local_intvn=len(local_intvn),
                    client_env_id=local_env.latest_data_id(),
                    client_intvn_id=local_intvn.latest_data_id(),
                )
                reason = upload_wd.record(ok)
                if reason:
                    ctl.trigger(reason)
                    ctl.result.extra["last_upload"] = report
                    break

            now = time.monotonic()
            if control_hz > 0:
                remain = (1.0 / control_hz) - (now - tick)
                if remain > 0:
                    time.sleep(remain)
            if now - last_status >= 1.0:
                age = net_wd.age_s
                print(
                    f"t={now - t0:5.1f}s step={step} env={len(local_env)} "
                    f"intvn={intvn_steps} uploads={upload_wd.attempts} "
                    f"net={net_wd.update_count} net_age={age}",
                    flush=True,
                )
                last_status = now

        if client is not None and not ctl.result.triggered:
            client.update()
            ok, report = confirm_server_counts(
                client,
                local_env=len(local_env),
                local_intvn=len(local_intvn),
                client_env_id=local_env.latest_data_id(),
                client_intvn_id=local_intvn.latest_data_id(),
            )
            reason = upload_wd.record(ok)
            ctl.result.extra["last_upload"] = report
            if reason:
                ctl.trigger(reason)

        if stop_flag["on"] and not ctl.result.triggered:
            ctl.trigger(stop_flag["reason"] or "sigint")

        if args.require_intervention_steps and intvn_steps < int(args.require_intervention_steps):
            if not ctl.result.triggered:
                ctl.trigger("insufficient_intervention")

        if args.require_network_update and net_wd.update_count < 1:
            if not ctl.result.triggered:
                ctl.trigger("no_network_update")

        if human and (time.monotonic() - t0) < min_seconds and not ctl.result.triggered:
            if not args.wait_for_sigint:
                ctl.trigger("human_window_too_short")

    except ActorSafetyError as exc:
        ctl.trigger(str(exc))
        print(f"SAFETY: {exc}", flush=True)
        raise
    except Exception as exc:
        if not ctl.result.triggered:
            ctl.trigger("actor_exception")
        ctl.result.extra["exception"] = f"{type(exc).__name__}: {exc}"
        print(f"ACTOR_EXCEPTION {type(exc).__name__}: {exc}", flush=True)
        if not args.expect_fault:
            raise
    finally:
        result = ctl.shutdown(env, client)

    last_upload = ctl.result.extra.get("last_upload") or {}
    summary = {
        "mode": args.mode,
        "policy": args.policy,
        "task_id": task.task_id,
        "exp_name": task.exp_name,
        "local_env_count": len(local_env),
        "local_intvn_count": len(local_intvn),
        "intervention_steps": intvn_steps,
        "intervention_count": intvn_count,
        "upload_attempts": upload_wd.attempts,
        "network_update_count": net_wd.update_count,
        "network_age_s": net_wd.age_s,
        "params_signature": net_wd.last_signature,
        "server_env_count": last_upload.get("server_env_count"),
        "server_intvn_count": last_upload.get("server_intvn_count"),
        "last_update_id_match": last_upload.get("last_update_id_match"),
        "fault_reason": result.reason,
        "env_closed": result.env_closed,
        "client_stopped": result.client_stopped,
        "stop_ok": result.stop_ok,
        "clear_ok": result.clear_ok,
        "steps_executed": result.steps_executed,
        "readonly_ignored": readonly_ignored,
        "real_images": real_images,
        "state_finite": state_finite,
        "translation_m": budget.translation_m,
        "rotation_deg": budget.rotation_deg,
        "motion_without_intervention": budget.motion_without_intervention,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(summary, indent=2, default=str, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"LOCAL_ENV_COUNT={len(local_env)}", flush=True)
    print(f"LOCAL_INTVN_COUNT={len(local_intvn)}", flush=True)
    print(f"INTERVENTION_STEPS={intvn_steps}", flush=True)
    print(f"UPLOAD_ATTEMPTS={upload_wd.attempts}", flush=True)
    print(f"NETWORK_UPDATE_COUNT={net_wd.update_count}", flush=True)
    if net_wd.age_s is not None:
        print(f"NETWORK_AGE_S={net_wd.age_s:.3f}", flush=True)
    if last_upload.get("server_env_count") is not None:
        print(f"SERVER_ENV_COUNT={last_upload['server_env_count']}", flush=True)
        print(f"SERVER_INTVN_COUNT={last_upload['server_intvn_count']}", flush=True)
        print(
            f"LAST_UPDATE_ID_MATCH={'PASS' if last_upload.get('last_update_id_match') else 'FAIL'}",
            flush=True,
        )
    print(f"ENV_CLOSED={str(result.env_closed).lower()}", flush=True)
    print(f"CLIENT_STOPPED={str(result.client_stopped).lower()}", flush=True)
    if result.stop_ok is not None:
        print(f"STOP_OK={str(result.stop_ok).lower()}", flush=True)
        print(f"CLEAR_OK={str(result.clear_ok).lower()}", flush=True)
    if result.reason:
        print(f"FAULT_REASON={result.reason}", flush=True)
    if args.mode == "readonly":
        print(f"ROBOT_MOTION={str(not readonly_ignored).lower()}", flush=True)
        print(f"REAL_IMAGES={str(real_images).lower()}", flush=True)
        print(f"STATE_FINITE={str(state_finite).lower()}", flush=True)
    if args.mode == "live-zero":
        print(f"POLICY={args.policy}", flush=True)
        print(f"MAX_TRANSLATION_OK={budget.translation_m <= budget.max_translation_m}", flush=True)
        print(f"MAX_ROTATION_OK={budget.rotation_deg <= budget.max_rotation_deg}", flush=True)
        print(
            f"MOTION_WITHOUT_INTERVENTION={str(budget.motion_without_intervention).lower()}",
            flush=True,
        )

    expected = args.expect_fault
    failed = bool(result.triggered)
    if expected:
        if result.reason != expected:
            print(f"R9_ACTOR: FAIL — expected fault {expected!r} got {result.reason!r}", flush=True)
            raise SystemExit(1)
        print("NO_FURTHER_STEPS=true", flush=True)
        if expected == "server_disconnect":
            print("R9_SERVER_DISCONNECT: PASS", flush=True)
        elif expected == "sigint":
            tag = "R9_LIVE_SIGINT" if args.mode == "live-zero" else "R9_ACTOR_LIFECYCLE"
            print(f"{tag}: PASS", flush=True)
        else:
            print("R9_ACTOR_LIFECYCLE: PASS", flush=True)
        return summary

    if failed:
        print(f"R9_ACTOR: FAIL — {result.reason}", flush=True)
        raise SystemExit(1)

    tag = {
        "fake": "R9_ACTOR_FAKE",
        "readonly": "R9_ACTOR_READONLY",
        "live-zero": "R9_ACTOR_LIVE_ZERO",
        "dry-run": "R9_ACTOR_FAKE",
    }[args.mode]
    if args.require_network_update:
        print("R9_NETWORK_CALLBACK: PASS", flush=True)
    print(f"{tag}: PASS", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    code = 0
    try:
        run_actor(args)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Agentlace ZMQ/Broadcast threads can block interpreter shutdown.
    os._exit(code)


if __name__ == "__main__":
    main()
