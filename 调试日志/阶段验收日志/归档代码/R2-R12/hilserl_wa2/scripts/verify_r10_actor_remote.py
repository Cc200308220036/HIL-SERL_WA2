#!/usr/bin/env python3
"""R10 remote Actor Gate: handshake then fake / readonly / live-zero against Learner.

Orin-only entry. Does not start r10_learner_server.py. Refuses 127.0.0.1.
Does not send Orin params pickle paths. Reuses R9 Env / transition / fail-closed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    find_wrapper,
    params_tree_signature,
)
from hilserl_wa2.experiments.env_factory import make_wa2_environment  # noqa: E402
from hilserl_wa2.experiments.r10_protocol import (  # noqa: E402
    AUTO_RECONNECT,
    R10ProtocolError,
    R10SessionGuard,
    assert_network_endpoints,
    build_handshake_request,
    confirm_r10_server_status,
    count_intervention_segments,
    estimate_wire_bytes,
    load_network_config,
    make_r10_trainer_config,
    make_session_id,
    ordered_transition_digest,
    rtt_stats_ms,
    transition_sha256,
)
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

FORWARD_JOY = [-0.9765625, 0.146484375, -0.04296875, -0.1015625, 0.087890625, 0.00390625]


def _resnet_encoder() -> str:
    pkl = Path.home() / ".serl" / "resnet10_params.pkl"
    if pkl.is_file():
        return "resnet-pretrained"
    raise SystemExit(
        "R10_ACTOR: FAIL — missing ~/.serl/resnet10_params.pkl (resnet-pretrained required)"
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
    rng, key = jax_mod.random.split(rng)
    actions = agent.sample_actions(
        observations=jax_mod.device_put(obs),
        seed=key,
        argmax=False,
    )
    return np.asarray(jax_mod.device_get(actions), dtype=np.float32), rng


def _load_json(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("R10_ACTOR: FAIL — manifest must be a JSON object")
    return data


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="R10 remote Actor Gate")
    p.add_argument("--task", default="bottle_pick")
    p.add_argument("--network-config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--server-ip", required=True, help="Learner LAN IPv4, never 127.0.0.1")
    p.add_argument("--mode", choices=("fake", "readonly", "live-zero"), default="fake")
    p.add_argument("--policy", choices=("zero", "sac"), default="zero")
    p.add_argument("--steps", type=int, default=1020)
    p.add_argument("--synthetic-intervention-start", type=int, default=1000)
    p.add_argument("--synthetic-intervention-steps", type=int, default=0)
    p.add_argument("--require-intervention-segments", type=int, default=0)
    p.add_argument("--require-network-update", action="store_true")
    p.add_argument("--require-intervention-steps", type=int, default=0)
    p.add_argument("--upload-every-steps", type=int, default=0)
    p.add_argument("--control-hz", type=float, default=0.0)
    p.add_argument("--min-seconds", type=float, default=0.0)
    p.add_argument("--max-seconds", type=float, default=300.0)
    p.add_argument("--max-total-translation-m", type=float, default=0.002)
    p.add_argument("--max-total-rotation-deg", type=float, default=0.2)
    p.add_argument("--skip-reset-motion", action="store_true", default=True)
    p.add_argument("--with-reset", action="store_true")
    p.add_argument("--output", default="")
    p.add_argument("--dump", default="")
    p.add_argument("--dump-on-fault", default="")
    p.add_argument("--expect-fault", default="")
    return p.parse_args(argv)


def _timed_update(client) -> Tuple[bool, float]:
    started = time.perf_counter()
    ok = bool(client.update())
    return ok, (time.perf_counter() - started) * 1000.0


def run_actor(args) -> Dict[str, Any]:
    cfg = load_network_config(args.network_config)
    try:
        server_ip = assert_network_endpoints(args.server_ip, cfg)
    except R10ProtocolError as exc:
        raise SystemExit(f"R10_ACTOR: FAIL — {exc}") from exc

    manifest = _load_json(args.manifest)
    if manifest.get("task_id") != args.task:
        raise SystemExit("R10_ACTOR: FAIL — task/manifest mismatch")

    assert_live_policy(args.mode, args.policy)
    if args.mode in ("readonly", "live-zero"):
        assert_no_teleop()
    assert_r4_confirm(args.mode)
    if args.mode == "live-zero" and args.policy != "zero":
        raise SystemExit("R10_ACTOR: FAIL — live-zero veto: policy must be zero")

    task = load_task(args.task)
    human = args.mode in ("readonly", "live-zero")
    control_hz = float(args.control_hz)
    if human and control_hz <= 0:
        control_hz = 50.0
    min_seconds = float(args.min_seconds)
    if human and min_seconds <= 0:
        min_seconds = 20.0
    upload_every = int(args.upload_every_steps) or int(cfg["upload_every_steps"])
    # ZMQ RCVTIMEO requires a Python int; YAML may load 3000 as int or float.
    timeout_ms = int(cfg["request_timeout_ms"])
    network_max_age_s = float(cfg["network_max_age_s"])

    session = R10SessionGuard(session_id=make_session_id())
    ctl = FailClosedController()
    upload_wd = UploadWatchdog(max_consecutive_failures=1)
    net_wd = NetworkWatchdog(max_age_s=network_max_age_s, enabled=False)
    budget = MotionBudget(
        max_translation_m=float(args.max_total_translation_m),
        max_rotation_deg=float(args.max_total_rotation_deg),
    )
    local_env = QueuedDataStore(50000)
    local_intvn = QueuedDataStore(50000)
    env_rows: List[dict] = []
    intvn_rows: List[dict] = []
    intervened_flags: List[bool] = []
    handshake_rtt_ms = None
    status_rtts: List[float] = []
    upload_rtts: List[float] = []
    last_upload: Dict[str, Any] = {}
    client = None
    agent = None
    jax_mod = None
    rng = None
    joy = None
    intvn_steps = 0
    intvn_count = 0
    readonly_ignored = True
    real_images = False
    state_finite = True
    reset_called = False
    t0 = time.monotonic()

    stop_flag = {"on": False, "reason": None}

    def _sigint(*_a):
        stop_flag["on"] = True
        stop_flag["reason"] = "sigint"

    signal.signal(signal.SIGINT, _sigint)

    def _digest_pair() -> Tuple[str, str]:
        return ordered_transition_digest(env_rows), ordered_transition_digest(intvn_rows)

    def _confirm() -> Tuple[bool, Dict[str, Any]]:
        env_digest, intvn_digest = _digest_pair()
        ok, report = confirm_r10_server_status(
            client,
            local_env=len(local_env),
            local_intvn=len(local_intvn),
            local_env_digest=env_digest,
            local_intvn_digest=intvn_digest,
            client_env_id=local_env.latest_data_id(),
            client_intvn_id=local_intvn.latest_data_id(),
        )
        if report.get("status_rtt_ms") is not None:
            status_rtts.append(float(report["status_rtt_ms"]))
        incoming = report.get("server_instance_id")
        if incoming:
            session.note_server_instance(str(incoming))
        return ok, report

    def _fault_network(detail: str) -> None:
        reason, normalized = session.trigger_network_loss(detail)
        ctl.trigger(reason)
        ctl.result.extra["fault_detail"] = normalized

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.mode == "fake":
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

    try:
        reset_opts: Dict[str, Any] = {"ready_timeout_s": 8.0, "camera_ready_timeout_s": 8.0}
        if args.with_reset:
            reset_opts["skip_reset_motion"] = False
            os.environ.setdefault("RESET_SCENE_OK", "YES")
            os.environ.setdefault("R5_CONFIRM", "YES")
        else:
            reset_opts["skip_reset_motion"] = True

        if human:
            target = find_wrapper(env, "WA2SpacemouseIntervention")
            if target is None:
                raise SystemExit("R10_ACTOR: FAIL — intervention wrapper missing")
            target.joy.start_ros()
            print("Waiting for /spacenav/joy ...", flush=True)
            target.joy.wait_ready(timeout_s=10.0)

        need_agent = args.policy == "sac" or bool(args.require_network_update)
        if need_agent:
            print("SAC_INIT_START", flush=True)
            agent, jax_mod = _make_agent(env, task)
            rng = jax_mod.random.PRNGKey(0)
            print("SAC_INIT_DONE", flush=True)

        trainer_cfg = make_r10_trainer_config(cfg["request_port"], cfg["broadcast_port"])
        client = TrainerClient(
            "actor_env",
            server_ip,
            trainer_cfg,
            data_stores={"actor_env": local_env, "actor_env_intvn": local_intvn},
            wait_for_server=False,
            timeout_ms=timeout_ms,
            log_level=logging.WARNING,
        )

        handshake = build_handshake_request(manifest, session.session_id)
        hs_started = time.perf_counter()
        hs_res = client.request("r10-handshake", handshake)
        handshake_rtt_ms = (time.perf_counter() - hs_started) * 1000.0
        hs_payload = {}
        accepted = False
        if isinstance(hs_res, dict):
            hs_payload = hs_res.get("payload") or {}
            accepted = bool(hs_res.get("success")) and bool(hs_payload.get("accepted", hs_res.get("success")))
        session.note_handshake(accepted, hs_payload.get("server_instance_id"))
        if not accepted:
            print("R10_HANDSHAKE: FAIL", flush=True)
            print("ENV_STEPS=0", flush=True)
            print("HANDSHAKE=FAIL", flush=True)
            mismatches = hs_payload.get("mismatches") or {"error": "handshake timeout/None"}
            print(json.dumps({"handshake_mismatches": mismatches}, default=str), flush=True)
            ctl.trigger("handshake_rejected")
            raise SystemExit("R10_ACTOR: FAIL — handshake rejected before env.reset/step")
        print("R10_HANDSHAKE: PASS", flush=True)
        print(f"SESSION_ID={session.session_id}", flush=True)
        print(f"SERVER_INSTANCE_ID={session.server_instance_id}", flush=True)
        print(f"handshake_rtt_ms={handshake_rtt_ms:.3f}", flush=True)

        expected_sig = manifest.get("params_tree_signature")

        def update_params(params):
            nonlocal agent
            try:
                sig = params_tree_signature(params)
                if expected_sig and sig != expected_sig:
                    ctl.trigger("params_signature_mismatch")
                    stop_flag["on"] = True
                    stop_flag["reason"] = "params_signature_mismatch"
                    return
                if agent is not None:
                    import jax.numpy as jnp

                    tree = jax_mod.tree_util.tree_map(lambda x: jnp.asarray(x), params)
                    agent = agent.replace(state=agent.state.replace(params=tree))
                net_wd.note_update(sig)
            except Exception as exc:
                ctl.trigger(f"callback_replace_failed:{exc}")
                stop_flag["on"] = True
                stop_flag["reason"] = "callback_replace_failed"

        client.recv_network_callback(update_params)
        if args.require_network_update:
            deadline = time.monotonic() + 30.0
            while net_wd.update_count < 1 and time.monotonic() < deadline and not ctl.result.triggered:
                time.sleep(0.05)
            if net_wd.update_count < 1:
                _fault_network("network_stale")
                raise SystemExit("R10_ACTOR: FAIL — no network callback after handshake")
            print(f"NETWORK_UPDATE_COUNT={net_wd.update_count}", flush=True)
            net_wd.enabled = True

        obs, reset_info = env.reset(seed=0, options=reset_opts)
        reset_called = True
        print(f"RESET_MODE={reset_info.get('reset_mode', reset_info.get('reset_ok'))}", flush=True)

        if args.policy == "sac":
            print("JIT_START", flush=True)
            _, rng = _sample_action(args.policy, agent, jax_mod, rng, env, obs)
            print("JIT_ACTION=PASS", flush=True)
            print("JIT_DONE", flush=True)
        else:
            print("JIT_DONE", flush=True)

        t0 = time.monotonic()
        step = 0
        last_status = t0
        start_intvn = int(args.synthetic_intervention_start)
        end_intvn = start_intvn + int(args.synthetic_intervention_steps)

        def should_stop() -> bool:
            if stop_flag["on"] or ctl.result.triggered:
                return True
            elapsed = time.monotonic() - t0
            # Disconnect Gates: keep running until fault or max-seconds.
            # Do NOT exit at the human min_seconds window, or network_loss never fires.
            if args.expect_fault:
                return elapsed >= float(args.max_seconds)
            if elapsed >= float(args.max_seconds) and human:
                return True
            if human:
                enough_time = elapsed >= min_seconds
                enough_intvn = intvn_steps >= int(args.require_intervention_steps)
                return bool(enough_time and enough_intvn)
            return step >= int(args.steps)

        while not should_stop():
            if not session.can_step():
                if session.fault_reason and not ctl.result.triggered:
                    ctl.trigger(session.fault_reason)
                break
            if joy is not None:
                if start_intvn <= step < end_intvn:
                    joy.clear_stale_injection()
                    joy.inject(FORWARD_JOY, buttons=[0, 1])
                else:
                    joy.inject(FORWARD_JOY, buttons=[0, 0])
                    if hasattr(env, "processor"):
                        env.processor.reset()

            tick = time.monotonic()
            stale = net_wd.check()
            if stale:
                _fault_network(stale)
                break

            session.note_env_step()
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
            env_rows.append(tr)
            session.register_transport("actor_env", len(env_rows) - 1, transition_sha256(tr))
            intervened_flags.append(bool(meta["intervened"]))
            if meta["intervened"]:
                intvn_rows.append(tr)
                session.register_transport(
                    "actor_env_intvn", len(intvn_rows) - 1, transition_sha256(tr)
                )
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

            do_upload = step > 0 and (
                step % upload_every == 0 or meta["episode_end"] or step >= int(args.steps)
            )
            if do_upload:
                _ok_update, upload_ms = _timed_update(client)
                upload_rtts.append(upload_ms)
                ok, report = _confirm()
                last_upload = report
                reason = upload_wd.record(ok)
                if reason:
                    _fault_network(reason)
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
            _ok_update, upload_ms = _timed_update(client)
            upload_rtts.append(upload_ms)
            ok, report = _confirm()
            last_upload = report
            reason = upload_wd.record(ok)
            ctl.result.extra["last_upload"] = report
            if reason:
                _fault_network(reason)

        if stop_flag["on"] and not ctl.result.triggered:
            ctl.trigger(stop_flag["reason"] or "sigint")

        if args.require_intervention_steps and intvn_steps < int(args.require_intervention_steps):
            if not ctl.result.triggered:
                ctl.trigger("insufficient_intervention")

        segments = count_intervention_segments(intervened_flags)
        if args.require_intervention_segments and segments != int(args.require_intervention_segments):
            if not ctl.result.triggered:
                ctl.trigger("intervention_segment_mismatch")

        if args.require_network_update and net_wd.update_count < 1:
            if not ctl.result.triggered:
                _fault_network("network_stale")

        if human and (time.monotonic() - t0) < min_seconds and not ctl.result.triggered:
            ctl.trigger("human_window_too_short")

    except ActorSafetyError as exc:
        ctl.trigger(str(exc))
        print(f"SAFETY: {exc}", flush=True)
        raise
    except SystemExit:
        raise
    except Exception as exc:
        if not ctl.result.triggered:
            ctl.trigger("actor_exception")
        ctl.result.extra["exception"] = f"{type(exc).__name__}: {exc}"
        print(f"ACTOR_EXCEPTION {type(exc).__name__}: {exc}", flush=True)
        if not args.expect_fault:
            raise
    finally:
        if ctl.result.triggered and args.dump_on_fault:
            session.write_fault_dump(
                args.dump_on_fault,
                env_rows,
                extra={"intvn_rows": len(intvn_rows), "mode": args.mode},
            )
        result = ctl.shutdown(env, client)

    elapsed = max(1e-6, time.monotonic() - t0)
    non_intvn = max(0, len(env_rows) - intvn_steps)
    wire_bytes = estimate_wire_bytes(env_rows)
    status_stats = rtt_stats_ms(status_rtts)
    upload_stats = rtt_stats_ms(upload_rtts)
    env_digest, intvn_digest = _digest_pair()
    summary = {
        "mode": args.mode,
        "policy": args.policy,
        "task_id": task.task_id,
        "exp_name": task.exp_name,
        "session_id": session.session_id,
        "server_instance_id": session.server_instance_id,
        "local_env_count": len(local_env),
        "local_intvn_count": len(local_intvn),
        "non_intervention_steps": non_intvn,
        "intervention_steps": intvn_steps,
        "intervention_segments": count_intervention_segments(intervened_flags),
        "upload_attempts": upload_wd.attempts,
        "network_update_count": net_wd.update_count,
        "network_age_s": net_wd.age_s,
        "params_signature": net_wd.last_signature,
        "server_env_count": last_upload.get("server_env_count"),
        "server_intvn_count": last_upload.get("server_intvn_count"),
        "last_update_id_match": last_upload.get("last_update_id_match"),
        "ordered_digest_match": last_upload.get("ordered_digest_match"),
        "ordered_digest": env_digest,
        "ordered_intvn_digest": intvn_digest,
        "handshake_rtt_ms": handshake_rtt_ms,
        "status_rtt_ms": status_stats,
        "upload_rtt_ms": upload_stats,
        "wire_bytes_estimate": wire_bytes,
        "transitions_per_second": len(env_rows) / elapsed,
        "effective_mib_per_second": (wire_bytes / elapsed) / (1024 * 1024),
        "fault_reason": result.reason,
        "fault_detail": session.fault_detail or ctl.result.extra.get("fault_detail"),
        "env_closed": result.env_closed,
        "client_stopped": result.client_stopped,
        "stop_ok": result.stop_ok,
        "clear_ok": result.clear_ok,
        "steps_executed": result.steps_executed,
        "env_steps": session.env_steps,
        "reset_called": reset_called,
        "auto_reconnect": AUTO_RECONNECT,
        "fault_dump_written": session.dump_written,
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
    if args.dump and env_rows:
        dump_path = Path(args.dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("wb") as handle:
            pickle.dump({"session_id": session.session_id, "rows": env_rows, "ledger": session.ledger}, handle)

    print(f"LOCAL_ENV_COUNT={len(local_env)}", flush=True)
    print(f"LOCAL_INTVN_COUNT={len(local_intvn)}", flush=True)
    print(f"NON_INTERVENTION_STEPS={non_intvn}", flush=True)
    print(f"INTERVENTION_STEPS={intvn_steps}", flush=True)
    print(f"INTERVENTION_SEGMENTS={count_intervention_segments(intervened_flags)}", flush=True)
    print(f"UPLOAD_ATTEMPTS={upload_wd.attempts}", flush=True)
    print(f"NETWORK_UPDATE_COUNT={net_wd.update_count}", flush=True)
    if net_wd.age_s is not None:
        print(f"network_age_s={net_wd.age_s:.3f}", flush=True)
        print(f"NETWORK_AGE_S={net_wd.age_s:.3f}", flush=True)
    if last_upload.get("server_env_count") is not None:
        print(f"SERVER_ENV_COUNT={last_upload['server_env_count']}", flush=True)
        print(f"SERVER_INTVN_COUNT={last_upload['server_intvn_count']}", flush=True)
        print(
            f"LAST_UPDATE_ID_MATCH={'PASS' if last_upload.get('last_update_id_match') else 'FAIL'}",
            flush=True,
        )
        digest_ok = last_upload.get("ordered_digest_match")
        if digest_ok is not None:
            print(f"ORDERED_DIGEST_MATCH={'PASS' if digest_ok else 'FAIL'}", flush=True)
    print(f"status_rtt_ms={status_stats}", flush=True)
    print(f"upload_rtt_ms={upload_stats}", flush=True)
    print(f"wire_bytes_estimate={wire_bytes}", flush=True)
    print(f"transitions_per_second={summary['transitions_per_second']:.4f}", flush=True)
    print(f"effective_mib_per_second={summary['effective_mib_per_second']:.4f}", flush=True)
    print(f"ENV_CLOSED={str(result.env_closed).lower()}", flush=True)
    print(f"CLIENT_STOPPED={str(result.client_stopped).lower()}", flush=True)
    print(f"AUTO_RECONNECT={str(AUTO_RECONNECT).lower()}", flush=True)
    if result.stop_ok is not None:
        print(f"STOP_OK={str(result.stop_ok).lower()}", flush=True)
        print(f"CLEAR_OK={str(result.clear_ok).lower()}", flush=True)
    if result.reason:
        print(f"FAULT_REASON={result.reason}", flush=True)
        detail = session.fault_detail or ctl.result.extra.get("fault_detail")
        if detail:
            print(f"FAULT_DETAIL={detail}", flush=True)
    if session.dump_written:
        print("FAULT_DUMP_WRITTEN=true", flush=True)
        print("NO_FURTHER_STEPS=true", flush=True)
    if args.mode == "readonly":
        print(f"ROBOT_MOTION={str(not readonly_ignored).lower()}", flush=True)
        print(f"REAL_IMAGES={str(real_images).lower()}", flush=True)
        print(f"STATE_FINITE={str(state_finite).lower()}", flush=True)
        print(f"action_ignored_for_motion={str(readonly_ignored).lower()}", flush=True)
    if args.mode == "live-zero":
        print(f"POLICY={args.policy}", flush=True)
        print(f"MAX_TRANSLATION_OK={budget.translation_m <= budget.max_translation_m}", flush=True)
        print(f"MAX_ROTATION_OK={budget.rotation_deg <= budget.max_rotation_deg}", flush=True)
        print(
            f"MOTION_WITHOUT_INTERVENTION={str(budget.motion_without_intervention).lower()}",
            flush=True,
        )
        print(f"translation_m={budget.translation_m:.6f}", flush=True)
        print(f"rotation_deg={budget.rotation_deg:.6f}", flush=True)

    expected = args.expect_fault
    failed = bool(result.triggered)
    if expected:
        if result.reason != expected:
            print(f"R10_ACTOR: FAIL — expected fault {expected!r} got {result.reason!r}", flush=True)
            raise SystemExit(1)
        print("NO_FURTHER_STEPS=true", flush=True)
        if expected == "network_loss":
            tag = "R10_LIVE_DISCONNECT" if args.mode == "live-zero" else "R10_SERVER_DISCONNECT"
            print(f"{tag}: PASS", flush=True)
        else:
            print("R10_ACTOR_LIFECYCLE: PASS", flush=True)
        return summary

    if failed:
        print(f"R10_ACTOR: FAIL — {result.reason}", flush=True)
        raise SystemExit(1)

    if args.mode == "fake":
        print("R10_ACTOR_FAKE: PASS", flush=True)
    elif args.mode == "readonly":
        print("R10_ACTOR_READONLY: PASS", flush=True)
    else:
        print("R10_ACTOR_LIVE_ZERO: PASS", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    code = 0
    try:
        run_actor(args)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            code = exc.code
        else:
            if exc.code:
                print(exc.code, flush=True)
            code = 1
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
