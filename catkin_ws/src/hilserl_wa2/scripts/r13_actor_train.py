#!/usr/bin/env python3
"""R13 Actor HIL loop. Orin-only. Handshake before reset/step. Fail-closed."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
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

from agentlace.trainer import TrainerClient  # noqa: E402

from hilserl_wa2.experiments.actor_safety import (  # noqa: E402
    ActorSafetyError,
    FailClosedController,
    NetworkWatchdog,
    UploadWatchdog,
    assert_no_teleop,
    assert_r13_hardware_confirm,
    find_wrapper,
    params_tree_signature,
    unwrap_env,
)
from hilserl_wa2.experiments.env_factory import make_wa2_environment  # noqa: E402
from hilserl_wa2.experiments.r10_protocol import (  # noqa: E402
    AUTO_RECONNECT,
    R10ProtocolError,
    R10SessionGuard,
    assert_network_endpoints,
    load_network_config,
    make_session_id,
)
from hilserl_wa2.experiments.r13_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    build_handshake_request,
    make_r13_trainer_config,
    scale_arm_action,
)
from hilserl_wa2.experiments.task_config import load_task  # noqa: E402
from hilserl_wa2.envs.contracts import WA2EnvContract  # noqa: E402
from hilserl_wa2.experiments.transition import (  # noqa: E402
    build_actor_transition,
    is_intervened,
    maybe_note_episode,
    route_transition,
)
from hilserl_wa2.experiments.async_transition import TransitionPipeline  # noqa: E402
from hilserl_wa2.interventions.actor_upload_queue import (  # noqa: E402
    DEFAULT_CAPACITY,
    DEFAULT_MAX_BATCH,
    DEFAULT_SOFT_WATERMARK,
    DrainingQueuedDataStore,
    align_stores_to_server,
    confirm_upload_by_last_id,
    upload_datastores,
)
from hilserl_wa2.ros_adapters.servo_session import DEFAULT_LATCH_MAX_AGE_S  # noqa: E402
from hilserl_wa2.wrappers.reward_classifier import (  # noqa: E402
    ClassifierHoldDumpGate,
    image_obs_stats,
    save_classifier_dump,
)

# If the Actor loop gaps longer than the Servo latch age, clear latch first
# (mechanism A: upload/GIL stall must not free-flight on the previous axis).
SERVO_STALL_HOLD_S = float(DEFAULT_LATCH_MAX_AGE_S)


def _resnet_encoder() -> str:
    pkl = Path.home() / ".serl" / "resnet10_params.pkl"
    if pkl.is_file():
        return "resnet-pretrained"
    raise SystemExit(
        "R13_ACTOR: FAIL — missing ~/.serl/resnet10_params.pkl (resnet-pretrained required)"
    )


def _load_json(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("R13_ACTOR: FAIL — manifest must be a JSON object")
    return data


def _make_hybrid_agent(env, task):
    import jax
    from serl_launcher.utils.launcher import make_sac_pixel_agent_hybrid_single_arm

    agent = make_sac_pixel_agent_hybrid_single_arm(
        seed=0,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=list(task.image_keys),
        encoder_type=_resnet_encoder(),
        discount=float(task.discount),
    )
    return agent, jax


def _find_classifier_success_pkl(ckpt: str) -> Optional[Path]:
    override = str(os.environ.get("WA2_CLASSIFIER_SANITY_BUNDLE") or "").strip()
    if override:
        bundle = Path(override).expanduser()
        hit = bundle / "success.pkl" if bundle.is_dir() else bundle
        return hit if hit.is_file() else None
    ckpt_path = Path(ckpt).expanduser().resolve()
    for parent in ckpt_path.parents:
        r12 = parent / "0819_1426_r12_clean" / "classifier"
        if r12.is_dir():
            hits = sorted(r12.glob("*/success.pkl"))
            if hits:
                return hits[0]
        hits = sorted(parent.glob("classifier/*/success.pkl"))
        if hits:
            return hits[0]
    return None


def _run_classifier_sanity(cls_wrap, ckpt: str, tag: str) -> None:
    success_pkl = _find_classifier_success_pkl(ckpt)
    if success_pkl is None:
        print(f"CLASSIFIER_SANITY {tag} SKIP no success.pkl", flush=True)
        return
    import pickle

    with success_pkl.open("rb") as handle:
        success_rows = pickle.load(handle)
    fail_pkl = success_pkl.with_name("failure.pkl")
    fail_rows = []
    if fail_pkl.is_file():
        with fail_pkl.open("rb") as handle:
            fail_rows = pickle.load(handle)
    p_s = float(cls_wrap.predict_fn(dict(success_rows[0]["observations"])))
    p_f = float("nan")
    if fail_rows:
        p_f = float(cls_wrap.predict_fn(dict(fail_rows[0]["observations"])))
    print(
        f"CLASSIFIER_SANITY {tag} success_p={p_s:.4f} failure_p={p_f:.4f} "
        f"pkl={success_pkl}",
        flush=True,
    )
    if p_s < 0.9:
        raise SystemExit(
            f"R13_ACTOR: FAIL — classifier sanity {tag} success_p={p_s:.4f} < 0.9 "
            "(weights not restored or inference path broken)"
        )
        if fail_rows and p_f > 0.1:
            raise SystemExit(
                f"R13_ACTOR: FAIL — classifier sanity {tag} failure_p={p_f:.4f} > 0.1"
            )


def _poll_dump_key() -> bool:
    if not sys.stdin.isatty():
        return False
    import select

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False
    token = sys.stdin.readline().strip().lower()
    return token in ("d", "dump")


def _classifier_dump_dir(args) -> Path:
    if args.output:
        out = Path(args.output).expanduser()
        if out.suffix.lower() == ".json":
            return out.parent / "classifier_live_dumps"
        return out / "classifier_live_dumps"
    return Path(args.manifest).expanduser().resolve().parent / "classifier_live_dumps"


def _sample_action(agent, jax_mod, rng, obs, *, grasp_eps: float = 0.0) -> Tuple[np.ndarray, Any]:
    rng, key = jax_mod.random.split(rng)
    actions = agent.sample_actions(
        observations=jax_mod.device_put(obs),
        seed=key,
        argmax=False,
        grasp_eps=float(grasp_eps),
    )
    return np.asarray(jax_mod.device_get(actions), dtype=np.float32), rng


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="R13 Actor HIL train")
    p.add_argument("--task", default="bottle_pick")
    p.add_argument("--network-config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--server-ip", required=True)
    p.add_argument("--mode", choices=("fake", "live"), default="fake")
    p.add_argument("--policy", choices=("sac",), default="sac")
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--episode-max-steps", type=int, default=600)
    p.add_argument("--synthetic-intervention-start", type=int, default=100)
    p.add_argument("--synthetic-intervention-steps", type=int, default=0)
    p.add_argument("--require-network-update", action="store_true")
    p.add_argument("--upload-every-steps", type=int, default=0)
    p.add_argument(
        "--request-timeout-ms",
        type=int,
        default=60000,
        help="ZMQ request timeout; YAML 3000ms is too short for 7D image batches",
    )
    p.add_argument(
        "--network-max-age-s",
        type=float,
        default=None,
        help=(
            "Seconds since last policy broadcast before fail-closed. "
            "Omit: YAML 5s in fake, 120s floor in live. 0 disables this "
            "watchdog (upload failure still fail-closes). Do not edit local.yaml."
        ),
    )
    p.add_argument(
        "--initial-policy-timeout-s",
        type=float,
        default=60.0,
        help=(
            "Maximum seconds to wait for the first Learner parameter broadcast "
            "after handshake; env.reset/step are forbidden before it arrives."
        ),
    )
    p.add_argument("--control-hz", type=float, default=0.0)
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--classifier-checkpoint", default="")
    p.add_argument(
        "--classifier-consecutive-n",
        type=int,
        default=1,
        help="10 Hz stage-three default: one positive high-level observation",
    )
    p.add_argument("--end-episode", action="store_true", default=True)
    p.add_argument("--debug", action="store_true", default=True)
    p.add_argument("--output", default="")
    p.add_argument("--skip-reset-motion", action="store_true")
    p.add_argument("--with-reset", action="store_true")
    p.add_argument(
        "--grasp-eps",
        type=float,
        default=0.0,
        help=(
            "Epsilon-greedy on discrete gripper during HIL (explore open/close only). "
            "Default 0: rely on Grasp Critic + human right-button; no random toggles. "
            "Set >0 only if you explicitly want forced gripper exploration."
        ),
    )
    return p.parse_args(argv)


def _resolve_network_max_age_s(yaml_s: float, *, live: bool, override: Optional[float]) -> Optional[float]:
    """Params-broadcast freshness. None disables this watchdog (upload fail still fail-closes).

    YAML 5.0s is R10 smoke. R13 live publishes every steps_per_update SAC updates,
    often >>5s, so live defaults to at least 120s. Do not edit local.yaml.
    """

    if override is not None:
        if float(override) <= 0:
            return None
        return float(override)
    age = float(yaml_s)
    if live:
        return max(age, 120.0)
    return age


def run_actor(args) -> Dict[str, Any]:
    cfg = load_network_config(args.network_config)
    try:
        server_ip = assert_network_endpoints(args.server_ip, cfg)
    except R10ProtocolError as exc:
        raise SystemExit(f"R13_ACTOR: FAIL — {exc}") from exc

    manifest = _load_json(args.manifest)
    if manifest.get("task_id") != args.task:
        raise SystemExit("R13_ACTOR: FAIL — task/manifest mismatch")
    if str(manifest.get("protocol_version")) != PROTOCOL_VERSION:
        raise SystemExit("R13_ACTOR: FAIL — legacy/incompatible protocol manifest")
    if int(manifest.get("action_dim") or 0) != 7:
        raise SystemExit("R13_ACTOR: FAIL — manifest action_dim must be 7")

    assert_no_teleop()
    assert_r13_hardware_confirm(args.mode)

    task = load_task(args.task)
    contract = WA2EnvContract.from_yaml(task.contract_path)
    expected_timebase = {
        "policy_hz": float(contract.policy_hz),
        "servo_hz": float(contract.control_hz),
        "servo_ticks_per_action": int(contract.servo_ticks_per_action),
        "discount": float(task.discount),
        "classifier_consecutive_n": int(task.classifier_consecutive_n),
    }
    for key, expected in expected_timebase.items():
        try:
            matched = abs(float(manifest.get(key)) - float(expected)) <= 1e-6
        except (TypeError, ValueError):
            matched = False
        if not matched:
            raise SystemExit(
                f"R13_ACTOR: FAIL — manifest {key} expected={expected} "
                f"got={manifest.get(key)}"
            )
    live = args.mode == "live"
    control_hz = float(args.control_hz)
    if live and control_hz <= 0:
        control_hz = float(contract.policy_hz)
    if live and abs(control_hz - float(contract.policy_hz)) > 1e-6:
        raise SystemExit(
            f"R13_ACTOR: FAIL — --control-hz must equal policy_hz="
            f"{contract.policy_hz}; Servo remains {contract.control_hz} Hz"
        )
    if int(args.classifier_consecutive_n) != int(task.classifier_consecutive_n):
        raise SystemExit(
            "R13_ACTOR: FAIL — --classifier-consecutive-n must match task/manifest "
            f"value {task.classifier_consecutive_n}"
        )
    upload_every = int(args.upload_every_steps) or int(cfg["upload_every_steps"])
    timeout_ms = int(args.request_timeout_ms) if int(args.request_timeout_ms) > 0 else int(cfg["request_timeout_ms"])
    network_max_age_s = _resolve_network_max_age_s(
        float(cfg["network_max_age_s"]),
        live=live,
        override=args.network_max_age_s,
    )
    params_watchdog = network_max_age_s is not None
    print(f"REQUEST_TIMEOUT_MS={timeout_ms} UPLOAD_EVERY={upload_every}", flush=True)
    print(
        f"NETWORK_MAX_AGE_S={'off' if not params_watchdog else network_max_age_s}",
        flush=True,
    )
    print(f"UPLOAD_ASYNC={str(live).lower()}", flush=True)
    print(
        "UPLOAD_DURING_SM_SESSION=true "
        "(periodic tick runs in intervention; watermark latch still applies)",
        flush=True,
    )
    print(
        f"UPLOAD_QUEUE capacity={DEFAULT_CAPACITY} max_batch={DEFAULT_MAX_BATCH} "
        f"soft_watermark={DEFAULT_SOFT_WATERMARK}",
        flush=True,
    )
    print("UPLOAD_WATCHDOG max_consecutive_hard_failures=5 soft_if_confirm_ok=true", flush=True)
    print(
        f"TIMEBASE policy_hz={contract.policy_hz:g} "
        f"servo_hz={contract.control_hz:g} "
        f"ticks_per_action={contract.servo_ticks_per_action} "
        f"discount={task.discount:g}",
        flush=True,
    )

    session = R10SessionGuard(session_id=make_session_id())
    ctl = FailClosedController()
    upload_wd = UploadWatchdog(max_consecutive_failures=5)
    net_wd = NetworkWatchdog(
        max_age_s=float(network_max_age_s) if params_watchdog else 1e9,
        enabled=False,
    )
    local_env = DrainingQueuedDataStore(DEFAULT_CAPACITY)
    local_intvn = DrainingQueuedDataStore(DEFAULT_CAPACITY)
    data_stores = {"actor_env": local_env, "actor_env_intvn": local_intvn}
    policy_version = 0
    intvn_steps = 0
    intvn_count = 0
    episode_return = 0.0
    episode_steps = 0
    succeed_episodes = 0
    client = None
    agent = None
    jax_mod = None
    rng = None
    env = None
    t0 = time.monotonic()
    stop_flag = {"on": False, "reason": None}
    upload_stop = threading.Event()
    upload_thread = None
    # maxsize=2: one in-flight + one coalesced "go" while worker is busy
    upload_requests: "queue.Queue[str]" = queue.Queue(maxsize=2)

    def _sigint(*_a):
        stop_flag["on"] = True
        stop_flag["reason"] = "sigint"

    signal.signal(signal.SIGINT, _sigint)

    def _confirm() -> Tuple[bool, Dict[str, Any]]:
        ok, report = confirm_upload_by_last_id(client, data_stores)
        incoming = report.get("server_instance_id")
        if incoming:
            session.note_server_instance(str(incoming))
        return bool(ok), report

    def _fault_network(detail: str) -> None:
        reason, _normalized = session.trigger_network_loss(detail)
        ctl.trigger(reason)
        ctl.result.extra["fault_detail"] = detail

    fake_env = not live
    classifier = live
    ckpt = args.classifier_checkpoint or os.environ.get("WA2_CLASSIFIER_CKPT") or ""
    if live and not ckpt:
        raise SystemExit("R13_ACTOR: FAIL — live needs --classifier-checkpoint or WA2_CLASSIFIER_CKPT")

    env = make_wa2_environment(
        task,
        fake_env=fake_env,
        classifier=classifier,
        grasp_action=True,
        enable_intervention=True if live else False,
        classifier_checkpoint=ckpt if classifier else None,
        classifier_consecutive_n=int(args.classifier_consecutive_n),
        end_episode=bool(args.end_episode),
    )
    if tuple(env.action_space.shape) != (7,):
        raise SystemExit(f"R13_ACTOR: FAIL — action space {env.action_space.shape} is not 7D")

    try:
        reset_opts: Dict[str, Any] = {"ready_timeout_s": 8.0, "camera_ready_timeout_s": 8.0}
        if live or args.with_reset:
            reset_opts["skip_reset_motion"] = False
            os.environ.setdefault("RESET_SCENE_OK", "YES")
            os.environ.setdefault("R5_CONFIRM", "YES")
        else:
            reset_opts["skip_reset_motion"] = True if args.skip_reset_motion or not live else False

        cls_wrap = None
        if live:
            target = find_wrapper(env, "WA2SpacemouseIntervention")
            if target is None:
                raise SystemExit("R13_ACTOR: FAIL — intervention wrapper missing")
            target.joy.start_ros()
            print("Waiting for /spacenav/joy ...", flush=True)
            target.joy.wait_ready(timeout_s=10.0)
            cls_wrap = find_wrapper(env, "WA2RewardClassifierWrapper")
            if cls_wrap is None:
                raise SystemExit("R13_ACTOR: FAIL — classifier wrapper missing")
            print(
                f"CLASSIFIER_READY threshold={cls_wrap.threshold} "
                f"consecutive_n={cls_wrap.consecutive_n} "
                f"end_episode={cls_wrap.end_episode} ckpt={ckpt}",
                flush=True,
            )
            _run_classifier_sanity(cls_wrap, ckpt, "before_sac")

        print("HYBRID_SAC_INIT_START", flush=True)
        agent, jax_mod = _make_hybrid_agent(env, task)
        rng = jax_mod.random.PRNGKey(0)
        live_sig = params_tree_signature(agent.state.params)
        print(f"PARAMS_TREE_SIGNATURE={live_sig}", flush=True)
        expected_sig = str(manifest.get("params_tree_signature") or "")
        if expected_sig and expected_sig != live_sig:
            raise SystemExit(
                "R13_ACTOR: FAIL — local hybrid params_tree_signature != manifest"
            )
        print("HYBRID_SAC_INIT_DONE", flush=True)
        if live:
            _run_classifier_sanity(cls_wrap, ckpt, "after_sac")

        trainer_cfg = make_r13_trainer_config(cfg["request_port"], cfg["broadcast_port"])
        client = TrainerClient(
            "actor_env",
            server_ip,
            trainer_cfg,
            data_stores=data_stores,
            wait_for_server=False,
            timeout_ms=timeout_ms,
            log_level=logging.WARNING,
        )

        initial_policy_ready = threading.Event()

        def update_params(params):
            nonlocal agent, policy_version
            try:
                if agent is not None:
                    import jax.numpy as jnp

                    tree = jax_mod.tree_util.tree_map(lambda x: jnp.asarray(x), params)
                    agent = agent.replace(state=agent.state.replace(params=tree))
                policy_version += 1
                net_wd.note_update(params_tree_signature(params))
                if params_watchdog:
                    net_wd.enabled = True
                initial_policy_ready.set()
                print(f"POLICY_VERSION={policy_version}", flush=True)
            except Exception as exc:
                ctl.trigger(f"callback_replace_failed:{exc}")
                stop_flag["on"] = True
                stop_flag["reason"] = "callback_replace_failed"
                # Wake the startup barrier so callback failures fail immediately
                # instead of being reported as a misleading timeout.
                initial_policy_ready.set()

        # Register before handshake: an accepted handshake asks the Learner to
        # publish its current parameters immediately.
        client.recv_network_callback(update_params)

        handshake = build_handshake_request(manifest, session.session_id)
        hs_res = client.request("r13-handshake", handshake)
        hs_payload = {}
        accepted = False
        if isinstance(hs_res, dict):
            hs_payload = hs_res.get("payload") or {}
            accepted = bool(hs_res.get("success")) and bool(
                hs_payload.get("accepted", hs_res.get("success"))
            )
        session.note_handshake(accepted, hs_payload.get("server_instance_id"))
        if not accepted:
            print("R13_HANDSHAKE: FAIL", flush=True)
            mismatches = hs_payload.get("mismatches") or {"error": "handshake timeout/None"}
            print(json.dumps({"handshake_mismatches": mismatches}, default=str), flush=True)
            ctl.trigger("handshake_rejected")
            raise SystemExit("R13_ACTOR: FAIL — handshake rejected before env.reset/step")
        print("R13_HANDSHAKE: PASS", flush=True)
        print(f"SESSION_ID={session.session_id}", flush=True)
        print(f"SERVER_INSTANCE_ID={session.server_instance_id}", flush=True)
        print(f"AUTO_RECONNECT={str(AUTO_RECONNECT).lower()}", flush=True)
        # Learner keeps last_update_id across Actor restarts; local queues must
        # continue from that cursor or uploads become empty no-ops.
        aligned = align_stores_to_server(client, data_stores)
        print(
            "UPLOAD_CURSOR_ALIGN "
            + " ".join(f"{name}={sid}" for name, sid in sorted(aligned.items())),
            flush=True,
        )

        initial_policy_timeout_s = float(args.initial_policy_timeout_s)
        if initial_policy_timeout_s <= 0.0:
            raise SystemExit(
                "R13_ACTOR: FAIL — --initial-policy-timeout-s must be positive"
            )
        print(
            f"INITIAL_POLICY_WAIT timeout_s={initial_policy_timeout_s:g}",
            flush=True,
        )
        deadline = time.monotonic() + initial_policy_timeout_s
        while not initial_policy_ready.wait(timeout=0.1):
            if stop_flag["on"] or ctl.result.triggered:
                break
            if time.monotonic() >= deadline:
                ctl.trigger("initial_policy_timeout")
                break
        if ctl.result.triggered or policy_version < 1:
            reason = ctl.result.reason or "initial_policy_unavailable"
            raise SystemExit(
                f"R13_ACTOR: FAIL — no valid Learner parameters before env.reset "
                f"(reason={reason})"
            )
        print(f"INITIAL_POLICY_READY version={policy_version}", flush=True)

        obs, reset_info = env.reset(seed=0, options=reset_opts)
        print(f"RESET_MODE={reset_info.get('reset_mode', reset_info.get('reset_ok'))}", flush=True)
        print("JIT_START", flush=True)
        grasp_eps = max(0.0, float(args.grasp_eps))
        print(f"GRASP_EPS={grasp_eps}", flush=True)
        _, rng = _sample_action(agent, jax_mod, rng, obs, grasp_eps=grasp_eps)
        print("JIT_DONE", flush=True)

        t0 = time.monotonic()
        step = 0
        last_status = t0
        last_loop_t = t0
        # A-02: consecutive request_hand failures → fail-closed (labels stay 0).
        hand_fail_streak = 0
        hand_fail_max = 5
        start_intvn = int(args.synthetic_intervention_start)
        end_intvn = start_intvn + int(args.synthetic_intervention_steps)
        last_upload: Dict[str, Any] = {}
        intvn_wrap = find_wrapper(env, "WA2SpacemouseIntervention")
        dump_gate = ClassifierHoldDumpGate(enable_hold=False) if live else None
        dump_dir = _classifier_dump_dir(args) if live else None
        if live and dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"CLASSIFIER_DUMP_DIR={dump_dir} "
                "hold-idle dump OFF (idle in session is OK); "
                "type d + Enter to dump; succeed still dumps",
                flush=True,
            )
        if live:
            print(
                f"SERVO_STALL_HOLD_S={SERVO_STALL_HOLD_S:.3f} "
                "(clear latch after control-loop gap / upload backlog)",
                flush=True,
            )

        def _run_one_upload() -> None:
            nonlocal last_upload
            pending_env = local_env.pending()
            pending_intvn = local_intvn.pending()
            pending_total = int(pending_env) + int(pending_intvn)
            # Before a heavy upload (GIL), freeze Servo latch so teleop cannot
            # keep integrating the last non-zero command while the main loop stalls.
            if live and pending_total >= int(DEFAULT_SOFT_WATERMARK):
                base = unwrap_env(env)
                if hasattr(base, "hold_servo_latch") and base.hold_servo_latch():
                    print(
                        f"SERVO_LATCH_HOLD reason=upload_backlog "
                        f"pending={pending_total}",
                        flush=True,
                    )
            started = time.perf_counter()
            _ok_update, upload_report = upload_datastores(
                client, data_stores, max_batch=DEFAULT_MAX_BATCH
            )
            _upload_ms = (time.perf_counter() - started) * 1000.0
            ok, report = _confirm()
            last_upload = {**report, "upload": upload_report, "upload_ms": _upload_ms}
            pending_env = local_env.pending()
            pending_intvn = local_intvn.pending()
            print(
                f"UPLOAD step={step} update_ok={_ok_update} confirm_ok={ok} "
                f"ms={_upload_ms:.0f} pending_env={pending_env} "
                f"pending_intvn={pending_intvn}",
                flush=True,
            )
            gap_skips = upload_report.get("gap_skips") or []
            if gap_skips:
                print(
                    f"UPLOAD_GAP_SKIP {json.dumps(gap_skips, default=str)}",
                    flush=True,
                )
            if not ok:
                print(
                    f"UPLOAD_CONFIRM_WARN {json.dumps(report, default=str)}",
                    flush=True,
                )
            # Backpressure / gap recovery with a reachable Learner is soft.
            # Only repeated hard failures (status unreachable) fail-closed.
            if _ok_update:
                upload_wd.record(True)
            elif ok:
                print(
                    "UPLOAD_SOFT_FAIL learner_reachable=true "
                    "(not counting toward network_loss)",
                    flush=True,
                )
                upload_wd.record(True)
            else:
                reason = upload_wd.record(False)
                if reason:
                    _fault_network(reason)
                    ctl.result.extra["last_upload"] = last_upload
            if local_env.stats()["dropped_unacked"] or local_intvn.stats()["dropped_unacked"]:
                print(
                    f"UPLOAD_DROP_UNACKED env={local_env.stats()['dropped_unacked']} "
                    f"intvn={local_intvn.stats()['dropped_unacked']}",
                    flush=True,
                )

        def _upload_worker() -> None:
            while not upload_stop.is_set():
                try:
                    upload_requests.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    _run_one_upload()
                except Exception as exc:
                    print(f"UPLOAD_WORKER_ERR {type(exc).__name__}: {exc}", flush=True)
                    _fault_network(f"upload_worker:{exc}")

        if live:
            upload_thread = threading.Thread(
                target=_upload_worker, name="r13-upload", daemon=True
            )
            upload_thread.start()
            print("UPLOAD_WORKER=started", flush=True)

        tr_pipe = TransitionPipeline() if live else None
        if tr_pipe is not None:
            print("TRANSITION_PIPELINE=on (overlap with next step)", flush=True)

        def _apply_routed(meta: dict, info_b: dict) -> None:
            nonlocal intvn_steps, intvn_count
            if meta.get("intervened"):
                intvn_steps += 1
                if int(info_b.get("intervention_count") or 0) > intvn_count:
                    intvn_count = int(info_b["intervention_count"])
            maybe_note_episode(
                info_b,
                meta,
                intervention_count=intvn_count,
                intervention_steps=intvn_steps,
            )

        def should_stop() -> bool:
            if stop_flag["on"] or ctl.result.triggered:
                return True
            if float(args.max_seconds) > 0 and (time.monotonic() - t0) >= float(args.max_seconds):
                return True
            if not live:
                return step >= int(args.steps)
            return False

        while not should_stop():
            if not session.can_step():
                if session.fault_reason and not ctl.result.triggered:
                    ctl.trigger(session.fault_reason)
                break
            stale = net_wd.check()
            if stale:
                _fault_network(stale)
                break

            tick = time.monotonic()
            loop_gap = float(tick - last_loop_t)
            if live and loop_gap > SERVO_STALL_HOLD_S:
                base = unwrap_env(env)
                if hasattr(base, "hold_servo_latch") and base.hold_servo_latch():
                    print(
                        f"SERVO_LATCH_HOLD reason=loop_stall "
                        f"gap_ms={loop_gap * 1000.0:.0f}",
                        flush=True,
                    )
            session.note_env_step()
            if live and intvn_wrap is not None and bool(getattr(intvn_wrap, "_session_active", False)):
                sampled = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                sampled, rng = _sample_action(
                    agent, jax_mod, rng, obs, grasp_eps=grasp_eps
                )
            action = scale_arm_action(sampled, float(args.action_scale))
            nxt, reward, terminated, truncated, info = env.step(action)
            last_loop_t = time.monotonic()
            info = dict(info)
            if (not live) and start_intvn <= step < end_intvn:
                exec_7d = np.asarray(action, dtype=np.float32).reshape(-1).copy()
                if "grasp_command" in info:
                    exec_7d[6] = float(info["grasp_command"])
                info["intervene_action"] = exec_7d
            elif "grasp_command" in info and "intervene_action" not in info:
                action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
                action[6] = float(info["grasp_command"])

            episode_end = bool(terminated or truncated)
            obs_i, act_i, nxt_i = obs, action, nxt
            rew_i, term_i, trunc_i, info_i = reward, terminated, truncated, info
            obs_space, act_space = env.observation_space, env.action_space

            def _build_and_route(
                o=obs_i,
                a=act_i,
                n=nxt_i,
                r=rew_i,
                t=term_i,
                trc=trunc_i,
                inf=info_i,
            ):
                transition, meta_b = build_actor_transition(
                    o,
                    a,
                    n,
                    r,
                    t,
                    trc,
                    inf,
                    observation_space=obs_space,
                    action_space=act_space,
                )
                route_transition(transition, meta_b, local_env, local_intvn)
                return meta_b, inf

            if tr_pipe is not None:
                prev = tr_pipe.push(_build_and_route)
                if prev is not None:
                    meta_prev, info_prev = prev
                    _apply_routed(meta_prev, info_prev)
                meta = {
                    "episode_end": episode_end,
                    "intervened": is_intervened(info),
                }
            else:
                meta_b, info_b = _build_and_route()
                _apply_routed(meta_b, info_b)
                meta = meta_b

            episode_return += float(reward)
            episode_steps += 1
            if info.get("sm_session_enter"):
                print("SM_SESSION_ENTER", flush=True)
            if info.get("sm_session_exit"):
                print("SM_SESSION_EXIT", flush=True)
            if info.get("sm_session_dropped_stale"):
                print(
                    f"SM_SESSION_HOLD_STALE joy_age={float(info.get('joy_age') or -1):.3f}",
                    flush=True,
                )
            if info.get("succeed"):
                print(
                    #f"SUCCEED episode_return={episode_return:.3f} "
                    f"\033[91mSUCCEED episode_return={episode_return:.3f} "
                    f"intervention_steps={intvn_steps} policy_version={policy_version}\033[0m",
                    flush=True,
                )

            if info.get("hand_exec_failed"):
                hand_fail_streak += 1
                print(
                    f"HAND_EXEC_FAIL streak={hand_fail_streak}/{hand_fail_max} "
                    f"cmd={info.get('hand_command')} grasp_command=0",
                    flush=True,
                )
                if hand_fail_streak >= hand_fail_max:
                    ctl.trigger("hand_exec_failed")
                    break
            elif info.get("hand_fired") and info.get("hand_ok"):
                hand_fail_streak = 0

            if info.get("stale") or info.get("servo_faulted") or info.get("is_singular"):
                ctl.trigger("env_stale_or_fault")
                break
            if not np.isfinite(np.asarray(nxt.get("state", 0))).all():
                ctl.trigger("nan_state")
                break

            if live and dump_gate is not None and dump_dir is not None:
                now_dump = time.monotonic()
                tag = dump_gate.should_dump(
                    session=bool(info.get("sm_session")),
                    idle=str(info.get("sm_intent") or "") == "idle",
                    force=_poll_dump_key(),
                    succeed=bool(info.get("succeed")),
                    dt=max(0.0, now_dump - tick),
                    now=now_dump,
                )
                if tag:
                    try:
                        save_classifier_dump(
                            dump_dir,
                            nxt,
                            info,
                            tag=tag,
                            seq=dump_gate.count,
                            # Skip JAX rescore on the control path (Orin GIL/GPU).
                            predict_fn=None,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"CLASSIFIER_DUMP_ERR {type(exc).__name__}: {exc}", flush=True)

            obs = nxt
            step += 1
            ctl.result.steps_executed = step

            hit_cap = live and episode_steps >= int(args.episode_max_steps)
            if meta["episode_end"] or hit_cap:
                if tr_pipe is not None:
                    last = tr_pipe.flush()
                    if last is not None:
                        _apply_routed(last[0], last[1])
                if info.get("succeed"):
                    succeed_episodes += 1
                print(
                    f"EPISODE_END succeed={bool(info.get('succeed'))} "
                    f"return={episode_return:.3f} steps={episode_steps} "
                    f"intervention_steps={intvn_steps} policy_version={policy_version}",
                    flush=True,
                )
                episode_return = 0.0
                episode_steps = 0
                obs, _ = env.reset(seed=None, options=reset_opts)

            pending_total = local_env.pending() + local_intvn.pending()
            # Upload on the periodic tick even during SpaceMouse sessions so
            # pending does not pile up to the soft watermark (which freezes
            # Servo for a multi-second GIL hold). Worker still batches async;
            # latch-hold only when pending hits the watermark.
            do_upload = step > 0 and (
                meta["episode_end"]
                or pending_total >= DEFAULT_SOFT_WATERMARK
                or step % upload_every == 0
                or (not live and step >= int(args.steps))
            )
            if do_upload:
                if live:
                    try:
                        upload_requests.put_nowait("go")
                    except queue.Full:
                        pass
                else:
                    _run_one_upload()

            now = time.monotonic()
            if control_hz > 0:
                remain = (1.0 / control_hz) - (now - tick)
                if remain > 0:
                    time.sleep(remain)
            if now - last_status >= 1.0:
                ia = info.get("intervene_action")
                if ia is not None:
                    ia_arr = np.asarray(ia, dtype=np.float32).reshape(-1)
                    ia_norm = float(np.linalg.norm(ia_arr[:6])) if ia_arr.size >= 6 else float(
                        np.linalg.norm(ia_arr)
                    )
                else:
                    ia_norm = 0.0
                hz = float(step) / max(1e-3, now - t0)
                print(
                    f"t={now - t0:5.1f}s step={step} "
                    f"pending_env={local_env.pending()} pending_intvn={local_intvn.pending()} "
                    f"intvn={intvn_steps} net={net_wd.update_count} "
                    f"policy_version={policy_version} hz={hz:.1f} "
                    f"servo_ticks={info.get('servo_ticks_executed')}/"
                    f"{info.get('servo_ticks_requested')} "
                    f"interrupt={info.get('interrupted_by')} "
                    f"sess={int(bool(info.get('sm_session')))} "
                    f"intervened={int(bool(info.get('intervened')))} "
                    f"intent={info.get('sm_intent') or '-'} "
                    f"axis={info.get('sm_axis') or '-'} "
                    f"ia_norm={ia_norm:.3f} "
                    f"joy_age={float(info.get('joy_age') or -1):.3f} "
                    f"p={float(info.get('classifier_p') or 0.0):.3f} "
                    f"streak={int(info.get('classifier_streak') or 0)} "
                    f"{image_obs_stats(obs, ('head', 'wrist'))}",
                    flush=True,
                )
                last_status = now

        if tr_pipe is not None:
            try:
                last = tr_pipe.flush()
                if last is not None:
                    _apply_routed(last[0], last[1])
            finally:
                tr_pipe.close()

        upload_stop.set()
        if upload_thread is not None:
            try:
                upload_requests.put_nowait("flush")
            except queue.Full:
                pass
            upload_thread.join(timeout=8.0)

        if client is not None and not ctl.result.triggered and not live:
            _run_one_upload()

        if args.require_network_update and net_wd.update_count < 1 and not ctl.result.triggered:
            deadline = time.monotonic() + 60.0
            while net_wd.update_count < 1 and time.monotonic() < deadline and not ctl.result.triggered:
                upload_datastores(client, data_stores, max_batch=DEFAULT_MAX_BATCH)
                time.sleep(0.1)
            if net_wd.update_count < 1:
                _fault_network("network_stale")

        if stop_flag["on"] and not ctl.result.triggered:
            ctl.trigger(stop_flag["reason"] or "sigint")

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
        raise
    finally:
        upload_stop.set()
        if upload_thread is not None:
            upload_thread.join(timeout=2.0)
        result = ctl.shutdown(env, client)

    summary = {
        "mode": args.mode,
        "task_id": task.task_id,
        "session_id": session.session_id,
        "server_instance_id": session.server_instance_id,
        "ONLINE_N": int(local_env.stats()["total_inserted"]),
        "INTVN_N": int(local_intvn.stats()["total_inserted"]),
        "PENDING_ENV": local_env.pending(),
        "PENDING_INTVN": local_intvn.pending(),
        "POLICY_VERSION": policy_version,
        "AUTO_RECONNECT": AUTO_RECONNECT,
        "FAIL_CLOSED": result.triggered,
        "fault_reason": result.reason,
        "steps_executed": result.steps_executed,
        "succeed_episodes": succeed_episodes,
        "network_update_count": net_wd.update_count,
    }
    if args.output:
        out = Path(args.output)
        if out.suffix.lower() == ".json":
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2, default=str, sort_keys=True) + "\n")
        else:
            out.mkdir(parents=True, exist_ok=True)
            (out / "actor_status.json").write_text(
                json.dumps(summary, indent=2, default=str, sort_keys=True) + "\n"
            )

    print(f"ONLINE_N={int(local_env.stats()['total_inserted'])}", flush=True)
    print(f"INTVN_N={int(local_intvn.stats()['total_inserted'])}", flush=True)
    print(f"PENDING_ENV={local_env.pending()} PENDING_INTVN={local_intvn.pending()}", flush=True)
    print(f"POLICY_VERSION={policy_version}", flush=True)
    print(f"AUTO_RECONNECT={str(AUTO_RECONNECT).lower()}", flush=True)
    if result.triggered:
        print("FAIL_CLOSED=true", flush=True)
        print(f"FAULT_REASON={result.reason}", flush=True)
        print("R13_ACTOR: FAIL", flush=True)
    elif args.mode == "fake":
        fake_ok = (
            int(local_env.stats()["total_inserted"]) >= int(args.steps)
            and (
                int(args.synthetic_intervention_steps) <= 0
                or int(local_intvn.stats()["total_inserted"])
                >= int(args.synthetic_intervention_steps)
            )
            and (not args.require_network_update or policy_version >= 1)
        )
        print("R13_FAKE: PASS" if fake_ok else "R13_FAKE: FAIL", flush=True)
        if not fake_ok:
            raise SystemExit("R13_ACTOR: FAIL — fake preflight thresholds not met")
    else:
        print("R13_ACTOR_LIVE_STOP: PASS", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    run_actor(args)


if __name__ == "__main__":
    main()
