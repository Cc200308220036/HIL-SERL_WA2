#!/usr/bin/env python3
"""Finite R10 Learner TrainerServer with handshake, dual stores and params broadcast."""

from __future__ import annotations

import argparse
import importlib.metadata
import ipaddress
import json
import os
import signal
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "hil-serl-main" / "examples"), str(ROOT / "hil-serl-main" / "serl_launcher")]

from agentlace.data.data_store import QueuedDataStore  # noqa: E402
from agentlace.trainer import TrainerServer  # noqa: E402
from hilserl_wa2.experiments.actor_safety import params_tree_signature  # noqa: E402
from hilserl_wa2.experiments.r10_protocol import (  # noqa: E402
    HANDSHAKE_KEYS,
    PROTOCOL_VERSION,
    R10ProtocolError,
    TRANSITION_SCHEMA_VERSION,
    assert_remote_learner_ip,
    compare_handshake,
    load_network_config,
    make_r10_trainer_config,
    network_config_hash,
    ordered_transition_digest,
    sha256_file,
    source_tree_manifest,
    transition_sha256,
)
from hilserl_wa2.experiments.transition import validate_transition  # noqa: E402


class InspectableStore(QueuedDataStore):
    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.rows = []

    def insert(self, data: Any) -> None:
        validate_transition(data)
        super().insert(data)
        self.rows.append(data)


def _load_manifest(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest must be a JSON object")
    return data


def _validate_runtime_manifest(
    manifest: dict, *, task_id: str, network_config: str, allow_placeholders: bool
) -> None:
    from hilserl_wa2.experiments.env_factory import build_space_signature
    from hilserl_wa2.experiments.task_config import load_task

    missing = [key for key in HANDSHAKE_KEYS if key not in manifest]
    if missing:
        raise SystemExit(f"R10_LEARNER: FAIL — manifest missing keys: {missing}")
    task = load_task(task_id)
    _, current_source_hash = source_tree_manifest(REPO_ROOT)
    wheel = REPO_ROOT / "artifacts" / "wheels" / "agentlace-0.1.3-py3-none-any.whl"
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "task_id": task.task_id,
        "exp_name": task.exp_name,
        "config_bundle_hash": task.config_bundle_hash(),
        "network_config_hash": network_config_hash(network_config),
        "space_hash": build_space_signature(task, "learner")["space_hash"],
        "agentlace_version": importlib.metadata.version("agentlace"),
        "agentlace_wheel_sha256": sha256_file(wheel),
        "source_tree_sha256": current_source_hash,
    }
    mismatches = {
        key: {"expected": value, "manifest": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if manifest.get("role") not in (None, "learner"):
        mismatches["role"] = {"expected": "learner", "manifest": manifest.get("role")}
    if mismatches:
        raise SystemExit(
            "R10_LEARNER: FAIL — stale/incompatible manifest: "
            + json.dumps(mismatches, sort_keys=True)
        )

    if not allow_placeholders:
        cfg = load_network_config(network_config)
        try:
            learner_ip = assert_remote_learner_ip(cfg["learner_ip"])
            actor_ip = assert_remote_learner_ip(cfg["actor_ip"])
        except R10ProtocolError as exc:
            raise SystemExit(f"R10_LEARNER: FAIL — invalid network endpoint: {exc}") from exc
        try:
            ipaddress.IPv4Address(learner_ip)
            ipaddress.IPv4Address(actor_ip)
        except ipaddress.AddressValueError as exc:
            raise SystemExit(f"R10_LEARNER: FAIL — invalid IPv4 address: {exc}") from exc
        if learner_ip == actor_ip:
            raise SystemExit("R10_LEARNER: FAIL — learner_ip and actor_ip must differ")
        expected_cidr = f"{actor_ip}/32"
        if str(cfg["allowed_actor_cidr"]).strip() != expected_cidr:
            raise SystemExit(
                "R10_LEARNER: FAIL — allowed_actor_cidr must equal " + expected_cidr
            )


def _make_params(task_id: str):
    import jax
    from hilserl_wa2.experiments.env_factory import make_wa2_environment_from_id
    from hilserl_wa2.experiments.task_config import load_task
    from serl_launcher.utils.launcher import make_sac_pixel_agent

    task = load_task(task_id)
    env = make_wa2_environment_from_id(task_id, fake_env=True, classifier=False)
    try:
        agent = make_sac_pixel_agent(
            seed=0,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=list(task.image_keys),
            encoder_type=task.encoder_type,
            discount=float(task.discount),
        )
        params = jax.tree_util.tree_map(lambda x: __import__("numpy").asarray(jax.device_get(x)), agent.state.params)
        print(f"JAX_DEVICES={jax.devices()}", flush=True)
        return params
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="bottle_pick")
    p.add_argument("--network-config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--expect-env", type=int, default=0)
    p.add_argument("--expect-intvn", type=int, default=0)
    p.add_argument("--republish-s", type=float, default=1.0)
    p.add_argument("--capacity", type=int, default=5000)
    p.add_argument("--output", required=True)
    p.add_argument("--status-file", default="")
    p.add_argument("--no-agent", action="store_true", help="protocol unit test only; formal Gate forbids")
    args = p.parse_args()

    cfg = load_network_config(args.network_config)
    expected = _load_manifest(args.manifest)
    if expected.get("task_id") != args.task:
        raise SystemExit("R10_LEARNER: FAIL — task/manifest mismatch")
    if args.capacity <= 0:
        raise SystemExit("R10_LEARNER: FAIL — capacity must be positive")
    if args.expect_env < 0 or args.expect_intvn < 0 or args.expect_intvn > args.expect_env:
        raise SystemExit("R10_LEARNER: FAIL — invalid expected store counts")
    if args.expect_env > args.capacity or args.expect_intvn > args.capacity:
        raise SystemExit("R10_LEARNER: FAIL — capacity is smaller than expected counts")
    if args.republish_s <= 0:
        raise SystemExit("R10_LEARNER: FAIL — republish-s must be positive")
    _validate_runtime_manifest(
        expected,
        task_id=args.task,
        network_config=args.network_config,
        allow_placeholders=args.no_agent,
    )
    env_store = InspectableStore(args.capacity)
    intvn_store = InspectableStore(args.capacity)
    lock = threading.RLock()
    state: Dict[str, Any] = {
        "server_instance_id": uuid.uuid4().hex,
        "accepted_session_id": None,
        "handshake_accepted": False,
        "handshake_mismatches": {},
        "schema_ok": True,
        "published": False,
        "params_tree_signature": None,
        "started_unix": time.time(),
        "expect_met": False,
    }
    params = None if args.no_agent else _make_params(args.task)
    if params is not None:
        state["params_tree_signature"] = params_tree_signature(params)
        expected_sig = expected.get("params_tree_signature")
        if expected_sig and expected_sig != state["params_tree_signature"]:
            raise SystemExit("R10_LEARNER: FAIL — manifest params tree mismatch")
        expected["params_tree_signature"] = state["params_tree_signature"]
        print(f"PARAMS_TREE_SIGNATURE={state['params_tree_signature']}", flush=True)
        print("PARAMS_TREE_READY", flush=True)

    def payload() -> dict:
        with lock:
            env_hashes = [transition_sha256(x) for x in env_store.rows]
            intvn_hashes = [transition_sha256(x) for x in intvn_store.rows]
            return {
                **state,
                "actor_env_count": len(env_store),
                "actor_env_intvn_count": len(intvn_store),
                "last_update_id": dict(server.last_update_id_map),
                "unique_transitions": len(set(env_hashes)),
                "unique_intvn": len(set(intvn_hashes)),
                "ordered_digest": ordered_transition_digest(env_store.rows),
                "ordered_intvn_digest": ordered_transition_digest(intvn_store.rows),
                "uptime_s": time.time() - state["started_unix"],
            }

    def write_status() -> None:
        data = json.dumps(payload(), indent=2, sort_keys=True) + "\n"
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(data, encoding="utf-8")
        if args.status_file:
            status_path = Path(args.status_file)
            status_path.parent.mkdir(parents=True, exist_ok=True)
            status_path.write_text(data, encoding="utf-8")

    def data_callback(store_name: str, update: dict) -> None:
        with lock:
            if not state["handshake_accepted"]:
                state["schema_ok"] = False
                state["handshake_mismatches"] = {"ordering": "datastore before handshake"}
            if args.expect_env and len(env_store) >= args.expect_env and len(intvn_store) >= args.expect_intvn:
                state["expect_met"] = bool(state["handshake_accepted"] and state["schema_ok"])
        write_status()
        print(f"STORE {store_name} batch={len(update.get('data') or [])} env={len(env_store)} intvn={len(intvn_store)}", flush=True)
        if state["expect_met"]:
            print(f"EXPECT_MET env={len(env_store)} intvn={len(intvn_store)}", flush=True)
            print("R10_AGENTLACE_REMOTE: PASS", flush=True)

    def request_callback(kind: str, request: dict) -> dict:
        if kind == "send-stats":
            return {"success": True}
        if kind == "r10-ping":
            return {"success": True, "payload": {"server_time_ns": time.time_ns(), "server_instance_id": state["server_instance_id"]}}
        if kind == "r10-status":
            return {"success": True, "payload": payload()}
        if kind == "r10-handshake":
            with lock:
                result = compare_handshake(expected, request or {})
                if state["accepted_session_id"] not in (None, result["session_id"]):
                    result["accepted"] = False
                    result["mismatches"]["session_id"] = {"expected": state["accepted_session_id"], "received": result["session_id"]}
                state["handshake_accepted"] = bool(result["accepted"])
                state["handshake_mismatches"] = result["mismatches"]
                if result["accepted"]:
                    state["accepted_session_id"] = result["session_id"]
            print(f"R10_HANDSHAKE: {'PASS' if result['accepted'] else 'FAIL'}", flush=True)
            return {"success": bool(result["accepted"]), "payload": {**result, "server_instance_id": state["server_instance_id"], "manifest": expected}}
        return {"success": False, "message": f"unknown request: {kind}"}

    trainer_cfg = make_r10_trainer_config(cfg["request_port"], cfg["broadcast_port"])
    server = TrainerServer(trainer_cfg, data_callback=data_callback, request_callback=request_callback)
    server.register_data_store("actor_env", env_store)
    server.register_data_store("actor_env_intvn", intvn_store)
    server.start(threaded=True)
    print(f"R10_LEARNER_READY request={cfg['request_port']} broadcast={cfg['broadcast_port']}", flush=True)
    print(f"SERVER_INSTANCE_ID={state['server_instance_id']}", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def publish_loop() -> None:
        while not stop.is_set():
            if params is not None and state["handshake_accepted"]:
                server.publish_network(params)
                state["published"] = True
            time.sleep(max(0.2, args.republish_s))

    threading.Thread(target=publish_loop, daemon=True).start()
    try:
        while not stop.wait(0.5):
            write_status()
    finally:
        write_status()
        server.stop()
        print("R10_LEARNER_STOPPED", flush=True)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
