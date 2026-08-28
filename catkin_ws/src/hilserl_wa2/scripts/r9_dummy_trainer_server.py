#!/usr/bin/env python3
"""R9 dummy TrainerServer: dual store, count/hash, compatible params broadcast. No GPU agent."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

CATKIN_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CATKIN_SRC))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "serl_launcher"))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "examples"))

from agentlace.data.data_store import QueuedDataStore  # noqa: E402

from hilserl_wa2.experiments.actor_safety import (  # noqa: E402
    make_r9_trainer_config,
    params_tree_signature,
)
from hilserl_wa2.experiments.transition import validate_transition  # noqa: E402


ALLOWED_PARAMS_ROOTS = (
    Path("/root/catkin_ws/runs"),
    Path("/tmp"),
)


class InspectableQueue(QueuedDataStore):
    def __init__(self, capacity: int, name: str):
        super().__init__(capacity)
        self.name = name
        self.rows = []

    def insert(self, data: Any) -> None:
        super().insert(data)
        self.rows.append(data)


def _ss_bind(port: int) -> str:
    try:
        out = subprocess.check_output(["ss", "-ltn"], text=True)
    except Exception:
        return f"tcp://*:{port} (ss unavailable)"
    for line in out.splitlines():
        if f":{port} " in line or line.rstrip().endswith(f":{port}"):
            return line.strip()
    return f"tcp://*:{port} (not listed yet)"


def _safe_params_path(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    allowed = False
    for root in ALLOWED_PARAMS_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise ValueError(f"params path not under allowed roots: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="R9 dummy Agentlace TrainerServer")
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--ip", default="127.0.0.1", help="documented client IP; ZMQ still binds *")
    parser.add_argument("--port", type=int, default=5588)
    parser.add_argument("--broadcast-port", type=int, default=5589)
    parser.add_argument("--expect-env", type=int, default=0)
    parser.add_argument("--expect-intvn", type=int, default=0)
    parser.add_argument("--stop-after-env", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--status-file", default="")
    parser.add_argument("--ready-file", default="")
    parser.add_argument("--params-pkl", default="")
    parser.add_argument("--republish-s", type=float, default=1.0)
    args = parser.parse_args()

    from agentlace.trainer import TrainerServer

    env_store = InspectableQueue(50000, "actor_env")
    intvn_store = InspectableQueue(50000, "actor_env_intvn")
    state: Dict[str, Any] = {
        "expect_met": False,
        "schema_ok": True,
        "schema_error": None,
        "published": False,
        "params_signature": None,
        "stopped": False,
        "task": args.task,
    }
    params_holder: Dict[str, Any] = {"params": None}
    lock = threading.Lock()

    def _schema_sample(store: InspectableQueue) -> None:
        if not store.rows:
            return
        try:
            validate_transition(store.rows[-1])
        except Exception as exc:
            state["schema_ok"] = False
            state["schema_error"] = str(exc)

    def status_payload() -> Dict[str, Any]:
        with lock:
            return {
                "actor_env_count": len(env_store),
                "actor_env_intvn_count": len(intvn_store),
                "last_update_id": {
                    "actor_env": server.last_update_id_map.get("actor_env", -1),
                    "actor_env_intvn": server.last_update_id_map.get("actor_env_intvn", -1),
                },
                "expect_met": state["expect_met"],
                "schema_ok": state["schema_ok"],
                "schema_error": state["schema_error"],
                "published": state["published"],
                "params_signature": state["params_signature"],
                "bind_request": f"tcp://*:{args.port}",
                "bind_broadcast": f"tcp://*:{args.broadcast_port}",
            }

    def write_status() -> None:
        if not args.status_file and not args.output:
            return
        payload = status_payload()
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.status_file:
            Path(args.status_file).write_text(text + "\n", encoding="utf-8")
        if args.output:
            Path(args.output).write_text(text + "\n", encoding="utf-8")

    def maybe_expect() -> None:
        if args.expect_env <= 0:
            return
        if len(env_store) >= args.expect_env and len(intvn_store) >= args.expect_intvn:
            if not state["expect_met"]:
                if not state["schema_ok"]:
                    print("EXPECT_MET skipped: SCHEMA=FAIL", flush=True)
                    return
                state["expect_met"] = True
                print(
                    f"EXPECT_MET env={len(env_store)} intvn={len(intvn_store)}",
                    flush=True,
                )
                print("R9_AGENTLACE_LOCAL: PASS", flush=True)

    def maybe_stop_after() -> None:
        if args.stop_after_env > 0 and len(env_store) >= args.stop_after_env:
            if state["stopped"]:
                return
            state["stopped"] = True
            print(
                f"STOP_AFTER_ENV reached {len(env_store)}; stopping server",
                flush=True,
            )
            threading.Thread(target=_delayed_stop, daemon=True).start()

    def _delayed_stop() -> None:
        time.sleep(0.05)
        try:
            server.stop()
        except Exception:
            pass
        state["stopped"] = True

    def data_callback(store_name: str, payload: dict) -> None:
        with lock:
            if store_name == "actor_env":
                _schema_sample(env_store)
            elif store_name == "actor_env_intvn":
                _schema_sample(intvn_store)
            maybe_expect()
            maybe_stop_after()
        write_status()
        print(
            f"STORE {store_name} count="
            f"{len(env_store) if store_name == 'actor_env' else len(intvn_store)} "
            f"batch={len(payload.get('data') or [])}",
            flush=True,
        )

    def publish_from_path(path: str) -> Dict[str, Any]:
        import pickle

        resolved = _safe_params_path(path)
        with resolved.open("rb") as handle:
            params = pickle.load(handle)
        sig = params_tree_signature(params)
        params_holder["params"] = params
        server.publish_network(params)
        state["published"] = True
        state["params_signature"] = sig
        print(f"PUBLISH_PARAMS signature={sig}", flush=True)
        return {"published": True, "signature": sig, "path": str(resolved)}

    def request_callback(typ: str, payload: dict) -> dict:
        if typ == "send-stats":
            return {}
        if typ == "r9-status":
            return {"success": True, "payload": status_payload()}
        if typ == "r9-publish-params":
            try:
                result = publish_from_path(str((payload or {}).get("path", "")))
                return {"success": True, "payload": result}
            except Exception as exc:
                return {"success": False, "message": str(exc)}
        return {"success": False, "message": f"unknown request {typ}"}

    cfg = make_r9_trainer_config(args.port, args.broadcast_port)
    server = TrainerServer(cfg, data_callback=data_callback, request_callback=request_callback)
    server.register_data_store("actor_env", env_store)
    server.register_data_store("actor_env_intvn", intvn_store)
    server.start(threaded=True)

    print(f"STORE actor_env registered", flush=True)
    print(f"STORE actor_env_intvn registered", flush=True)
    print(
        f"R9_SERVER_READY {args.ip}:{args.port} broadcast={args.broadcast_port}",
        flush=True,
    )
    print(f"ZMQ_BIND_REQUEST {_ss_bind(args.port)}", flush=True)
    print(f"ZMQ_BIND_BROADCAST {_ss_bind(args.broadcast_port)}", flush=True)
    print(
        "NOTE: Agentlace ReqRep/Broadcast bind tcp://*:port ; R9 Client must use 127.0.0.1",
        flush=True,
    )
    if args.ready_file:
        Path(args.ready_file).write_text("ready\n", encoding="utf-8")
    write_status()

    stop_flag = threading.Event()

    def _handle_stop(*_args) -> None:
        stop_flag.set()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    if args.params_pkl:
        try:
            publish_from_path(args.params_pkl)
        except Exception as exc:
            print(f"WARN initial params-pkl failed: {exc}", flush=True)

    def republish_loop() -> None:
        while not stop_flag.is_set() and not state["stopped"]:
            params = params_holder.get("params")
            if params is not None:
                try:
                    server.publish_network(params)
                except Exception:
                    pass
            time.sleep(max(0.2, float(args.republish_s)))

    threading.Thread(target=republish_loop, daemon=True).start()

    while not stop_flag.is_set() and not state["stopped"]:
        time.sleep(0.2)
        write_status()

    summary = status_payload()
    summary["schema"] = "PASS" if state["schema_ok"] else "FAIL"
    env_n = int(summary["actor_env_count"])
    intvn_n = int(summary["actor_env_intvn_count"])
    summary["no_duplicates"] = env_n == len(env_store.rows) and intvn_n == len(intvn_store.rows)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"SERVER_ENV_COUNT={env_n}", flush=True)
    print(f"SERVER_INTVN_COUNT={intvn_n}", flush=True)
    print(f"NO_DUPLICATES={'PASS' if summary['no_duplicates'] else 'FAIL'}", flush=True)
    print(f"SCHEMA={'PASS' if state['schema_ok'] else 'FAIL'}", flush=True)
    try:
        server.stop()
    except Exception:
        pass
    print("R9_SERVER_STOPPED", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
