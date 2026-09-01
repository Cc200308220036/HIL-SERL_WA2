#!/usr/bin/env python3
"""Learner-only localhost smoke for R10 handshake and dual data stores.

This deliberately does not initialize SAC params and is not a cross-machine Gate.
The formal Actor still uses verify_r10_actor_remote.py from the Orin container.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "hil-serl-main" / "examples"),
    str(ROOT / "hil-serl-main" / "serl_launcher"),
]

from agentlace.data.data_store import QueuedDataStore  # noqa: E402
from agentlace.trainer import TrainerClient  # noqa: E402
from hilserl_wa2.experiments.env_factory import make_wa2_environment_from_id  # noqa: E402
from hilserl_wa2.experiments.r10_protocol import (  # noqa: E402
    build_handshake_request,
    confirm_r10_server_status,
    load_network_config,
    make_r10_trainer_config,
    ordered_transition_digest,
)
from hilserl_wa2.experiments.transition import (  # noqa: E402
    build_actor_transition,
    route_transition,
)


def _load_manifest(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("R10_LEARNER_LOOPBACK: FAIL — manifest must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--network-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()

    cfg = load_network_config(args.network_config)
    manifest = _load_manifest(args.manifest)
    if manifest.get("task_id") != args.task:
        raise SystemExit("R10_LEARNER_LOOPBACK: FAIL — task/manifest mismatch")

    env_store = QueuedDataStore(16)
    intvn_store = QueuedDataStore(16)
    trainer_cfg = make_r10_trainer_config(cfg["request_port"], cfg["broadcast_port"])
    client = TrainerClient(
        "actor_env",
        args.server_ip,
        trainer_cfg,
        data_stores={"actor_env": env_store, "actor_env_intvn": intvn_store},
        wait_for_server=False,
        timeout_ms=int(float(args.timeout_s) * 1000),
    )
    env = make_wa2_environment_from_id(args.task, fake_env=True, classifier=False)
    try:
        session_id = f"learner-loopback-{uuid.uuid4().hex[:8]}"
        response = client.request(
            "r10-handshake", build_handshake_request(manifest, session_id)
        )
        payload = response.get("payload", {}) if isinstance(response, dict) else {}
        if not (
            isinstance(response, dict)
            and response.get("success")
            and payload.get("accepted")
        ):
            raise SystemExit(
                f"R10_LEARNER_LOOPBACK: FAIL — handshake rejected: {response}"
            )

        observation, _ = env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)
        info = dict(info)
        info["intervene_action"] = action.copy()
        transition, metadata = build_actor_transition(
            observation,
            action,
            next_observation,
            reward,
            terminated,
            truncated,
            info,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        if not metadata["intervened"]:
            raise SystemExit("R10_LEARNER_LOOPBACK: FAIL — intervention not routed")
        route_transition(transition, metadata, env_store, intvn_store)
        client.update()

        deadline = time.monotonic() + args.timeout_s
        report = {}
        ok = False
        while time.monotonic() < deadline:
            ok, report = confirm_r10_server_status(
                client,
                local_env=len(env_store),
                local_intvn=len(intvn_store),
                local_env_digest=ordered_transition_digest([transition]),
                local_intvn_digest=ordered_transition_digest([transition]),
                client_env_id=env_store.latest_data_id(),
                client_intvn_id=intvn_store.latest_data_id(),
            )
            if ok:
                break
            time.sleep(0.1)
        if not ok:
            raise SystemExit(f"R10_LEARNER_LOOPBACK: FAIL — status mismatch: {report}")

        print("R10_HANDSHAKE: PASS")
        print(f"SERVER_INSTANCE_ID={payload.get('server_instance_id')}")
        print(f"SERVER_ENV_COUNT={report.get('server_env_count')}")
        print(f"SERVER_INTVN_COUNT={report.get('server_intvn_count')}")
        print(
            "LAST_UPDATE_ID_MATCH="
            f"{'PASS' if report.get('last_update_id_match') else 'FAIL'}"
        )
        print(
            "ORDERED_DIGEST_MATCH="
            f"{'PASS' if report.get('ordered_digest_match') else 'FAIL'}"
        )
        print("R10_LEARNER_LOOPBACK: PASS")
    finally:
        env.close()
        client.stop()


if __name__ == "__main__":
    main()
