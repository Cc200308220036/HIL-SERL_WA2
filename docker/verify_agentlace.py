#!/usr/bin/env python3
"""Exercise Agentlace transition upload and policy broadcast locally."""

from __future__ import annotations

import time

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig, TrainerServer


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def main() -> None:
    config = TrainerConfig(
        port_number=5598,
        broadcast_port=5599,
        request_types=["send-stats"],
    )
    server_store = QueuedDataStore(10)
    client_store = QueuedDataStore(10)
    server = TrainerServer(config)
    server.register_data_store("actor_test", server_store)
    server.start(threaded=True)
    client = None

    try:
        time.sleep(0.5)
        client = TrainerClient(
            "actor_test",
            "127.0.0.1",
            config,
            data_store=client_store,
            wait_for_server=True,
            timeout_ms=3000,
        )
        networks: list[dict] = []
        client.recv_network_callback(networks.append)

        client_store.insert(
            {
                "observation": [1, 2, 3],
                "action": [0.1, 0.2],
                "reward": 1.0,
            }
        )
        if not client.update():
            raise RuntimeError("Agentlace transition upload failed.")
        if not wait_until(lambda: len(server_store) == 1):
            raise RuntimeError("TrainerServer did not receive transition.")

        time.sleep(0.5)
        server.publish_network({"step": 123, "params": "test"})
        if not wait_until(lambda: bool(networks)):
            raise RuntimeError("TrainerClient did not receive broadcast.")
        if networks[-1]["step"] != 123:
            raise RuntimeError(f"Unexpected broadcast: {networks[-1]}")

        print("AGENTLACE LOCAL COMMUNICATION: PASS")
    finally:
        if client is not None:
            client.stop()
        server.stop()


if __name__ == "__main__":
    main()
