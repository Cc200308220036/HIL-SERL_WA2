"""Unit tests for draining Actor upload queue (no ROS)."""

from __future__ import annotations

import pathlib
import sys
import unittest
from typing import Any, Dict, Optional

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.interventions.actor_upload_queue import (  # noqa: E402
    DrainingQueuedDataStore,
    align_stores_to_server,
    upload_datastores,
)


class FakeClient:
    def __init__(self) -> None:
        self.last_ids: Dict[str, int] = {"actor_env": -1, "actor_env_intvn": -1}
        self.received: Dict[str, list] = {"actor_env": [], "actor_env_intvn": []}

    def get_server_last_update_id(self, name: str) -> Optional[int]:
        return self.last_ids.get(name)

    def _update_ds(self, name: str, data: dict) -> dict:
        batch = data["data"]
        last_id = int(data["last_id"])
        self.received[name].extend(batch)
        self.last_ids[name] = last_id
        return {"success": True}


class DrainQueueTests(unittest.TestCase):
    def test_insert_and_discard(self):
        store = DrainingQueuedDataStore(capacity=8)
        for i in range(5):
            store.insert({"i": i})
        self.assertEqual(store.pending(), 5)
        self.assertEqual(store.latest_data_id(), 4)
        dropped = store.discard_through(2)
        self.assertEqual(dropped, 3)
        self.assertEqual(store.pending(), 2)
        batch, last, gap = store.peek_batch_after(2, max_batch=10)
        self.assertIsNone(gap)
        self.assertEqual(len(batch), 2)
        self.assertEqual(last, 4)

    def test_max_batch_last_id(self):
        store = DrainingQueuedDataStore(capacity=64)
        for i in range(20):
            store.insert({"i": i})
        batch, last, gap = store.peek_batch_after(-1, max_batch=5)
        self.assertIsNone(gap)
        self.assertEqual(len(batch), 5)
        self.assertEqual(last, 4)

    def test_capacity_drops_oldest(self):
        store = DrainingQueuedDataStore(capacity=8)
        for i in range(10):
            store.insert({"i": i})
        self.assertEqual(store.pending(), 8)
        self.assertGreater(store.stats()["dropped_unacked"], 0)
        # Default: skip the hole and continue from local head.
        batch, last, gap = store.peek_batch_after(-1, max_batch=10)
        self.assertIsNotNone(gap)
        self.assertEqual(gap["local_first"], 2)
        self.assertEqual(len(batch), 8)
        self.assertEqual(last, 9)
        with self.assertRaises(RuntimeError):
            store.peek_batch_after(-1, max_batch=10, allow_gap=False)

    def test_upload_drains_acked(self):
        env = DrainingQueuedDataStore(capacity=64)
        intvn = DrainingQueuedDataStore(capacity=64)
        for i in range(10):
            env.insert({"e": i})
            intvn.insert({"v": i})
        client = FakeClient()
        ok, report = upload_datastores(
            client, {"actor_env": env, "actor_env_intvn": intvn}, max_batch=4
        )
        self.assertTrue(ok)
        self.assertEqual(report["stores"]["actor_env"]["sent"], 4)
        self.assertEqual(env.pending(), 6)
        self.assertEqual(client.last_ids["actor_env"], 3)
        ok2, _ = upload_datastores(
            client, {"actor_env": env, "actor_env_intvn": intvn}, max_batch=100
        )
        self.assertTrue(ok2)
        self.assertEqual(env.pending(), 0)
        self.assertEqual(intvn.pending(), 0)

    def test_align_allows_actor_only_restart(self):
        """Learner still at high last_id; new Actor must append, not no-op."""

        client = FakeClient()
        client.last_ids = {"actor_env": 1000, "actor_env_intvn": 800}
        env = DrainingQueuedDataStore(capacity=64)
        intvn = DrainingQueuedDataStore(capacity=64)
        # Stale local ids from a fresh process (would trigger server-ahead).
        for i in range(5):
            env.insert({"stale": i})
        aligned = align_stores_to_server(
            client, {"actor_env": env, "actor_env_intvn": intvn}
        )
        self.assertEqual(aligned["actor_env"], 1000)
        self.assertEqual(aligned["actor_env_intvn"], 800)
        self.assertEqual(env.pending(), 0)
        env.insert({"e": "new"})
        intvn.insert({"v": "new"})
        self.assertEqual(env.latest_data_id(), 1001)
        self.assertEqual(intvn.latest_data_id(), 801)
        ok, report = upload_datastores(
            client, {"actor_env": env, "actor_env_intvn": intvn}, max_batch=10
        )
        self.assertTrue(ok)
        self.assertEqual(report["stores"]["actor_env"]["sent"], 1)
        self.assertEqual(report["stores"]["actor_env_intvn"]["sent"], 1)
        self.assertEqual(client.last_ids["actor_env"], 1001)
        self.assertEqual(env.pending(), 0)

    def test_upload_skips_capacity_gap(self):
        """After DROP_UNACKED, upload must jump the hole instead of failing."""

        env = DrainingQueuedDataStore(capacity=8)
        for i in range(10):
            env.insert({"i": i})
        client = FakeClient()
        # Server still thinks nothing was acked (points before dropped ids 0,1).
        client.last_ids["actor_env"] = -1
        client.last_ids["actor_env_intvn"] = -1
        intvn = DrainingQueuedDataStore(capacity=8)
        ok, report = upload_datastores(
            client, {"actor_env": env, "actor_env_intvn": intvn}, max_batch=4
        )
        self.assertTrue(ok)
        self.assertTrue(report["gap_skips"])
        self.assertEqual(report["gap_skips"][0]["store"], "actor_env")
        self.assertEqual(report["stores"]["actor_env"]["sent"], 4)
        self.assertEqual(client.last_ids["actor_env"], 5)  # ids 2,3,4,5


if __name__ == "__main__":
    unittest.main()
