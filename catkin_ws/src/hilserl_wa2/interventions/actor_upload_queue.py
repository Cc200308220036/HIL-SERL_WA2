"""Actor-side upload queue: bounded, drainable, capped batch size.

Keeps the 50 Hz control loop from fighting a multi-10k image deque + GIL.
Not part of the R13 handshake source tree (``interventions/``).
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agentlace.data.data_store import DataStoreBase

# Keep pending images small. Learner replay capacity is separate.
DEFAULT_CAPACITY = 2048
# Smaller batches → shorter GIL holds on the upload worker (helps ~20 Hz teleop).
DEFAULT_MAX_BATCH = 64
# Trigger an upload sooner when the pending queue climbs.
# Lower than the old 512 so a stalled periodic drain still latches early
# instead of dumping a multi-second backlog onto the control path.
DEFAULT_SOFT_WATERMARK = 256
# Yield the GIL between store uploads so the control thread can run.
UPLOAD_GIL_YIELD_S = 0.002


class DrainingQueuedDataStore(DataStoreBase):
    """Queue with explicit discard of ids already accepted by the Learner."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        if int(capacity) < 8:
            raise ValueError("capacity must be >= 8")
        DataStoreBase.__init__(self, int(capacity))
        self._seq_id_queue: deque[int] = deque()
        self._data_queue: deque[Any] = deque()
        self.latest_seq_id = -1
        self._lock = Lock()
        self._total_inserted = 0
        self._total_discarded = 0
        self._dropped_unacked = 0

    def latest_data_id(self) -> int:
        return self.latest_seq_id

    def insert(self, data: Any) -> None:
        with self._lock:
            # If full, drop oldest. Count as unacked loss when we still had pending.
            if len(self._data_queue) >= self.capacity:
                self._seq_id_queue.popleft()
                self._data_queue.popleft()
                self._dropped_unacked += 1
            self.latest_seq_id += 1
            self._seq_id_queue.append(self.latest_seq_id)
            self._data_queue.append(data)
            self._total_inserted += 1

    def get_latest_data(self, from_id: int) -> List[Any]:
        batch, _last, _gap = self.peek_batch_after(from_id, max_batch=None)
        return batch

    def peek_batch_after(
        self, from_id: int, max_batch: Optional[int], *, allow_gap: bool = True
    ) -> Tuple[List[Any], int, Optional[Dict[str, int]]]:
        """Return (batch, last_id_in_batch, gap_info).

        Empty batch → last_id=from_id, gap_info=None.

        When capacity eviction dropped ids the server still points at, ``allow_gap``
        (default True) skips the hole and sends from the local queue head so the
        Learner cursor can jump forward. Set ``allow_gap=False`` to raise instead.
        """

        with self._lock:
            if not self._seq_id_queue or from_id >= self.latest_seq_id:
                return [], int(from_id), None
            first = int(self._seq_id_queue[0])
            gap_info: Optional[Dict[str, int]] = None
            if int(from_id) + 1 < first:
                if not allow_gap:
                    raise RuntimeError(
                        f"upload queue gap: server_from_id={from_id} local_first={first} "
                        f"(unacked rows were dropped)"
                    )
                gap_info = {
                    "server_from_id": int(from_id),
                    "local_first": first,
                    "skipped": int(first - int(from_id) - 1),
                }
                start_idx = 0
            else:
                start_idx = max(0, int(from_id) - first + 1)
            if start_idx >= len(self._data_queue):
                return [], int(from_id), gap_info
            end_idx = len(self._data_queue)
            if max_batch is not None:
                end_idx = min(end_idx, start_idx + int(max_batch))
            if end_idx <= start_idx:
                return [], int(from_id), gap_info
            batch = list(self._data_queue)[start_idx:end_idx]
            last_id = int(self._seq_id_queue[end_idx - 1])
            return batch, last_id, gap_info

    def discard_through(self, seq_id: int) -> int:
        """Drop all items with id <= seq_id. Returns number dropped."""

        with self._lock:
            dropped = 0
            while self._seq_id_queue and self._seq_id_queue[0] <= int(seq_id):
                self._seq_id_queue.popleft()
                self._data_queue.popleft()
                dropped += 1
            self._total_discarded += dropped
            return dropped

    def pending(self) -> int:
        with self._lock:
            return len(self._data_queue)

    def align_to_server_id(self, server_id: int) -> int:
        """Continue local ids after Learner's last_update_id (Actor-only restart safe).

        Clears any pending rows from a previous process identity. Next ``insert``
        uses ``server_id + 1``. Returns the aligned latest id.
        """

        sid = int(server_id)
        with self._lock:
            self._seq_id_queue.clear()
            self._data_queue.clear()
            self.latest_seq_id = sid
            return int(self.latest_seq_id)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "pending": len(self._data_queue),
                "latest_seq_id": int(self.latest_seq_id),
                "total_inserted": int(self._total_inserted),
                "total_discarded": int(self._total_discarded),
                "dropped_unacked": int(self._dropped_unacked),
                "capacity": int(self.capacity),
            }

    def __len__(self) -> int:
        return self.pending()


def align_stores_to_server(
    client: Any,
    stores: Mapping[str, DrainingQueuedDataStore],
) -> Dict[str, int]:
    """Seed each local queue cursor from TrainerServer last_update_id.

    Call once after a successful handshake so a new Actor process can append
    without restarting the Learner. Missing server ids default to -1.
    """

    aligned: Dict[str, int] = {}
    for name, store in stores.items():
        sid = client.get_server_last_update_id(name)
        if sid is None:
            sid = -1
        aligned[name] = store.align_to_server_id(int(sid))
    return aligned


def upload_datastores(
    client: Any,
    stores: Mapping[str, DrainingQueuedDataStore],
    *,
    max_batch: int = DEFAULT_MAX_BATCH,
) -> Tuple[bool, Dict[str, Any]]:
    """Send <= max_batch rows per store with correct last_id, then drain acked ids.

    Control thread must not call this; run on the upload worker only.

    Capacity eviction may leave a hole vs Learner ``last_update_id``. Those gaps
    are skipped (cursor jumps); dropped rows are gone but newer pending still
    upload so the Actor is not fail-closed solely for backpressure.
    """

    report: Dict[str, Any] = {"stores": {}, "gap_skips": []}
    all_ok = True
    store_items = list(stores.items())
    for idx, (name, store) in enumerate(store_items):
        from_id = client.get_server_last_update_id(name)
        if from_id is None:
            all_ok = False
            report["stores"][name] = {"ok": False, "error": "no_server_last_id"}
            continue
        batch, last_id, gap_info = store.peek_batch_after(
            int(from_id), int(max_batch), allow_gap=True
        )
        if gap_info is not None:
            report["gap_skips"].append({"store": name, **gap_info})
        if not batch:
            report["stores"][name] = {
                "ok": True,
                "sent": 0,
                "drained": 0,
                "from_id": int(from_id),
                "pending": store.pending(),
                "gap_skip": gap_info,
            }
            if idx + 1 < len(store_items) and UPLOAD_GIL_YIELD_S > 0:
                time.sleep(float(UPLOAD_GIL_YIELD_S))
            continue
        payload = {"data": batch, "last_id": int(last_id)}
        res = client._update_ds(name, payload)
        ok = bool(res is not None and res.get("success"))
        drained = 0
        if ok:
            # Confirm server advanced at least through this batch.
            server_id = client.get_server_last_update_id(name)
            if server_id is None:
                ok = False
            else:
                # Drain through the batch we sent (not necessarily full latest).
                drain_to = min(int(last_id), int(server_id))
                drained = store.discard_through(drain_to)
                if int(server_id) < int(last_id):
                    ok = False
        if not ok:
            all_ok = False
        report["stores"][name] = {
            "ok": ok,
            "sent": len(batch),
            "drained": drained,
            "from_id": int(from_id),
            "last_id": int(last_id),
            "pending": store.pending(),
            "dropped_unacked": store.stats()["dropped_unacked"],
            "gap_skip": gap_info,
        }
        if idx + 1 < len(store_items) and UPLOAD_GIL_YIELD_S > 0:
            time.sleep(float(UPLOAD_GIL_YIELD_S))
    report["ok"] = all_ok
    return all_ok, report


def confirm_upload_by_last_id(
    client: Any,
    stores: Mapping[str, DrainingQueuedDataStore],
) -> Tuple[bool, Dict[str, Any]]:
    """Ack check via last_update_id. Do not compare pending len to Learner replay len."""

    import time

    started = time.perf_counter()
    try:
        status = client.request("r13-status", {})
    except Exception as exc:  # noqa: BLE001
        return False, {"ok": False, "error": f"r13-status exception: {exc}"}
    rtt_ms = (time.perf_counter() - started) * 1000.0
    if status is None:
        return False, {"ok": False, "error": "r13-status timeout/None", "status_rtt_ms": rtt_ms}
    payload = status.get("payload", status) if isinstance(status, dict) else {}
    if isinstance(status, dict) and "success" in status and not status.get("success"):
        return False, {
            "ok": False,
            "error": status.get("message", "r13-status unsuccessful"),
            "status_rtt_ms": rtt_ms,
        }

    report: Dict[str, Any] = {
        "ok": True,
        "status_rtt_ms": rtt_ms,
        "server_env_count": payload.get("actor_env_count", payload.get("ONLINE_N")),
        "server_intvn_count": payload.get("actor_env_intvn_count", payload.get("INTVN_N")),
        "server_instance_id": payload.get("server_instance_id"),
        "handshake_accepted": payload.get("handshake_accepted"),
        "publish_count": payload.get("PUBLISH_COUNT", payload.get("publish_count")),
        "nan": payload.get("NAN_OR_INF", payload.get("nan")),
        "learner_step": payload.get("learner_step"),
        "last_update_id": {},
        "pending": {},
    }
    for name, store in stores.items():
        sid = client.get_server_last_update_id(name)
        report["last_update_id"][name] = sid
        report["pending"][name] = store.pending()
        if sid is None:
            report["ok"] = False
            report["error"] = f"missing last_update_id for {name}"
            continue
        # Server must not be ahead of what we have produced.
        if int(sid) > int(store.latest_data_id()):
            report["ok"] = False
            report["error"] = f"server ahead of client for {name}"
    return bool(report["ok"]), report
