"""Reusable R10 Actor/Learner protocol helpers (no ROS/JAX imports)."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

PROTOCOL_VERSION = "wa2-r10-v1"
TRANSITION_SCHEMA_VERSION = "r9-v1"
REQUEST_TYPES = ("send-stats", "r10-handshake", "r10-status", "r10-ping")
HANDSHAKE_KEYS = (
    "protocol_version",
    "task_id",
    "exp_name",
    "config_bundle_hash",
    "network_config_hash",
    "space_hash",
    "params_tree_signature",
    "agentlace_version",
    "agentlace_wheel_sha256",
    "source_tree_sha256",
    "transition_schema_version",
)


class R10ProtocolError(RuntimeError):
    pass


FORBIDDEN_LEARNER_IPS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
NETWORK_FAULT_DETAILS = {"server_disconnect", "network_stale"}
AUTO_RECONNECT = False


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_network_config(path: Path | str) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise R10ProtocolError("network config must be a mapping")
    required = {
        "schema_version",
        "profile_id",
        "learner_ip",
        "actor_ip",
        "request_port",
        "broadcast_port",
        "request_timeout_ms",
        "network_max_age_s",
        "upload_every_steps",
        "allowed_actor_cidr",
    }
    missing = required - set(cfg)
    if missing:
        raise R10ProtocolError(f"network config missing keys: {sorted(missing)}")
    if int(cfg["request_port"]) == int(cfg["broadcast_port"]):
        raise R10ProtocolError("request and broadcast ports must differ")
    if not (1 <= int(cfg["request_port"]) <= 65535):
        raise R10ProtocolError("invalid request_port")
    if not (1 <= int(cfg["broadcast_port"]) <= 65535):
        raise R10ProtocolError("invalid broadcast_port")
    return cfg


def network_config_hash(path: Path | str) -> str:
    return canonical_json_sha256(load_network_config(path))


def make_r10_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    from agentlace.trainer import TrainerConfig

    return TrainerConfig(
        port_number=int(port_number),
        broadcast_port=int(broadcast_port),
        request_types=list(REQUEST_TYPES),
        version=PROTOCOL_VERSION,
    )


def compare_handshake(expected: Mapping[str, Any], received: Mapping[str, Any]) -> Dict[str, Any]:
    mismatches: Dict[str, Any] = {}
    for key in HANDSHAKE_KEYS:
        left = expected.get(key)
        right = received.get(key)
        if left != right:
            mismatches[key] = {"expected": left, "received": right}
    session_id = received.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        mismatches["session_id"] = {"expected": "non-empty string", "received": session_id}
    return {
        "accepted": not mismatches,
        "mismatches": mismatches,
        "session_id": session_id,
    }


def _hash_value(digest: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, Mapping):
        digest.update(b"dict\0")
        for key in sorted(value, key=str):
            digest.update(str(key).encode("utf-8") + b"\0")
            _hash_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"seq\0")
        for item in value:
            _hash_value(digest, item)
        return
    arr = np.asarray(value)
    digest.update(str(arr.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(arr.shape)).encode("ascii") + b"\0")
    digest.update(np.ascontiguousarray(arr).tobytes())


def transition_sha256(transition: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, transition)
    return digest.hexdigest()


def ordered_transition_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(bytes.fromhex(transition_sha256(row)))
    return digest.hexdigest()


def _exclude_source_file(rel_posix: str) -> bool:
    """§3.4: skip machine-local YAML and bytecode; IP must not enter source_tree."""

    parts = Path(rel_posix).parts
    if "__pycache__" in parts:
        return True
    name = Path(rel_posix).name.lower()
    if Path(rel_posix).suffix.lower() in {".yaml", ".yml"} and "local" in name:
        return True
    return False


def source_tree_manifest(repo: Path | str) -> Tuple[Sequence[Dict[str, str]], str]:
    """Hash only Actor/Learner semantic sources shared by both deployments."""

    root = Path(repo).resolve()
    includes = (
        root / "src" / "hilserl_wa2" / "configs",
        root / "src" / "hilserl_wa2" / "envs",
        root / "src" / "hilserl_wa2" / "experiments",
        root / "src" / "hilserl_wa2" / "wrappers",
        root / "src" / "hil-serl-main" / "serl_launcher" / "serl_launcher",
        root / "src" / "hil-serl-main" / "examples" / "experiments" / "wa2",
    )
    singles = (
        root / "src" / "hil-serl-main" / "examples" / "experiments" / "config.py",
        root / "src" / "hil-serl-main" / "examples" / "experiments" / "mappings.py",
    )
    files = []
    for base in includes:
        if not base.is_dir():
            raise R10ProtocolError(f"source directory missing: {base}")
        files.extend(p for p in base.rglob("*") if p.is_file())
    files.extend(p for p in singles if p.is_file())
    selected = []
    for path in files:
        if path.suffix not in {".py", ".yaml", ".yml"}:
            continue
        rel = path.resolve().relative_to(root).as_posix()
        if _exclude_source_file(rel):
            continue
        selected.append(path.resolve())
    selected = sorted(set(selected), key=lambda p: p.relative_to(root).as_posix())
    rows = [
        {"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p)}
        for p in selected
    ]
    return rows, canonical_json_sha256(rows)


def make_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def is_forbidden_learner_ip(ip: str) -> bool:
    value = str(ip).strip().lower()
    if not value or value.startswith("replace_with"):
        return True
    if value in FORBIDDEN_LEARNER_IPS or value.startswith("127."):
        return True
    if value.startswith("172.17."):
        return True
    return False


def assert_remote_learner_ip(ip: str) -> str:
    value = str(ip).strip()
    if is_forbidden_learner_ip(value):
        raise R10ProtocolError(
            f"R10 Actor rejects loopback/placeholder/docker-bridge server-ip: {ip!r}"
        )
    return value


def assert_network_endpoints(server_ip: str, cfg: Mapping[str, Any]) -> str:
    learner = assert_remote_learner_ip(server_ip)
    profile_ip = assert_remote_learner_ip(cfg.get("learner_ip", ""))
    actor_ip = assert_remote_learner_ip(cfg.get("actor_ip", ""))
    if learner != profile_ip:
        raise R10ProtocolError(
            f"--server-ip {learner} != local.yaml learner_ip {profile_ip}"
        )
    if learner == actor_ip:
        raise R10ProtocolError("learner_ip and actor_ip must differ")
    return learner


def build_handshake_request(manifest: Mapping[str, Any], session_id: str) -> Dict[str, Any]:
    missing = [key for key in HANDSHAKE_KEYS if key not in manifest]
    if missing:
        raise R10ProtocolError(f"manifest missing handshake keys: {missing}")
    if not isinstance(session_id, str) or not session_id.strip():
        raise R10ProtocolError("session_id must be a non-empty string")
    payload = {key: manifest[key] for key in HANDSHAKE_KEYS}
    payload["session_id"] = session_id
    return payload


def normalize_network_fault(detail: str) -> Tuple[str, str]:
    value = str(detail)
    if value in NETWORK_FAULT_DETAILS:
        return "network_loss", value
    if value == "network_loss":
        return "network_loss", "server_disconnect"
    return value, value


def count_intervention_segments(flags: Iterable[bool]) -> int:
    segments = 0
    prev = False
    for flag in flags:
        current = bool(flag)
        if current and not prev:
            segments += 1
        prev = current
    return segments


def rtt_stats_ms(samples: Sequence[float]) -> Dict[str, float]:
    if not samples:
        return {"min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
    arr = sorted(float(item) for item in samples)

    def pct(p: float) -> float:
        if len(arr) == 1:
            return arr[0]
        idx = min(len(arr) - 1, max(0, int(round((p / 100.0) * (len(arr) - 1)))))
        return arr[idx]

    return {
        "min": arr[0],
        "median": pct(50),
        "p95": pct(95),
        "max": arr[-1],
        "n": len(arr),
    }


def estimate_wire_bytes(rows: Sequence[Mapping[str, Any]]) -> int:
    return len(pickle.dumps(list(rows), protocol=pickle.HIGHEST_PROTOCOL))


@dataclass
class R10SessionGuard:
    """Same-process session is valid; Server restart invalidates and forbids auto-replay."""

    session_id: str
    server_instance_id: Optional[str] = None
    handshake_accepted: bool = False
    env_steps: int = 0
    invalidated: bool = False
    fault_reason: Optional[str] = None
    fault_detail: Optional[str] = None
    ledger: List[Dict[str, Any]] = field(default_factory=list)
    dump_written: bool = False
    auto_reconnect: bool = AUTO_RECONNECT

    def note_handshake(self, accepted: bool, server_instance_id: Optional[str] = None) -> bool:
        self.handshake_accepted = bool(accepted)
        if accepted:
            if server_instance_id:
                self.server_instance_id = str(server_instance_id)
            return True
        self.invalidated = True
        return False

    def can_step(self) -> bool:
        return (
            self.handshake_accepted
            and not self.invalidated
            and self.fault_reason is None
            and not self.auto_reconnect
        )

    def note_env_step(self) -> int:
        if not self.can_step():
            raise R10ProtocolError("ENV_STEPS=0: handshake failed or session invalidated")
        self.env_steps += 1
        return self.env_steps

    def register_transport(self, stream: str, sequence_id: int, content_sha256: str) -> Dict[str, Any]:
        record = {
            "session_id": self.session_id,
            "stream": str(stream),
            "sequence_id": int(sequence_id),
            "content_sha256": str(content_sha256),
        }
        self.ledger.append(record)
        return record

    def note_server_instance(self, server_instance_id: str) -> None:
        incoming = str(server_instance_id)
        if not incoming:
            return
        if self.server_instance_id is None:
            self.server_instance_id = incoming
            return
        if incoming != self.server_instance_id:
            self.invalidated = True
            self.handshake_accepted = False
            self.trigger_network_loss("server_disconnect")

    def trigger_network_loss(self, detail: str) -> Tuple[str, str]:
        reason, normalized = normalize_network_fault(detail)
        if self.fault_reason is None:
            self.fault_reason = reason
            self.fault_detail = normalized
        return self.fault_reason, self.fault_detail

    def write_fault_dump(
        self,
        dump_dir: Path | str,
        transitions: Sequence[Mapping[str, Any]],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "server_instance_id": self.server_instance_id,
            "ledger": list(self.ledger),
            "transitions": list(transitions),
            "fault_reason": self.fault_reason,
            "fault_detail": self.fault_detail,
            "env_steps": self.env_steps,
            "auto_reconnect": False,
        }
        if extra:
            payload.update(dict(extra))
        with (path / "fault_dump.pkl").open("wb") as handle:
            pickle.dump(payload, handle)
        meta = {key: value for key, value in payload.items() if key != "transitions"}
        (path / "fault_meta.json").write_text(
            json.dumps(meta, indent=2, default=str, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.dump_written = True
        return path

    @staticmethod
    def refuse_old_dump_import(old_session_id: str, new_session_id: str) -> None:
        raise R10ProtocolError(
            "R10 does not auto-import old dump "
            f"{old_session_id} into new session {new_session_id}"
        )


def confirm_r10_server_status(
    client: Any,
    *,
    local_env: int,
    local_intvn: int,
    local_env_digest: str,
    local_intvn_digest: str,
    client_env_id: Optional[int] = None,
    client_intvn_id: Optional[int] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Authoritative upload check via r10-status, not client.update() bool."""

    started = time.perf_counter()
    try:
        status = client.request("r10-status", {})
    except Exception as exc:
        return False, {"error": f"r10-status exception: {exc}"}
    rtt_ms = (time.perf_counter() - started) * 1000.0
    if status is None:
        return False, {"error": "r10-status timeout/None", "status_rtt_ms": rtt_ms}

    payload = status.get("payload", status) if isinstance(status, dict) else {}
    if isinstance(status, dict) and "success" in status and not status.get("success"):
        return False, {
            "error": status.get("message", "r10-status unsuccessful"),
            "raw": status,
            "status_rtt_ms": rtt_ms,
        }

    env_count = payload.get("actor_env_count")
    intvn_count = payload.get("actor_env_intvn_count")
    last_ids = payload.get("last_update_id") or {}
    server_digest = payload.get("ordered_digest")
    server_intvn_digest = payload.get("ordered_intvn_digest")
    report: Dict[str, Any] = {
        "server_env_count": env_count,
        "server_intvn_count": intvn_count,
        "last_update_id": last_ids,
        "local_env": local_env,
        "local_intvn": local_intvn,
        "status_rtt_ms": rtt_ms,
        "server_instance_id": payload.get("server_instance_id"),
        "schema_ok": payload.get("schema_ok"),
        "ordered_digest_match": None,
        "ordered_intvn_digest_match": None,
        "last_update_id_match": False,
    }
    if payload.get("schema_ok") is False:
        return False, {**report, "error": "schema_ok=false (data before handshake or invalid)"}
    if env_count is None or intvn_count is None:
        return False, {**report, "error": "missing store counts"}
    if int(env_count) != int(local_env) or int(intvn_count) != int(local_intvn):
        return False, {**report, "error": "store count mismatch"}
    if server_digest:
        report["ordered_digest_match"] = server_digest == local_env_digest
        if not report["ordered_digest_match"]:
            return False, {**report, "error": "ordered_digest mismatch"}
    if server_intvn_digest:
        report["ordered_intvn_digest_match"] = server_intvn_digest == local_intvn_digest
        if not report["ordered_intvn_digest_match"]:
            return False, {**report, "error": "ordered_intvn_digest mismatch"}

    env_id = last_ids.get("actor_env")
    intvn_id = last_ids.get("actor_env_intvn")
    if env_id is None or intvn_id is None:
        report["last_update_id_fallback"] = "monotonic_store_length"
        report["last_update_id_match"] = True
        return True, report
    if client_env_id is not None and int(env_id) != int(client_env_id):
        return False, {**report, "error": "actor_env last_update_id mismatch"}
    if client_intvn_id is not None and int(local_intvn) > 0 and int(intvn_id) != int(client_intvn_id):
        return False, {**report, "error": "actor_env_intvn last_update_id mismatch"}
    report["last_update_id_match"] = True
    return True, report

