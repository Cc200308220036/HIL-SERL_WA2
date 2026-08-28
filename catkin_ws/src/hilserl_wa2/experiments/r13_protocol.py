"""R13 handshake, NaN guards, and stats helpers. No ROS / JAX imports."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from hilserl_wa2.experiments.r10_protocol import (
    AUTO_RECONNECT,
    compare_handshake as compare_r10_handshake,
    make_r10_trainer_config,
    sha256_file,
)

PROTOCOL_VERSION = "wa2-r13-timescale-v2"
TRANSITION_SCHEMA_VERSION = "r13-timescale-v2"
ACTION_DIM = 7
END_EPISODE = True
REQUEST_TYPES = ("send-stats", "r13-handshake", "r13-status", "r13-ping")
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
    "demo_pkl_sha256",
    "action_dim",
    "end_episode",
    "action_scale",
    "policy_hz",
    "servo_hz",
    "servo_ticks_per_action",
    "discount",
    "classifier_consecutive_n",
)


class R13ProtocolError(RuntimeError):
    pass


def make_r13_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    from agentlace.trainer import TrainerConfig

    return TrainerConfig(
        port_number=int(port_number),
        broadcast_port=int(broadcast_port),
        request_types=list(REQUEST_TYPES),
        version=PROTOCOL_VERSION,
    )


def build_handshake_request(manifest: Mapping[str, Any], session_id: str) -> Dict[str, Any]:
    missing = [key for key in HANDSHAKE_KEYS if key not in manifest]
    if missing:
        raise R13ProtocolError(f"manifest missing handshake keys: {missing}")
    if not isinstance(session_id, str) or not session_id.strip():
        raise R13ProtocolError("session_id must be a non-empty string")
    payload = {key: manifest[key] for key in HANDSHAKE_KEYS}
    payload["session_id"] = session_id
    return payload


def compare_handshake(expected: Mapping[str, Any], received: Mapping[str, Any]) -> Dict[str, Any]:
    mismatches: Dict[str, Any] = {}
    for key in HANDSHAKE_KEYS:
        left = expected.get(key)
        right = received.get(key)
        if key in {"action_dim", "end_episode"}:
            if _canon(left) != _canon(right):
                mismatches[key] = {"expected": left, "received": right}
            continue
        if key in {"action_scale", "policy_hz", "servo_hz", "discount"}:
            try:
                if abs(float(left) - float(right)) > 1e-6:
                    mismatches[key] = {"expected": left, "received": right}
            except (TypeError, ValueError):
                mismatches[key] = {"expected": left, "received": right}
            continue
        if left != right:
            mismatches[key] = {"expected": left, "received": right}
    session_id = received.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        mismatches["session_id"] = {"expected": "non-empty string", "received": session_id}
    if str(received.get("protocol_version") or "") != PROTOCOL_VERSION:
        mismatches["protocol_version"] = {
            "expected": PROTOCOL_VERSION,
            "received": received.get("protocol_version"),
        }
    return {
        "accepted": not mismatches,
        "mismatches": mismatches,
        "session_id": session_id,
    }


def _canon(value: Any) -> Any:
    if isinstance(value, bool) or value in ("true", "false", True, False):
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def tree_has_nan_or_inf(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        return any(tree_has_nan_or_inf(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(tree_has_nan_or_inf(v) for v in obj)
    try:
        arr = np.asarray(obj)
    except Exception:
        return False
    if arr.dtype == object:
        return any(tree_has_nan_or_inf(v) for v in arr.reshape(-1))
    if np.issubdtype(arr.dtype, np.number):
        return bool(np.any(~np.isfinite(arr)))
    return False


def update_info_has_nan(info: Mapping[str, Any]) -> bool:
    for key, value in info.items():
        if "loss" in str(key).lower() or "grad" in str(key).lower() or "q" in str(key).lower():
            if tree_has_nan_or_inf(value):
                return True
    return tree_has_nan_or_inf(info)


def scale_arm_action(action: np.ndarray, scale: float) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    scale = float(scale)
    if scale < 0:
        raise R13ProtocolError("action_scale must be >= 0")
    n = min(ARM_LIMIT, arr.shape[0])
    arr[:n] = np.clip(arr[:n] * scale, -1.0, 1.0)
    return arr


ARM_LIMIT = 6


def confirm_r13_server_status(
    client: Any,
    *,
    local_env: int,
    local_intvn: int,
) -> Dict[str, Any]:
    """Upload check via r13-status. Demo-file rows are excluded from intvn count."""

    import time

    started = time.perf_counter()
    try:
        status = client.request("r13-status", {})
    except Exception as exc:
        return {"ok": False, "error": f"r13-status exception: {exc}"}
    rtt_ms = (time.perf_counter() - started) * 1000.0
    if status is None:
        return {"ok": False, "error": "r13-status timeout/None", "status_rtt_ms": rtt_ms}
    payload = status.get("payload", status) if isinstance(status, dict) else {}
    if isinstance(status, dict) and "success" in status and not status.get("success"):
        return {
            "ok": False,
            "error": status.get("message", "r13-status unsuccessful"),
            "status_rtt_ms": rtt_ms,
            "raw": status,
        }
    env_count = payload.get("actor_env_count", payload.get("ONLINE_N"))
    intvn_count = payload.get("actor_env_intvn_count", payload.get("INTVN_N"))
    report = {
        "ok": True,
        "status_rtt_ms": rtt_ms,
        "server_env_count": env_count,
        "server_intvn_count": intvn_count,
        "server_instance_id": payload.get("server_instance_id"),
        "handshake_accepted": payload.get("handshake_accepted"),
        "publish_count": payload.get("PUBLISH_COUNT", payload.get("publish_count")),
        "nan": payload.get("NAN_OR_INF", payload.get("nan")),
        "learner_step": payload.get("learner_step"),
    }
    if env_count is None or intvn_count is None:
        report["ok"] = False
        report["error"] = "missing store counts"
        return report
    if int(env_count) != int(local_env) or int(intvn_count) != int(local_intvn):
        report["ok"] = False
        report["error"] = "store count mismatch"
    return report


# Re-export a few R10 helpers used by manifests.
sha256_file = sha256_file
compare_r10_handshake = compare_r10_handshake
make_r10_trainer_config = make_r10_trainer_config
AUTO_RECONNECT = AUTO_RECONNECT
