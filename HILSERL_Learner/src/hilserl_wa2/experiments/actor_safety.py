"""R9 Actor upload / network watchdog and fail-closed lifecycle (no JAX/ROS import)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

R9_REQUEST_TYPES = ("send-stats", "r9-status", "r9-publish-params")
TELEOP_PATTERNS = (
    "spacemouse_wa2_teleop",
    "run_spacemouse_teleop_from_yaml",
)
LIVE_MODES = ("live-zero",)
HUMAN_MODES = ("readonly", "live-zero")


class ActorSafetyError(RuntimeError):
    """R9 safety gate refused to start or continue."""


def make_r9_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    from agentlace.trainer import TrainerConfig

    return TrainerConfig(
        port_number=int(port_number),
        broadcast_port=int(broadcast_port),
        request_types=list(R9_REQUEST_TYPES),
    )


def teleop_pids() -> List[str]:
    hits: List[str] = []
    pattern = "|".join(TELEOP_PATTERNS)
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", pattern],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    for line in out.splitlines():
        if any(name in line for name in TELEOP_PATTERNS):
            if "pgrep" in line:
                continue
            hits.append(line.strip())
    return hits


def assert_no_teleop() -> None:
    hits = teleop_pids()
    if hits:
        raise ActorSafetyError(
            "teleop script is running; stop it before R9 Actor:\n" + "\n".join(hits)
        )


def assert_live_policy(mode: str, policy: str) -> None:
    if mode in LIVE_MODES and policy != "zero":
        raise ActorSafetyError(
            f"mode={mode} forbids policy={policy!r}; untrained SAC must not move the arm"
        )


def assert_r4_confirm(mode: str) -> None:
    if mode in LIVE_MODES and os.environ.get("R4_CONFIRM") != "YES":
        raise ActorSafetyError("live-zero requires R4_CONFIRM=YES")


def assert_r13_hardware_confirm(mode: str) -> None:
    """R13 live training / eval must have R4_CONFIRM; fake does not.

    Do not fold ``live`` into ``LIVE_MODES``: that list also vetoes SAC.
    """

    if mode in ("live", "eval") and os.environ.get("R4_CONFIRM") != "YES":
        raise ActorSafetyError(f"R13 {mode} requires R4_CONFIRM=YES")


def unwrap_env(env: Any) -> Any:
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ == "WA2Env":
            return cur
        inner = getattr(cur, "unwrapped", None)
        if inner is not None and inner is not cur:
            cur = inner
            continue
        inner = getattr(cur, "env", None)
        if inner is None or inner is cur:
            return cur
        cur = inner
    return env


def find_wrapper(env: Any, class_name: str) -> Optional[Any]:
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ == class_name:
            return cur
        inner = getattr(cur, "env", None)
        if inner is None or inner is cur:
            return None
        cur = inner
    return None


def extract_servo_health(env: Any) -> Optional[Dict[str, Any]]:
    base = unwrap_env(env)
    servo = getattr(base, "_servo", None)
    if servo is None or not hasattr(servo, "health"):
        return None
    try:
        return dict(servo.health())
    except Exception:
        return None


@dataclass
class UploadWatchdog:
    max_consecutive_failures: int = 1
    consecutive_failures: int = 0
    attempts: int = 0
    successes: int = 0

    def record(self, ok: bool) -> Optional[str]:
        self.attempts += 1
        if ok:
            self.consecutive_failures = 0
            self.successes += 1
            return None
        self.consecutive_failures += 1
        if self.consecutive_failures >= int(self.max_consecutive_failures):
            return "server_disconnect"
        return None


@dataclass
class NetworkWatchdog:
    max_age_s: float = 5.0
    enabled: bool = False
    update_count: int = 0
    last_monotonic: Optional[float] = None
    last_signature: Optional[str] = None

    def note_update(self, signature: Optional[str] = None) -> None:
        self.update_count += 1
        self.last_monotonic = time.monotonic()
        if signature is not None:
            self.last_signature = signature

    def check(self) -> Optional[str]:
        if not self.enabled:
            return None
        if self.last_monotonic is None:
            return "network_stale"
        age = time.monotonic() - self.last_monotonic
        if age > float(self.max_age_s):
            return "network_stale"
        return None

    @property
    def age_s(self) -> Optional[float]:
        if self.last_monotonic is None:
            return None
        return time.monotonic() - self.last_monotonic


@dataclass
class MotionBudget:
    max_translation_m: float = 0.020
    max_rotation_deg: float = 2.0
    translation_m: float = 0.0
    rotation_deg: float = 0.0
    motion_without_intervention: bool = False

    def note(self, info: Dict[str, Any], intervened: bool) -> Optional[str]:
        dp = float(info.get("delta_pos_m") or 0.0)
        dr = float(info.get("delta_rot_rad") or 0.0)
        self.translation_m += max(0.0, dp)
        self.rotation_deg += abs(dr) * 180.0 / 3.141592653589793
        published = bool(info.get("published"))
        if published and dp > 1e-5 and not intervened:
            self.motion_without_intervention = True
            return "motion_without_intervention"
        if self.translation_m > self.max_translation_m + 1e-9:
            return "motion_budget_translation"
        if self.rotation_deg > self.max_rotation_deg + 1e-9:
            return "motion_budget_rotation"
        return None


@dataclass
class FailClosedResult:
    reason: Optional[str] = None
    triggered: bool = False
    env_closed: bool = False
    client_stopped: bool = False
    stop_ok: Optional[bool] = None
    clear_ok: Optional[bool] = None
    steps_executed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class FailClosedController:
    def __init__(self) -> None:
        self.result = FailClosedResult()

    def trigger(self, reason: str) -> None:
        if not self.result.triggered:
            self.result.triggered = True
            self.result.reason = str(reason)

    def shutdown(self, env: Any = None, client: Any = None) -> FailClosedResult:
        if env is not None:
            try:
                env.close()
                self.result.env_closed = True
            except Exception as exc:
                self.result.extra["env_close_error"] = str(exc)
            health = extract_servo_health(env)
            if health is not None:
                self.result.stop_ok = bool(health.get("stop_ok"))
                self.result.clear_ok = bool(health.get("clear_ok"))
                self.result.extra["servo_health"] = health
        if client is not None:
            try:
                client.stop()
                self.result.client_stopped = True
            except Exception as exc:
                self.result.extra["client_stop_error"] = str(exc)
        else:
            self.result.client_stopped = True
        return self.result


def confirm_server_counts(
    client: Any,
    *,
    local_env: int,
    local_intvn: int,
    client_env_id: Optional[int] = None,
    client_intvn_id: Optional[int] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Authoritative upload check: Server store counts / last_update_id, not update() bool."""

    status = None
    try:
        status = client.request("r9-status", {})
    except Exception as exc:
        return False, {"error": f"r9-status exception: {exc}"}
    if status is None:
        return False, {"error": "r9-status timeout/None"}

    payload = status.get("payload", status) if isinstance(status, dict) else {}
    if isinstance(status, dict) and "success" in status and not status.get("success"):
        return False, {"error": status.get("message", "r9-status unsuccessful"), "raw": status}

    env_count = payload.get("actor_env_count")
    intvn_count = payload.get("actor_env_intvn_count")
    last_ids = payload.get("last_update_id") or {}
    report = {
        "server_env_count": env_count,
        "server_intvn_count": intvn_count,
        "last_update_id": last_ids,
        "local_env": local_env,
        "local_intvn": local_intvn,
    }
    if env_count is None or intvn_count is None:
        return False, {**report, "error": "missing store counts"}
    if int(env_count) != int(local_env) or int(intvn_count) != int(local_intvn):
        return False, {**report, "error": "store count mismatch"}

    env_id = last_ids.get("actor_env")
    intvn_id = last_ids.get("actor_env_intvn")
    if env_id is None or intvn_id is None:
        report["last_update_id_fallback"] = "monotonic_store_length"
        return True, report
    if client_env_id is not None and int(env_id) != int(client_env_id):
        return False, {**report, "error": "actor_env last_update_id mismatch"}
    if client_intvn_id is not None and int(local_intvn) > 0 and int(intvn_id) != int(client_intvn_id):
        return False, {**report, "error": "actor_env_intvn last_update_id mismatch"}
    report["last_update_id_match"] = True
    return True, report


def _flatten_params(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    if isinstance(obj, dict) or (hasattr(obj, "items") and hasattr(obj, "keys")):
        try:
            items = list(obj.items())
        except Exception:
            return [(prefix, obj)]
        out: List[Tuple[str, Any]] = []
        for key, value in items:
            out.extend(_flatten_params(value, f"{prefix}/{key}"))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for idx, value in enumerate(obj):
            out.extend(_flatten_params(value, f"{prefix}/{idx}"))
        return out
    return [(prefix, obj)]


def params_tree_signature(params: Any) -> str:
    import numpy as np

    meta = []
    for key, leaf in _flatten_params(params):
        try:
            arr = np.asarray(leaf)
            meta.append([key, list(arr.shape), str(arr.dtype)])
        except Exception:
            meta.append([key, str(type(leaf))])
    meta.sort(key=lambda row: str(row[0]))
    payload = json.dumps({"n": len(meta), "meta": meta}, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def host_params_tree(params: Any) -> Any:
    import numpy as np

    try:
        import jax

        return jax.tree_util.tree_map(
            lambda x: np.asarray(jax.device_get(x)),
            params,
        )
    except Exception:
        return params
