"""Read-only WA2 state monitor: subscribe, cache, age, copy-out. No commands."""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

LEFT_JOINT_NAMES = [
    "Chest_Z_L",
    "Shoulder_Y_L",
    "Shoulder_X_L",
    "Shoulder_Z_L",
    "Elbow_L",
    "Wrist_Z_L",
    "Wrist_Y_L",
    "Wrist_X_L",
]

LEFT_HAND_NAMES = [
    "THUMB_MP_LEFT",
    "THUMB_CMC_LEFT",
    "INDEX_MCP_LEFT",
    "MIDDLE_MCP_LEFT",
    "RING_MCP_LEFT",
    "LITTLE_MCP_LEFT",
]

STATE_KEYS = ("tcp_pose", "tcp_vel", "joint_pos", "hand_joints", "uplimb_state")


class StateCache:
    """Thread-safe monotonic-aged state cache (ROS-free, unit-testable)."""

    def __init__(self, state_max_age_s: float = 0.2):
        self.state_max_age_s = float(state_max_age_s)
        self._lock = threading.Lock()
        self._tcp_pose: Optional[np.ndarray] = None
        self._tcp_vel: Optional[np.ndarray] = None
        self._joint_pos: Optional[np.ndarray] = None
        self._hand_joints: Optional[np.ndarray] = None
        self._stamp: Dict[str, Optional[float]] = {k: None for k in STATE_KEYS}
        self._is_singular: Optional[bool] = None
        self._cmd_num: Optional[int] = None
        self._cmd_name: Optional[str] = None
        self._iddp_status: Optional[bool] = None
        # Test / fault-injection overrides: field -> forced age seconds
        self._forced_age: Dict[str, float] = {}

    def update_tcp_pose(self, pose7: Sequence[float], stamp: Optional[float] = None) -> None:
        arr = _normalize_pose(pose7)
        with self._lock:
            self._tcp_pose = arr
            self._stamp["tcp_pose"] = time.monotonic() if stamp is None else float(stamp)
            self._forced_age.pop("tcp_pose", None)

    def update_tcp_vel(self, vel6: Sequence[float], stamp: Optional[float] = None) -> None:
        arr = np.asarray(vel6, dtype=np.float32).reshape(6)
        if not np.all(np.isfinite(arr)):
            raise ValueError("tcp_vel must be finite")
        with self._lock:
            self._tcp_vel = arr.copy()
            self._stamp["tcp_vel"] = time.monotonic() if stamp is None else float(stamp)
            self._forced_age.pop("tcp_vel", None)

    def update_joint_pos(self, joints8: Sequence[float], stamp: Optional[float] = None) -> None:
        arr = np.asarray(joints8, dtype=np.float32).reshape(8)
        if not np.all(np.isfinite(arr)):
            raise ValueError("joint_pos must be finite")
        with self._lock:
            self._joint_pos = arr.copy()
            self._stamp["joint_pos"] = time.monotonic() if stamp is None else float(stamp)
            self._forced_age.pop("joint_pos", None)

    def update_hand_joints(self, hand6: Sequence[float], stamp: Optional[float] = None) -> None:
        arr = np.asarray(hand6, dtype=np.float32).reshape(6)
        if not np.all(np.isfinite(arr)):
            raise ValueError("hand_joints must be finite")
        with self._lock:
            self._hand_joints = arr.copy()
            self._stamp["hand_joints"] = time.monotonic() if stamp is None else float(stamp)
            self._forced_age.pop("hand_joints", None)

    def update_uplimb_state(
        self,
        *,
        is_singular: bool,
        cmd_num: int,
        cmd_name: str,
        iddp_status: bool,
        stamp: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._is_singular = bool(is_singular)
            self._cmd_num = int(cmd_num)
            self._cmd_name = str(cmd_name)
            self._iddp_status = bool(iddp_status)
            self._stamp["uplimb_state"] = (
                time.monotonic() if stamp is None else float(stamp)
            )
            self._forced_age.pop("uplimb_state", None)

    def get_state(self) -> Dict[str, np.ndarray]:
        with self._lock:
            if any(
                x is None
                for x in (
                    self._tcp_pose,
                    self._tcp_vel,
                    self._joint_pos,
                    self._hand_joints,
                )
            ):
                raise RuntimeError("state cache is not ready")
            return {
                "tcp_pose": self._tcp_pose.copy(),
                "tcp_vel": self._tcp_vel.copy(),
                "joint_pos": self._joint_pos.copy(),
                "hand_joints": self._hand_joints.copy(),
            }

    def get_ages(self) -> Dict[str, Optional[float]]:
        now = time.monotonic()
        with self._lock:
            ages: Dict[str, Optional[float]] = {}
            for key in ("tcp_pose", "tcp_vel", "joint_pos", "hand_joints", "uplimb_state"):
                if key in self._forced_age:
                    ages[key] = float(self._forced_age[key])
                    continue
                stamp = self._stamp[key]
                ages[key] = None if stamp is None else max(0.0, now - stamp)
            return ages

    def stale_fields(self) -> List[str]:
        ages = self.get_ages()
        stale = []
        for key in ("tcp_pose", "tcp_vel", "joint_pos", "hand_joints", "uplimb_state"):
            age = ages.get(key)
            if age is None or age > self.state_max_age_s:
                stale.append(key)
        return stale

    def is_fresh(self) -> bool:
        return len(self.stale_fields()) == 0

    def is_ready(self) -> bool:
        with self._lock:
            return all(
                x is not None
                for x in (
                    self._tcp_pose,
                    self._tcp_vel,
                    self._joint_pos,
                    self._hand_joints,
                    self._stamp["uplimb_state"],
                )
            )

    def get_info(self) -> Dict[str, object]:
        ages = self.get_ages()
        with self._lock:
            state_age = max(
                (ages[k] for k in ("tcp_pose", "tcp_vel", "joint_pos", "hand_joints") if ages[k] is not None),
                default=None,
            )
            return {
                "is_singular": self._is_singular,
                "cmd_num": self._cmd_num,
                "cmd_name": self._cmd_name,
                "iddp_status": self._iddp_status,
                "state_age": state_age,
                "ages": ages,
            }

    def inject_stale_for_test(
        self, fields: Optional[Sequence[str]] = None, age_s: float = 1.0
    ) -> None:
        fields = list(fields) if fields is not None else ["tcp_pose"]
        with self._lock:
            for field in fields:
                if field not in self._stamp:
                    raise ValueError(f"unknown field {field}")
                self._forced_age[field] = float(age_s)

    def clear_stale_injection(self) -> None:
        with self._lock:
            self._forced_age.clear()


class WA2StateMonitor:
    """ROS subscriber facade over :class:`StateCache`. Never publishes commands."""

    def __init__(
        self,
        arm: str = "left",
        state_max_age_s: float = 0.2,
        joint_names: Optional[Sequence[str]] = None,
        hand_names: Optional[Sequence[str]] = None,
        cache: Optional[StateCache] = None,
    ):
        if arm != "left":
            raise ValueError("R3 StateMonitor only supports arm='left'")
        self.arm = arm
        self.state_max_age_s = float(state_max_age_s)
        self._joint_names = list(joint_names or LEFT_JOINT_NAMES)
        self._hand_names = list(hand_names or LEFT_HAND_NAMES)
        self.cache = cache or StateCache(state_max_age_s=self.state_max_age_s)
        self._started = False
        self._subs = []
        self._rospy = None

    def left_joint_names(self) -> List[str]:
        return list(self._joint_names)

    def left_hand_names(self) -> List[str]:
        return list(self._hand_names)

    def start(self) -> None:
        if self._started:
            return
        import rospy
        from sensor_msgs.msg import JointState
        from upperlimb.msg import Pose as UplimbPose
        from upperlimb.msg import TcpSpeed, UplimbState

        self._rospy = rospy
        if not rospy.core.is_initialized():
            rospy.init_node("wa2_state_monitor", anonymous=True, disable_signals=True)

        self._subs = [
            rospy.Subscriber(
                "/zj_humanoid/upperlimb/tcp_pose/left_arm",
                UplimbPose,
                self._tcp_cb,
                queue_size=1,
            ),
            rospy.Subscriber(
                "/zj_humanoid/upperlimb/tcp_speed/dual_arm",
                TcpSpeed,
                self._speed_cb,
                queue_size=1,
            ),
            rospy.Subscriber(
                "/zj_humanoid/upperlimb/joint_states",
                JointState,
                self._joints_cb,
                queue_size=1,
            ),
            rospy.Subscriber(
                "/zj_humanoid/hand/joint_states",
                JointState,
                self._hand_cb,
                queue_size=1,
            ),
            rospy.Subscriber(
                "/zj_humanoid/upperlimb/uplimb_state",
                UplimbState,
                self._uplimb_cb,
                queue_size=1,
            ),
        ]
        self._started = True

    def stop(self) -> None:
        for sub in self._subs:
            try:
                sub.unregister()
            except Exception:
                pass
        self._subs = []
        self._started = False

    def wait_ready(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._rospy is not None and self._rospy.is_shutdown():
                raise RuntimeError("ROS shutdown while waiting for state")
            if self.cache.is_ready() and self.cache.is_fresh():
                return
            time.sleep(0.01)
        raise TimeoutError(
            f"WA2StateMonitor not ready within {timeout_s}s; "
            f"stale={self.cache.stale_fields()} ready={self.cache.is_ready()}"
        )

    def get_state(self) -> Dict[str, np.ndarray]:
        return self.cache.get_state()

    def get_ages(self) -> Dict[str, Optional[float]]:
        return self.cache.get_ages()

    def get_info(self) -> Dict[str, object]:
        return self.cache.get_info()

    def is_fresh(self) -> bool:
        return self.cache.is_fresh()

    def stale_fields(self) -> List[str]:
        return self.cache.stale_fields()

    def inject_stale_for_test(
        self, fields: Optional[Sequence[str]] = None, age_s: float = 1.0
    ) -> None:
        self.cache.inject_stale_for_test(fields=fields, age_s=age_s)

    def clear_stale_injection(self) -> None:
        self.cache.clear_stale_injection()

    def _tcp_cb(self, msg) -> None:
        pose = [
            msg.position.x,
            msg.position.y,
            msg.position.z,
            msg.quaternion.x,
            msg.quaternion.y,
            msg.quaternion.z,
            msg.quaternion.w,
        ]
        self.cache.update_tcp_pose(pose)

    def _speed_cb(self, msg) -> None:
        self.cache.update_tcp_vel(list(msg.left_arm))

    def _joints_cb(self, msg) -> None:
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        try:
            joints = [float(name_to_pos[n]) for n in self._joint_names]
        except KeyError as exc:
            # Fall back to first 8 positions if names missing (should not happen on WA2).
            if len(msg.position) < 8:
                return
            joints = [float(x) for x in msg.position[:8]]
            _ = exc
        self.cache.update_joint_pos(joints)

    def _hand_cb(self, msg) -> None:
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        try:
            hand = [float(name_to_pos[n]) for n in self._hand_names]
        except KeyError:
            if len(msg.position) < 6:
                return
            hand = [float(x) for x in msg.position[:6]]
        self.cache.update_hand_joints(hand)

    def _uplimb_cb(self, msg) -> None:
        self.cache.update_uplimb_state(
            is_singular=bool(getattr(msg, "left_arm_is_singular", False)),
            cmd_num=int(getattr(msg, "cmd_num", -1)),
            cmd_name=str(getattr(msg, "cmd_name", "")),
            iddp_status=bool(getattr(msg, "iddp_status", False)),
        )


def _normalize_pose(pose7: Sequence[float]) -> np.ndarray:
    arr = np.asarray(pose7, dtype=np.float32).reshape(7)
    if not np.all(np.isfinite(arr)):
        raise ValueError("tcp_pose must be finite")
    quat = arr[3:].astype(np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError("quaternion must not be zero")
    arr[3:] = (quat / norm).astype(np.float32)
    return arr.copy()
