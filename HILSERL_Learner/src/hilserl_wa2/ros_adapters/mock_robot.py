"""Mock robot backend for R2: no ROS, no hardware."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from hilserl_wa2.envs.contracts import (
    DEFAULT_HOME_HAND,
    DEFAULT_HOME_JOINTS,
    DEFAULT_HOME_TCP,
    DEFAULT_HOME_VEL,
    WA2EnvContract,
)


class MockRobot:
    """Integrate clipped 6D actions into an in-memory left-arm state."""

    def __init__(self, contract: WA2EnvContract):
        self.contract = contract
        self.tcp_pose = DEFAULT_HOME_TCP.copy()
        self.tcp_vel = DEFAULT_HOME_VEL.copy()
        self.joint_pos = DEFAULT_HOME_JOINTS.copy()
        self.hand_joints = DEFAULT_HOME_HAND.copy()
        self.is_singular = False
        self.cmd_num = 0
        self.cmd_name = "MOCK_IDLE"
        self.iddp_status = True
        self._home_tcp = DEFAULT_HOME_TCP.copy()

    def reset(
        self,
        tcp_pose: Optional[Sequence[float]] = None,
        joint_pos: Optional[Sequence[float]] = None,
        hand_joints: Optional[Sequence[float]] = None,
    ) -> None:
        self.tcp_pose = self._validate_pose(
            DEFAULT_HOME_TCP if tcp_pose is None else tcp_pose
        )
        self._home_tcp = self.tcp_pose.copy()
        self.tcp_vel = DEFAULT_HOME_VEL.copy()
        self.joint_pos = np.asarray(
            DEFAULT_HOME_JOINTS if joint_pos is None else joint_pos,
            dtype=np.float32,
        ).reshape(8)
        self.hand_joints = np.asarray(
            DEFAULT_HOME_HAND if hand_joints is None else hand_joints,
            dtype=np.float32,
        ).reshape(6)
        self.is_singular = False
        self.cmd_num = 0
        self.cmd_name = "MOCK_RESET"
        self.iddp_status = True

    def apply_action(self, action: Sequence[float]) -> Dict[str, float]:
        """Apply one normalized action; returns applied physical deltas."""

        action_arr = np.asarray(action, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(action_arr)):
            raise ValueError("action must be finite")
        clipped = np.clip(
            action_arr,
            self.contract.action_low,
            self.contract.action_high,
        )

        delta_pos = clipped[:3] * self.contract.max_pos_delta_m
        delta_rot = clipped[3:] * self.contract.max_rot_delta_rad

        # Enforce per-step physical caps (already matched to full-scale action).
        pos_norm = float(np.linalg.norm(delta_pos))
        if pos_norm > self.contract.max_pos_delta_m + 1e-12:
            delta_pos *= self.contract.max_pos_delta_m / pos_norm
        rot_norm = float(np.linalg.norm(delta_rot))
        if rot_norm > self.contract.max_rot_delta_rad + 1e-12:
            delta_rot *= self.contract.max_rot_delta_rad / rot_norm

        # Translation in base frame.
        if self.contract.position_frame != "base":
            raise ValueError("MockRobot only implements position_frame=base")
        new_pose = self.tcp_pose.astype(np.float64).copy()
        new_pose[:3] = new_pose[:3] + delta_pos

        # Rotation in tool frame (right-multiply).
        current = Rotation.from_quat(new_pose[3:])
        delta = Rotation.from_rotvec(delta_rot)
        if self.contract.rotation_frame == "tool":
            updated = current * delta
        elif self.contract.rotation_frame == "base":
            updated = delta * current
        else:
            raise ValueError(f"unknown rotation_frame={self.contract.rotation_frame}")
        quat = updated.as_quat()
        quat = quat / np.linalg.norm(quat)
        if float(np.dot(quat, new_pose[3:])) < 0.0:
            quat = -quat
        new_pose[3:] = quat

        dt = self.contract.control_dt
        self.tcp_vel = np.asarray(
            [
                delta_pos[0] / dt,
                delta_pos[1] / dt,
                delta_pos[2] / dt,
                delta_rot[0] / dt,
                delta_rot[1] / dt,
                delta_rot[2] / dt,
            ],
            dtype=np.float32,
        )
        # Joints are not integrated from Cartesian mock; keep last reset value.
        self.tcp_pose = new_pose.astype(np.float32)
        self.cmd_num += 1
        self.cmd_name = "MOCK_STEP"
        return {
            "delta_pos_m": float(np.linalg.norm(delta_pos)),
            "delta_rot_rad": float(np.linalg.norm(delta_rot)),
            "delta_pos_xyz": delta_pos,
            "delta_rot_rpy": delta_rot,
        }

    def get_state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "tcp_pose": self.tcp_pose.copy(),
            "tcp_vel": self.tcp_vel.copy(),
            "joint_pos": self.joint_pos.copy(),
            "hand_joints": self.hand_joints.copy(),
        }

    def get_info_fields(self) -> Dict[str, object]:
        return {
            "is_singular": bool(self.is_singular),
            "cmd_num": int(self.cmd_num),
            "cmd_name": str(self.cmd_name),
            "iddp_status": bool(self.iddp_status),
            "state_age": 0.0,
        }

    @staticmethod
    def _validate_pose(pose: Sequence[float]) -> np.ndarray:
        arr = np.asarray(pose, dtype=np.float32).reshape(7)
        if not np.all(np.isfinite(arr)):
            raise ValueError("pose must be finite")
        quat = arr[3:].astype(np.float64)
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            raise ValueError("quaternion must not be zero")
        arr[3:] = (quat / norm).astype(np.float32)
        return arr
