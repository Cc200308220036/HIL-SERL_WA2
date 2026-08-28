"""Pure-Python bounded Cartesian pose integration for SpaceMouse commands."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseIntegratorConfig:
    """Physical scaling and safety bounds for one deadman session."""

    linear_scale: float = 0.010
    angular_scale: float = 0.08
    max_linear_step: float = 0.0005
    max_angular_step: float = 0.002
    translation_limit: float = 0.03
    rotation_limit_rad: float = math.radians(2.0)
    rotation_frame: str = "tool"
    allow_mixed_motion: bool = False

    def __post_init__(self) -> None:
        numeric = (
            self.linear_scale,
            self.angular_scale,
            self.max_linear_step,
            self.max_angular_step,
            self.translation_limit,
            self.rotation_limit_rad,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("all pose integrator values must be finite")
        if self.linear_scale < 0.0 or self.angular_scale < 0.0:
            raise ValueError("linear_scale and angular_scale must be non-negative")
        if self.max_linear_step <= 0.0 or self.max_angular_step <= 0.0:
            raise ValueError("maximum step sizes must be greater than zero")
        if self.translation_limit <= 0.0 or self.rotation_limit_rad <= 0.0:
            raise ValueError("workspace limits must be greater than zero")
        if self.rotation_frame not in ("tool", "base"):
            raise ValueError("rotation_frame must be 'tool' or 'base'")


class PoseIntegrator:
    """Integrate normalized motion into a bounded ``xyz + quaternion`` target."""

    def __init__(self, config: Optional[PoseIntegratorConfig] = None):
        self.config = config or PoseIntegratorConfig()
        self._origin_pose: Optional[np.ndarray] = None
        self._target_pose: Optional[np.ndarray] = None
        self._origin_rotation: Optional[Rotation] = None

    @property
    def initialized(self) -> bool:
        return self._target_pose is not None

    @property
    def origin_pose(self) -> np.ndarray:
        if self._origin_pose is None:
            raise RuntimeError("pose integrator is not initialized")
        return self._origin_pose.copy()

    @property
    def target_pose(self) -> np.ndarray:
        if self._target_pose is None:
            raise RuntimeError("pose integrator is not initialized")
        return self._target_pose.copy()

    @property
    def relative_translation(self) -> float:
        if self._origin_pose is None or self._target_pose is None:
            raise RuntimeError("pose integrator is not initialized")
        return float(np.linalg.norm(self._target_pose[:3] - self._origin_pose[:3]))

    @property
    def relative_rotation_rad(self) -> float:
        if self._origin_rotation is None or self._target_pose is None:
            raise RuntimeError("pose integrator is not initialized")
        target_rotation = Rotation.from_quat(self._target_pose[3:])
        relative = self._origin_rotation.inv() * target_rotation
        return float(np.linalg.norm(relative.as_rotvec()))

    def reset(self, pose: Sequence[float]) -> np.ndarray:
        """Start a new deadman session from a measured TCP pose."""

        validated = self._validate_pose(pose)
        self._origin_pose = validated.copy()
        self._target_pose = validated.copy()
        self._origin_rotation = Rotation.from_quat(validated[3:])
        return self.target_pose

    def step(self, normalized_motion: Sequence[float], dt: float) -> np.ndarray:
        """Integrate one normalized ``[XYZ, RollPitchYaw]`` command sample."""

        if self._target_pose is None or self._origin_pose is None or self._origin_rotation is None:
            raise RuntimeError("reset(pose) must be called before step()")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and greater than zero")

        motion = np.asarray(normalized_motion, dtype=np.float64)
        if motion.shape != (6,) or not np.all(np.isfinite(motion)):
            raise ValueError("normalized_motion must be 6 finite values")
        motion = np.clip(motion, -1.0, 1.0)

        has_translation = bool(np.linalg.norm(motion[:3]) > 1e-12)
        has_rotation = bool(np.linalg.norm(motion[3:]) > 1e-12)
        if has_translation and has_rotation and not self.config.allow_mixed_motion:
            raise ValueError("mixed translation and rotation are disabled in the first safety phase")

        candidate = self._target_pose.copy()
        if has_translation:
            delta_xyz = motion[:3] * self.config.linear_scale * dt
            delta_xyz = self._clip_norm(delta_xyz, self.config.max_linear_step)
            candidate[:3] += delta_xyz
            candidate[:3] = self._clip_translation_workspace(candidate[:3])

        if has_rotation:
            delta_rotvec = motion[3:] * self.config.angular_scale * dt
            delta_rotvec = self._clip_norm(delta_rotvec, self.config.max_angular_step)
            current_rotation = Rotation.from_quat(candidate[3:])
            delta_rotation = Rotation.from_rotvec(delta_rotvec)
            if self.config.rotation_frame == "tool":
                candidate_rotation = current_rotation * delta_rotation
            else:
                candidate_rotation = delta_rotation * current_rotation
            candidate_rotation = self._clip_rotation_workspace(candidate_rotation)
            quaternion = candidate_rotation.as_quat()
            quaternion /= np.linalg.norm(quaternion)
            if float(np.dot(quaternion, self._target_pose[3:])) < 0.0:
                quaternion = -quaternion
            candidate[3:] = quaternion

        self._target_pose = self._validate_pose(candidate)
        return self.target_pose

    def _clip_translation_workspace(self, position: np.ndarray) -> np.ndarray:
        assert self._origin_pose is not None
        offset = np.asarray(position, dtype=np.float64) - self._origin_pose[:3]
        return self._origin_pose[:3] + self._clip_norm(
            offset, self.config.translation_limit
        )

    def _clip_rotation_workspace(self, candidate: Rotation) -> Rotation:
        assert self._origin_rotation is not None
        relative = self._origin_rotation.inv() * candidate
        rotvec = relative.as_rotvec()
        clipped = self._clip_norm(rotvec, self.config.rotation_limit_rad)
        return self._origin_rotation * Rotation.from_rotvec(clipped)

    @staticmethod
    def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm > maximum and norm > 0.0:
            return vector * (maximum / norm)
        return vector

    @staticmethod
    def _validate_pose(pose: Sequence[float]) -> np.ndarray:
        value = np.asarray(pose, dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("pose must be 7 finite values: xyz + qx qy qz qw")
        quaternion_norm = float(np.linalg.norm(value[3:]))
        if quaternion_norm < 1e-8:
            raise ValueError("pose quaternion must not be zero")
        value = value.copy()
        value[3:] /= quaternion_norm
        return value
