"""Load WA2Env contract YAML into spaces and typed constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from gymnasium import spaces

DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "wa2_env_contract.yaml"
)

# Mock home near typical left-arm workspace (not a real reset pose).
DEFAULT_HOME_TCP = np.asarray(
    [0.0, 0.265, 0.384, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
)
DEFAULT_HOME_JOINTS = np.zeros(8, dtype=np.float32)
DEFAULT_HOME_HAND = np.zeros(6, dtype=np.float32)
DEFAULT_HOME_VEL = np.zeros(6, dtype=np.float32)


@dataclass(frozen=True)
class WA2EnvContract:
    """Frozen R1 contract fields needed by Mock/ROS Env."""

    version: str
    arm: str
    action_dim: int
    action_low: float
    action_high: float
    position_frame: str
    rotation_frame: str
    max_pos_delta_m: float
    max_rot_delta_deg: float
    control_dt: float
    control_hz: float
    policy_hz: float
    servo_ticks_per_action: int
    max_steps: int
    image_shape: Tuple[int, int, int]
    wrist_enabled: bool
    missing_policy: str
    state_max_age_s: float
    image_max_age_s: float
    raw: Mapping[str, Any]

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "WA2EnvContract":
        contract_path = Path(path) if path is not None else DEFAULT_CONTRACT_PATH
        with contract_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        action = raw["action"]
        images = raw["observation"]["images"]
        head_shape = tuple(int(x) for x in images["head"]["shape"])
        wrist_shape = tuple(int(x) for x in images["wrist"]["shape"])
        if head_shape != wrist_shape:
            raise ValueError("head and wrist observation shapes must match")
        control_hz = float(action["control_hz"])
        control_dt = float(action["control_dt"])
        policy_hz = float(action.get("policy_hz", 10.0))
        servo_ticks = int(action.get("servo_ticks_per_action", 5))
        if control_hz <= 0.0 or policy_hz <= 0.0 or servo_ticks < 1:
            raise ValueError("control/policy rates and servo_ticks_per_action must be positive")
        if not np.isclose(control_dt * control_hz, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("action.control_dt must equal 1 / action.control_hz")
        if not np.isclose(policy_hz * servo_ticks, control_hz, rtol=0.0, atol=1e-6):
            raise ValueError(
                "policy_hz * servo_ticks_per_action must equal control_hz"
            )
        return cls(
            version=str(raw["version"]),
            arm=str(raw["arm"]),
            action_dim=int(action["dim"]),
            action_low=float(action["low"]),
            action_high=float(action["high"]),
            position_frame=str(action["position_frame"]),
            rotation_frame=str(action["rotation_frame"]),
            max_pos_delta_m=float(action["max_pos_delta_m"]),
            max_rot_delta_deg=float(action["max_rot_delta_deg"]),
            control_dt=control_dt,
            control_hz=control_hz,
            policy_hz=policy_hz,
            servo_ticks_per_action=servo_ticks,
            max_steps=int(raw["episode"]["max_steps"]),
            image_shape=head_shape,  # type: ignore[arg-type]
            wrist_enabled=bool(images["wrist"]["enabled"]),
            missing_policy=str(
                images["wrist"].get("missing_policy", "zero_image")
            ),
            state_max_age_s=float(raw["freshness"]["state_max_age_s"]),
            image_max_age_s=float(raw["freshness"]["image_max_age_s"]),
            raw=raw,
        )

    @property
    def head_topic(self) -> str:
        return str(self.raw["observation"]["images"]["head"]["topic"])

    @property
    def wrist_topic(self) -> str:
        return str(self.raw["observation"]["images"]["wrist"]["topic"])

    @property
    def max_rot_delta_rad(self) -> float:
        return float(np.deg2rad(self.max_rot_delta_deg))

    def build_action_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.full((self.action_dim,), self.action_low, dtype=np.float32),
            high=np.full((self.action_dim,), self.action_high, dtype=np.float32),
            shape=(self.action_dim,),
            dtype=np.float32,
        )

    def build_observation_space(self) -> spaces.Dict:
        # Conservative physical bounds for gymnasium check_env.
        tcp_low = np.asarray([-2, -2, -2, -1, -1, -1, -1], dtype=np.float32)
        tcp_high = np.asarray([2, 2, 2, 1, 1, 1, 1], dtype=np.float32)
        vel_low = np.full((6,), -2.0, dtype=np.float32)
        vel_high = np.full((6,), 2.0, dtype=np.float32)
        joint_low = np.full((8,), -np.pi, dtype=np.float32)
        joint_high = np.full((8,), np.pi, dtype=np.float32)
        hand_low = np.full((6,), -np.pi, dtype=np.float32)
        hand_high = np.full((6,), np.pi, dtype=np.float32)
        h, w, c = self.image_shape
        image_space = spaces.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )
        return spaces.Dict(
            {
                "state": spaces.Dict(
                    {
                        "tcp_pose": spaces.Box(tcp_low, tcp_high, dtype=np.float32),
                        "tcp_vel": spaces.Box(vel_low, vel_high, dtype=np.float32),
                        "joint_pos": spaces.Box(joint_low, joint_high, dtype=np.float32),
                        "hand_joints": spaces.Box(hand_low, hand_high, dtype=np.float32),
                    }
                ),
                "images": spaces.Dict(
                    {
                        "head": image_space,
                        "wrist": image_space,
                    }
                ),
            }
        )


def resolve_contract_path(path: Optional[str | Path] = None) -> Path:
    if path is None:
        return DEFAULT_CONTRACT_PATH
    return Path(path)
