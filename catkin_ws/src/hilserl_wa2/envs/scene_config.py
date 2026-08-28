"""Load per-task scene calibration YAML (home / waist / TCP / timeouts)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import yaml

DEFAULT_SCENES_DIR = Path(__file__).resolve().parents[1] / "configs" / "scenes"


@dataclass(frozen=True)
class WA2SceneConfig:
    """Task-specific reset poses and tolerances (swap YAML to change scene)."""

    scene_id: str
    display_name: str
    arm: str
    home_joints_left: np.ndarray
    home_joints_neck: np.ndarray
    home_joints_waist: np.ndarray
    task_reset_tcp: np.ndarray
    hand_reset: np.ndarray
    waist_policy: str
    neck_policy: str
    workspace_policy: str
    do_movel_to_tcp: bool
    require_human_confirm: bool
    max_retries: int
    total_timeout_s: float
    hand_timeout_s: float
    waist_timeout_s: float
    neck_timeout_s: float
    arm_timeout_s: float
    movej_v: float
    movej_acc: float
    joint_tol_rad: float
    waist_tol_rad: float
    neck_tol_rad: float
    hand_tol_rad: float
    tcp_pos_tol_m: float
    tcp_rot_tol_deg: float
    max_steps: Optional[int]
    episode_trans_limit_m: Optional[float]
    episode_rot_limit_deg: Optional[float]
    raw: Mapping[str, Any]

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "WA2SceneConfig":
        scene_path = Path(path)
        with scene_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "WA2SceneConfig":
        reset = raw.get("reset") or {}
        tols = reset.get("tolerances") or {}
        workspace = raw.get("workspace") or {}
        episode = raw.get("episode") or {}

        left = _as_f64(raw["home_joints_left"], 8, "home_joints_left")
        neck = _as_f64(raw["home_joints_neck"], 2, "home_joints_neck")
        waist = _as_f64(raw["home_joints_waist"], 4, "home_joints_waist")
        tcp = _as_f64(raw["task_reset_tcp"], 7, "task_reset_tcp")
        hand = _as_f64(raw.get("hand_reset", [0.0] * 6), 6, "hand_reset")
        # Normalize quaternion
        qn = float(np.linalg.norm(tcp[3:]))
        if qn < 1e-8:
            raise ValueError("task_reset_tcp quaternion must not be zero")
        tcp = tcp.copy()
        tcp[3:] = tcp[3:] / qn

        allowed_policies = ("auto_movej", "check_only", "manual")
        waist_policy = str(raw.get("waist_policy", "auto_movej"))
        if waist_policy not in allowed_policies:
            raise ValueError(
                f"waist_policy must be auto_movej|check_only|manual, got {waist_policy}"
            )
        neck_policy = str(raw.get("neck_policy", "auto_movej"))
        if neck_policy not in allowed_policies:
            raise ValueError(
                f"neck_policy must be auto_movej|check_only|manual, got {neck_policy}"
            )

        max_steps = episode.get("max_steps")
        trans_limit = episode.get("trans_limit_m")
        rot_limit = episode.get("rot_limit_deg")
        return cls(
            scene_id=str(raw.get("scene_id", "unnamed")),
            display_name=str(raw.get("display_name", raw.get("scene_id", "unnamed"))),
            arm=str(raw.get("arm", "left")),
            home_joints_left=left.astype(np.float64),
            home_joints_neck=neck.astype(np.float64),
            home_joints_waist=waist.astype(np.float64),
            task_reset_tcp=tcp.astype(np.float64),
            hand_reset=hand.astype(np.float64),
            waist_policy=waist_policy,
            neck_policy=neck_policy,
            workspace_policy=str(workspace.get("policy", "head_camera_visible_desktop")),
            do_movel_to_tcp=bool(reset.get("do_movel_to_tcp", False)),
            require_human_confirm=bool(reset.get("require_human_confirm", True)),
            max_retries=int(reset.get("max_retries", 1)),
            total_timeout_s=float(reset.get("total_timeout_s", 60.0)),
            hand_timeout_s=float(reset.get("hand_timeout_s", 10.0)),
            waist_timeout_s=float(reset.get("waist_timeout_s", 30.0)),
            neck_timeout_s=float(reset.get("neck_timeout_s", 15.0)),
            arm_timeout_s=float(reset.get("arm_timeout_s", 30.0)),
            movej_v=float(reset.get("movej_v", 0.15)),
            movej_acc=float(reset.get("movej_acc", 0.25)),
            joint_tol_rad=float(tols.get("joint_tol_rad", 0.05)),
            waist_tol_rad=float(tols.get("waist_tol_rad", 0.05)),
            neck_tol_rad=float(tols.get("neck_tol_rad", 0.05)),
            hand_tol_rad=float(tols.get("hand_tol_rad", 0.1)),
            tcp_pos_tol_m=float(tols.get("tcp_pos_tol_m", 0.01)),
            tcp_rot_tol_deg=float(tols.get("tcp_rot_tol_deg", 5.0)),
            max_steps=int(max_steps) if max_steps is not None else None,
            episode_trans_limit_m=(
                float(trans_limit) if trans_limit is not None else None
            ),
            episode_rot_limit_deg=(
                float(rot_limit) if rot_limit is not None else None
            ),
            raw=raw,
        )

    @property
    def tcp_rot_tol_rad(self) -> float:
        return float(np.deg2rad(self.tcp_rot_tol_deg))


def resolve_scene_path(
    scene_path: Optional[Union[str, Path]] = None,
    scene_name: Optional[str] = None,
    scenes_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Resolve scene YAML. Prefer explicit path; else ``<scenes_dir>/<name>.yaml``."""

    if scene_path is not None:
        return Path(scene_path)
    if scene_name is None:
        return None
    base = Path(scenes_dir) if scenes_dir is not None else DEFAULT_SCENES_DIR
    path = base / f"{scene_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"scene not found: {path}")
    return path


def load_scene(
    scene_path: Optional[Union[str, Path]] = None,
    scene_name: Optional[str] = None,
    scenes_dir: Optional[Union[str, Path]] = None,
) -> Optional[WA2SceneConfig]:
    path = resolve_scene_path(scene_path, scene_name, scenes_dir)
    if path is None:
        return None
    return WA2SceneConfig.from_yaml(path)


def _as_f64(values: Sequence[Any], n: int, name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} must have length {n}, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr
