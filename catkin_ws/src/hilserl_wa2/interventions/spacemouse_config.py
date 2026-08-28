"""Load SpaceMouse YAML config for Intervention / teleop / joy launch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml

from hilserl_wa2.interventions.spacemouse_input import SpaceMouseInputConfig

DEFAULT_SPACEMOUSE_YAML = (
    Path(__file__).resolve().parents[1] / "configs" / "spacemouse" / "default.yaml"
)


def resolve_spacemouse_config_path(
    path: Optional[Union[str, Path]] = None,
) -> Path:
    if path is None:
        return DEFAULT_SPACEMOUSE_YAML
    p = Path(path)
    if p.is_file():
        return p
    # Allow stem under configs/spacemouse/
    stem = Path(__file__).resolve().parents[1] / "configs" / "spacemouse" / f"{path}.yaml"
    if stem.is_file():
        return stem
    raise FileNotFoundError(f"SpaceMouse config not found: {path}")


@dataclass(frozen=True)
class SpaceMouseRuntimeConfig:
    """Parsed SpaceMouse stack config."""

    path: Path
    version: str
    joy_topic: str
    joy_max_age_s: float
    deadman_button: int
    hand_button: int
    session_mode: str
    input_config: SpaceMouseInputConfig
    intervene_eps: float
    control_dt: float
    auto_start_ros: bool
    action_gain: float
    teleop: Dict[str, Any]
    launch: Dict[str, Any]
    raw: Dict[str, Any]

    def teleop_ros_params(self) -> Dict[str, Any]:
        """Private-node params for spacemouse_wa2_teleop (keys without ~)."""
        t = dict(self.teleop)
        inp = self.input_config
        params: Dict[str, Any] = {
            "joy_topic": self.joy_topic,
            "deadman_button": self.deadman_button,
            "hand_button": self.hand_button,
            "axis_map": list(inp.axis_map),
            "axis_sign": list(inp.axis_sign),
            "translation_deadzone": inp.translation_deadzone,
            "rotation_deadzone": inp.rotation_deadzone,
            "translation_curve_mix": inp.translation_curve_mix,
            "rotation_curve_mix": inp.rotation_curve_mix,
            "translation_filter_tau": inp.translation_filter_tau,
            "rotation_filter_tau": inp.rotation_filter_tau,
            "translation_enter_threshold": inp.translation_enter_threshold,
            "rotation_enter_threshold": inp.rotation_enter_threshold,
            "intent_exit_threshold": inp.intent_exit_threshold,
            "group_switch_hysteresis": inp.group_switch_hysteresis,
            "axis_switch_hysteresis": inp.axis_switch_hysteresis,
            "secondary_axis_ratio": inp.secondary_axis_ratio,
        }
        # teleop block overrides / adds scales etc.
        for key, value in t.items():
            params[key] = value
        return params


def _as_tuple_float(seq: Sequence[Any], n: int, name: str) -> Tuple[float, ...]:
    vals = tuple(float(x) for x in seq)
    if len(vals) != n:
        raise ValueError(f"{name} must have length {n}, got {len(vals)}")
    return vals


def _as_tuple_int(seq: Sequence[Any], n: int, name: str) -> Tuple[int, ...]:
    vals = tuple(int(x) for x in seq)
    if len(vals) != n:
        raise ValueError(f"{name} must have length {n}, got {len(vals)}")
    return vals


def _parse_action_gain(value: Any) -> float:
    gain = float(value)
    if not (0.0 < gain <= 3.0):
        raise ValueError(f"intervention.action_gain must be in (0, 3], got {gain}")
    return gain


def _parse_session_mode(value: Any) -> str:
    mode = str(value or "toggle").strip().lower()
    if mode not in ("toggle", "hold"):
        raise ValueError(f"buttons.session_mode must be toggle|hold, got {value}")
    return mode


def load_spacemouse_config(
    path: Optional[Union[str, Path]] = None,
) -> SpaceMouseRuntimeConfig:
    cfg_path = resolve_spacemouse_config_path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"invalid spacemouse yaml root: {cfg_path}")

    joy = raw.get("joy") or {}
    buttons = raw.get("buttons") or {}
    inp = raw.get("input") or {}
    intervention = raw.get("intervention") or {}
    teleop = dict(raw.get("teleop") or {})
    launch = dict(raw.get("launch") or {})
    hand = dict(raw.get("hand") or {})
    for key in ("grasp_target", "release_target"):
        if key in hand and key not in teleop:
            teleop[key] = hand[key]

    input_config = SpaceMouseInputConfig(
        axis_map=_as_tuple_int(inp.get("axis_map", [0, 1, 2, 3, 4, 5]), 6, "axis_map"),
        axis_sign=_as_tuple_float(
            inp.get("axis_sign", [-1, -1, 1, 1, -1, -1]), 6, "axis_sign"
        ),
        translation_deadzone=float(inp.get("translation_deadzone", 0.15)),
        rotation_deadzone=float(inp.get("rotation_deadzone", 0.18)),
        translation_curve_mix=float(inp.get("translation_curve_mix", 0.25)),
        rotation_curve_mix=float(inp.get("rotation_curve_mix", 0.45)),
        translation_filter_tau=float(inp.get("translation_filter_tau", 0.06)),
        rotation_filter_tau=float(inp.get("rotation_filter_tau", 0.10)),
        translation_enter_threshold=float(
            inp.get("translation_enter_threshold", 0.35)
        ),
        rotation_enter_threshold=float(inp.get("rotation_enter_threshold", 0.65)),
        intent_exit_threshold=float(inp.get("intent_exit_threshold", 0.20)),
        group_switch_hysteresis=float(inp.get("group_switch_hysteresis", 0.15)),
        axis_switch_hysteresis=float(inp.get("axis_switch_hysteresis", 0.25)),
        secondary_axis_ratio=float(inp.get("secondary_axis_ratio", 0.90)),
    )

    return SpaceMouseRuntimeConfig(
        path=cfg_path,
        version=str(raw.get("version", "")),
        joy_topic=str(joy.get("topic", "/spacenav/joy")),
        joy_max_age_s=float(joy.get("max_age_s", 0.2)),
        deadman_button=int(buttons.get("deadman", 1)),
        hand_button=int(buttons.get("hand", 0)),
        session_mode=_parse_session_mode(buttons.get("session_mode", "toggle")),
        input_config=input_config,
        intervene_eps=float(intervention.get("intervene_eps", 1e-3)),
        control_dt=float(intervention.get("control_dt", 0.02)),
        auto_start_ros=bool(intervention.get("auto_start_ros", True)),
        action_gain=_parse_action_gain(intervention.get("action_gain", 1.0)),
        teleop=teleop,
        launch=launch,
        raw=raw,
    )
