"""Load / validate / hash WA2 task YAML (R8). No ROS, no serl_launcher."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS_DIR = PACKAGE_ROOT / "configs"
DEFAULT_TASKS_DIR = DEFAULT_CONFIGS_DIR / "tasks"
DEFAULT_SCENES_DIR = DEFAULT_CONFIGS_DIR / "scenes"
DEFAULT_CAMERAS_DIR = DEFAULT_CONFIGS_DIR / "cameras"
DEFAULT_SPACEMOUSE_DIR = DEFAULT_CONFIGS_DIR / "spacemouse"
DEFAULT_CONTRACT_PATH = DEFAULT_CONFIGS_DIR / "wa2_env_contract.yaml"

TASK_ID_RE = re.compile(r"^[a-z0-9_]+$")
ALLOWED_TOP_LEVEL = {
    "schema_version",
    "task_id",
    "display_name",
    "env",
    "observation",
    "action",
    "wrappers",
    "reward",
    "training",
    "notes",
}
ALLOWED_ENV = {"scene", "contract", "camera", "spacemouse"}
ALLOWED_OBSERVATION = {"image_keys", "proprio_keys"}
ALLOWED_ACTION = {"mode"}
ALLOWED_WRAPPERS = {
    "intervention",
    "serl_obs",
    "obs_horizon",
    "act_exec_horizon",
}
ALLOWED_REWARD = {"mode", "classifier_keys", "classifier_consecutive_n"}
ALLOWED_TRAINING = {
    "agent",
    "setup_mode",
    "encoder_type",
    "discount",
    "batch_size",
    "max_steps",
    "replay_buffer_capacity",
    "random_steps",
    "training_starts",
    "steps_per_update",
    "buffer_period",
    "checkpoint_period",
}
FORBIDDEN_KEYS = {
    "server_url",
    "SERVER_URL",
    "serial_number",
    "serial",
    "max_pos_delta_m",
    "max_rot_delta_deg",
    "servo_gain",
    "servo_time",
    "compliance_param",
    "COMPLIANCE_PARAM",
}
ALLOWED_ACTION_MODES = {"left_arm_6d"}
ALLOWED_REWARD_MODES = {"constant_zero"}
ALLOWED_SETUP_MODES = {
    "single-arm-fixed-gripper",
    "single-arm-learned-gripper",
}
BUNDLE_FILE_ORDER = ("task", "scene", "contract", "camera", "spacemouse")


class TaskConfigError(ValueError):
    """Invalid task YAML or task id."""


def sanitize_task_id(task_id: str) -> str:
    if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
        raise TaskConfigError(
            f"invalid task_id {task_id!r}; expected [a-z0-9_]+ (no path separators)"
        )
    return task_id


def exp_name_for_task(task_id: str) -> str:
    return f"wa2_{sanitize_task_id(task_id)}"


def task_id_from_exp_name(exp_name: str) -> str:
    if not isinstance(exp_name, str) or not exp_name.startswith("wa2_"):
        raise TaskConfigError(
            f"exp_name must be 'wa2_<task_id>', got {exp_name!r}"
        )
    return sanitize_task_id(exp_name[len("wa2_") :])


def resolve_task_path(
    task_id: str,
    tasks_dir: Optional[Union[str, Path]] = None,
) -> Path:
    tid = sanitize_task_id(task_id)
    base = Path(tasks_dir) if tasks_dir is not None else DEFAULT_TASKS_DIR
    path = (base / f"{tid}.yaml").resolve()
    tasks_root = Path(base).resolve()
    if tasks_root not in path.parents and path.parent != tasks_root:
        raise TaskConfigError(f"task path escaped tasks dir: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"task YAML not found: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_unknown(raw: Mapping[str, Any], allowed: set, where: str) -> None:
    extra = set(raw) - allowed
    if extra:
        raise TaskConfigError(f"unknown key(s) in {where}: {sorted(extra)}")


def _walk_forbidden(obj: Any, trail: str = "") -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            loc = f"{trail}.{key}" if trail else str(key)
            if key in FORBIDDEN_KEYS:
                raise TaskConfigError(f"forbidden key {key!r} at {loc}")
            _walk_forbidden(value, loc)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            _walk_forbidden(value, f"{trail}[{idx}]")


def _require_str_list(value: Any, name: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TaskConfigError(f"{name} must be a non-empty list")
    out = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise TaskConfigError(f"{name} entries must be non-empty strings")
        out.append(item)
    return tuple(out)


def _resolve_named_yaml(
    name: str,
    directory: Path,
    *,
    what: str,
    default_name: Optional[str] = None,
) -> Path:
    if not isinstance(name, str) or not name:
        raise TaskConfigError(f"env.{what} must be a non-empty string")
    if name == "default" and default_name is not None:
        name = default_name
    if not TASK_ID_RE.fullmatch(name) and name != "default":
        # camera/scene names also [a-z0-9_]; contract default handled above
        if what != "contract":
            raise TaskConfigError(f"env.{what} {name!r} is not a safe id")
    if what == "contract" and name == "default":
        path = DEFAULT_CONTRACT_PATH
    else:
        path = directory / f"{name}.yaml"
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{what} YAML not found: {path}")
    return path


@dataclass(frozen=True)
class WA2TaskConfig:
    task_id: str
    display_name: str
    scene: str
    contract_name: str
    camera: str
    spacemouse: str
    image_keys: Tuple[str, ...]
    proprio_keys: Tuple[str, ...]
    action_mode: str
    intervention: bool
    serl_obs: bool
    obs_horizon: int
    act_exec_horizon: Optional[int]
    reward_mode: str
    classifier_keys: Optional[Tuple[str, ...]]
    classifier_consecutive_n: int
    agent: str
    setup_mode: str
    encoder_type: str
    discount: float
    batch_size: int
    max_steps: int
    replay_buffer_capacity: int
    random_steps: int
    training_starts: int
    steps_per_update: int
    buffer_period: int
    checkpoint_period: int
    task_path: Path
    scene_path: Path
    contract_path: Path
    camera_path: Path
    spacemouse_path: Path
    raw: Mapping[str, Any]

    @property
    def exp_name(self) -> str:
        return exp_name_for_task(self.task_id)

    @property
    def proprio_dim(self) -> int:
        dims = {
            "tcp_pose": 7,
            "tcp_vel": 6,
            "joint_pos": 8,
            "hand_joints": 6,
        }
        return int(sum(dims[k] for k in self.proprio_keys))

    def resolved_paths(self) -> Dict[str, str]:
        return {
            "task": str(self.task_path),
            "scene": str(self.scene_path),
            "contract": str(self.contract_path),
            "camera": str(self.camera_path),
            "spacemouse": str(self.spacemouse_path),
        }

    def file_hashes(self) -> Dict[str, str]:
        return {
            "task_hash": _sha256_file(self.task_path),
            "scene_hash": _sha256_file(self.scene_path),
            "contract_hash": _sha256_file(self.contract_path),
            "camera_hash": _sha256_file(self.camera_path),
            "spacemouse_hash": _sha256_file(self.spacemouse_path),
        }

    def config_bundle_hash(self) -> str:
        hashes = self.file_hashes()
        mapping = {
            "task": hashes["task_hash"],
            "scene": hashes["scene_hash"],
            "contract": hashes["contract_hash"],
            "camera": hashes["camera_hash"],
            "spacemouse": hashes["spacemouse_hash"],
        }
        blob = "".join(f"{name}:{mapping[name]}\n" for name in BUNDLE_FILE_ORDER)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_task(
    task_id: str,
    *,
    tasks_dir: Optional[Union[str, Path]] = None,
    configs_dir: Optional[Union[str, Path]] = None,
) -> WA2TaskConfig:
    path = resolve_task_path(task_id, tasks_dir=tasks_dir)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TaskConfigError(f"{path} must be a mapping")
    return WA2TaskConfig.from_dict(  # type: ignore[attr-defined]
        raw, task_path=path, configs_dir=configs_dir
    )


def _from_dict(
    raw: Mapping[str, Any],
    *,
    task_path: Path,
    configs_dir: Optional[Union[str, Path]] = None,
) -> WA2TaskConfig:
    _reject_unknown(raw, ALLOWED_TOP_LEVEL, "task YAML")
    _walk_forbidden(raw)
    cfg_root = Path(configs_dir) if configs_dir is not None else DEFAULT_CONFIGS_DIR

    task_id = sanitize_task_id(str(raw.get("task_id", "")))
    stem = task_path.stem
    if task_id != stem:
        raise TaskConfigError(
            f"task_id {task_id!r} must match filename stem {stem!r}"
        )
    if str(raw.get("schema_version", "")) != "1":
        raise TaskConfigError("schema_version must be '1'")

    env = raw.get("env") or {}
    observation = raw.get("observation") or {}
    action = raw.get("action") or {}
    wrappers = raw.get("wrappers") or {}
    reward = raw.get("reward") or {}
    training = raw.get("training") or {}
    if not isinstance(env, Mapping):
        raise TaskConfigError("env must be a mapping")
    _reject_unknown(env, ALLOWED_ENV, "env")
    _reject_unknown(observation, ALLOWED_OBSERVATION, "observation")
    _reject_unknown(action, ALLOWED_ACTION, "action")
    _reject_unknown(wrappers, ALLOWED_WRAPPERS, "wrappers")
    _reject_unknown(reward, ALLOWED_REWARD, "reward")
    _reject_unknown(training, ALLOWED_TRAINING, "training")

    scene = str(env.get("scene", ""))
    contract_name = str(env.get("contract", "default"))
    camera = str(env.get("camera", ""))
    spacemouse = str(env.get("spacemouse", "default"))
    if not TASK_ID_RE.fullmatch(scene):
        raise TaskConfigError(f"env.scene {scene!r} is not a safe id")
    if not TASK_ID_RE.fullmatch(camera):
        raise TaskConfigError(f"env.camera {camera!r} is not a safe id")
    if not TASK_ID_RE.fullmatch(spacemouse):
        raise TaskConfigError(f"env.spacemouse {spacemouse!r} is not a safe id")

    image_keys = _require_str_list(observation.get("image_keys"), "image_keys")
    proprio_keys = _require_str_list(observation.get("proprio_keys"), "proprio_keys")
    known_proprio = {"tcp_pose", "tcp_vel", "joint_pos", "hand_joints"}
    known_images = {"head", "wrist"}
    bad_p = [k for k in proprio_keys if k not in known_proprio]
    bad_i = [k for k in image_keys if k not in known_images]
    if bad_p:
        raise TaskConfigError(f"unknown proprio_keys: {bad_p}")
    if bad_i:
        raise TaskConfigError(f"unknown image_keys: {bad_i}")

    action_mode = str(action.get("mode", ""))
    if action_mode not in ALLOWED_ACTION_MODES:
        raise TaskConfigError(
            f"action.mode must be one of {sorted(ALLOWED_ACTION_MODES)}, got {action_mode!r}"
        )

    if "serl_obs" in wrappers and not bool(wrappers["serl_obs"]):
        raise TaskConfigError("wrappers.serl_obs must be true in R8")
    obs_horizon = int(wrappers.get("obs_horizon", 1))
    if obs_horizon < 1:
        raise TaskConfigError("obs_horizon must be >= 1")
    raw_horizon = wrappers.get("act_exec_horizon", None)
    if raw_horizon not in (None, "null"):
        raise TaskConfigError("R8 act_exec_horizon must be null")
    act_exec_horizon = None

    reward_mode = str(reward.get("mode", ""))
    if reward_mode not in ALLOWED_REWARD_MODES:
        raise TaskConfigError(
            f"reward.mode must be one of {sorted(ALLOWED_REWARD_MODES)}"
        )
    raw_ck = reward.get("classifier_keys", None)
    if raw_ck in (None, "null"):
        classifier_keys = None
    else:
        classifier_keys = _require_str_list(raw_ck, "classifier_keys")
        extra = [key for key in classifier_keys if key not in image_keys]
        if extra:
            raise TaskConfigError(
                f"classifier_keys {extra} are not in image_keys {list(image_keys)}"
            )
    classifier_consecutive_n = int(reward.get("classifier_consecutive_n", 1))
    if classifier_consecutive_n < 1:
        raise TaskConfigError("reward.classifier_consecutive_n must be >= 1")

    setup_mode = str(training.get("setup_mode", "single-arm-fixed-gripper"))
    if setup_mode not in ALLOWED_SETUP_MODES:
        raise TaskConfigError(f"unsupported setup_mode {setup_mode!r}")

    scene_path = _resolve_named_yaml(
        scene, cfg_root / "scenes", what="scene"
    )
    if contract_name == "default":
        contract_path = (cfg_root / "wa2_env_contract.yaml").resolve()
        if not contract_path.is_file():
            raise FileNotFoundError(f"contract YAML not found: {contract_path}")
    else:
        raise TaskConfigError("env.contract must be 'default' in R8")
    camera_path = _resolve_named_yaml(
        camera, cfg_root / "cameras", what="camera"
    )
    spacemouse_path = _resolve_named_yaml(
        spacemouse, cfg_root / "spacemouse", what="spacemouse"
    )

    return WA2TaskConfig(
        task_id=task_id,
        display_name=str(raw.get("display_name", task_id)),
        scene=scene,
        contract_name=contract_name,
        camera=camera,
        spacemouse=spacemouse,
        image_keys=image_keys,
        proprio_keys=proprio_keys,
        action_mode=action_mode,
        intervention=bool(wrappers.get("intervention", True)),
        serl_obs=True,
        obs_horizon=obs_horizon,
        act_exec_horizon=act_exec_horizon,
        reward_mode=reward_mode,
        classifier_keys=classifier_keys,
        classifier_consecutive_n=classifier_consecutive_n,
        agent=str(training.get("agent", "drq")),
        setup_mode=setup_mode,
        encoder_type=str(training.get("encoder_type", "resnet-pretrained")),
        discount=float(training.get("discount", 0.97)),
        batch_size=int(training.get("batch_size", 256)),
        max_steps=int(training.get("max_steps", 1_000_000)),
        replay_buffer_capacity=int(training.get("replay_buffer_capacity", 200000)),
        random_steps=int(training.get("random_steps", 0)),
        training_starts=int(training.get("training_starts", 100)),
        steps_per_update=int(training.get("steps_per_update", 50)),
        buffer_period=int(training.get("buffer_period", 1000)),
        checkpoint_period=int(training.get("checkpoint_period", 5000)),
        task_path=task_path.resolve(),
        scene_path=scene_path,
        contract_path=contract_path,
        camera_path=camera_path,
        spacemouse_path=spacemouse_path,
        raw=raw,
    )


WA2TaskConfig.from_dict = staticmethod(_from_dict)  # type: ignore[attr-defined]


def iter_task_files(tasks_dir: Optional[Union[str, Path]] = None) -> Iterable[Path]:
    base = Path(tasks_dir) if tasks_dir is not None else DEFAULT_TASKS_DIR
    if not base.is_dir():
        raise FileNotFoundError(f"tasks dir not found: {base}")
    for path in sorted(base.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        yield path


def discover_task_ids(
    tasks_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    ids: List[str] = []
    for path in iter_task_files(tasks_dir):
        cfg = load_task(path.stem, tasks_dir=tasks_dir or path.parent)
        ids.append(cfg.task_id)
    return ids


def check_wa2_task_override(task_id: str, wa2_task_env: Optional[str]) -> None:
    """Reject WA2_TASK when it disagrees with the resolved task_id."""

    if not wa2_task_env:
        return
    env_id = sanitize_task_id(wa2_task_env)
    tid = sanitize_task_id(task_id)
    if env_id != tid:
        raise TaskConfigError(
            f"WA2_TASK={env_id!r} does not match task_id={tid!r}"
        )


def serialize_space(space: Any) -> Dict[str, Any]:
    """Deterministic JSON-able gymnasium space description."""

    import numpy as np
    from gymnasium import spaces

    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": str(space.dtype),
            "low": np.asarray(space.low).tolist(),
            "high": np.asarray(space.high).tolist(),
        }
    if isinstance(space, spaces.Dict):
        return {
            "type": "Dict",
            "keys": list(space.spaces.keys()),
            "spaces": {k: serialize_space(v) for k, v in space.spaces.items()},
        }
    if isinstance(space, spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n)}
    raise TypeError(f"unsupported space type {type(space)!r}")


def space_sha256(observation_space: Any, action_space: Any) -> str:
    payload = {
        "observation": serialize_space(observation_space),
        "action": serialize_space(action_space),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
