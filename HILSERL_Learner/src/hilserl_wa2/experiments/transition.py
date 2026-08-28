"""R9 Actor transition schema, terminated/truncated semantics, dual-store routing."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

TRANSITION_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "masks",
    "dones",
)

ACTION_DIM = 6
ACTION_DIMS = (6, 7)
ACTION_LOW = -1.0
ACTION_HIGH = 1.0


class TransitionError(ValueError):
    """Invalid Actor transition."""


def dones_and_mask(
    terminated: bool, truncated: bool
) -> Tuple[bool, np.floating, bool]:
    """Frozen R9 semantics: dones follows terminated only; truncated still ends episode."""

    done = bool(terminated)
    mask = np.float32(0.0 if terminated else 1.0)
    episode_end = bool(terminated or truncated)
    return done, mask, episode_end


def executed_action(policy_action: Any, info: Optional[Mapping[str, Any]]) -> np.ndarray:
    if info is not None and "intervene_action" in info:
        action = np.asarray(info["intervene_action"], dtype=np.float32).reshape(-1).copy()
    else:
        action = np.asarray(policy_action, dtype=np.float32).reshape(-1).copy()
    # A shortened finite Servo window represents a smaller effective
    # high-level continuous action over the nominal 100 ms transition. Apply
    # the same rule to policy and human actions; the discrete grasp edge is not
    # duration-scaled.
    if info is not None:
        requested = int(info.get("servo_ticks_requested") or 0)
        executed = int(info.get("servo_ticks_executed") or requested)
        if requested > 0 and 0 <= executed < requested:
            action[: min(6, action.size)] *= np.float32(executed / requested)
    return action


def is_intervened(info: Optional[Mapping[str, Any]]) -> bool:
    return bool(info is not None and "intervene_action" in info)


def _require_finite_tree(name: str, obj: Any) -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            _require_finite_tree(f"{name}.{key}", value)
        return
    arr = np.asarray(obj)
    if arr.dtype == object:
        raise TransitionError(f"{name} has object dtype")
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise TransitionError(f"{name} contains NaN/Inf")


def _require_finite(name: str, array: np.ndarray) -> None:
    _require_finite_tree(name, array)


def _as_scalar_float32(name: str, value: Any) -> np.floating:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape not in ((), (1,)):
        raise TransitionError(f"{name} must be a scalar, got shape {arr.shape}")
    _require_finite(name, arr)
    return np.float32(arr.reshape(()))


def validate_transition(
    transition: Mapping[str, Any],
    *,
    observation_space: Any = None,
    action_space: Any = None,
) -> None:
    if set(transition.keys()) != set(TRANSITION_KEYS):
        raise TransitionError(
            f"transition keys {sorted(transition.keys())} != {list(TRANSITION_KEYS)}"
        )

    actions = np.asarray(transition["actions"], dtype=np.float32).reshape(-1)
    if int(actions.shape[0]) not in ACTION_DIMS:
        raise TransitionError(
            f"actions shape must be 6D or 7D, got {actions.shape}"
        )
    _require_finite("actions", actions)
    if np.any(actions < ACTION_LOW - 1e-6) or np.any(actions > ACTION_HIGH + 1e-6):
        raise TransitionError("actions outside [-1, 1]")
    if int(actions.shape[0]) == 7:
        grasp = float(actions[-1])
        if not any(np.isclose(grasp, value, atol=1e-6) for value in (-1.0, 0.0, 1.0)):
            raise TransitionError(
                f"7D grasp action must be one of -1, 0, +1, got {grasp}"
            )
    if action_space is not None and hasattr(action_space, "contains"):
        if not action_space.contains(actions):
            raise TransitionError("actions not in action_space")

    rewards = _as_scalar_float32("rewards", transition["rewards"])
    masks = _as_scalar_float32("masks", transition["masks"])
    if float(masks) not in (0.0, 1.0):
        raise TransitionError(f"masks must be 0 or 1, got {float(masks)}")

    dones = np.asarray(transition["dones"])
    if dones.shape not in ((), (1,)):
        raise TransitionError(f"dones must be a scalar, got shape {dones.shape}")
    dones_b = bool(np.asarray(dones).reshape(()))
    if dones_b and float(masks) != 0.0:
        raise TransitionError("terminated transition must have masks=0")
    if (not dones_b) and float(masks) != 1.0:
        raise TransitionError("non-terminated transition must have masks=1")
    _ = rewards

    for key in ("observations", "next_observations"):
        obs = transition[key]
        if not isinstance(obs, Mapping):
            raise TransitionError(f"{key} must be a dict")
        if "state" in obs:
            _require_finite_tree(f"{key}.state", obs["state"])
        if observation_space is not None and hasattr(observation_space, "contains"):
            if not observation_space.contains(obs):
                raise TransitionError(f"{key} not in observation_space")


def build_actor_transition(
    observation: Any,
    policy_action: Any,
    next_observation: Any,
    reward: Any,
    terminated: bool,
    truncated: bool,
    info: Optional[Mapping[str, Any]] = None,
    *,
    observation_space: Any = None,
    action_space: Any = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    done, mask, episode_end = dones_and_mask(terminated, truncated)
    action = executed_action(policy_action, info)
    transition = {
        "observations": copy.deepcopy(observation),
        "actions": action.astype(np.float32),
        "next_observations": copy.deepcopy(next_observation),
        "rewards": np.float32(reward),
        "masks": np.float32(mask),
        "dones": np.bool_(done),
    }
    validate_transition(
        transition,
        observation_space=observation_space,
        action_space=action_space,
    )
    intervened = is_intervened(info)
    if intervened:
        ia = np.asarray(info["intervene_action"], dtype=np.float32).reshape(-1)
        requested = int(info.get("servo_ticks_requested") or 0)
        executed = int(info.get("servo_ticks_executed") or requested)
        if requested > 0 and 0 <= executed < requested:
            ia = ia.copy()
            ia[: min(6, ia.size)] *= np.float32(executed / requested)
        if not np.allclose(action, ia, atol=1e-6):
            raise TransitionError("executed action != intervene_action")
    meta = {
        "episode_end": bool(episode_end),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "intervened": bool(intervened),
        "action_ignored_for_motion": bool(
            info.get("action_ignored_for_motion") if info else False
        ),
    }
    return transition, meta


def route_transition(
    transition: Mapping[str, Any],
    meta: Mapping[str, Any],
    actor_env: Any,
    actor_env_intvn: Any,
) -> None:
    actor_env.insert(transition)
    if meta.get("intervened"):
        actor_env_intvn.insert(transition)


def transition_rows_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(rows)).encode())
    for row in rows:
        validate_transition(row)
        digest.update(np.asarray(row["actions"], dtype=np.float32).tobytes())
        digest.update(np.asarray(row["rewards"], dtype=np.float32).tobytes())
        digest.update(np.asarray(row["masks"], dtype=np.float32).tobytes())
        digest.update(np.asarray(row["dones"]).astype(np.uint8).tobytes())
        obs = row["observations"]
        nxt = row["next_observations"]
        for _blob_name, blob in (("obs", obs), ("nxt", nxt)):
            state = blob["state"]
            if isinstance(state, Mapping):
                for key, value in state.items():
                    digest.update(np.asarray(value).tobytes())
            else:
                digest.update(np.asarray(state).tobytes())
            for cam_group in ("head", "wrist"):
                if cam_group in blob:
                    digest.update(np.asarray(blob[cam_group]).tobytes())
            images = blob.get("images")
            if isinstance(images, Mapping):
                for key, value in images.items():
                    digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


class ListStore:
    """Minimal insert-only store for unit tests and offline dumps."""

    def __init__(self) -> None:
        self.rows: list = []

    def insert(self, data: Any) -> None:
        self.rows.append(copy.deepcopy(data))

    def __len__(self) -> int:
        return len(self.rows)


def maybe_note_episode(
    info: MutableMapping[str, Any],
    meta: Mapping[str, Any],
    *,
    intervention_count: int,
    intervention_steps: int,
) -> None:
    if not meta.get("episode_end"):
        info.pop("episode", None)
        return
    episode = dict(info.get("episode") or {})
    episode["intervention_count"] = int(intervention_count)
    episode["intervention_steps"] = int(intervention_steps)
    episode["terminated"] = bool(meta.get("terminated"))
    episode["truncated"] = bool(meta.get("truncated"))
    info["episode"] = episode
