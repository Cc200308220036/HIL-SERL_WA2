"""Augment R11 6D demo transitions with a discrete grasp channel."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

from hilserl_wa2.experiments.demo_io import (
    DemoIOError,
    dump_json,
    dump_transitions,
    load_bundle,
    load_json,
)
from hilserl_wa2.experiments.transition import validate_transition
from hilserl_wa2.wrappers.grasp_action import ARM_DIM, GRASP_DIM

DEFAULT_GRASP_TARGET = np.asarray([0.1, 0.9, 0.7, 0.7, 0.4, 0.4], dtype=np.float32)
DEFAULT_RELEASE_TARGET = np.asarray([0.1, 0.9, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)


def hand_joints_from_obs(obs: Mapping[str, Any]) -> np.ndarray:
    state = obs["state"]
    if isinstance(state, Mapping):
        joints = np.asarray(state["hand_joints"], dtype=np.float32).reshape(-1)
    else:
        joints = np.asarray(state, dtype=np.float32).reshape(-1)[-6:]
    if joints.shape[0] < 6:
        raise DemoIOError(f"hand joints dim {joints.shape} too small")
    return joints[-6:].astype(np.float32)


def infer_grasp_dim(
    current: np.ndarray,
    nxt: np.ndarray,
    *,
    grasp_target: np.ndarray = DEFAULT_GRASP_TARGET,
    release_target: np.ndarray = DEFAULT_RELEASE_TARGET,
    min_delta: float = 0.05,
) -> float:
    """+1 approaching grasp target, -1 approaching release, else 0."""

    d_g0 = float(np.linalg.norm(current - grasp_target))
    d_g1 = float(np.linalg.norm(nxt - grasp_target))
    d_r0 = float(np.linalg.norm(current - release_target))
    d_r1 = float(np.linalg.norm(nxt - release_target))
    if d_g0 - d_g1 >= min_delta and d_g1 < d_r1:
        return 1.0
    if d_r0 - d_r1 >= min_delta and d_r1 < d_g1:
        return -1.0
    return 0.0


def augment_action(action: Any, grasp: float) -> np.ndarray:
    arm = np.asarray(action, dtype=np.float32).reshape(-1)
    if arm.shape[0] == GRASP_DIM:
        arm = arm[:ARM_DIM]
    if arm.shape[0] != ARM_DIM:
        raise DemoIOError(f"demo action dim {arm.shape} is not 6 or 7")
    out = np.concatenate([arm, np.asarray([float(grasp)], dtype=np.float32)])
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def augment_transitions(
    transitions: Sequence[Mapping[str, Any]],
    *,
    grasp_target: np.ndarray = DEFAULT_GRASP_TARGET,
    release_target: np.ndarray = DEFAULT_RELEASE_TARGET,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    nonzero = 0
    for row in transitions:
        payload = copy.deepcopy(dict(row))
        curr = hand_joints_from_obs(payload["observations"])
        nxt = hand_joints_from_obs(payload["next_observations"])
        grasp = infer_grasp_dim(
            curr, nxt, grasp_target=grasp_target, release_target=release_target
        )
        payload["actions"] = augment_action(payload["actions"], grasp)
        validate_transition(payload)
        if abs(grasp) > 0:
            nonzero += 1
        rows.append(payload)
    if not rows:
        raise DemoIOError("no transitions to augment")
    return rows


def augment_bundle(
    bundle_dir: Union[str, Path],
    out_dir: Union[str, Path],
) -> Dict[str, Any]:
    packed = load_bundle(bundle_dir)
    rows = augment_transitions(packed["transitions"])
    n_grasp = int(sum(abs(float(np.asarray(r["actions"]).reshape(-1)[-1])) > 0 for r in rows))
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dump_transitions(out / "demo.pkl", rows)
    manifest = dict(packed.get("manifest") or load_json(Path(bundle_dir) / "bundle.json"))
    manifest["bundle_name"] = str(manifest.get("bundle_name") or out.name) + "_7d"
    manifest["action_dim"] = GRASP_DIM
    manifest["transition_schema_version"] = "r13-v1"
    manifest["n_transitions"] = len(rows)
    manifest["n_grasp_nonzero"] = n_grasp
    dump_json(out / "bundle.json", manifest)
    sidecar_src = Path(bundle_dir) / "episodes"
    if sidecar_src.is_dir():
        dest = out / "episodes"
        dest.mkdir(exist_ok=True)
        for path in sorted(sidecar_src.glob("*.json")):
            payload = load_json(path)
            dump_json(dest / path.name, payload)
    return {
        "n_transitions": len(rows),
        "n_grasp_nonzero": n_grasp,
        "out_dir": str(out.resolve()),
    }
