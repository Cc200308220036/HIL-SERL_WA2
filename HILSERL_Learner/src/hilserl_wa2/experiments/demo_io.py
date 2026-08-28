"""R11 demo bundle IO: pickle, sidecar, schema checks. No ROS / recorder / Env."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from hilserl_wa2.experiments.transition import (
    TRANSITION_KEYS,
    TransitionError,
    validate_transition,
)

TRANSITION_SCHEMA_VERSION = "r9-v1"
TRANSITION_SCHEMA_VERSIONS = ("r9-v1", "r13-v1", "r13-timescale-v2")
MIN_INTERVENED_STEPS = 30
BUNDLE_FILES = ("demo.pkl", "bundle.json")
DEMO_PKL_FORMAT_FLAT = "flat-list-v0"
DEMO_PKL_FORMAT_STREAM = "episode-stream-v1"
REQUIRED_SIDECAR_KEYS = (
    "episode_index",
    "label",
    "operator",
    "task_id",
    "exp_name",
    "config_bundle_hash",
    "space_hash",
    "transition_schema_version",
    "started_at",
    "n_steps",
    "intervened_steps",
    "intervention_count",
    "hand_toggles",
    "reset_ok",
    "human_success",
    "discard_reason",
)
REQUIRED_BUNDLE_KEYS = (
    "bundle_name",
    "pkl",
    "n_episodes",
    "label",
    "episode_sidecars",
)


class DemoIOError(ValueError):
    """Invalid demo bundle, sidecar, or transition list."""


def _as_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def assert_not_failed_path(path: Union[str, Path]) -> Path:
    resolved = _as_path(path)
    if "failed" in resolved.parts:
        raise DemoIOError(f"refusing failed/ path: {resolved}")
    return resolved


def dump_transitions(path: Union[str, Path], transitions: Sequence[Mapping[str, Any]]) -> None:
    out = _as_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [dict(row) for row in transitions]
    with out.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)


def dump_episode_stream(
    path: Union[str, Path], episodes: Sequence[Sequence[Mapping[str, Any]]]
) -> None:
    """Write one pickle list per episode so later loads can stream (not one 5GB object)."""
    out = _as_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        for episode in episodes:
            pickle.dump([dict(row) for row in episode], handle, protocol=4)


def iter_pickle_lists(path: Union[str, Path]):
    src = _as_path(path)
    with src.open("rb") as handle:
        while True:
            try:
                payload = pickle.load(handle)
            except EOFError:
                break
            if not isinstance(payload, list):
                raise DemoIOError(f"{src} pickle object is not a list of transitions")
            yield payload


def load_transitions(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load every pickle list in the file. Do not use on a multi-GB concatenated demo.pkl."""
    rows: List[Dict[str, Any]] = []
    for payload in iter_pickle_lists(path):
        rows.extend(payload)
    return rows


def episode_pkl_rel(index: int) -> str:
    return f"episodes/ep{int(index):03d}.pkl"


def resolve_bundle_dir(path: Union[str, Path]) -> Path:
    resolved = _as_path(path)
    if resolved.is_dir():
        return assert_not_failed_path(resolved)
    if resolved.is_file() and resolved.name == "demo.pkl":
        return assert_not_failed_path(resolved.parent)
    raise DemoIOError(f"expected a bundle directory or demo.pkl, got {resolved}")


def list_episode_pkl_paths(bundle_dir: Union[str, Path]) -> List[Path]:
    root = assert_not_failed_path(bundle_dir)
    manifest_path = root / "bundle.json"
    if not manifest_path.is_file():
        raise DemoIOError(f"missing bundle.json: {root}")
    manifest = load_json(manifest_path)
    rels = manifest.get("episode_pkls")
    if isinstance(rels, list) and rels:
        return [(root / str(rel)).resolve() for rel in rels]
    episodes_dir = root / "episodes"
    if not episodes_dir.is_dir():
        return []
    return sorted(episodes_dir.glob("ep[0-9][0-9][0-9].pkl"))


def require_episode_pkls(bundle_dir: Union[str, Path]) -> List[Path]:
    root = resolve_bundle_dir(bundle_dir)
    manifest = load_json(root / "bundle.json")
    validate_bundle_manifest(manifest)
    expected = int(manifest["n_episodes"])
    paths = list_episode_pkl_paths(root)
    present = [path for path in paths if path.is_file()]
    if len(present) != expected:
        raise DemoIOError(
            f"bundle {root} needs {expected} per-episode pkls, found {len(present)}. "
            "On Actor (enough RAM) run split_r13_demo_pkl.py --bundle <dir>. "
            "Do not pickle.load the concatenated demo.pkl on a 16GB Learner."
        )
    return present


def split_flat_demo_into_episode_pkls(
    bundle_dir: Union[str, Path],
    *,
    progress=print,
) -> Dict[str, Any]:
    """One-shot split of a legacy concatenated demo.pkl. Run where RAM ≥ ~2× file size."""
    root = resolve_bundle_dir(bundle_dir)
    manifest = load_json(root / "bundle.json")
    validate_bundle_manifest(manifest)
    n_ep = int(manifest["n_episodes"])
    rels = [episode_pkl_rel(index) for index in range(n_ep)]
    paths = [root / rel for rel in rels]
    if all(path.is_file() for path in paths):
        manifest["episode_pkls"] = rels
        manifest.setdefault("demo_pkl_format", DEMO_PKL_FORMAT_FLAT)
        dump_json(root / "bundle.json", manifest)
        if progress is not None:
            progress(f"R13_DEMO_SPLIT: already have {n_ep} episode pkls")
        return {"wrote": 0, "existed": n_ep, "paths": paths}

    pkl = root / "demo.pkl"
    if progress is not None:
        progress(f"R13_DEMO_SPLIT: loading {pkl} (RAM ≈ 2× file; Actor only)")
    transitions = load_transitions(pkl)
    episodes = split_episodes(transitions)
    del transitions
    if len(episodes) != n_ep:
        raise DemoIOError(
            f"demo.pkl episodes {len(episodes)} != bundle n_episodes {n_ep}"
        )
    wrote = 0
    for index, episode in enumerate(episodes):
        dest = paths[index]
        if not dest.is_file():
            dump_transitions(dest, episode)
            wrote += 1
        if progress is not None:
            progress(f"EP{index:03d} n_steps={len(episode)} file={dest.name}")
        episodes[index] = []
    del episodes
    manifest["episode_pkls"] = rels
    manifest.setdefault("demo_pkl_format", DEMO_PKL_FORMAT_FLAT)
    dump_json(root / "bundle.json", manifest)
    return {"wrote": wrote, "existed": n_ep - wrote, "paths": paths}


def dump_json(path: Union[str, Path], payload: Mapping[str, Any]) -> None:
    out = _as_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    src = _as_path(path)
    with src.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise DemoIOError(f"{src} is not a JSON object")
    return payload


def mark_human_success(transitions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Finalize a human-confirmed success episode (upstream-aligned sparse reward).

    Intermediate steps stay reward 0 with masks=1. The last step is the success
    absorbing transition: dones=True, masks=0, **rewards=1** (HIL-SERL
    ``record_demos.py`` keeps classifier terminal reward; WA2 previously forced 0).
    """

    if not transitions:
        raise DemoIOError("cannot mark an empty episode as success")
    rows = [dict(row) for row in transitions]
    for row in rows[:-1]:
        row["dones"] = np.bool_(False)
        row["masks"] = np.float32(1.0)
        row["rewards"] = np.float32(0.0)
        validate_transition(row)
    last = rows[-1]
    last["dones"] = np.bool_(True)
    last["masks"] = np.float32(0.0)
    last["rewards"] = np.float32(1.0)
    validate_transition(last)
    return rows


def _reward_is_binary(value: Any) -> bool:
    reward = float(np.asarray(value).reshape(()))
    return abs(reward) < 1e-6 or abs(reward - 1.0) < 1e-6


def validate_demo_list(
    transitions: Sequence[Mapping[str, Any]],
    *,
    observation_space: Any = None,
    action_space: Any = None,
) -> None:
    if not transitions:
        raise DemoIOError("demo list is empty")
    for index, row in enumerate(transitions):
        if not isinstance(row, Mapping):
            raise DemoIOError(f"transition {index} is not a mapping")
        if "infos" in row:
            raise DemoIOError("infos must not appear in demo transitions")
        if set(row.keys()) != set(TRANSITION_KEYS):
            raise DemoIOError(
                f"transition {index} keys {sorted(row.keys())} != {list(TRANSITION_KEYS)}"
            )
        validate_transition(
            row,
            observation_space=observation_space,
            action_space=action_space,
        )
        if not _reward_is_binary(row["rewards"]):
            raise DemoIOError(
                f"transition {index} reward must be 0 or 1, "
                f"got {float(np.asarray(row['rewards']).reshape(()))}"
            )

def grasp_action_counts(transitions: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Count discrete grasp-channel edges in 7D actions: +1 grasp, -1 release."""

    plus = 0
    minus = 0
    for row in transitions:
        action = np.asarray(row["actions"], dtype=np.float32).reshape(-1)
        if int(action.shape[0]) != 7:
            continue
        value = float(np.round(action[-1]))
        if value >= 1.0:
            plus += 1
        elif value <= -1.0:
            minus += 1
    return {"plus": int(plus), "minus": int(minus), "nonzero": int(plus + minus)}


def validate_r13_grasp_edges(transitions: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """R13 success demo must record real grasp and release commands, not inferred."""

    for index, row in enumerate(transitions):
        action = np.asarray(row["actions"], dtype=np.float32).reshape(-1)
        if int(action.shape[0]) != 7:
            raise DemoIOError(f"R13 transition {index} actions must be 7D, got {action.shape}")
    counts = grasp_action_counts(transitions)
    if counts["plus"] < 1 or counts["minus"] < 1:
        raise DemoIOError(
            "R13 success episode needs at least one grasp (+1) and one release (-1) "
            f"in actions[6], got plus={counts['plus']} minus={counts['minus']}"
        )
    return counts


def split_episodes(transitions: Sequence[Mapping[str, Any]]) -> List[List[Dict[str, Any]]]:
    validate_demo_list(transitions)
    episodes: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for row in transitions:
        current.append(dict(row))
        if bool(np.asarray(row["dones"]).reshape(())):
            episodes.append(current)
            current = []
    if current:
        raise DemoIOError("demo list must end on a terminated (dones=True) step")
    return episodes


def validate_success_episode(
    transitions: Sequence[Mapping[str, Any]],
    *,
    require_intervention: bool = False,
    min_intervened_steps: int = MIN_INTERVENED_STEPS,
) -> Dict[str, int]:
    validate_demo_list(transitions)
    n_steps = len(transitions)
    for index, row in enumerate(transitions):
        done = bool(np.asarray(row["dones"]).reshape(()))
        mask = float(np.asarray(row["masks"]).reshape(()))
        if index < n_steps - 1:
            if done or mask != 1.0:
                raise DemoIOError("only the last success step may be terminated")
        else:
            if (not done) or mask != 0.0:
                raise DemoIOError("success episode last step must have dones=True masks=0")
            last_reward = float(np.asarray(row["rewards"]).reshape(()))
            if abs(last_reward - 1.0) > 1e-6:
                raise DemoIOError(
                    "success episode last step must have reward=1 "
                    f"(upstream-aligned), got {last_reward}"
                )
    intervened = 0
    for row in transitions:
        action = np.asarray(row["actions"], dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(action)) > 1e-6:
            intervened += 1
    if require_intervention and intervened < int(min_intervened_steps):
        raise DemoIOError(
            f"intervened_steps {intervened} < {min_intervened_steps}"
        )
    return {"n_steps": n_steps, "intervened_steps": intervened}


def validate_sidecar(sidecar: Mapping[str, Any], *, label: str = "success") -> None:
    missing = [key for key in REQUIRED_SIDECAR_KEYS if key not in sidecar]
    if missing:
        raise DemoIOError(f"sidecar missing keys: {missing}")
    if str(sidecar["transition_schema_version"]) not in TRANSITION_SCHEMA_VERSIONS:
        raise DemoIOError(
            "sidecar transition_schema_version must be r9-v1, r13-v1 or "
            "r13-timescale-v2"
        )
    if str(sidecar["transition_schema_version"]) == "r13-timescale-v2":
        for key in (
            "policy_hz",
            "servo_hz",
            "servo_ticks_per_action",
            "discount",
            "classifier_consecutive_n",
        ):
            if key not in sidecar:
                raise DemoIOError(f"timescale-v2 sidecar missing {key}")
    if str(sidecar["label"]) != label:
        raise DemoIOError(f"sidecar label must be {label!r}")


def validate_bundle_manifest(manifest: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_BUNDLE_KEYS if key not in manifest]
    if missing:
        raise DemoIOError(f"bundle.json missing keys: {missing}")
    if str(manifest.get("pkl")) != "demo.pkl":
        raise DemoIOError("bundle.json pkl must be demo.pkl")
    sidecars = manifest.get("episode_sidecars")
    if not isinstance(sidecars, list) or not sidecars:
        raise DemoIOError("bundle.json episode_sidecars must be a non-empty list")
    if int(manifest["n_episodes"]) != len(sidecars):
        raise DemoIOError("bundle.json n_episodes must match sidecar count")
    pkls = manifest.get("episode_pkls")
    if pkls is not None:
        if not isinstance(pkls, list) or len(pkls) != int(manifest["n_episodes"]):
            raise DemoIOError("bundle.json episode_pkls must match n_episodes")


def load_bundle(bundle_dir: Union[str, Path]) -> Dict[str, Any]:
    root = assert_not_failed_path(bundle_dir)
    if not root.is_dir():
        raise DemoIOError(f"bundle-dir is not a directory: {root}")
    pkl = root / "demo.pkl"
    manifest_path = root / "bundle.json"
    episodes_dir = root / "episodes"
    if not pkl.is_file() or not manifest_path.is_file() or not episodes_dir.is_dir():
        raise DemoIOError(
            f"bundle-dir must contain demo.pkl, bundle.json, episodes/: {root}"
        )
    manifest = load_json(manifest_path)
    validate_bundle_manifest(manifest)
    pkl_rels = manifest.get("episode_pkls")
    if isinstance(pkl_rels, list) and pkl_rels:
        episodes = [load_transitions(root / str(rel)) for rel in pkl_rels]
        transitions = [row for episode in episodes for row in episode]
    else:
        transitions = load_transitions(pkl)
        episodes = split_episodes(transitions)
    validate_demo_list(transitions)
    sidecars: List[Dict[str, Any]] = []
    for rel in manifest["episode_sidecars"]:
        side_path = (root / str(rel)).resolve()
        if root not in side_path.parents and side_path != root:
            raise DemoIOError(f"sidecar escapes bundle: {rel}")
        sidecar = load_json(side_path)
        validate_sidecar(sidecar, label=str(manifest.get("label", "success")))
        sidecars.append(sidecar)
    if len(episodes) != len(sidecars):
        raise DemoIOError(
            f"pkl episodes {len(episodes)} != sidecar count {len(sidecars)}"
        )
    if int(manifest["n_episodes"]) != len(episodes):
        raise DemoIOError("bundle n_episodes does not match pkl episode count")
    success_sidecars = [row for row in sidecars if str(row["label"]) == "success"]
    if str(manifest.get("label", "success")) == "success":
        if len(success_sidecars) != len(sidecars):
            raise DemoIOError("success bundle sidecar labels must all be success")
    return {
        "bundle_dir": root,
        "manifest": manifest,
        "transitions": transitions,
        "episodes": episodes,
        "sidecars": sidecars,
    }


def write_success_bundle(
    bundle_dir: Union[str, Path],
    *,
    bundle_name: str,
    episodes: Sequence[Sequence[Mapping[str, Any]]],
    sidecars: Sequence[Mapping[str, Any]],
) -> Path:
    root = assert_not_failed_path(bundle_dir)
    if root.exists() and any(root.iterdir()):
        raise DemoIOError(f"refusing to overwrite non-empty bundle: {root}")
    if len(episodes) != len(sidecars):
        raise DemoIOError("episodes and sidecars length mismatch")
    marked_episodes: List[List[Dict[str, Any]]] = []
    rels: List[str] = []
    pkl_rels: List[str] = []
    written_sides: List[Dict[str, Any]] = []
    for index, (episode, sidecar) in enumerate(zip(episodes, sidecars)):
        marked = mark_human_success(episode)
        validate_success_episode(marked)
        marked_episodes.append(marked)
        payload = dict(sidecar)
        payload["episode_index"] = int(payload.get("episode_index", index))
        payload["label"] = "success"
        payload["human_success"] = True
        payload["n_steps"] = int(len(marked))
        payload["transition_schema_version"] = str(
            payload.get("transition_schema_version") or TRANSITION_SCHEMA_VERSION
        )
        if payload["transition_schema_version"] in ("r13-v1", "r13-timescale-v2"):
            counts = validate_r13_grasp_edges(marked)
            payload["action_dim"] = 7
            payload["n_grasp_plus"] = int(counts["plus"])
            payload["n_grasp_minus"] = int(counts["minus"])
        validate_sidecar(payload, label="success")
        rel = f"episodes/ep{index:03d}.json"
        dump_json(root / rel, payload)
        rels.append(rel)
        pkl_rel = episode_pkl_rel(index)
        dump_transitions(root / pkl_rel, marked)
        pkl_rels.append(pkl_rel)
        written_sides.append(payload)
    dump_episode_stream(root / "demo.pkl", marked_episodes)
    manifest = {
        "bundle_name": str(bundle_name),
        "pkl": "demo.pkl",
        "n_episodes": int(len(episodes)),
        "label": "success",
        "episode_sidecars": rels,
        "episode_pkls": pkl_rels,
        "demo_pkl_format": DEMO_PKL_FORMAT_STREAM,
    }
    dump_json(root / "bundle.json", manifest)
    return root


def write_failed_episode(
    failed_dir: Union[str, Path],
    *,
    episode_index: int,
    transitions: Sequence[Mapping[str, Any]],
    sidecar: Mapping[str, Any],
) -> Path:
    root = Path(failed_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"ep{int(episode_index):03d}"
    dump_transitions(root / f"{stem}.pkl", list(transitions))
    payload = dict(sidecar)
    payload["episode_index"] = int(episode_index)
    payload["label"] = str(payload.get("label", "failed"))
    payload["human_success"] = False
    payload["transition_schema_version"] = str(
        payload.get("transition_schema_version") or TRANSITION_SCHEMA_VERSION
    )
    dump_json(root / f"{stem}.json", payload)
    return root / f"{stem}.pkl"


def tcp_xyz(obs: Mapping[str, Any]) -> np.ndarray:
    state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
    if state.size < 3:
        raise DemoIOError("observation state too short for TCP xyz")
    return state[:3].copy()


def tcp_deltas(transitions: Sequence[Mapping[str, Any]]) -> List[List[float]]:
    deltas: List[List[float]] = []
    for index in range(1, len(transitions)):
        prev = tcp_xyz(transitions[index - 1]["observations"])
        cur = tcp_xyz(transitions[index]["observations"])
        deltas.append((cur - prev).astype(np.float32).tolist())
    return deltas


def count_intervened_steps(transitions: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in transitions:
        action = np.asarray(row["actions"], dtype=np.float32).reshape(-1)
        if float(np.linalg.norm(action)) > 1e-6:
            count += 1
    return count


def images_are_real(transitions: Iterable[Mapping[str, Any]], sample: int = 8) -> bool:
    rows = list(transitions)
    if not rows:
        return False
    picks = rows[:: max(1, len(rows) // sample)][:sample]
    for row in picks:
        for key in ("head", "wrist"):
            image = np.asarray(row["observations"][key])
            if image.size == 0 or not np.any(image):
                return False
    return True
