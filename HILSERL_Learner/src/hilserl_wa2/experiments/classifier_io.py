"""R12 classifier image-bundle IO. No ROS / JAX / Env / recorder."""

from __future__ import annotations

import glob
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SCHEMA_VERSION = "r12-v1"
FROZEN_SPACE_HASH = (
    "b7a0b860d94b648a56cc453d772c478886dc4d0acecb89d6e8eb527b6831b367"
)
CLASSIFIER_KEYS = ("head", "wrist")
IMAGE_HW = (128, 128)
SAMPLE_KEYS = ("episode_id", "label", "index", "created_at", "observations")
OBS_KEYS = ("head", "wrist")
FORBIDDEN_SAMPLE_KEYS = (
    "actions",
    "next_observations",
    "rewards",
    "masks",
    "dones",
    "infos",
)
R11_TRANSITION_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "masks",
    "dones",
)
REQUIRED_BUNDLE_KEYS = (
    "schema_version",
    "task_id",
    "exp_name",
    "n_success",
    "n_failure",
    "n_episodes",
    "space_hash",
    "config_bundle_hash",
    "classifier_keys",
    "image_hw",
    "operator",
    "mode",
    "bundle_name",
    "episode_sidecars",
)
THRESHOLD_CANDIDATES = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
MIN_TEST_PRECISION = 0.85
MIN_TEST_RECALL = 0.80
SPLIT_RATIOS = (0.70, 0.15, 0.15)


class ClassifierIOError(ValueError):
    """Invalid R12 classifier sample, bundle, or split."""


def _as_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def resolve_single_bundle_dir(raw: str) -> Path:
    text = str(raw)
    if any(ch in text for ch in "*?["):
        matches = sorted(glob.glob(text))
        if len(matches) != 1:
            raise ClassifierIOError(
                f"glob matched {len(matches)} paths (need exactly 1): {text}"
            )
        return _as_path(matches[0])
    return _as_path(text)


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
        raise ClassifierIOError(f"{src} is not a JSON object")
    return payload


def dump_samples(path: Union[str, Path], samples: Sequence[Mapping[str, Any]]) -> None:
    out = _as_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [dict(row) for row in samples]
    with out.open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)


def load_samples(path: Union[str, Path]) -> List[Dict[str, Any]]:
    src = _as_path(path)
    with src.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list):
        raise ClassifierIOError(f"{src} is not a list of samples")
    return payload


def _copy_image(image: Any, *, key: str) -> np.ndarray:
    arr = np.asarray(image)
    if arr.shape != (1, IMAGE_HW[0], IMAGE_HW[1], 3):
        raise ClassifierIOError(
            f"observations[{key}] shape {arr.shape} != (1, 128, 128, 3)"
        )
    if arr.dtype != np.uint8:
        raise ClassifierIOError(f"observations[{key}] dtype {arr.dtype} != uint8")
    return np.array(arr, copy=True, dtype=np.uint8)


def extract_images(obs: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    if not isinstance(obs, Mapping):
        raise ClassifierIOError("observations must be a mapping")
    if "images" in obs:
        raise ClassifierIOError("nested observations['images'] is forbidden")
    missing = [key for key in OBS_KEYS if key not in obs]
    if missing:
        raise ClassifierIOError(f"observations missing {missing}")
    return {key: _copy_image(obs[key], key=key) for key in OBS_KEYS}


def make_sample(
    *,
    episode_id: str,
    label: int,
    index: int,
    created_at: str,
    observations: Mapping[str, Any],
) -> Dict[str, Any]:
    if int(label) not in (0, 1):
        raise ClassifierIOError(f"label must be 0 or 1, got {label!r}")
    sample = {
        "episode_id": str(episode_id),
        "label": int(label),
        "index": int(index),
        "created_at": str(created_at),
        "observations": extract_images(observations),
    }
    validate_sample(sample)
    return sample


def validate_sample(sample: Mapping[str, Any]) -> None:
    if not isinstance(sample, Mapping):
        raise ClassifierIOError("sample is not a mapping")
    keys = set(sample.keys())
    if keys == set(R11_TRANSITION_KEYS) or (
        set(R11_TRANSITION_KEYS).issubset(keys) and "label" not in keys
    ):
        raise ClassifierIOError(
            "R11 demo transition is not a classifier sample; refuse demo.pkl contents"
        )
    extra_forbidden = [key for key in FORBIDDEN_SAMPLE_KEYS if key in sample]
    if extra_forbidden:
        raise ClassifierIOError(f"sample has forbidden keys: {extra_forbidden}")
    missing = [key for key in SAMPLE_KEYS if key not in sample]
    if missing:
        raise ClassifierIOError(f"sample missing keys: {missing}")
    if set(sample.keys()) != set(SAMPLE_KEYS):
        raise ClassifierIOError(
            f"sample keys {sorted(sample.keys())} != {list(SAMPLE_KEYS)}"
        )
    if int(sample["label"]) not in (0, 1):
        raise ClassifierIOError("sample label must be 0 or 1")
    if not str(sample["episode_id"]):
        raise ClassifierIOError("episode_id must be non-empty")
    extract_images(sample["observations"])


def validate_bundle_manifest(manifest: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_BUNDLE_KEYS if key not in manifest]
    if missing:
        raise ClassifierIOError(f"bundle.json missing keys: {missing}")
    if str(manifest.get("schema_version")) != SCHEMA_VERSION:
        raise ClassifierIOError("bundle.json schema_version must be r12-v1")
    if str(manifest.get("space_hash")) != FROZEN_SPACE_HASH:
        raise ClassifierIOError(
            f"bundle.json space_hash must stay {FROZEN_SPACE_HASH}"
        )
    keys = manifest.get("classifier_keys")
    if list(keys) != list(CLASSIFIER_KEYS):
        raise ClassifierIOError("bundle.json classifier_keys must be [head, wrist]")
    if list(manifest.get("image_hw")) != [128, 128]:
        raise ClassifierIOError("bundle.json image_hw must be [128, 128]")


def write_sha256sums(bundle_dir: Union[str, Path]) -> Path:
    root = _as_path(bundle_dir)
    rels = ["success.pkl", "failure.pkl", "bundle.json"]
    episodes = root / "episodes"
    if episodes.is_dir():
        rels.extend(
            sorted(
                str(path.relative_to(root))
                for path in episodes.glob("*.json")
                if path.is_file()
            )
        )
    lines = []
    for rel in rels:
        path = root / rel
        if not path.is_file():
            raise ClassifierIOError(f"cannot hash missing file: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    out = root / "SHA256SUMS"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_classifier_bundle(
    bundle_dir: Union[str, Path],
    *,
    bundle_name: str,
    success: Sequence[Mapping[str, Any]],
    failure: Sequence[Mapping[str, Any]],
    manifest_extra: Mapping[str, Any],
    episode_sidecars: Sequence[Mapping[str, Any]],
) -> Path:
    root = _as_path(bundle_dir)
    if root.exists() and any(root.iterdir()):
        raise ClassifierIOError(f"refusing to overwrite non-empty bundle: {root}")
    for sample in list(success) + list(failure):
        validate_sample(sample)
    dump_samples(root / "success.pkl", success)
    dump_samples(root / "failure.pkl", failure)
    rels: List[str] = []
    for index, sidecar in enumerate(episode_sidecars):
        payload = dict(sidecar)
        rel = str(payload.get("rel") or f"episodes/ep{index:03d}.json")
        payload.pop("rel", None)
        dump_json(root / rel, payload)
        rels.append(rel)
    episode_ids = {
        str(row["episode_id"]) for row in list(success) + list(failure)
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bundle_name": str(bundle_name),
        "n_success": int(len(success)),
        "n_failure": int(len(failure)),
        "n_episodes": int(len(episode_ids)),
        "classifier_keys": list(CLASSIFIER_KEYS),
        "image_hw": [IMAGE_HW[0], IMAGE_HW[1]],
        "episode_sidecars": rels,
        **dict(manifest_extra),
    }
    manifest["n_success"] = int(len(success))
    manifest["n_failure"] = int(len(failure))
    manifest["n_episodes"] = int(len(episode_ids))
    validate_bundle_manifest(manifest)
    dump_json(root / "bundle.json", manifest)
    write_sha256sums(root)
    return root


def load_classifier_bundle(bundle_dir: Union[str, Path]) -> Dict[str, Any]:
    root = _as_path(bundle_dir)
    if not root.is_dir():
        raise ClassifierIOError(f"bundle-dir is not a directory: {root}")
    if (root / "demo.pkl").is_file():
        raise ClassifierIOError(f"refusing bundle that contains demo.pkl: {root}")
    if "failed" in root.parts and root.name == "failed":
        raise ClassifierIOError(f"refusing failed/ path: {root}")
    success_path = root / "success.pkl"
    failure_path = root / "failure.pkl"
    manifest_path = root / "bundle.json"
    if not success_path.is_file() or not failure_path.is_file() or not manifest_path.is_file():
        raise ClassifierIOError(
            f"bundle-dir must contain success.pkl, failure.pkl, bundle.json: {root}"
        )
    manifest = load_json(manifest_path)
    validate_bundle_manifest(manifest)
    success = load_samples(success_path)
    failure = load_samples(failure_path)
    for sample in success:
        validate_sample(sample)
        if int(sample["label"]) != 1:
            raise ClassifierIOError("success.pkl sample label must be 1")
    for sample in failure:
        validate_sample(sample)
        if int(sample["label"]) != 0:
            raise ClassifierIOError("failure.pkl sample label must be 0")
    if int(manifest["n_success"]) != len(success):
        raise ClassifierIOError("bundle n_success does not match success.pkl")
    if int(manifest["n_failure"]) != len(failure):
        raise ClassifierIOError("bundle n_failure does not match failure.pkl")
    samples = success + failure
    episode_ids = {str(row["episode_id"]) for row in samples}
    if int(manifest["n_episodes"]) != len(episode_ids):
        raise ClassifierIOError("bundle n_episodes does not match unique episode_id count")
    sidecars: List[Dict[str, Any]] = []
    for rel in manifest.get("episode_sidecars") or []:
        side_path = (root / str(rel)).resolve()
        if root not in side_path.parents and side_path != root:
            raise ClassifierIOError(f"sidecar escapes bundle: {rel}")
        sidecars.append(load_json(side_path))
    return {
        "bundle_dir": root,
        "manifest": manifest,
        "success": success,
        "failure": failure,
        "samples": samples,
        "sidecars": sidecars,
    }


def _allocate_episode_ids(ids: Sequence[str], rng: np.random.Generator) -> Dict[str, str]:
    items = list(ids)
    rng.shuffle(items)
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0]: "train"}
    if n == 2:
        return {items[0]: "train", items[1]: "val"}
    n_test = max(1, int(round(n * SPLIT_RATIOS[2])))
    n_val = max(1, int(round(n * SPLIT_RATIOS[1])))
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    n_train = n - n_val - n_test
    assigned: Dict[str, str] = {}
    for ep_id in items[:n_train]:
        assigned[ep_id] = "train"
    for ep_id in items[n_train : n_train + n_val]:
        assigned[ep_id] = "val"
    for ep_id in items[n_train + n_val :]:
        assigned[ep_id] = "test"
    return assigned


def split_by_episode(
    samples: Sequence[Mapping[str, Any]],
    *,
    seed: int = 12,
) -> Dict[str, Any]:
    if not samples:
        raise ClassifierIOError("cannot split an empty sample list")
    by_ep: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in samples:
        validate_sample(row)
        by_ep[str(row["episode_id"])].append(dict(row))
    pos_eps = [
        ep_id
        for ep_id, rows in by_ep.items()
        if any(int(row["label"]) == 1 for row in rows)
    ]
    neg_eps = [ep_id for ep_id in by_ep if ep_id not in pos_eps]
    rng = np.random.default_rng(int(seed))
    assigned = {}
    assigned.update(_allocate_episode_ids(pos_eps, rng))
    assigned.update(_allocate_episode_ids(neg_eps, rng))
    splits = {"train": [], "val": [], "test": []}
    episode_split = {}
    for ep_id, rows in by_ep.items():
        name = assigned[ep_id]
        episode_split[ep_id] = name
        splits[name].extend(rows)
    validate_no_episode_leakage(splits)
    return {
        "seed": int(seed),
        "episode_split": episode_split,
        "splits": splits,
        "counts": {
            name: {
                "n_samples": len(rows),
                "n_success": int(sum(int(r["label"]) == 1 for r in rows)),
                "n_failure": int(sum(int(r["label"]) == 0 for r in rows)),
                "n_episodes": len({r["episode_id"] for r in rows}),
            }
            for name, rows in splits.items()
        },
    }


def validate_no_episode_leakage(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    seen: Dict[str, str] = {}
    for name, rows in splits.items():
        for row in rows:
            ep_id = str(row["episode_id"])
            prev = seen.get(ep_id)
            if prev is not None and prev != name:
                raise ClassifierIOError(
                    f"episode_id {ep_id} leaked across {prev} and {name}"
                )
            seen[ep_id] = name


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, Any]:
    y_true_arr = np.asarray(list(y_true), dtype=np.int32).reshape(-1)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int32).reshape(-1)
    if y_true_arr.size != y_pred_arr.size:
        raise ClassifierIOError("y_true/y_pred length mismatch")
    tp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 1)))
    tn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 0)))
    fp = int(np.sum((y_pred_arr == 1) & (y_true_arr == 0)))
    fn = int(np.sum((y_pred_arr == 0) & (y_true_arr == 1)))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    acc = float((tp + tn) / y_true_arr.size) if y_true_arr.size else 0.0
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "n": int(y_true_arr.size),
    }


def select_threshold(
    y_true: Sequence[int],
    probs: Sequence[float],
    *,
    candidates: Sequence[float] = THRESHOLD_CANDIDATES,
    min_precision: float = MIN_TEST_PRECISION,
    precision_beta: float = 0.5,
) -> Dict[str, Any]:
    """Pick a decision threshold on val.

    False positives are more costly than false negatives, but an overly high
    threshold (tiny val recall) also fails the test Gate. Among candidates with
    precision >= min_precision, maximize precision-weighted F-beta (default
    beta=0.5), then prefer a higher threshold on ties.
    """

    y_true_arr = np.asarray(list(y_true), dtype=np.int32).reshape(-1)
    p_arr = np.asarray(list(probs), dtype=np.float64).reshape(-1)
    beta = float(precision_beta)
    beta2 = beta * beta
    scored = []
    for threshold in candidates:
        pred = (p_arr >= float(threshold)).astype(np.int32)
        metrics = binary_metrics(y_true_arr, pred)
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        if precision + recall <= 0.0:
            f_beta = 0.0
        else:
            f_beta = (1.0 + beta2) * precision * recall / (beta2 * precision + recall)
        metrics["threshold"] = float(threshold)
        metrics["f_beta"] = float(f_beta)
        scored.append(metrics)
    eligible = [row for row in scored if row["precision"] >= float(min_precision)]
    if eligible:
        chosen = max(
            eligible,
            key=lambda row: (row["f_beta"], row["precision"], row["threshold"]),
        )
        ok = True
    else:
        chosen = max(
            scored,
            key=lambda row: (row["precision"], row["f_beta"], row["threshold"]),
        )
        ok = False
    return {"ok": ok, "chosen": chosen, "all": scored}


def stack_observations(
    samples: Sequence[Mapping[str, Any]],
    image_keys: Sequence[str] = CLASSIFIER_KEYS,
) -> Dict[str, np.ndarray]:
    if not samples:
        raise ClassifierIOError("cannot stack empty sample list")
    return {
        str(key): np.stack(
            [np.asarray(row["observations"][key]) for row in samples],
            axis=0,
        )
        for key in image_keys
    }


def labels_array(samples: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in samples], dtype=np.int32)


def episode_counts(samples: Sequence[Mapping[str, Any]]) -> Tuple[int, int, int]:
    success_eps = {str(row["episode_id"]) for row in samples if int(row["label"]) == 1}
    failure_eps = {str(row["episode_id"]) for row in samples if int(row["label"]) == 0}
    return len(success_eps), len(failure_eps), len(success_eps | failure_eps)


def _retag_episode_id(episode_id: str, source_tag: str) -> str:
    tag = str(source_tag).strip()
    if not tag:
        raise ClassifierIOError("source_tag must be non-empty")
    return f"{tag}__{episode_id}"


def merge_classifier_bundles(
    bundle_dirs: Sequence[Union[str, Path]],
    *,
    out_dir: Union[str, Path],
    bundle_name: str,
    operator: str = "merge",
    mode: str = "live",
) -> Path:
    """Merge multiple r12-v1 bundles. Episode IDs are prefixed per source to avoid collisions."""

    if len(bundle_dirs) < 2:
        raise ClassifierIOError("merge needs at least two bundle dirs")
    packed_list = [load_classifier_bundle(path) for path in bundle_dirs]
    space_hashes = {str(p["manifest"]["space_hash"]) for p in packed_list}
    if space_hashes != {FROZEN_SPACE_HASH}:
        raise ClassifierIOError(f"space_hash mismatch across bundles: {sorted(space_hashes)}")
    key_sets = {tuple(p["manifest"]["classifier_keys"]) for p in packed_list}
    if key_sets != {CLASSIFIER_KEYS}:
        raise ClassifierIOError(f"classifier_keys mismatch: {sorted(key_sets)}")
    task_ids = {str(p["manifest"]["task_id"]) for p in packed_list}
    if len(task_ids) != 1:
        raise ClassifierIOError(f"task_id mismatch: {sorted(task_ids)}")
    exp_names = {str(p["manifest"]["exp_name"]) for p in packed_list}
    if len(exp_names) != 1:
        raise ClassifierIOError(f"exp_name mismatch: {sorted(exp_names)}")

    success: List[Dict[str, Any]] = []
    failure: List[Dict[str, Any]] = []
    sources: List[str] = []
    for packed in packed_list:
        src = Path(packed["bundle_dir"]).name
        sources.append(src)
        for row in packed["success"]:
            sample = dict(row)
            sample["episode_id"] = _retag_episode_id(str(row["episode_id"]), src)
            validate_sample(sample)
            success.append(sample)
        for row in packed["failure"]:
            sample = dict(row)
            sample["episode_id"] = _retag_episode_id(str(row["episode_id"]), src)
            validate_sample(sample)
            failure.append(sample)

    counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"n_success": 0, "n_failure": 0}
    )
    for row in success:
        counts[str(row["episode_id"])]["n_success"] += 1
    for row in failure:
        counts[str(row["episode_id"])]["n_failure"] += 1
    sidecars = []
    for index, episode_id in enumerate(sorted(counts)):
        sidecars.append(
            {
                "episode_id": episode_id,
                "n_success": counts[episode_id]["n_success"],
                "n_failure": counts[episode_id]["n_failure"],
                "rel": f"episodes/ep{index:03d}.json",
            }
        )

    task_id = next(iter(task_ids))
    exp_name = next(iter(exp_names))
    # Prefer first bundle's config hash (YAML may be identical across captures).
    config_bundle_hash = str(packed_list[0]["manifest"]["config_bundle_hash"])
    return write_classifier_bundle(
        out_dir,
        bundle_name=str(bundle_name),
        success=success,
        failure=failure,
        manifest_extra={
            "task_id": task_id,
            "exp_name": exp_name,
            "space_hash": FROZEN_SPACE_HASH,
            "config_bundle_hash": config_bundle_hash,
            "operator": str(operator),
            "mode": str(mode),
            "merged_from": sources,
        },
        episode_sidecars=sidecars,
    )
