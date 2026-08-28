#!/usr/bin/env python3
"""R13 Learner: RLPD + hybrid SAC, dual stores, handshake. No ROS / no robot."""

from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT),
    str(REPO / "src") if (REPO / "src").is_dir() else str(ROOT),
    str((REPO / "src" / "hil-serl-main" / "examples") if (REPO / "src").is_dir() else ROOT / "hil-serl-main" / "examples"),
    str((REPO / "src" / "hil-serl-main" / "serl_launcher") if (REPO / "src").is_dir() else ROOT / "hil-serl-main" / "serl_launcher"),
]


CONTINUOUS_TARGET_ENTROPY = -3.0  # Six continuous EEF dimensions, using HIL-SERL's -dim/2 rule.
# Soft cost on executed grasp/release edges. -0.02 was collapsing grasp_critic to
# "always hold" under sparse success; keep a tiny anti-chatter term only.
GRASP_ACTION_PENALTY = np.float32(-0.002)
# Hybrid SAC temperature_init in make_sac_pixel_agent_hybrid_single_arm.
SAC_TEMPERATURE_INIT = 1e-2
# Floor for α: raised from 0.01 — HIL was pinning the floor and under-exploring.
DEFAULT_MIN_TEMPERATURE = 5e-2
# Resume kick: same as floor (lift collapsed ckpts straight to the working band).
DEFAULT_RESUME_TEMPERATURE = 5e-2


def set_grasp_action_penalty(value: float) -> np.floating:
    """Update the process-wide grasp penalty used by PenaltyStore / cache migrate."""

    global GRASP_ACTION_PENALTY
    GRASP_ACTION_PENALTY = np.float32(value)
    return GRASP_ACTION_PENALTY


def softplus_inv(y: float) -> float:
    """Inverse of softplus for GeqLagrangeMultiplier(parameterization='softplus')."""

    y = float(max(y, 1e-8))
    return float(np.log(np.expm1(y)))


def _find_temperature_lagrange(params: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Return the key path to the temperature lagrange leaf.

    ModuleDict scopes it as ``modules_temperature/lagrange`` (not ``temperature/lagrange``).
    """

    candidates = (
        ("modules_temperature", "lagrange"),
        ("temperature", "lagrange"),
    )
    for path in candidates:
        cur: Any = params
        ok = True
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return path
    # Fallback: scan one level for *temperature*/lagrange
    for key, val in params.items():
        if "temperature" in str(key).lower() and isinstance(val, Mapping) and "lagrange" in val:
            return (key, "lagrange")
    raise KeyError(
        "agent params missing temperature lagrange "
        f"(top-level keys={list(params.keys())[:20]})"
    )


def _replace_mapping_path(tree: Any, path: Tuple[Any, ...], value: Any) -> Any:
    """Replace a nested mapping leaf while preserving FrozenDict containers.

    ``unfreeze``/``freeze`` of the whole tree turns params into plain dicts and
    then mismatches Adam ``opt_state`` (FrozenDict) → Optax ValueError.
    """

    try:
        from flax.core.frozen_dict import FrozenDict
    except ImportError:  # pragma: no cover
        FrozenDict = ()  # type: ignore

    key = path[0]
    if len(path) == 1:
        if FrozenDict and isinstance(tree, FrozenDict):
            return tree.copy(add_or_replace={key: value})
        out = dict(tree)
        out[key] = value
        return out
    child = tree[key]
    new_child = _replace_mapping_path(child, path[1:], value)
    if FrozenDict and isinstance(tree, FrozenDict):
        return tree.copy(add_or_replace={key: new_child})
    out = dict(tree)
    out[key] = new_child
    return out


def read_sac_temperature(agent) -> float:
    """Effective α = softplus(lagrange)."""

    value = agent.forward_temperature()
    return float(np.asarray(value).reshape(()))


def set_sac_temperature(agent, alpha: float):
    """Write temperature so softplus(lagrange) == alpha (actor/critic trees untouched)."""

    import jax.numpy as jnp

    alpha = float(max(alpha, 1e-8))
    raw = softplus_inv(alpha)
    path = _find_temperature_lagrange(agent.state.params)
    leaf = agent.state.params
    for key in path:
        leaf = leaf[key]
    new_leaf = jnp.asarray(raw, dtype=jnp.asarray(leaf).dtype)
    # Keep scalar shape if the stored lagrange is 0-d / () 
    if hasattr(leaf, "shape") and tuple(leaf.shape) != ():
        new_leaf = jnp.full(np.shape(leaf), raw, dtype=jnp.asarray(leaf).dtype)

    new_params = _replace_mapping_path(agent.state.params, path, new_leaf)

    new_target = getattr(agent.state, "target_params", None)
    if new_target is not None:
        try:
            tpath = _find_temperature_lagrange(new_target)
            tleaf = new_target
            for key in tpath:
                tleaf = tleaf[key]
            tval = jnp.asarray(raw, dtype=jnp.asarray(tleaf).dtype)
            if hasattr(tleaf, "shape") and tuple(tleaf.shape) != ():
                tval = jnp.full(np.shape(tleaf), raw, dtype=jnp.asarray(tleaf).dtype)
            new_target = _replace_mapping_path(new_target, tpath, tval)
        except KeyError:
            pass

    # Re-init temperature Adam state so stale momentum does not yank α back,
    # and so opt_state tree type stays aligned with params.
    new_opt_states = agent.state.opt_states
    txs = getattr(agent.state, "txs", None)
    if isinstance(new_opt_states, Mapping) and isinstance(txs, Mapping) and "temperature" in txs:
        try:
            opt_map = (
                dict(new_opt_states)
                if not hasattr(new_opt_states, "copy")
                else dict(new_opt_states)
            )
            opt_map["temperature"] = txs["temperature"].init(new_params)
            new_opt_states = opt_map
        except Exception:  # noqa: BLE001
            new_opt_states = agent.state.opt_states

    return agent.replace(
        state=agent.state.replace(
            params=new_params,
            target_params=new_target,
            opt_states=new_opt_states,
        )
    )


def ensure_min_temperature(agent, min_alpha: float) -> Tuple[Any, bool, float]:
    """Clamp α up to ``min_alpha``. Returns (agent, bumped, alpha_after)."""

    min_alpha = float(min_alpha)
    if min_alpha <= 0.0:
        alpha = read_sac_temperature(agent)
        return agent, False, alpha
    alpha = read_sac_temperature(agent)
    if alpha + 1e-12 >= min_alpha:
        return agent, False, alpha
    agent = set_sac_temperature(agent, min_alpha)
    return agent, True, min_alpha


def maybe_reset_temperature_on_resume(
    agent,
    *,
    min_alpha: float,
    resume_alpha: float,
) -> Tuple[Any, Optional[float], float]:
    """If loaded α is below the floor, kick to ``resume_alpha`` (or floor if resume≤0)."""

    loaded = read_sac_temperature(agent)
    min_alpha = float(min_alpha)
    resume_alpha = float(resume_alpha)
    if min_alpha <= 0.0:
        return agent, None, loaded
    if loaded + 1e-12 >= min_alpha:
        return agent, None, loaded
    target = resume_alpha if resume_alpha > 0.0 else min_alpha
    target = max(target, min_alpha)
    agent = set_sac_temperature(agent, target)
    return agent, loaded, target


def _with_penalty(row: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the grasp action cost from the executed 7-D action.

    The actor stores the command that was actually applied by the grasp wrapper,
    so only a non-zero final dimension represents a physical grasp/release event.
    Existing values are deliberately replaced to keep demos, resumed buffers and
    online intervention data on the same reward-shaping rule.
    """
    payload = dict(row)
    action = np.asarray(payload["actions"], dtype=np.float32).reshape(-1)
    operated = action.size == 7 and not np.isclose(action[-1], 0.0, atol=1e-6)
    payload["grasp_penalty"] = np.float32(GRASP_ACTION_PENALTY if operated else 0.0)
    return payload


def _refresh_buffer_grasp_penalties(buffer) -> None:
    """Migrate cached buffers to the current grasp-penalty rule in-place."""
    size = int(len(buffer))
    dataset = buffer.dataset_dict
    if size < 1 or "actions" not in dataset or "grasp_penalty" not in dataset:
        return
    actions = np.asarray(dataset["actions"][:size])
    penalties = np.zeros((size,), dtype=np.float32)
    if actions.ndim == 2 and actions.shape[1] == 7:
        operated = ~np.isclose(actions[:, -1], 0.0, atol=1e-6)
        penalties[operated] = GRASP_ACTION_PENALTY
    dataset["grasp_penalty"][:size] = penalties


class PenaltyStore:
    def __init__(self, inner, validate):
        self.inner = inner
        self._validate = validate

    def insert(self, data):
        payload = dict(data)
        core = {
            key: payload[key]
            for key in (
                "observations",
                "actions",
                "next_observations",
                "rewards",
                "masks",
                "dones",
            )
        }
        self._validate(core)
        self.inner.insert(_with_penalty(payload))

    def batch_insert(self, batch_data):
        # agentlace calls batch_insert; do not let __getattr__ bind inner.insert
        # (that path skips grasp_penalty and KeyError's the hybrid buffer).
        if isinstance(batch_data, dict):
            rows = (batch_data,)
        else:
            rows = batch_data
        for data in rows:
            self.insert(data)

    def __len__(self):
        return len(self.inner)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _insert_demo_episodes(demo_path: str, demo_buffer, validate) -> Tuple[int, int]:
    from hilserl_wa2.experiments.demo_io import (
        load_transitions,
        require_episode_pkls,
        resolve_bundle_dir,
    )

    bundle = resolve_bundle_dir(demo_path)
    paths = require_episode_pkls(bundle)
    print(f"DEMO_BUNDLE={bundle}", flush=True)
    print(f"DEMO_EPISODE_PKLS={len(paths)}", flush=True)
    n_rows = 0
    n_grasp = 0
    for index, pkl_path in enumerate(paths):
        print(f"DEMO_EP_LOAD i={index:03d} file={pkl_path.name}", flush=True)
        rows = load_transitions(pkl_path)
        try:
            for row in rows:
                validate(row)
                action = np.asarray(row["actions"]).reshape(-1)
                if int(action.shape[0]) != 7:
                    raise SystemExit("R13_LEARNER: FAIL — demo actions must be 7D")
                if abs(float(action[-1])) > 0:
                    n_grasp += 1
                demo_buffer.insert(_with_penalty(row))
                n_rows += 1
        finally:
            del rows
            gc.collect()
    return n_rows, n_grasp


_DEMO_BUFFER_CACHE_FORMAT = "r13-timescale-v2-demo-buffer-v1"
_REPLAY_BUFFER_CACHE_FORMAT = "r13-timescale-v2-replay-buffer-v1"


def _iter_leaf_arrays(tree: Any, prefix: str = ""):
    if isinstance(tree, np.ndarray):
        yield prefix, tree
        return
    if isinstance(tree, dict):
        for key in sorted(tree):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_leaf_arrays(tree[key], child)
        return
    raise TypeError(f"unsupported buffer tree type: {type(tree)}")


def _leaf_path(root: Path, dotted: str) -> Path:
    return root / (dotted.replace(".", "__") + ".npy")


def _get_leaf(tree: Any, dotted: str):
    cur = tree
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _progress_bar(current: int, total: int, *, width: int = 28) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    filled = int(round(width * current / total))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _print_progress_line(prefix: str, current: int, total: int, suffix: str = "") -> None:
    bar = _progress_bar(current, total)
    line = f"\r{prefix} {bar} {current}/{total}"
    if suffix:
        line += f" {suffix}"
    # Pad to clear leftovers from longer previous lines.
    print(f"{line:<120}", end="", flush=True)


def save_replay_buffer(
    cache_dir: Path,
    buffer,
    *,
    kind: str,
    extra: Optional[Mapping[str, Any]] = None,
    log_prefix: str = "BUFFER_CACHE",
    cancel_event: Optional[threading.Event] = None,
) -> bool:
    """Atomically serialize a MemoryEfficientReplayBufferDataStore prefix to disk.

    Returns False if cancelled via ``cancel_event`` (tmp dir cleaned up).
    """

    import shutil

    size = int(len(buffer))
    if size < 1:
        raise ValueError("refusing to cache an empty buffer")
    tmp = cache_dir.with_name(cache_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    arrays_dir = tmp / "arrays"
    arrays_dir.mkdir()
    names = []
    leaves = list(_iter_leaf_arrays(buffer.dataset_dict))
    # +1 for is_correct_index
    total_parts = len(leaves) + 1
    t0 = time.perf_counter()
    print(flush=True)
    for idx, (dotted, arr) in enumerate(leaves, start=1):
        if cancel_event is not None and cancel_event.is_set():
            shutil.rmtree(tmp, ignore_errors=True)
            print(
                f"\n{log_prefix}=cancelled kind={kind} at {dotted} "
                f"({idx - 1}/{total_parts})",
                flush=True,
            )
            return False
        view = arr[:size]
        mb = float(view.nbytes) / (1024.0 * 1024.0)
        _print_progress_line(
            f"{log_prefix}",
            idx,
            total_parts,
            suffix=f"kind={kind} {dotted} {mb:.0f}MiB",
        )
        np.save(_leaf_path(arrays_dir, dotted), view, allow_pickle=False)
        names.append(dotted)
    if cancel_event is not None and cancel_event.is_set():
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\n{log_prefix}=cancelled kind={kind} before index flags", flush=True)
        return False
    _print_progress_line(
        f"{log_prefix}",
        total_parts,
        total_parts,
        suffix=f"kind={kind} is_correct_index",
    )
    np.save(
        tmp / "is_correct_index.npy",
        np.asarray(buffer._is_correct_index[:size]),
        allow_pickle=False,
    )
    meta: Dict[str, Any] = {
        "format": _REPLAY_BUFFER_CACHE_FORMAT,
        "legacy_demo_format": _DEMO_BUFFER_CACHE_FORMAT,
        "kind": str(kind),
        "size": size,
        "insert_index": int(buffer._insert_index),
        "first": bool(buffer._first),
        "num_stack": int(buffer._num_stack),
        "pixel_keys": [str(k) for k in buffer.pixel_keys],
        "array_keys": names,
        "capacity": int(buffer._capacity),
    }
    if extra:
        meta.update({str(k): v for k, v in extra.items()})
    (tmp / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    tmp.rename(cache_dir)
    print(
        f"\n{log_prefix}=saved kind={kind} size={size} "
        f"path={cache_dir} sec={time.perf_counter() - t0:.1f}",
        flush=True,
    )
    return True


def try_load_replay_buffer(
    cache_dir: Path,
    buffer,
    *,
    kind: str,
    require: Optional[Mapping[str, Any]] = None,
    log_prefix: str = "BUFFER_CACHE",
) -> Optional[Dict[str, Any]]:
    """Load buffer snapshot. Returns meta dict on success."""

    meta_path = cache_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = str(meta.get("format") or "")
    if fmt not in (_REPLAY_BUFFER_CACHE_FORMAT, _DEMO_BUFFER_CACHE_FORMAT):
        print(f"{log_prefix}: miss — format", flush=True)
        return None
    # Legacy demo-only caches omit kind.
    meta_kind = str(meta.get("kind") or "demo")
    if meta_kind != str(kind):
        print(f"{log_prefix}: miss — kind want={kind} got={meta_kind}", flush=True)
        return None
    if require:
        for key, expected in require.items():
            if meta.get(key) != expected:
                print(f"{log_prefix}: miss — {key}", flush=True)
                return None
    size = int(meta["size"])
    if size < 1 or size > int(buffer._capacity):
        print(f"{log_prefix}: miss — size", flush=True)
        return None
    if int(meta.get("num_stack", -1)) != int(buffer._num_stack):
        print(f"{log_prefix}: miss — num_stack", flush=True)
        return None
    if [str(k) for k in buffer.pixel_keys] != list(meta.get("pixel_keys") or []):
        print(f"{log_prefix}: miss — pixel_keys", flush=True)
        return None
    arrays_dir = cache_dir / "arrays"
    try:
        for dotted in meta["array_keys"]:
            dest = _get_leaf(buffer.dataset_dict, dotted)
            src = np.load(_leaf_path(arrays_dir, dotted), mmap_mode="r")
            if tuple(src.shape[1:]) != tuple(dest.shape[1:]) or int(src.shape[0]) != size:
                print(f"{log_prefix}: miss — shape {dotted}", flush=True)
                return None
            print(f"{log_prefix}_LOAD {dotted} shape={tuple(src.shape)}", flush=True)
            dest[:size] = src
            del src
        flags = np.load(cache_dir / "is_correct_index.npy")
        if int(flags.shape[0]) != size:
            print(f"{log_prefix}: miss — is_correct_index", flush=True)
            return None
        buffer._is_correct_index[:] = False
        buffer._is_correct_index[:size] = flags.astype(bool, copy=False)
        buffer._size = size
        buffer._insert_index = int(meta["insert_index"]) % int(buffer._capacity)
        buffer._first = bool(meta["first"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"{log_prefix}: miss — {exc}", flush=True)
        return None
    _refresh_buffer_grasp_penalties(buffer)
    return meta


def save_demo_buffer(
    cache_dir: Path,
    buffer,
    *,
    demo_n: int,
    n_grasp: int,
    demo_pkl_sha256: str,
    cancel_event: Optional[threading.Event] = None,
) -> bool:
    ok = save_replay_buffer(
        cache_dir,
        buffer,
        kind="demo",
        extra={
            "demo_pkl_sha256": str(demo_pkl_sha256),
            "demo_n": int(demo_n),
            "n_grasp": int(n_grasp),
            # Keep legacy readers happy if they only check format string in old code paths.
            "format": _DEMO_BUFFER_CACHE_FORMAT,
        },
        log_prefix="DEMO_BUFFER_CACHE",
        cancel_event=cancel_event,
    )
    if not ok:
        return False
    # Ensure format field is the legacy one for existing unit tests / old loaders.
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["format"] = _DEMO_BUFFER_CACHE_FORMAT
    meta["kind"] = "demo"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def try_load_demo_buffer(
    cache_dir: Path,
    buffer,
    *,
    demo_pkl_sha256: str,
) -> Tuple[int, int] | None:
    meta = try_load_replay_buffer(
        cache_dir,
        buffer,
        kind="demo",
        require={"demo_pkl_sha256": str(demo_pkl_sha256)},
        log_prefix="DEMO_BUFFER_CACHE",
    )
    if meta is None:
        return None
    return int(meta["demo_n"]), int(meta["n_grasp"])


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_timescale_demo_bundle(bundle: Path, task, contract) -> None:
    """Reject legacy 20 ms demonstrations before allocating replay memory."""

    from hilserl_wa2.experiments.r13_protocol import TRANSITION_SCHEMA_VERSION

    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    expected = {
        "transition_schema_version": TRANSITION_SCHEMA_VERSION,
        "config_bundle_hash": task.config_bundle_hash(),
        "policy_hz": float(contract.policy_hz),
        "servo_hz": float(contract.control_hz),
        "servo_ticks_per_action": int(contract.servo_ticks_per_action),
        "discount": float(task.discount),
        "classifier_consecutive_n": int(task.classifier_consecutive_n),
    }
    rels = manifest.get("episode_sidecars") or []
    if not rels:
        raise SystemExit("R13_LEARNER: FAIL — demo bundle has no episode sidecars")
    for rel in rels:
        side = json.loads((bundle / str(rel)).read_text(encoding="utf-8"))
        for key, value in expected.items():
            got = side.get(key)
            if isinstance(value, float):
                try:
                    matched = abs(float(got) - value) <= 1e-6
                except (TypeError, ValueError):
                    matched = False
            else:
                matched = got == value
            if not matched:
                raise SystemExit(
                    f"R13_LEARNER: FAIL — legacy/incompatible demo {rel}: "
                    f"{key} expected={value!r} got={got!r}"
                )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="bottle_pick")
    p.add_argument("--network-config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--demo-path", required=True)
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--status-file", default="")
    p.add_argument("--mode", choices=("fake", "live"), default="fake")
    p.add_argument("--debug", action="store_true", default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--training-starts", type=int, default=100)
    p.add_argument("--steps-per-update", type=int, default=50)
    p.add_argument("--checkpoint-period", type=int, default=1000)
    p.add_argument("--max-learner-steps", type=int, default=0)
    p.add_argument("--cta-ratio", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--episode-max-steps", type=int, default=600)
    p.add_argument("--end-episode", action="store_true", default=True)
    p.add_argument("--capacity", type=int, default=200000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--demo-buffer-cache",
        default="",
        help="serialized demo buffer dir; default $output/demo_buffer_cache",
    )
    p.add_argument(
        "--online-buffer-cache",
        default="",
        help="serialized online replay dir; default $output/online_buffer_cache",
    )
    p.add_argument(
        "--rebuild-demo-buffer-cache",
        action="store_true",
        help="ignore cache and re-insert episode pkls",
    )
    p.add_argument(
        "--buffer-snapshot-every",
        type=int,
        default=0,
        help="also save online+demo every N learner steps (0=shutdown only, recommended)",
    )
    p.add_argument(
        "--no-buffer-snapshot",
        action="store_true",
        help="do not persist online/demo buffers even on shutdown",
    )
    p.add_argument(
        "--grasp-penalty",
        type=float,
        default=float(GRASP_ACTION_PENALTY),
        help="reward shaping on executed grasp/release edges (default -0.002; was -0.02)",
    )
    p.add_argument(
        "--min-temperature",
        type=float,
        default=float(DEFAULT_MIN_TEMPERATURE),
        help=(
            "floor for SAC α after each update (default 0.05). "
            "0 disables. Prevents arm policy collapse to near-deterministic."
        ),
    )
    p.add_argument(
        "--resume-temperature",
        type=float,
        default=float(DEFAULT_RESUME_TEMPERATURE),
        help=(
            "on --resume, if loaded α < --min-temperature, set α to this "
            f"(default {DEFAULT_RESUME_TEMPERATURE}; 0 = only lift to min-temperature)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_grasp_action_penalty(float(args.grasp_penalty))
    print(f"GRASP_ACTION_PENALTY={float(GRASP_ACTION_PENALTY)}", flush=True)
    import jax
    from flax.training import checkpoints
    from agentlace.trainer import TrainerServer
    from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
    from serl_launcher.utils.launcher import make_sac_pixel_agent_hybrid_single_arm
    from serl_launcher.utils.train_utils import concat_batches
    from hilserl_wa2.experiments.actor_safety import params_tree_signature
    from hilserl_wa2.experiments.demo_io import (
        DemoIOError,
        require_episode_pkls,
        resolve_bundle_dir,
    )
    from hilserl_wa2.experiments.env_factory import make_wa2_environment
    from hilserl_wa2.experiments.r10_protocol import load_network_config
    from hilserl_wa2.experiments.r13_protocol import (
        PROTOCOL_VERSION,
        compare_handshake,
        make_r13_trainer_config,
        tree_has_nan_or_inf,
        update_info_has_nan,
    )
    from hilserl_wa2.experiments.task_config import load_task
    from hilserl_wa2.experiments.transition import validate_transition

    devices = jax.devices()
    print(f"JAX_DEVICES={devices}", flush=True)
    if not any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices):
        if args.mode == "live":
            raise SystemExit("R13_LEARNER: FAIL — GPU required")
        print("R13_LEARNER: WARN — no GPU (fake may continue)", flush=True)

    cfg = load_network_config(args.network_config)
    manifest = _load_json(args.manifest)
    task = load_task(args.task)
    if manifest.get("task_id") != args.task:
        raise SystemExit("R13_LEARNER: FAIL — task/manifest mismatch")
    if str(manifest.get("protocol_version")) != PROTOCOL_VERSION:
        raise SystemExit("R13_LEARNER: FAIL — legacy/incompatible protocol manifest")

    env = make_wa2_environment(task, fake_env=True, classifier=False, grasp_action=True)
    contract = env.unwrapped.contract
    expected_manifest_timebase = {
        "config_bundle_hash": task.config_bundle_hash(),
        "policy_hz": float(contract.policy_hz),
        "servo_hz": float(contract.control_hz),
        "servo_ticks_per_action": int(contract.servo_ticks_per_action),
        "discount": float(task.discount),
        "classifier_consecutive_n": int(task.classifier_consecutive_n),
    }
    for key, expected in expected_manifest_timebase.items():
        got = manifest.get(key)
        if isinstance(expected, float):
            try:
                matched = abs(float(got) - expected) <= 1e-6
            except (TypeError, ValueError):
                matched = False
        else:
            matched = got == expected
        if not matched:
            raise SystemExit(
                f"R13_LEARNER: FAIL — manifest {key} expected={expected!r} got={got!r}"
            )
    demo_bundle = resolve_bundle_dir(args.demo_path)
    _require_timescale_demo_bundle(demo_bundle, task, contract)
    try:
        require_episode_pkls(demo_bundle)
    except DemoIOError as exc:
        raise SystemExit(f"R13_LEARNER: FAIL — {exc}") from exc
    print(f"DEMO_BUNDLE={demo_bundle}", flush=True)

    rng = jax.random.PRNGKey(int(args.seed))
    agent = make_sac_pixel_agent_hybrid_single_arm(
        seed=int(args.seed),
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=list(task.image_keys),
        encoder_type=task.encoder_type,
        discount=float(task.discount),
        target_entropy=CONTINUOUS_TARGET_ENTROPY,
    )
    print(f"PARAMS_TREE_SIGNATURE={params_tree_signature(agent.state.params)}", flush=True)
    print("PARAMS_TREE_READY", flush=True)
    print("debug=true", flush=True)

    ckpt_dir = Path(args.checkpoint_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    timescale_meta = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_hz": float(contract.policy_hz),
        "servo_hz": float(contract.control_hz),
        "servo_ticks_per_action": int(contract.servo_ticks_per_action),
        "discount": float(task.discount),
        "classifier_consecutive_n": int(task.classifier_consecutive_n),
        "config_bundle_hash": task.config_bundle_hash(),
    }
    timescale_path = ckpt_dir / "timescale.json"
    start_step = 0
    if args.resume:
        if not timescale_path.is_file():
            raise SystemExit(
                "R13_LEARNER: FAIL — resume checkpoint has no timescale.json; "
                "legacy 50 Hz checkpoint is not compatible"
            )
        loaded_timescale = json.loads(timescale_path.read_text(encoding="utf-8"))
        if loaded_timescale != timescale_meta:
            raise SystemExit(
                "R13_LEARNER: FAIL — checkpoint time scale/config does not match current run"
            )
        latest = checkpoints.latest_checkpoint(os.path.abspath(ckpt_dir))
        if not latest:
            raise SystemExit(
                "R13_LEARNER: FAIL — --resume requested but no checkpoint exists"
            )
        agent = agent.replace(
            state=checkpoints.restore_checkpoint(os.path.abspath(ckpt_dir), agent.state)
        )
        start_step = int(os.path.basename(latest).split("_")[-1])
        print(f"RESUMED_STEP={start_step}", flush=True)
        loaded_alpha = read_sac_temperature(agent)
        print(f"TEMPERATURE_LOADED={loaded_alpha:.6g}", flush=True)
        agent, prev_alpha, after_alpha = maybe_reset_temperature_on_resume(
            agent,
            min_alpha=float(args.min_temperature),
            resume_alpha=float(args.resume_temperature),
        )
        if prev_alpha is not None:
            print(
                f"TEMPERATURE_RESUME_RESET {prev_alpha:.6g} -> {after_alpha:.6g}",
                flush=True,
            )
        else:
            print(f"TEMPERATURE_RESUME_KEEP={after_alpha:.6g}", flush=True)
    else:
        if any(ckpt_dir.iterdir()):
            raise SystemExit(
                "R13_LEARNER: FAIL — checkpoint directory is non-empty; "
                "use a new stage-three run directory"
            )
        timescale_path.write_text(
            json.dumps(timescale_meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        f"TEMPERATURE_FLOOR min={float(args.min_temperature)} "
        f"resume_kick={float(args.resume_temperature)} "
        f"init={SAC_TEMPERATURE_INIT}",
        flush=True,
    )

    replay = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=int(args.capacity),
        image_keys=list(task.image_keys),
        include_grasp_penalty=True,
    )
    demo_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=int(args.capacity),
        image_keys=list(task.image_keys),
        include_grasp_penalty=True,
    )
    cache_dir = Path(args.demo_buffer_cache) if str(args.demo_buffer_cache).strip() else (
        Path(args.output) / "demo_buffer_cache"
    )
    online_cache_dir = (
        Path(args.online_buffer_cache)
        if str(args.online_buffer_cache).strip()
        else (Path(args.output) / "online_buffer_cache")
    )
    demo_sha = str(manifest.get("demo_pkl_sha256") or "")
    snapshot_every = int(args.buffer_snapshot_every)
    do_buffer_snapshot = not bool(args.no_buffer_snapshot)

    loaded = None
    if not args.rebuild_demo_buffer_cache:
        loaded = try_load_demo_buffer(cache_dir, demo_buffer, demo_pkl_sha256=demo_sha)
    if loaded is None:
        print("DEMO_BUFFER_CACHE=miss", flush=True)
        demo_n, n_grasp = _insert_demo_episodes(args.demo_path, demo_buffer, validate_transition)
        try:
            save_demo_buffer(
                cache_dir,
                demo_buffer,
                demo_n=demo_n,
                n_grasp=n_grasp,
                demo_pkl_sha256=demo_sha,
            )
        except (OSError, ValueError) as exc:
            print(f"DEMO_BUFFER_CACHE: WARN save failed — {exc}", flush=True)
    else:
        demo_n, n_grasp = loaded
        print(
            f"DEMO_BUFFER_CACHE=hit path={cache_dir} "
            f"size={len(demo_buffer)} demo_n={demo_n} "
            f"intvn_cached={max(0, len(demo_buffer) - int(demo_n))}",
            flush=True,
        )
    print(f"DEMO_FILE_N={demo_n}", flush=True)
    print(f"N_GRASP_NONZERO={n_grasp}", flush=True)
    if n_grasp <= 0:
        raise SystemExit("R13_LEARNER: FAIL — 7D demo grasp channel is all zeros")
    if int(len(demo_buffer)) < int(demo_n):
        raise SystemExit(
            f"R13_LEARNER: FAIL — demo buffer size mismatch "
            f"buffer={len(demo_buffer)} demo_n={demo_n}"
        )
    print(f"DEMO_BUFFER_N={len(demo_buffer)}", flush=True)

    # Original file-demo count is the INTVN baseline (not current len after resume).
    demo_buffer_baseline = int(demo_n)

    if args.resume and do_buffer_snapshot:
        online_meta = try_load_replay_buffer(
            online_cache_dir,
            replay,
            kind="online",
            log_prefix="ONLINE_BUFFER_CACHE",
        )
        if online_meta is None:
            print("ONLINE_BUFFER_CACHE=miss", flush=True)
        else:
            print(
                f"ONLINE_BUFFER_CACHE=hit path={online_cache_dir} "
                f"size={len(replay)}",
                flush=True,
            )
    print(f"ONLINE_N={len(replay)}", flush=True)
    print(
        f"BUFFER_SNAPSHOT enabled={str(do_buffer_snapshot).lower()} "
        f"mode={'shutdown_only' if snapshot_every <= 0 else f'every_{snapshot_every}_and_shutdown'} "
        f"demo_cache={cache_dir} online_cache={online_cache_dir}",
        flush=True,
    )

    env_store = PenaltyStore(replay, validate_transition)
    intvn_store = PenaltyStore(demo_buffer, validate_transition)

    lock = threading.RLock()
    state: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": args.mode,
        "handshake_accepted": False,
        "accepted_session_id": None,
        "server_instance_id": uuid.uuid4().hex,
        "online_n": int(len(replay)),
        "intvn_n": max(0, int(len(demo_buffer)) - demo_buffer_baseline),
        "demo_file_n": demo_n,
        "publish_count": 0,
        "learner_step": start_step,
        "nan": False,
        "grasp_critic_update": False,
        "temperature": None,
        "started_unix": time.time(),
    }

    snapshot_state = {"thread": None, "guard": threading.Lock()}

    def persist_buffers(
        reason: str,
        *,
        blocking: bool = False,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Persist online+demo buffers.

        Checkpoint-time saves run in a background thread so RLPD/publish keep
        going (multi-GB image dumps can take minutes). Shutdown uses
        ``blocking=True`` and waits for any in-flight snapshot first.
        During shutdown, pass ``cancel_event`` so a second Ctrl+C aborts the dump.
        """

        if not do_buffer_snapshot:
            return

        def _run() -> None:
            try:
                online_n = int(len(replay))
                demo_n_now = int(len(demo_buffer))
                step_now = int(state.get("learner_step") or 0)
                print(
                    f"\nBUFFER_SNAPSHOT_BEGIN reason={reason} "
                    f"online={online_n} demo={demo_n_now} "
                    f"(Ctrl+C again to cancel dump)",
                    flush=True,
                )
                # Do not hold ``lock`` across np.save: Actor inserts and the
                # train loop must keep running. Snapshot may be slightly torn.
                if online_n >= 1:
                    ok = save_replay_buffer(
                        online_cache_dir,
                        replay,
                        kind="online",
                        extra={
                            "learner_step": step_now,
                            "task_id": str(args.task),
                        },
                        log_prefix="ONLINE_BUFFER_CACHE",
                        cancel_event=cancel_event,
                    )
                    if not ok:
                        print(f"BUFFER_SNAPSHOT reason={reason} cancelled=online", flush=True)
                        return
                if cancel_event is not None and cancel_event.is_set():
                    print(f"BUFFER_SNAPSHOT reason={reason} cancelled=before_demo", flush=True)
                    return
                if demo_n_now >= 1:
                    ok = save_demo_buffer(
                        cache_dir,
                        demo_buffer,
                        demo_n=int(demo_n),
                        n_grasp=int(n_grasp),
                        demo_pkl_sha256=demo_sha,
                        cancel_event=cancel_event,
                    )
                    if not ok:
                        print(f"BUFFER_SNAPSHOT reason={reason} cancelled=demo", flush=True)
                        return
                print(f"BUFFER_SNAPSHOT reason={reason}", flush=True)
            except (OSError, ValueError) as exc:
                print(f"BUFFER_SNAPSHOT_WARN {reason}: {exc}", flush=True)

        with snapshot_state["guard"]:
            alive = (
                snapshot_state["thread"] is not None
                and snapshot_state["thread"].is_alive()
            )
            if blocking:
                if alive:
                    print("BUFFER_SNAPSHOT waiting for in-flight save…", flush=True)
                    snapshot_state["thread"].join(timeout=3600)
                _run()
                return
            if alive:
                print(
                    f"BUFFER_SNAPSHOT_SKIP reason={reason} (previous save still running)",
                    flush=True,
                )
                return
            th = threading.Thread(target=_run, name="r13-buffer-snapshot", daemon=True)
            snapshot_state["thread"] = th
            th.start()
            print(f"BUFFER_SNAPSHOT_ASYNC reason={reason}", flush=True)

    def payload() -> dict:
        with lock:
            return {
                **state,
                "actor_env_count": len(replay),
                "actor_env_intvn_count": max(0, len(demo_buffer) - demo_buffer_baseline),
                "DEMO_FILE_N": demo_n,
                "DEMO_BUFFER_N": len(demo_buffer),
                "ONLINE_N": len(replay),
                "INTVN_N": max(0, len(demo_buffer) - demo_buffer_baseline),
                "PUBLISH_COUNT": state["publish_count"],
                "NAN_OR_INF": state["nan"],
                "GRASP_CRITIC_UPDATE": state["grasp_critic_update"],
            }

    def write_status() -> None:
        data = json.dumps(payload(), indent=2, sort_keys=True, default=str) + "\n"
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "status.json").write_text(data, encoding="utf-8")
        if args.status_file:
            Path(args.status_file).write_text(data, encoding="utf-8")

    def data_callback(store_name: str, update: dict) -> None:
        with lock:
            state["online_n"] = len(replay)
            state["intvn_n"] = max(0, len(demo_buffer) - demo_buffer_baseline)
            last = float(state.get("_last_store_print", 0.0) or 0.0)
            now = time.time()
            # Avoid spamming newlines that wipe the training progress bar.
            if now - last >= 5.0:
                state["_last_store_print"] = now
                print(
                    f"\nSTORE {store_name} env={len(replay)} demo={len(demo_buffer)}",
                    flush=True,
                )
        write_status()

    def request_callback(kind: str, request: dict) -> dict:
        if kind == "send-stats":
            return {"success": True}
        if kind == "r13-ping":
            return {"success": True, "payload": {"server_instance_id": state["server_instance_id"]}}
        if kind == "r13-status":
            return {"success": True, "payload": payload()}
        if kind == "r13-handshake":
            result = compare_handshake(manifest, request or {})
            with lock:
                state["handshake_accepted"] = bool(result["accepted"])
                if result["accepted"]:
                    state["accepted_session_id"] = result["session_id"]
            print(f"R13_HANDSHAKE: {'PASS' if result['accepted'] else 'FAIL'}", flush=True)
            if not result["accepted"]:
                print(json.dumps(result["mismatches"], default=str), flush=True)
            return {
                "success": bool(result["accepted"]),
                "payload": {**result, "server_instance_id": state["server_instance_id"]},
            }
        return {"success": False, "message": f"unknown request: {kind}"}

    trainer_cfg = make_r13_trainer_config(cfg["request_port"], cfg["broadcast_port"])
    server = TrainerServer(trainer_cfg, data_callback=data_callback, request_callback=request_callback)
    server.register_data_store("actor_env", env_store)
    server.register_data_store("actor_env_intvn", intvn_store)
    server.start(threaded=True)
    print("R13_SERVER: LISTEN 5588/5589", flush=True)

    stop = threading.Event()
    force_cancel = threading.Event()

    def _on_stop_signal(*_args) -> None:
        if stop.is_set():
            force_cancel.set()
            print("\nR13_LEARNER force-cancel dump (2nd signal)", flush=True)
        else:
            stop.set()
            print("\nR13_LEARNER stopping…", flush=True)

    signal.signal(signal.SIGINT, _on_stop_signal)
    signal.signal(signal.SIGTERM, _on_stop_signal)

    sharding = jax.sharding.PositionalSharding(jax.local_devices())
    replay_iterator = None
    demo_iterator = None
    training_started = False
    step = int(start_step)
    last_ckpt_step = int(start_step)
    ckpt_period = max(1, int(args.checkpoint_period)) if int(args.checkpoint_period) > 0 else 0
    critic_set = frozenset({"critic", "grasp_critic"})
    all_set = frozenset({"critic", "grasp_critic", "actor", "temperature"})
    metrics_path = Path(args.output) / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"R13_LEARNER_READY request={cfg['request_port']} broadcast={cfg['broadcast_port']}", flush=True)
    if ckpt_period > 0:
        print(
            f"CKPT_PROGRESS period={ckpt_period} "
            f"(bar fills → checkpoint; interrupt saves mid-period)",
            flush=True,
        )

    try:
        while not stop.is_set():
            write_status()
            online_n = len(replay)
            if (not training_started) and online_n >= int(args.training_starts) and len(demo_buffer) > 0:
                replay_iterator = replay.get_iterator(
                    sample_args={
                        "batch_size": int(args.batch_size) // 2,
                        "pack_obs_and_next_obs": True,
                    },
                    device=sharding.replicate(),
                )
                demo_iterator = demo_buffer.get_iterator(
                    sample_args={
                        "batch_size": int(args.batch_size) // 2,
                        "pack_obs_and_next_obs": True,
                    },
                    device=sharding.replicate(),
                )
                server.publish_network(agent.state.params)
                state["publish_count"] += 1
                training_started = True
                print("R13_RLPD_START", flush=True)
            if training_started:
                update_info = {}
                for critic_step in range(max(1, int(args.cta_ratio))):
                    batch = next(replay_iterator)
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)
                    nets = all_set if critic_step == int(args.cta_ratio) - 1 else critic_set
                    agent, update_info = agent.update(batch, networks_to_update=nets)
                if float(args.min_temperature) > 0.0:
                    agent, bumped, alpha_now = ensure_min_temperature(
                        agent, float(args.min_temperature)
                    )
                    state["temperature"] = alpha_now
                    if bumped:
                        hits = int(state.get("_temp_floor_hits", 0) or 0) + 1
                        state["_temp_floor_hits"] = hits
                        if hits == 1 or hits % 200 == 0:
                            print(
                                f"\nTEMPERATURE_FLOOR_HIT count={hits} α→{alpha_now:.6g}",
                                flush=True,
                            )
                else:
                    state["temperature"] = read_sac_temperature(agent)
                if any("grasp" in str(key).lower() for key in update_info):
                    state["grasp_critic_update"] = True
                if update_info_has_nan(update_info) or tree_has_nan_or_inf(agent.state.params):
                    state["nan"] = True
                    print("\nR13_LEARNER: FAIL — NAN_OR_INF", flush=True)
                    write_status()
                    break
                step += 1
                state["learner_step"] = step
                try:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {"step": step, **{k: _jsonable(v) for k, v in update_info.items()}}
                            )
                            + "\n"
                        )
                except (TypeError, ValueError) as exc:
                    print(f"\nMETRICS_WARN {exc}", flush=True)
                if step > 0 and step % int(args.steps_per_update) == 0:
                    agent = jax.block_until_ready(agent)
                    server.publish_network(agent.state.params)
                    state["publish_count"] += 1
                if ckpt_period > 0:
                    within = step % ckpt_period
                    if within == 0:
                        within = ckpt_period
                    grasp_flag = "g" if state.get("grasp_critic_update") else "-"
                    temp_now = state.get("temperature")
                    temp_s = f"α={float(temp_now):.4g}" if temp_now is not None else "α=?"
                    _print_progress_line(
                        "CKPT",
                        within,
                        ckpt_period,
                        suffix=(
                            f"step={step} online={len(replay)} "
                            f"demo={len(demo_buffer)} pub={state['publish_count']} "
                            f"grasp={grasp_flag} {temp_s}"
                        ),
                    )
                    if step % ckpt_period == 0:
                        checkpoints.save_checkpoint(
                            os.path.abspath(ckpt_dir), agent.state, step=step, keep=20
                        )
                        last_ckpt_step = step
                        print(f"\nCHECKPOINT_STEP={step}", flush=True)
                # Optional periodic buffer dump (default off: shutdown-only).
                if (
                    do_buffer_snapshot
                    and snapshot_every > 0
                    and step > 0
                    and step % int(snapshot_every) == 0
                ):
                    persist_buffers(f"step_{step}")
                if int(args.max_learner_steps) > 0 and (step - start_step) >= int(args.max_learner_steps):
                    print(flush=True)
                    print(f"ONLINE_N={len(replay)}", flush=True)
                    print(f"INTVN_N={max(0, len(demo_buffer) - demo_buffer_baseline)}", flush=True)
                    print(f"PUBLISH_COUNT={state['publish_count']}", flush=True)
                    print(f"NAN_OR_INF={str(state['nan']).lower()}", flush=True)
                    print(f"GRASP_CRITIC_UPDATE={str(state['grasp_critic_update']).lower()}", flush=True)
                    print("R13_LEARNER_FAKE: PASS" if args.mode == "fake" else "R13_LEARNER_LIVE_STOP: PASS", flush=True)
                    break
            else:
                time.sleep(0.2)
    finally:
        # Mid-period / interrupt: still save so --resume continues from here.
        try:
            if training_started and step > int(start_step) and step != last_ckpt_step:
                print(flush=True)
                checkpoints.save_checkpoint(
                    os.path.abspath(ckpt_dir), agent.state, step=step, keep=20
                )
                last_ckpt_step = step
                print(f"CHECKPOINT_STEP={step} reason=interrupt", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"CHECKPOINT_WARN interrupt: {exc}", flush=True)
        try:
            persist_buffers("shutdown", blocking=True, cancel_event=force_cancel)
        except Exception as exc:  # noqa: BLE001
            print(f"BUFFER_SNAPSHOT_WARN shutdown: {exc}", flush=True)
        write_status()
        server.stop()
        env.close()
        print("R13_LEARNER_STOPPED", flush=True)


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    arr = np.asarray(value)
    if arr.dtype == object:
        if arr.shape == ():
            return _jsonable(arr.item())
        return [_jsonable(item) for item in arr.tolist()]
    if arr.shape == ():
        return arr.item()
    return arr.tolist()


if __name__ == "__main__":
    main()
