#!/usr/bin/env python3
"""Read-only offline policy audit for WA2 R13 hybrid SAC checkpoints.

This script never constructs a live environment and never imports ROS/Agentlace.
It restores checkpoints against the fake WA2 environment solely to recreate the
network structure, then evaluates fixed observations reconstructed from an R13
memory-efficient replay-buffer snapshot.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT),
    str(REPO / "src"),
    str(REPO / "src" / "hil-serl-main" / "examples"),
    str(REPO / "src" / "hil-serl-main" / "serl_launcher"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline WA2 checkpoint policy audit")
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--demo-bundle", required=True)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[2000, 5000, 7856, 10000, 11509],
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--reference-observations", type=int, default=32)
    parser.add_argument("--temporal-observations", type=int, default=96)
    parser.add_argument("--distribution-samples", type=int, default=32)
    parser.add_argument("--progress-bins", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summary(array: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


class BufferSnapshot:
    """Memory-map an R13 cache and reconstruct inference observations."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.meta = json.loads((self.directory / "meta.json").read_text(encoding="utf-8"))
        self.size = int(self.meta["size"])
        arrays = self.directory / "arrays"
        self.actions = np.load(arrays / "actions.npy", mmap_mode="r")[: self.size]
        self.rewards = np.load(arrays / "rewards.npy", mmap_mode="r")[: self.size]
        self.dones = np.load(arrays / "dones.npy", mmap_mode="r")[: self.size]
        self.state = np.load(arrays / "observations__state.npy", mmap_mode="r")[: self.size]
        self.head = np.load(arrays / "observations__head.npy", mmap_mode="r")[: self.size]
        self.wrist = np.load(arrays / "observations__wrist.npy", mmap_mode="r")[: self.size]
        self.correct = np.load(self.directory / "is_correct_index.npy", mmap_mode="r")[: self.size]

    @property
    def valid_indices(self) -> np.ndarray:
        # A valid memory-efficient row i reconstructs current pixels from i-1.
        idx = np.flatnonzero(np.asarray(self.correct, dtype=bool))
        return idx[idx > 0]

    def observations(self, indices: Sequence[int]) -> Dict[str, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("indices must be a non-empty 1-D sequence")
        if np.any(idx <= 0) or np.any(~np.asarray(self.correct[idx], dtype=bool)):
            raise ValueError("all requested cache indices must be valid and greater than zero")
        return {
            "head": np.asarray(self.head[idx - 1])[:, None, ...],
            "wrist": np.asarray(self.wrist[idx - 1])[:, None, ...],
            "state": np.asarray(self.state[idx]),
        }

    def action_batch(self, indices: Sequence[int]) -> np.ndarray:
        return np.asarray(self.actions[np.asarray(indices, dtype=np.int64)], dtype=np.float32)


def _evenly_spaced(indices: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or len(indices) <= count:
        return np.asarray(indices, dtype=np.int64)
    positions = np.linspace(0, len(indices) - 1, num=count, dtype=np.int64)
    return np.asarray(indices[positions], dtype=np.int64)


def _latest_contiguous_valid(snapshot: BufferSnapshot, count: int) -> np.ndarray:
    valid = snapshot.valid_indices
    valid_set = set(int(v) for v in valid)
    end = int(valid[-1])
    start = end
    while start - 1 in valid_set and end - start + 1 < count:
        start -= 1
    return np.arange(start, end + 1, dtype=np.int64)


def _episode_progress_indices(
    snapshot: BufferSnapshot,
    demo_bundle: Path,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    sidecars = sorted((demo_bundle / "episodes").glob("ep*.json"))
    if not sidecars:
        raise RuntimeError(f"no episode sidecars under {demo_bundle / 'episodes'}")

    cursor = 0
    indices: List[int] = []
    progress_ids: List[int] = []
    episodes: List[Dict[str, Any]] = []
    for path in sidecars:
        payload = json.loads(path.read_text(encoding="utf-8"))
        n_steps = int(payload["n_steps"])
        valid_start = cursor + 1  # one invalid image-stack seed row per episode
        valid_end = cursor + n_steps
        if valid_end >= snapshot.size:
            raise RuntimeError(
                f"demo cache shorter than sidecars: {path.name} ends at {valid_end}, "
                f"cache size={snapshot.size}"
            )
        if not np.all(np.asarray(snapshot.correct[valid_start : valid_end + 1], dtype=bool)):
            raise RuntimeError(f"unexpected invalid row inside {path.name}")
        positions = np.rint(np.linspace(valid_start, valid_end, num=bins)).astype(np.int64)
        indices.extend(int(v) for v in positions)
        progress_ids.extend(range(bins))
        episodes.append(
            {
                "file": path.name,
                "n_steps": n_steps,
                "cache_start": valid_start,
                "cache_end": valid_end,
            }
        )
        cursor = valid_end + 1
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(progress_ids, dtype=np.int64),
        episodes,
    )


def _slice_obs(obs: Mapping[str, np.ndarray], start: int, end: int) -> Dict[str, np.ndarray]:
    return {key: np.asarray(value[start:end]) for key, value in obs.items()}


def _concat_outputs(chunks: List[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = chunks[0].keys()
    return {key: np.concatenate([np.asarray(chunk[key]) for chunk in chunks], axis=0) for key in keys}


def main() -> None:
    args = parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    from flax.training import checkpoints
    from serl_launcher.utils.launcher import make_sac_pixel_agent_hybrid_single_arm
    from hilserl_wa2.experiments.env_factory import make_wa2_environment
    from hilserl_wa2.experiments.task_config import load_task

    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "offline_policy_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_bundle = Path(args.demo_bundle).expanduser().resolve()

    online = BufferSnapshot(run_dir / "online_buffer_cache")
    demo = BufferSnapshot(run_dir / "demo_buffer_cache")
    reference_indices = _evenly_spaced(online.valid_indices, int(args.reference_observations))
    temporal_indices = _latest_contiguous_valid(online, int(args.temporal_observations))
    progress_indices, progress_ids, episode_rows = _episode_progress_indices(
        demo, demo_bundle, int(args.progress_bins)
    )

    reference_obs = online.observations(reference_indices)
    reference_actions = online.action_batch(reference_indices)[:, :6]
    temporal_obs = online.observations(temporal_indices)
    progress_obs = demo.observations(progress_indices)
    progress_actions = demo.action_batch(progress_indices)[:, :6]

    task = load_task(args.task)
    env = make_wa2_environment(task, fake_env=True, classifier=False, grasp_action=True)
    agent = make_sac_pixel_agent_hybrid_single_arm(
        seed=42,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=list(task.image_keys),
        encoder_type=task.encoder_type,
        discount=float(task.discount),
        target_entropy=-3.0,
    )

    @jax.jit
    def policy_eval(agent_obj, observations, keys):
        dist = agent_obj.forward_policy(observations, rng=keys[0], train=False)
        mode = dist.mode()
        samples = jax.vmap(lambda key: dist.sample(seed=key))(keys)
        log_probs = jax.vmap(lambda action: dist.log_prob(action))(samples)
        return mode, samples, log_probs

    @jax.jit
    def critic_eval(agent_obj, observations, actions, key):
        values = agent_obj.forward_critic(observations, actions, rng=key, train=False)
        return values

    @jax.jit
    def grasp_eval(agent_obj, observations, key):
        return agent_obj.forward_grasp_critic(observations, rng=key, train=False)

    def run_policy_batched(agent_obj, obs: Mapping[str, np.ndarray], *, samples: int):
        count = next(iter(obs.values())).shape[0]
        chunks = []
        for start in range(0, count, int(args.batch_size)):
            end = min(count, start + int(args.batch_size))
            keys = jax.random.split(
                jax.random.PRNGKey(int(args.seed) + start + count), samples
            )
            mode, draws, log_probs = policy_eval(
                agent_obj, jax.device_put(_slice_obs(obs, start, end)), keys
            )
            chunks.append(
                {
                    "mode": np.asarray(jax.device_get(mode)),
                    # Move sample dimension behind observation dimension before concat.
                    "samples": np.asarray(jax.device_get(draws)).transpose(1, 0, 2),
                    "log_probs": np.asarray(jax.device_get(log_probs)).transpose(1, 0),
                }
            )
        return _concat_outputs(chunks)

    def run_critic_batched(agent_obj, obs: Mapping[str, np.ndarray], actions: np.ndarray):
        count = len(actions)
        chunks = []
        for start in range(0, count, int(args.batch_size)):
            end = min(count, start + int(args.batch_size))
            values = critic_eval(
                agent_obj,
                jax.device_put(_slice_obs(obs, start, end)),
                jax.device_put(np.asarray(actions[start:end], dtype=np.float32)),
                jax.random.PRNGKey(int(args.seed) + 100000 + start),
            )
            chunks.append(np.asarray(jax.device_get(values)))
        return np.concatenate(chunks, axis=-1)

    def run_grasp_batched(agent_obj, obs: Mapping[str, np.ndarray]):
        count = next(iter(obs.values())).shape[0]
        chunks = []
        for start in range(0, count, int(args.batch_size)):
            end = min(count, start + int(args.batch_size))
            values = grasp_eval(
                agent_obj,
                jax.device_put(_slice_obs(obs, start, end)),
                jax.random.PRNGKey(int(args.seed) + 200000 + start),
            )
            chunks.append(np.asarray(jax.device_get(values)))
        return np.concatenate(chunks, axis=0)

    metadata = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "demo_bundle": str(demo_bundle),
        "steps": [int(v) for v in args.steps],
        "jax_devices": [str(v) for v in jax.devices()],
        "reference_indices": reference_indices.tolist(),
        "temporal_indices": temporal_indices.tolist(),
        "progress_bins": int(args.progress_bins),
        "episode_rows": episode_rows,
        "online_cache_meta": online.meta,
        "demo_cache_meta": demo.meta,
    }
    results: List[Dict[str, Any]] = []

    for step in args.steps:
        checkpoint_path = checkpoint_dir / f"checkpoint_{int(step)}"
        if not checkpoint_path.is_dir():
            raise RuntimeError(f"checkpoint missing: {checkpoint_path}")
        print(f"AUDIT_CHECKPOINT_BEGIN step={step}", flush=True)
        started = time.monotonic()
        restored_state = checkpoints.restore_checkpoint(
            os.path.abspath(checkpoint_dir), agent.state, step=int(step)
        )
        current = agent.replace(state=restored_state)

        ref = run_policy_batched(
            current, reference_obs, samples=int(args.distribution_samples)
        )
        temporal = run_policy_batched(current, temporal_obs, samples=1)
        progress = run_policy_batched(current, progress_obs, samples=1)

        mode = ref["mode"]
        draws = ref["samples"]
        within_std = np.std(draws, axis=1)
        sample_abs_delta = np.mean(np.abs(draws - mode[:, None, :]), axis=1)
        entropy_empirical = -np.mean(ref["log_probs"], axis=1)

        temporal_mode = temporal["mode"]
        temporal_sample = temporal["samples"][:, 0, :]

        q_mode = run_critic_batched(current, reference_obs, mode)
        q_executed = run_critic_batched(current, reference_obs, reference_actions)
        q_zero = run_critic_batched(current, reference_obs, np.zeros_like(mode))

        # Probe Q sensitivity by changing one normalized arm dimension by +/-0.5.
        probe_actions = []
        probe_obs = {key: [] for key in reference_obs}
        for axis in range(6):
            for sign in (-1.0, 1.0):
                candidate = mode.copy()
                candidate[:, axis] = np.clip(candidate[:, axis] + sign * 0.5, -1.0, 1.0)
                probe_actions.append(candidate)
                for key, value in reference_obs.items():
                    probe_obs[key].append(value)
        probe_actions_np = np.concatenate(probe_actions, axis=0)
        probe_obs_np = {key: np.concatenate(value, axis=0) for key, value in probe_obs.items()}
        q_probe = run_critic_batched(current, probe_obs_np, probe_actions_np)
        # critic output is [ensemble, candidates*B]
        q_probe_mean = np.mean(q_probe, axis=0).reshape(12, len(mode)).T
        q_probe_range = np.ptp(q_probe_mean, axis=1)

        grasp_q = run_grasp_batched(current, reference_obs)
        grasp_sorted = np.sort(grasp_q, axis=-1)
        grasp_choice = np.argmax(grasp_q, axis=-1) - 1

        progress_q_exec = run_critic_batched(current, progress_obs, progress_actions)
        progress_q_mode = run_critic_batched(current, progress_obs, progress["mode"])
        q_exec_mean = np.mean(progress_q_exec, axis=0)
        q_mode_mean = np.mean(progress_q_mode, axis=0)
        progress_table = []
        for progress_id in range(int(args.progress_bins)):
            mask = progress_ids == progress_id
            progress_table.append(
                {
                    "progress": float(progress_id / max(1, int(args.progress_bins) - 1)),
                    "executed_q_mean": float(np.mean(q_exec_mean[mask])),
                    "executed_q_std": float(np.std(q_exec_mean[mask])),
                    "mode_q_mean": float(np.mean(q_mode_mean[mask])),
                    "mode_q_std": float(np.std(q_mode_mean[mask])),
                }
            )

        def temporal_metrics(actions: np.ndarray) -> Dict[str, Any]:
            left = actions[:-1]
            right = actions[1:]
            return {
                "delta_abs_mean_per_axis": np.mean(np.abs(right - left), axis=0),
                "sign_flip_fraction_per_axis": np.mean((left * right) < 0.0, axis=0),
                "delta_l2": _summary(np.linalg.norm(right - left, axis=1)),
            }

        result = {
            "step": int(step),
            "elapsed_s": float(time.monotonic() - started),
            "temperature": float(np.asarray(jax.device_get(current.forward_temperature()))),
            "policy": {
                "mode_mean_per_axis": np.mean(mode, axis=0),
                "mode_abs_mean_per_axis": np.mean(np.abs(mode), axis=0),
                "mode_std_across_observations_per_axis": np.std(mode, axis=0),
                "sample_within_observation_std_per_axis": np.mean(within_std, axis=0),
                "sample_abs_delta_from_mode_per_axis": np.mean(sample_abs_delta, axis=0),
                "empirical_entropy": _summary(entropy_empirical),
                "temporal_mode": temporal_metrics(temporal_mode),
                "temporal_one_sample": temporal_metrics(temporal_sample),
            },
            "critic": {
                "q_mode": _summary(np.mean(q_mode, axis=0)),
                "q_executed": _summary(np.mean(q_executed, axis=0)),
                "q_zero": _summary(np.mean(q_zero, axis=0)),
                "mode_minus_executed": _summary(
                    np.mean(q_mode, axis=0) - np.mean(q_executed, axis=0)
                ),
                "axis_probe_q_range": _summary(q_probe_range),
                "ensemble_disagreement_mode": _summary(np.std(q_mode, axis=0)),
                "demo_progress": progress_table,
            },
            "grasp": {
                "q_mean_per_action_minus_hold_plus": np.mean(grasp_q, axis=0),
                "top_margin": _summary(grasp_sorted[:, -1] - grasp_sorted[:, -2]),
                "choice_counts": {
                    str(value): int(np.sum(grasp_choice == value)) for value in (-1, 0, 1)
                },
            },
        }
        results.append(_jsonable(result))
        (output_dir / f"checkpoint_{int(step)}.json").write_text(
            json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"AUDIT_CHECKPOINT_DONE step={step} sec={result['elapsed_s']:.1f} "
            f"alpha={result['temperature']:.6g} "
            f"sample_std={np.mean(within_std):.4f} "
            f"q_probe_range={np.mean(q_probe_range):.6g}",
            flush=True,
        )
        del current, restored_state
        gc.collect()

    payload = {"metadata": metadata, "results": results}
    (output_dir / "audit.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"AUDIT_COMPLETE output={output_dir / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
