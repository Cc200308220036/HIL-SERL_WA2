#!/usr/bin/env python3
"""Re-evaluate a saved R12 classifier ckpt on one split (Learner only)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
LEARNER_SRC = Path(__file__).resolve().parents[3]
for path in (
    SRC_ROOT,
    LEARNER_SRC,
    LEARNER_SRC / "hil-serl-main" / "examples",
    LEARNER_SRC / "hil-serl-main" / "serl_launcher",
    SRC_ROOT / "hil-serl-main" / "examples",
    SRC_ROOT / "hil-serl-main" / "serl_launcher",
):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

from hilserl_wa2.experiments.classifier_io import (  # noqa: E402
    CLASSIFIER_KEYS,
    MIN_TEST_PRECISION,
    MIN_TEST_RECALL,
    binary_metrics,
    labels_array,
    load_classifier_bundle,
    load_json,
    resolve_single_bundle_dir,
    split_by_episode,
    stack_observations,
)
from hilserl_wa2.experiments.task_config import load_task  # noqa: E402


def _predict_probs(classifier, samples, image_keys, eval_batch_size: int = 16):
    """Batched inference to avoid full-split GPU OOM."""
    if not samples:
        return np.zeros((0,), dtype=np.float64)
    bs = max(1, int(eval_batch_size))
    chunks = []
    for start in range(0, len(samples), bs):
        obs = stack_observations(samples[start : start + bs], image_keys)
        logits = classifier.apply_fn({"params": classifier.params}, obs, train=False)
        chunks.append(np.asarray(logits).reshape(-1))
    logits = np.concatenate(chunks, axis=0)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))).astype(np.float64)


def _evaluate_split(classifier, samples, image_keys, threshold: float):
    if not samples:
        return {
            "n": 0,
            "precision": 0.0,
            "recall": 0.0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    probs = _predict_probs(classifier, samples, image_keys)
    pred = (probs >= float(threshold)).astype(np.int32)
    metrics = binary_metrics(labels_array(samples), pred)
    metrics["threshold"] = float(threshold)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    bundle_dir = resolve_single_bundle_dir(args.bundle_dir)
    threshold_payload = load_json(run_dir / "threshold.json")
    splits_payload = load_json(run_dir / "splits.json")
    threshold = float(threshold_payload["threshold"])
    image_keys = list(threshold_payload.get("image_keys") or CLASSIFIER_KEYS)
    load_task(args.task)

    packed = load_classifier_bundle(bundle_dir)
    split_pack = split_by_episode(packed["samples"], seed=int(splits_payload["seed"]))
    rows = split_pack["splits"][args.split]

    import jax
    from flax.training import checkpoints
    from serl_launcher.networks.reward_classifier import create_classifier

    source = rows[:1] or packed["samples"][:1]
    sample_obs = {
        key: np.asarray(source[0]["observations"][key]) for key in image_keys
    }
    classifier = create_classifier(
        jax.random.PRNGKey(0), sample_obs, image_keys, n_way=2
    )
    classifier = checkpoints.restore_checkpoint(
        os.path.abspath(str(run_dir / "classifier_ckpt")),
        target=classifier,
    )
    metrics = _evaluate_split(classifier, rows, image_keys, threshold)
    print(f"SPLIT={args.split} N={metrics.get('n', 0)}")
    print(f"THRESHOLD={threshold}")
    print(f"PRECISION={metrics.get('precision', 0):.4f}")
    print(f"RECALL={metrics.get('recall', 0):.4f}")
    print(
        "CONFUSION "
        f"tn={metrics.get('tn')} fp={metrics.get('fp')} "
        f"fn={metrics.get('fn')} tp={metrics.get('tp')}"
    )
    if args.split == "test":
        if (
            float(metrics.get("precision", 0)) < MIN_TEST_PRECISION
            or float(metrics.get("recall", 0)) < MIN_TEST_RECALL
        ):
            print("R12_EVAL_SPLIT: FAIL")
            return 1
        print(f"TEST_PRECISION={metrics.get('precision', 0):.4f}")
        print(f"TEST_RECALL={metrics.get('recall', 0):.4f}")
    print("R12_EVAL_SPLIT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
