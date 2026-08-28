#!/usr/bin/env python3
"""R12 Learner: train reward classifier with episode splits and precision/recall."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

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
    ClassifierIOError,
    binary_metrics,
    dump_json,
    labels_array,
    load_classifier_bundle,
    resolve_single_bundle_dir,
    select_threshold,
    split_by_episode,
    stack_observations,
)
from hilserl_wa2.experiments.task_config import load_task  # noqa: E402


def _parse_image_keys(raw: str):
    keys = tuple(part.strip() for part in str(raw).split(",") if part.strip())
    if keys != CLASSIFIER_KEYS:
        raise SystemExit(f"--image-keys must be head,wrist, got {keys}")
    return list(keys)


def _balanced_batches(
    pos: Sequence[Dict[str, Any]],
    neg: Sequence[Dict[str, Any]],
    batch_size: int,
    rng: np.random.Generator,
):
    half = max(1, int(batch_size) // 2)
    if not pos or not neg:
        raise ClassifierIOError("train split needs both success and failure samples")
    while True:
        pi = rng.choice(len(pos), size=half, replace=len(pos) < half)
        ni = rng.choice(len(neg), size=half, replace=len(neg) < half)
        batch = [pos[int(i)] for i in pi] + [neg[int(i)] for i in ni]
        yield batch


def predict_probs(
    classifier,
    samples: Sequence[Dict[str, Any]],
    image_keys: List[str],
    eval_batch_size: int = 16,
):
    """Batched inference — full-split stacks OOM on laptop GPUs once n is large."""
    if not samples:
        return np.zeros((0,), dtype=np.float64)
    bs = max(1, int(eval_batch_size))
    chunks: List[np.ndarray] = []
    for start in range(0, len(samples), bs):
        obs = stack_observations(samples[start : start + bs], image_keys)
        logits = classifier.apply_fn({"params": classifier.params}, obs, train=False)
        chunks.append(np.asarray(logits).reshape(-1))
    logits = np.concatenate(chunks, axis=0)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
    return probs.astype(np.float64)


def evaluate_split(classifier, samples, image_keys, threshold: float) -> Dict[str, Any]:
    if not samples:
        return {"n": 0, "precision": 0.0, "recall": 0.0, "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    probs = predict_probs(classifier, samples, image_keys)
    y = labels_array(samples)
    pred = (probs >= float(threshold)).astype(np.int32)
    metrics = binary_metrics(y, pred)
    metrics["threshold"] = float(threshold)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-keys", default="head,wrist")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    image_keys = _parse_image_keys(args.image_keys)
    task = load_task(args.task)
    if list(task.classifier_keys or ()) != image_keys:
        raise SystemExit(
            f"task.classifier_keys={task.classifier_keys} != {image_keys}"
        )
    bundle_dir = resolve_single_bundle_dir(args.bundle_dir)
    if (bundle_dir / "demo.pkl").is_file():
        print("R12_TRAIN: FAIL — bundle contains demo.pkl")
        return 1
    packed = load_classifier_bundle(bundle_dir)
    split_pack = split_by_episode(packed["samples"], seed=int(args.split_seed))
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "classifier_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    splits_payload = {
        "seed": int(args.split_seed),
        "episode_split": split_pack["episode_split"],
        "counts": split_pack["counts"],
        "image_keys": image_keys,
        "bundle_dir": str(bundle_dir),
    }
    dump_json(run_dir / "splits.json", splits_payload)

    train_rows = split_pack["splits"]["train"]
    val_rows = split_pack["splits"]["val"]
    test_rows = split_pack["splits"]["test"]
    pos = [row for row in train_rows if int(row["label"]) == 1]
    neg = [row for row in train_rows if int(row["label"]) == 0]
    print(
        f"SPLIT train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
        f"pos={len(pos)} neg={len(neg)}",
        flush=True,
    )
    if not train_rows or not pos or not neg:
        print("R12_TRAIN: FAIL — train split missing success or failure samples")
        return 1

    import jax
    import optax
    from flax.training import checkpoints
    from serl_launcher.networks.reward_classifier import create_classifier
    from serl_launcher.vision.data_augmentations import batched_random_crop

    rng = jax.random.PRNGKey(0)
    rng, init_key = jax.random.split(rng)
    sample_obs = {
        key: np.asarray(train_rows[0]["observations"][key]) for key in image_keys
    }
    classifier = create_classifier(init_key, sample_obs, image_keys, n_way=2)

    def data_augmentation_fn(key, observations):
        out = dict(observations)
        for pixel_key in image_keys:
            out[pixel_key] = batched_random_crop(
                observations[pixel_key], key, padding=4, num_batch_dims=2
            )
        return out

    @jax.jit
    def train_step(state, batch_obs, batch_labels, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                batch_obs,
                rngs={"dropout": key},
                train=True,
            )
            logits = logits.reshape(-1)
            labels = batch_labels.reshape(-1)
            return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    np_rng = np.random.default_rng(int(args.split_seed))
    batches = _balanced_batches(pos, neg, int(args.batch_size), np_rng)
    epochs = int(args.epochs)
    for epoch in range(epochs):
        batch = next(batches)
        obs = stack_observations(batch, image_keys)
        labels = labels_array(batch).astype(np.float32)
        rng, aug_key, step_key = jax.random.split(rng, 3)
        obs = data_augmentation_fn(aug_key, obs)
        classifier, loss = train_step(classifier, obs, labels, step_key)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch+1}/{epochs} loss={float(loss):.4f}", flush=True)

    checkpoints.save_checkpoint(
        os.path.abspath(str(ckpt_dir)),
        classifier,
        step=epochs,
        overwrite=True,
    )

    val_probs = predict_probs(classifier, val_rows, image_keys) if val_rows else np.array([])
    val_y = labels_array(val_rows) if val_rows else np.array([], dtype=np.int32)
    selection = (
        select_threshold(val_y, val_probs)
        if val_rows
        else {"ok": False, "chosen": {"threshold": 0.85, "precision": 0.0, "recall": 0.0}, "all": []}
    )
    threshold = float(selection["chosen"]["threshold"])
    val_metrics = evaluate_split(classifier, val_rows, image_keys, threshold)
    test_metrics = evaluate_split(classifier, test_rows, image_keys, threshold)
    train_metrics = evaluate_split(classifier, train_rows, image_keys, threshold)

    threshold_payload = {
        "threshold": threshold,
        "consecutive_n": 3,
        "image_keys": image_keys,
        "val": selection["chosen"],
        "val_constraint_ok": bool(selection["ok"]),
    }
    metrics_payload = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "threshold": threshold,
        "image_keys": image_keys,
        "epochs": epochs,
        "smoke": bool(args.smoke),
        "confusion_test": {
            "tn": test_metrics.get("tn", 0),
            "fp": test_metrics.get("fp", 0),
            "fn": test_metrics.get("fn", 0),
            "tp": test_metrics.get("tp", 0),
        },
    }
    dump_json(run_dir / "threshold.json", threshold_payload)
    dump_json(run_dir / "metrics.json", metrics_payload)

    print(f"THRESHOLD={threshold}")
    print(f"VAL_CONSTRAINT_OK={str(bool(selection['ok'])).lower()}")
    print(
        f"VAL_PRECISION={val_metrics.get('precision', 0):.4f} "
        f"VAL_RECALL={val_metrics.get('recall', 0):.4f}"
    )
    print(
        f"TEST_PRECISION={test_metrics.get('precision', 0):.4f} "
        f"TEST_RECALL={test_metrics.get('recall', 0):.4f}"
    )
    print(
        "TEST_CONFUSION "
        f"tn={test_metrics.get('tn')} fp={test_metrics.get('fp')} "
        f"fn={test_metrics.get('fn')} tp={test_metrics.get('tp')}"
    )
    def _print_sweep(title: str, scored, chosen_thr: float) -> None:
        print(f"{title}:")
        for row in scored:
            mark = "*" if abs(float(row["threshold"]) - chosen_thr) < 1e-9 else " "
            gate = ""
            if (
                float(row["precision"]) >= MIN_TEST_PRECISION
                and float(row["recall"]) >= MIN_TEST_RECALL
            ):
                gate = " GATE_OK"
            print(
                f"  {mark} thr={row['threshold']:.2f} "
                f"P={row['precision']:.4f} R={row['recall']:.4f} "
                f"Fβ={row.get('f_beta', 0):.4f} "
                f"fp={row['fp']} fn={row['fn']}{gate}",
                flush=True,
            )

    _print_sweep("VAL_THRESHOLD_SWEEP", selection.get("all") or [], threshold)
    test_selection = select_threshold(
        labels_array(test_rows),
        predict_probs(classifier, test_rows, image_keys) if test_rows else np.array([]),
    )
    if test_rows:
        _print_sweep("TEST_THRESHOLD_SWEEP", test_selection.get("all") or [], threshold)
    print(f"CKPT={ckpt_dir}")

    if args.smoke:
        print("R12_TRAIN_FAKE: PASS")
        return 0

    test_p = float(test_metrics.get("precision", 0.0))
    test_r = float(test_metrics.get("recall", 0.0))
    reasons = []
    if not selection["ok"]:
        reasons.append("val_constraint")
    if test_p < MIN_TEST_PRECISION:
        reasons.append(f"test_precision<{MIN_TEST_PRECISION}")
    if test_r < MIN_TEST_RECALL:
        reasons.append(f"test_recall<{MIN_TEST_RECALL}")
    if reasons:
        print("R12_TRAIN: FAIL — " + ",".join(reasons))
        return 1
    print("R12_TRAIN: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
