#!/usr/bin/env python3
"""Re-select R12 threshold on an existing ckpt (no retrain). Learner only."""

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
    dump_json,
    labels_array,
    load_classifier_bundle,
    load_json,
    resolve_single_bundle_dir,
    select_threshold,
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


def _evaluate(classifier, samples, image_keys, threshold: float):
    if not samples:
        return {
            "n": 0,
            "precision": 0.0,
            "recall": 0.0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "threshold": float(threshold),
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
    parser.add_argument("--write", action="store_true", help="Overwrite threshold.json/metrics.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    bundle_dir = resolve_single_bundle_dir(args.bundle_dir)
    splits_payload = load_json(run_dir / "splits.json")
    old_thr = None
    if (run_dir / "threshold.json").is_file():
        old = load_json(run_dir / "threshold.json")
        old_thr = float(old.get("threshold", 0.0))
        image_keys = list(old.get("image_keys") or CLASSIFIER_KEYS)
        consecutive_n = int(old.get("consecutive_n", 3))
    else:
        image_keys = list(CLASSIFIER_KEYS)
        consecutive_n = 3
    load_task(args.task)

    packed = load_classifier_bundle(bundle_dir)
    split_pack = split_by_episode(packed["samples"], seed=int(splits_payload["seed"]))
    train_rows = split_pack["splits"]["train"]
    val_rows = split_pack["splits"]["val"]
    test_rows = split_pack["splits"]["test"]

    import jax
    from flax.training import checkpoints
    from serl_launcher.networks.reward_classifier import create_classifier

    source = train_rows[:1] or packed["samples"][:1]
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

    val_probs = _predict_probs(classifier, val_rows, image_keys)
    selection = select_threshold(labels_array(val_rows), val_probs)
    threshold = float(selection["chosen"]["threshold"])
    val_metrics = _evaluate(classifier, val_rows, image_keys, threshold)
    test_metrics = _evaluate(classifier, test_rows, image_keys, threshold)
    train_metrics = _evaluate(classifier, train_rows, image_keys, threshold)

    print(f"OLD_THRESHOLD={old_thr}")
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

    test_probs = _predict_probs(classifier, test_rows, image_keys)
    test_selection = select_threshold(labels_array(test_rows), test_probs)
    _print_sweep("TEST_THRESHOLD_SWEEP", test_selection.get("all") or [], threshold)
    # FP ids at the Gate-chosen (val) threshold — for next hardneg round
    y_test = labels_array(test_rows)
    pred_test = (test_probs >= float(threshold)).astype(np.int32)
    print("TEST_FP_AT_CHOSEN_THRESHOLD:")
    n_fp_listed = 0
    for row, y_i, p_i, pred_i in zip(test_rows, y_test, test_probs, pred_test):
        if int(y_i) == 0 and int(pred_i) == 1:
            n_fp_listed += 1
            print(
                f"  fp#{n_fp_listed} episode_id={row.get('episode_id')} "
                f"p={float(p_i):.4f}",
                flush=True,
            )
    if n_fp_listed == 0:
        print("  (none)", flush=True)
    # Also report whether any candidate would pass Gate on test (diagnostic only)
    test_gate_ok = [
        row
        for row in (test_selection.get("all") or [])
        if float(row["precision"]) >= MIN_TEST_PRECISION
        and float(row["recall"]) >= MIN_TEST_RECALL
    ]
    if test_gate_ok:
        best = max(test_gate_ok, key=lambda r: (r["precision"], r["recall"], r["threshold"]))
        print(
            f"TEST_DIAG: some thr would hit Gate on test "
            f"(best thr={best['threshold']:.2f} P={best['precision']:.4f} "
            f"R={best['recall']:.4f}) — still selected on VAL only",
            flush=True,
        )
    else:
        print(
            "TEST_DIAG: no candidate thr hits Gate on test — need more/better hardneg",
            flush=True,
        )

    if args.write:
        dump_json(
            run_dir / "threshold.json",
            {
                "threshold": threshold,
                "consecutive_n": consecutive_n,
                "image_keys": image_keys,
                "val": selection["chosen"],
                "val_constraint_ok": bool(selection["ok"]),
                "reselected": True,
            },
        )
        dump_json(
            run_dir / "metrics.json",
            {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "threshold": threshold,
                "image_keys": image_keys,
                "reselected": True,
                "confusion_test": {
                    "tn": test_metrics.get("tn", 0),
                    "fp": test_metrics.get("fp", 0),
                    "fn": test_metrics.get("fn", 0),
                    "tp": test_metrics.get("tp", 0),
                },
            },
        )
        print("WROTE=threshold.json,metrics.json")

    reasons = []
    if not selection["ok"]:
        reasons.append("val_constraint")
    if float(test_metrics.get("precision", 0)) < MIN_TEST_PRECISION:
        reasons.append(f"test_precision<{MIN_TEST_PRECISION}")
    if float(test_metrics.get("recall", 0)) < MIN_TEST_RECALL:
        reasons.append(f"test_recall<{MIN_TEST_RECALL}")
    if reasons:
        print("R12_RESELECT: FAIL — " + ",".join(reasons))
        return 1
    print("R12_RESELECT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
