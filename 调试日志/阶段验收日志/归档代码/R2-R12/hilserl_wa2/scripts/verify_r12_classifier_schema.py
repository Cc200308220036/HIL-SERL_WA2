#!/usr/bin/env python3
"""Validate one R12 classifier bundle without starting Env / ROS / JAX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.classifier_io import (  # noqa: E402
    ClassifierIOError,
    episode_counts,
    load_classifier_bundle,
    resolve_single_bundle_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--expect-success", type=int, required=True)
    parser.add_argument("--expect-failure", type=int, required=True)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--task", default="bottle_pick")
    args = parser.parse_args()
    _ = args.task

    try:
        bundle_dir = resolve_single_bundle_dir(args.bundle_dir)
        packed = load_classifier_bundle(bundle_dir)
    except ClassifierIOError as exc:
        print(f"R12_CLASSIFIER_SCHEMA: FAIL {exc}")
        return 1

    n_s = len(packed["success"])
    n_f = len(packed["failure"])
    if n_s < int(args.expect_success):
        print(f"R12_CLASSIFIER_SCHEMA: FAIL success={n_s} < {args.expect_success}")
        return 1
    if n_f < int(args.expect_failure):
        print(f"R12_CLASSIFIER_SCHEMA: FAIL failure={n_f} < {args.expect_failure}")
        return 1
    n_s_eps, n_f_eps, n_eps = episode_counts(packed["samples"])
    if n_s_eps < int(args.min_episodes) or n_f_eps < int(args.min_episodes):
        print(
            f"R12_CLASSIFIER_SCHEMA: FAIL success_eps={n_s_eps} "
            f"failure_eps={n_f_eps} min={args.min_episodes}"
        )
        return 1
    if any("demo.pkl" == path.name for path in bundle_dir.iterdir()):
        print("R12_CLASSIFIER_SCHEMA: FAIL demo.pkl present")
        return 1
    print(f"BUNDLE={bundle_dir}")
    print(f"SUCCESS_SNAPSHOTS={n_s}")
    print(f"FAILURE_SNAPSHOTS={n_f}")
    print(f"SUCCESS_EPISODES={n_s_eps}")
    print(f"FAILURE_EPISODES={n_f_eps}")
    print(f"EPISODES={n_eps}")
    print("R12_CLASSIFIER_SCHEMA: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
