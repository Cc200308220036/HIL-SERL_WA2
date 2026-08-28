#!/usr/bin/env python3
"""Merge multiple R12 classifier bundles into one (episode IDs retagged per source)."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
LEARNER_SRC = Path(__file__).resolve().parents[3]
for path in (SRC_ROOT, LEARNER_SRC):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hilserl_wa2.experiments.classifier_io import (  # noqa: E402
    ClassifierIOError,
    episode_counts,
    load_classifier_bundle,
    merge_classifier_bundles,
    resolve_single_bundle_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-dir",
        action="append",
        required=True,
        help="Source bundle (repeatable). Glob must match exactly one path.",
    )
    parser.add_argument("--out-dir", required=True, help="Parent dir for the new bundle")
    parser.add_argument("--out-name", required=True, help="Bundle name stem")
    parser.add_argument("--operator", default="merge")
    parser.add_argument("--mode", default="live", choices=("fake", "live", "merge"))
    args = parser.parse_args()

    try:
        sources = [resolve_single_bundle_dir(raw) for raw in args.bundle_dir]
        out_parent = Path(args.out_dir).expanduser().resolve()
        out_parent.mkdir(parents=True, exist_ok=True)
        out_dir = out_parent / f"{args.out_name}_{uuid.uuid4().hex[:8]}"
        if out_dir.exists():
            raise ClassifierIOError(f"out bundle already exists: {out_dir}")
        merge_classifier_bundles(
            sources,
            out_dir=out_dir,
            bundle_name=str(args.out_name),
            operator=str(args.operator),
            mode=str(args.mode),
        )
        packed = load_classifier_bundle(out_dir)
    except ClassifierIOError as exc:
        print(f"R12_MERGE: FAIL {exc}")
        return 1

    n_s = len(packed["success"])
    n_f = len(packed["failure"])
    n_s_eps, n_f_eps, n_eps = episode_counts(packed["samples"])
    print(f"SOURCES={packed['manifest'].get('merged_from')}")
    print(f"BUNDLE={out_dir}")
    print(f"SUCCESS_SNAPSHOTS={n_s}")
    print(f"FAILURE_SNAPSHOTS={n_f}")
    print(f"SUCCESS_EPISODES={n_s_eps}")
    print(f"FAILURE_EPISODES={n_f_eps}")
    print(f"EPISODES={n_eps}")
    print("R12_MERGE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
