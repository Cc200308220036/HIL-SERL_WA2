#!/usr/bin/env python3
"""Augment an R11 6D success demo bundle with a 7th grasp dim. Learner entry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[3]
for path in (ROOT, REPO / "src" if (REPO / "src").is_dir() else ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from hilserl_wa2.experiments.demo_grasp import augment_bundle

    report = augment_bundle(args.bundle, args.out)
    print(f"N_TRANSITIONS={report['n_transitions']}")
    print(f"N_GRASP_NONZERO={report['n_grasp_nonzero']}")
    print(f"OUT={report['out_dir']}")
    if int(report["n_grasp_nonzero"]) <= 0:
        print("R13_DEMO_AUGMENT: FAIL — no grasp edges inferred")
        raise SystemExit(1)
    print("R13_DEMO_AUGMENT: PASS")


if __name__ == "__main__":
    main()
