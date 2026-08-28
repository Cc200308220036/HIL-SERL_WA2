#!/usr/bin/env python3
"""Split a concatenated R13 demo.pkl into episodes/epXXX.pkl. Run on Actor (RAM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hilserl_wa2.experiments.demo_io import (  # noqa: E402
    DemoIOError,
    resolve_bundle_dir,
    split_flat_demo_into_episode_pkls,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split legacy concatenated demo.pkl into per-episode pkls"
    )
    parser.add_argument("--bundle", required=True, help="bundle dir or path to demo.pkl")
    args = parser.parse_args()
    try:
        bundle = resolve_bundle_dir(args.bundle)
        result = split_flat_demo_into_episode_pkls(bundle, progress=print)
    except (DemoIOError, OSError, ValueError) as exc:
        print(f"R13_DEMO_SPLIT: FAIL — {exc}")
        return 1
    print(f"BUNDLE={bundle}", flush=True)
    print(f"WROTE={result['wrote']}", flush=True)
    print(f"EXISTED={result['existed']}", flush=True)
    print(f"EPISODE_PKLS={len(result['paths'])}", flush=True)
    print("R13_DEMO_SPLIT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
