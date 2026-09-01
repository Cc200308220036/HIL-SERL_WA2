#!/usr/bin/env python3
"""Validate one R11 demo bundle without starting Env / ROS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.demo_io import (  # noqa: E402
    MIN_INTERVENED_STEPS,
    count_intervened_steps,
    load_bundle,
    tcp_deltas,
    validate_success_episode,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--expect-episodes", type=int, required=True)
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--episode-index", default="")
    parser.add_argument("--print-tcp-delta", action="store_true")
    parser.add_argument("--min-intervened", type=int, default=MIN_INTERVENED_STEPS)
    args = parser.parse_args()
    _ = args.task

    packed = load_bundle(args.bundle_dir)
    n_eps = len(packed["episodes"])
    if n_eps < int(args.expect_episodes):
        print(f"R11_DEMO_SCHEMA: FAIL episodes={n_eps} < {args.expect_episodes}")
        return 1
    for episode, sidecar in zip(packed["episodes"], packed["sidecars"]):
        stats = validate_success_episode(episode)
        intervened = count_intervened_steps(episode)
        if intervened < int(args.min_intervened):
            print(
                f"R11_DEMO_SCHEMA: FAIL ep={sidecar['episode_index']} "
                f"intervened={intervened}"
            )
            return 1
        if int(sidecar["n_steps"]) != stats["n_steps"]:
            print("R11_DEMO_SCHEMA: FAIL sidecar n_steps mismatch")
            return 1

    if args.episode_index:
        wanted = [int(x.strip()) for x in str(args.episode_index).split(",") if x.strip()]
        for index in wanted:
            episode = packed["episodes"][index]
            print(f"EPISODE={index} STEPS={len(episode)}")
            if args.print_tcp_delta:
                deltas = tcp_deltas(episode)
                intervened_idx = [
                    i
                    for i, row in enumerate(episode)
                    if float(np.linalg.norm(np.asarray(row["actions"]))) > 1e-6
                ]
                print(f"INTERVENED_STEPS={intervened_idx[:12]}")
                print(f"TCP_DELTA_HEAD={deltas[:8]}")

    print(f"EPISODES={n_eps}")
    print("R11_DEMO_SCHEMA: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
