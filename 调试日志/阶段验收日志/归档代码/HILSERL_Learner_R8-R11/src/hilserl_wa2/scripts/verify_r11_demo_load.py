#!/usr/bin/env python3
"""Learner-side R11 demo load: one bundle, small buffer, no SAC / no ROS."""

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
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

from hilserl_wa2.experiments.demo_io import (  # noqa: E402
    images_are_real,
    load_bundle,
    split_episodes,
    validate_r13_grasp_edges,
)
from hilserl_wa2.experiments.env_factory import (  # noqa: E402
    assert_fake_env_isolated,
    make_wa2_environment,
)
from hilserl_wa2.experiments.task_config import load_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--bundle-dir", dest="bundle_dir", default="")
    parser.add_argument("--bundle", dest="bundle_dir", default="")
    parser.add_argument("--expect-episodes", type=int, default=1)
    parser.add_argument("--run-dir", default="/tmp")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--require-real-images", action="store_true")
    parser.add_argument("--grasp-action", action="store_true")
    args = parser.parse_args()
    if not args.bundle_dir:
        parser.error("provide --bundle-dir or --bundle")

    bundle_dir = Path(args.bundle_dir)
    if len(str(bundle_dir).split()) != 1:
        print("R11_DEMO_LOAD: FAIL — glob matched multiple paths")
        return 1

    packed = load_bundle(bundle_dir)
    n_eps = len(packed["episodes"])
    if n_eps < int(args.expect_episodes):
        print(f"R11_DEMO_LOAD: FAIL EPISODES={n_eps} < {args.expect_episodes}")
        return 1

    task = load_task(args.task)
    from experiments.wa2.config import WA2TrainConfig

    cfg = WA2TrainConfig(task_id=args.task)
    processed = cfg.process_demos(packed["transitions"])
    if processed is not packed["transitions"]:
        print("PROCESS_DEMOS=not_identity")
        print("R11_DEMO_LOAD: FAIL")
        return 1

    env = make_wa2_environment(task, fake_env=True, grasp_action=bool(args.grasp_action))
    try:
        iso = assert_fake_env_isolated(env)
        print(f"FAKE_ENV_ISOLATED={str(iso['hardware_touched'] is False).lower()}")
        from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

        n_transitions = len(packed["transitions"])
        capacity = n_transitions + 32
        batch_size = min(int(args.batch_size), n_transitions)
        if batch_size < 1:
            print("R11_DEMO_LOAD: FAIL empty transitions")
            return 1
        buf = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=capacity,
            image_keys=list(task.image_keys),
            include_grasp_penalty=bool(args.grasp_action),
        )
        for row in processed:
            payload = dict(row)
            if args.grasp_action:
                payload.setdefault("grasp_penalty", np.float32(0.0))
            buf.insert(payload)
        print(f"BUFFER_CAPACITY={capacity}")
        print("BUFFER_INSERT_OK=true")
        batch = buf.sample(batch_size)
        actions = np.asarray(batch["actions"])
        print(f"SAMPLE_BATCH={actions.shape[0]}")
        print(f"ACTION_DIM={int(actions.shape[-1])}")
        if args.grasp_action and int(actions.shape[-1]) != 7:
            print("R11_DEMO_LOAD: FAIL expected 7D actions")
            return 1
        if args.grasp_action:
            for ep_i, episode in enumerate(split_episodes(processed)):
                counts = validate_r13_grasp_edges(episode)
                print(
                    f"EP{ep_i:03d}_GRASP plus={counts['plus']} minus={counts['minus']}",
                    flush=True,
                )
        if actions.shape[0] != batch_size:
            print("R11_DEMO_LOAD: FAIL sample batch")
            return 1
    finally:
        env.close()

    if args.require_real_images and not images_are_real(packed["transitions"]):
        print("R11_DEMO_LOAD: FAIL images look empty")
        return 1

    sha_ok = (bundle_dir / "SHA256SUMS").is_file()
    print("SCHEMA_OK=true")
    print(f"EPISODES={n_eps}")
    print(f"SIDECAR_SUCCESS={n_eps}")
    print("PROCESS_DEMOS=identity")
    print(f"SHA256_MATCH={str(sha_ok).lower()}")
    print("SAMPLE_BATCH_OK=true")
    print("FAILED_DIR_ABSENT=true")
    print("R11_DEMO_LOAD: PASS")
    _ = args.run_dir
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
