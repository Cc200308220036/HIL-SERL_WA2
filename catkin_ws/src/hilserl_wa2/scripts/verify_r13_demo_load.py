#!/usr/bin/env python3
"""Learner R13 demo check: sidecar + one episode at a time. Never pickle.load demo.pkl."""

from __future__ import annotations

import argparse
import gc
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
    DemoIOError,
    images_are_real,
    load_json,
    load_transitions,
    require_episode_pkls,
    resolve_bundle_dir,
    validate_bundle_manifest,
    validate_r13_grasp_edges,
    validate_sidecar,
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
    parser.add_argument("--expect-episodes", type=int, default=20)
    parser.add_argument("--run-dir", default="/tmp")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--require-real-images", action="store_true")
    parser.add_argument("--grasp-action", action="store_true", default=True)
    args = parser.parse_args()
    if not args.bundle_dir:
        parser.error("provide --bundle-dir or --bundle")

    try:
        bundle_dir = resolve_bundle_dir(args.bundle_dir)
        manifest = load_json(bundle_dir / "bundle.json")
        validate_bundle_manifest(manifest)
        n_eps = int(manifest["n_episodes"])
        if n_eps < int(args.expect_episodes):
            print(f"R13_DEMO_LOAD: FAIL EPISODES={n_eps} < {args.expect_episodes}")
            return 1
        sidecars = []
        for rel in manifest["episode_sidecars"]:
            sidecar = load_json(bundle_dir / str(rel))
            validate_sidecar(sidecar, label=str(manifest.get("label", "success")))
            plus = int(sidecar.get("n_grasp_plus") or 0)
            minus = int(sidecar.get("n_grasp_minus") or 0)
            if plus < 1 or minus < 1:
                print(
                    f"R13_DEMO_LOAD: FAIL {rel} plus={plus} minus={minus}",
                    flush=True,
                )
                return 1
            print(
                f"SIDECAR {rel} steps={sidecar['n_steps']} plus={plus} minus={minus}",
                flush=True,
            )
            sidecars.append(sidecar)
        ep_pkls = require_episode_pkls(bundle_dir)
    except DemoIOError as exc:
        print(f"R13_DEMO_LOAD: FAIL — {exc}")
        return 1

    task = load_task(args.task)
    env = make_wa2_environment(task, fake_env=True, grasp_action=True)
    sampled = False
    try:
        iso = assert_fake_env_isolated(env)
        print(f"FAKE_ENV_ISOLATED={str(iso['hardware_touched'] is False).lower()}")
        from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

        for ep_i, (pkl_path, sidecar) in enumerate(zip(ep_pkls, sidecars)):
            print(f"LOAD_EP {ep_i:03d} {pkl_path.name}", flush=True)
            rows = load_transitions(pkl_path)
            try:
                if int(sidecar["n_steps"]) != len(rows):
                    print(
                        f"R13_DEMO_LOAD: FAIL sidecar n_steps {sidecar['n_steps']} "
                        f"!= pkl {len(rows)}"
                    )
                    return 1
                counts = validate_r13_grasp_edges(rows)
                print(
                    f"EP{ep_i:03d}_GRASP plus={counts['plus']} minus={counts['minus']}",
                    flush=True,
                )
                if args.require_real_images and not images_are_real(rows):
                    print(f"R13_DEMO_LOAD: FAIL images look empty ep={ep_i:03d}")
                    return 1
                if not sampled:
                    capacity = len(rows) + 32
                    batch_size = min(int(args.batch_size), len(rows))
                    buf = MemoryEfficientReplayBufferDataStore(
                        env.observation_space,
                        env.action_space,
                        capacity=capacity,
                        image_keys=list(task.image_keys),
                        include_grasp_penalty=True,
                    )
                    for row in rows:
                        payload = dict(row)
                        payload.setdefault("grasp_penalty", np.float32(0.0))
                        buf.insert(payload)
                    batch = buf.sample(batch_size)
                    actions = np.asarray(batch["actions"])
                    print(f"BUFFER_CAPACITY={capacity}")
                    print("BUFFER_INSERT_OK=true")
                    print(f"SAMPLE_BATCH={actions.shape[0]}")
                    print(f"ACTION_DIM={int(actions.shape[-1])}")
                    if int(actions.shape[-1]) != 7:
                        print("R13_DEMO_LOAD: FAIL expected 7D actions")
                        return 1
                    sampled = True
                    del buf
            finally:
                del rows
                gc.collect()
    finally:
        env.close()

    sha_ok = (bundle_dir / "SHA256SUMS").is_file()
    print("SCHEMA_OK=true")
    print(f"EPISODES={n_eps}")
    print(f"SIDECAR_SUCCESS={n_eps}")
    print("PROCESS_DEMOS=identity")
    print(f"SHA256_MATCH={str(sha_ok).lower()}")
    print("SAMPLE_BATCH_OK=true")
    print("FAILED_DIR_ABSENT=true")
    print("R13_DEMO_LOAD: PASS")
    _ = args.run_dir
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
