#!/usr/bin/env python3
"""R8 Gate: task YAML, fake_env, hardware isolation, Actor/Learner spaces."""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

CATKIN_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CATKIN_SRC))
EXAMPLES = CATKIN_SRC / "hil-serl-main" / "examples"
sys.path.insert(0, str(EXAMPLES))
SERL = CATKIN_SRC / "hil-serl-main" / "serl_launcher"
sys.path.insert(0, str(SERL))


def _load(task_id: str):
    from hilserl_wa2.experiments.task_config import (
        TaskConfigError,
        check_wa2_task_override,
        load_task,
        sanitize_task_id,
    )

    try:
        tid = sanitize_task_id(task_id)
        check_wa2_task_override(tid, os.environ.get("WA2_TASK"))
        return load_task(tid)
    except TaskConfigError as exc:
        raise SystemExit(f"R8_CONFIG: FAIL — {exc}") from exc


def cmd_config_only(task_id: str) -> None:
    task = _load(task_id)
    print(f"exp_name={task.exp_name}")
    print(f"task_id={task.task_id}")
    print(f"action_mode={task.action_mode}")
    print(f"image_keys={','.join(task.image_keys)}")
    print(f"proprio_dim={task.proprio_dim}")
    for key, path in task.resolved_paths().items():
        print(f"{key}_path={path}")
    hashes = task.file_hashes()
    for name, digest in hashes.items():
        print(f"{name}={digest}")
    print(f"config_bundle_hash={task.config_bundle_hash()}")
    print("R8_CONFIG_ONLY: PASS")


def cmd_fake_env(task_id: str, steps: int) -> None:
    from hilserl_wa2.experiments.env_factory import (
        assert_fake_env_isolated,
        make_wa2_environment,
    )

    task = _load(task_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = make_wa2_environment(
            task, fake_env=True, save_video=True, classifier=False
        )
    try:
        report = assert_fake_env_isolated(env)
        obs, _ = env.reset(seed=0)
        print(f"observation_keys={','.join(sorted(obs))}")
        print(f"state_shape={obs['state'].shape}")
        print(f"head_shape={obs['head'].shape}")
        print(f"wrist_shape={obs['wrist'].shape}")
        print(f"action_shape={env.action_space.shape}")
        print(f"hardware_touched={report['hardware_touched']}")
        assert set(obs) == {"state", "head", "wrist"}
        assert obs["state"].shape == (1, 27)
        assert obs["head"].shape == (1, 128, 128, 3)
        assert obs["wrist"].shape == (1, 128, 128, 3)
        assert env.action_space.shape == (6,)
        n = 0
        while n < steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs), n
            assert float(reward) == 0.0
            assert np.isfinite(obs["state"]).all()
            n += 1
            if terminated or truncated:
                obs, _ = env.reset()
        print(f"steps={n}")
    finally:
        env.close()
    print("R8_CONFIG: PASS")


def cmd_assert_no_hardware(task_id: str) -> None:
    from hilserl_wa2.experiments.env_factory import (
        assert_fake_env_isolated,
        make_wa2_environment,
    )

    task = _load(task_id)
    env = make_wa2_environment(task, fake_env=True)
    try:
        report = assert_fake_env_isolated(env)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        print(json.dumps(report, sort_keys=True))
    finally:
        env.close()
    print("R8_HARDWARE_ISOLATION: PASS")


def cmd_compare_spaces(task_id: str) -> None:
    from hilserl_wa2.experiments.env_factory import build_space_signature

    task = _load(task_id)
    actor = build_space_signature(task, "actor")
    learner = build_space_signature(task, "learner")
    print(f"ACTOR_SPACE_SHA256={actor['space_hash']}")
    print(f"LEARNER_SPACE_SHA256={learner['space_hash']}")
    if actor["space_hash"] != learner["space_hash"]:
        raise SystemExit("R8_SPACE_MATCH: FAIL — actor/learner space hash differ")
    print("INTERVENTION_SPACE_UNCHANGED: PASS")
    print("R8_SPACE_MATCH: PASS")


def cmd_print_hashes(task_id: str) -> None:
    from hilserl_wa2.experiments.env_factory import build_space_signature

    task = _load(task_id)
    sig = build_space_signature(task, "learner")
    print(f"exp_name={task.exp_name}")
    print(f"task_id={task.task_id}")
    for name, digest in task.file_hashes().items():
        print(f"{name}={digest}")
    print(f"config_bundle_hash={task.config_bundle_hash()}")
    print(f"space_hash={sig['space_hash']}")


def cmd_all(task_id: str) -> None:
    cmd_config_only(task_id)
    cmd_fake_env(task_id, steps=1000)
    cmd_assert_no_hardware(task_id)
    cmd_compare_spaces(task_id)
    from experiments.mappings import CONFIG_MAPPING

    task = _load(task_id)
    if task.exp_name not in CONFIG_MAPPING:
        raise SystemExit(f"FAIL: {task.exp_name} not in CONFIG_MAPPING")
    if "wa2" in CONFIG_MAPPING:
        raise SystemExit("FAIL: bare key 'wa2' must not be registered")
    cfg = CONFIG_MAPPING[task.exp_name]()
    env = cfg.get_environment(fake_env=True, save_video=False, classifier=False)
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
    finally:
        env.close()
    print(f"CONFIG_MAPPING[{task.exp_name}]: PASS")
    print("R8_CONFIG: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="R8 WA2 config / fake_env Gate")
    parser.add_argument("--task", required=True)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--fake-env", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--assert-no-hardware", action="store_true")
    parser.add_argument("--assert-no-hardware-imports", action="store_true")
    parser.add_argument("--compare-actor-learner-spaces", action="store_true")
    parser.add_argument("--print-hashes", action="store_true")
    parser.add_argument("--print-bundle-hash", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        cmd_all(args.task)
        return
    ran = False
    if args.config_only:
        cmd_config_only(args.task)
        ran = True
    if args.fake_env:
        cmd_fake_env(args.task, steps=args.steps)
        ran = True
    if args.assert_no_hardware or args.assert_no_hardware_imports:
        cmd_assert_no_hardware(args.task)
        ran = True
    if args.compare_actor_learner_spaces:
        cmd_compare_spaces(args.task)
        ran = True
    if args.print_hashes or args.print_bundle_hash:
        cmd_print_hashes(args.task)
        ran = True
    if not ran:
        parser.error("select a mode, e.g. --config-only or --all")


if __name__ == "__main__":
    main()
