#!/usr/bin/env python3
"""R9 offline Gate: fake env + synthetic intervention + dump/reload. No hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

CATKIN_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CATKIN_SRC))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "serl_launcher"))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "examples"))

from hilserl_wa2.experiments.env_factory import (  # noqa: E402
    assert_fake_env_isolated,
    make_wa2_environment,
    wrapper_names,
)
from hilserl_wa2.experiments.task_config import load_task  # noqa: E402
from hilserl_wa2.experiments.transition import (  # noqa: E402
    ListStore,
    build_actor_transition,
    route_transition,
    transition_rows_hash,
)
from hilserl_wa2.interventions.joy_watchdog import JoyWatchdog  # noqa: E402
from hilserl_wa2.interventions.spacemouse_input import SpaceMouseInputConfig  # noqa: E402
from hilserl_wa2.interventions.wa2_spacemouse_intervention import (  # noqa: E402
    WA2SpacemouseIntervention,
)
from hilserl_wa2.tests.unit.test_spacemouse_input import SAMPLES  # noqa: E402


def _wrap_synthetic(env):
    joy = JoyWatchdog(max_age_s=0.2)
    wrapped = WA2SpacemouseIntervention(
        env,
        joy_watchdog=joy,
        auto_start_ros=False,
        input_config=SpaceMouseInputConfig(
            translation_filter_tau=0.0, rotation_filter_tau=0.0
        ),
        intervene_eps=1e-3,
    )
    return wrapped, joy


def _schema_case(label: str, env, terminated: bool, truncated: bool) -> None:
    base = env.unwrapped
    obs, _ = env.reset(seed=1)
    if terminated:
        base.inject_success()
    if truncated:
        base.inject_truncate()
    nxt, reward, term, trunc, info = env.step(np.zeros(6, np.float32))
    tr, meta = build_actor_transition(
        obs,
        np.zeros(6, np.float32),
        nxt,
        reward,
        term,
        trunc,
        info,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    if terminated:
        assert bool(tr["dones"]) and float(tr["masks"]) == 0.0 and meta["episode_end"]
        assert term is True
    if truncated and not terminated:
        assert (not bool(tr["dones"])) and float(tr["masks"]) == 1.0 and meta["episode_end"]
        assert trunc is True
    print(f"{label}_SCHEMA=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="R9 offline transition dump/reload")
    parser.add_argument("--task", default="bottle_pick")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--synthetic-intervention-steps", type=int, default=20)
    parser.add_argument("--output", default="")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    task = load_task(args.task)
    env = make_wa2_environment(task, fake_env=True, classifier=False)
    try:
        report = assert_fake_env_isolated(env)
        print(f"wrappers_before={','.join(wrapper_names(env))}")
        print(f"hardware_touched={report['hardware_touched']}")
        env, joy = _wrap_synthetic(env)
        print(f"wrappers_after={','.join(wrapper_names(env))}")

        actor_env = ListStore()
        actor_intvn = ListStore()
        obs, _ = env.reset(seed=0)
        n_intvn = 0
        start_intvn = 10
        end_intvn = start_intvn + int(args.synthetic_intervention_steps)
        policy = np.asarray([0.1, 0, 0, 0, 0, 0], dtype=np.float32)

        for step in range(int(args.steps)):
            if start_intvn <= step < end_intvn:
                joy.clear_stale_injection()
                joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
            else:
                joy.inject(SAMPLES["forward_translation"], buttons=[0, 0])
                env.processor.reset()
            nxt, reward, terminated, truncated, info = env.step(policy)
            tr, meta = build_actor_transition(
                obs,
                policy,
                nxt,
                reward,
                terminated,
                truncated,
                info,
                observation_space=env.observation_space,
                action_space=env.action_space,
            )
            route_transition(tr, meta, actor_env, actor_intvn)
            if meta["intervened"]:
                n_intvn += 1
                np.testing.assert_allclose(tr["actions"], info["intervene_action"])
            obs = nxt
            if meta["episode_end"]:
                obs, _ = env.reset()

        if n_intvn != int(args.synthetic_intervention_steps):
            raise SystemExit(
                f"R9_TRANSITION_OFFLINE: FAIL — intervention_steps={n_intvn} "
                f"expected {args.synthetic_intervention_steps}"
            )
        if len(actor_env) != int(args.steps):
            raise SystemExit(
                f"R9_TRANSITION_OFFLINE: FAIL — total={len(actor_env)} expected {args.steps}"
            )
        if len(actor_intvn) != n_intvn:
            raise SystemExit("R9_TRANSITION_OFFLINE: FAIL — intvn store count mismatch")

        digest = transition_rows_hash(actor_env.rows)
        out_path = Path(args.output) if args.output else None
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as handle:
                pickle.dump(actor_env.rows, handle)
            with out_path.open("rb") as handle:
                reloaded = pickle.load(handle)
            if len(reloaded) != len(actor_env.rows):
                raise SystemExit("DUMP_RELOAD: FAIL — length")
            if transition_rows_hash(reloaded) != digest:
                raise SystemExit("DUMP_RELOAD: FAIL — hash")
            print("DUMP_RELOAD=PASS")
            print(f"DUMP_PATH={out_path}")
            print(f"DUMP_SHA256={hashlib.sha256(out_path.read_bytes()).hexdigest()}")
        else:
            print("DUMP_RELOAD=SKIP")

        print(f"TOTAL_TRANSITIONS={len(actor_env)}")
        print(f"INTERVENTION_STEPS={n_intvn}")
        print(f"INTERVENTION_COUNT={int(info.get('intervention_count') or 1)}")
        print(f"TRANSITION_HASH={digest}")
        print("NORMAL_SCHEMA=PASS")

        _schema_case("TERMINATED", env, terminated=True, truncated=False)
        _schema_case("TRUNCATED", env, terminated=False, truncated=True)

        summary = {
            "task_id": task.task_id,
            "exp_name": task.exp_name,
            "config_bundle_hash": task.config_bundle_hash(),
            "total": len(actor_env),
            "intervention_steps": n_intvn,
            "transition_hash": digest,
        }
        if args.summary:
            Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary).write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(f"SUMMARY_PATH={args.summary}")
        print("R9_TRANSITION_OFFLINE: PASS")
    finally:
        env.close()


if __name__ == "__main__":
    main()
