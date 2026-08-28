#!/usr/bin/env python3
"""R13 hands-off eval. Orin-only. Intervention wrapper off. Frozen ckpt."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

CATKIN_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CATKIN_SRC))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "examples"))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "serl_launcher"))

from hilserl_wa2.experiments.actor_safety import (  # noqa: E402
    ActorSafetyError,
    assert_no_teleop,
    assert_r13_hardware_confirm,
    find_wrapper,
)
from hilserl_wa2.experiments.env_factory import make_wa2_environment  # noqa: E402
from hilserl_wa2.experiments.r13_protocol import scale_arm_action  # noqa: E402
from hilserl_wa2.experiments.task_config import load_task  # noqa: E402
from hilserl_wa2.experiments.transition import is_intervened  # noqa: E402


def _resnet_encoder() -> str:
    pkl = Path.home() / ".serl" / "resnet10_params.pkl"
    if pkl.is_file():
        return "resnet-pretrained"
    raise SystemExit("R13_EVAL: FAIL — missing ~/.serl/resnet10_params.pkl")


def _ask_human() -> bool:
    try:
        answer = input("human_success? [y/n] ").strip().lower()
    except EOFError:
        answer = "n"
    return answer in {"y", "yes"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="R13 hands-off eval")
    p.add_argument("--task", default="bottle_pick")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint-step", type=int, required=True)
    p.add_argument("--classifier-checkpoint", default="")
    p.add_argument("--classifier-consecutive-n", type=int, default=1)
    p.add_argument("--end-episode", action="store_true", default=True)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--episode-max-steps", type=int, default=600)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--no-intervention", action="store_true", default=True)
    p.add_argument("--min-success", type=int, default=8)
    p.add_argument("--output", default="")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    assert_no_teleop()
    assert_r13_hardware_confirm("eval")
    if abs(float(args.action_scale) - 1.0) > 1e-6:
        raise SystemExit("R13_EVAL: FAIL — eval action_scale must be 1.0")
    if not args.no_intervention:
        raise SystemExit("R13_EVAL: FAIL — eval must pass --no-intervention")

    ckpt_cls = args.classifier_checkpoint or os.environ.get("WA2_CLASSIFIER_CKPT") or ""
    if not ckpt_cls:
        raise SystemExit("R13_EVAL: FAIL — classifier checkpoint required")

    import jax
    from flax.training import checkpoints
    from serl_launcher.utils.launcher import make_sac_pixel_agent_hybrid_single_arm

    task = load_task(args.task)
    if int(args.classifier_consecutive_n) != int(task.classifier_consecutive_n):
        raise SystemExit(
            "R13_EVAL: FAIL — classifier consecutive count must match task config"
        )
    env = make_wa2_environment(
        task,
        fake_env=False,
        classifier=True,
        grasp_action=True,
        enable_intervention=False,
        classifier_checkpoint=ckpt_cls,
        classifier_consecutive_n=int(args.classifier_consecutive_n),
        end_episode=True,
    )
    if find_wrapper(env, "WA2SpacemouseIntervention") is not None:
        raise SystemExit("R13_EVAL: FAIL — intervention wrapper still present")
    if tuple(env.action_space.shape) != (7,):
        raise SystemExit("R13_EVAL: FAIL — eval action is not 7D")

    agent = make_sac_pixel_agent_hybrid_single_arm(
        seed=0,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=list(task.image_keys),
        encoder_type=_resnet_encoder(),
        discount=float(task.discount),
    )
    restored = checkpoints.restore_checkpoint(
        os.path.abspath(args.checkpoint),
        agent.state,
        step=int(args.checkpoint_step),
    )
    agent = agent.replace(state=restored)
    print(f"LOADED_CHECKPOINT_STEP={int(args.checkpoint_step)}", flush=True)

    reset_opts: Dict[str, Any] = {
        "ready_timeout_s": 8.0,
        "camera_ready_timeout_s": 8.0,
        "skip_reset_motion": False,
    }
    os.environ.setdefault("RESET_SCENE_OK", "YES")
    os.environ.setdefault("R5_CONFIRM", "YES")

    rng = jax.random.PRNGKey(0)
    rows: List[Dict[str, Any]] = []
    hard_fail = False
    try:
        for ep in range(int(args.n_episodes)):
            obs, reset_info = env.reset(seed=None, options=reset_opts)
            reset_ok = bool(reset_info.get("reset_ok", True))
            singular = bool(reset_info.get("is_singular", False))
            intervened = False
            classifier_success = False
            collision = False
            ep_return = 0.0
            steps = 0
            for _ in range(int(args.episode_max_steps)):
                rng, key = jax.random.split(rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                    grasp_eps=0.0,
                )
                action = scale_arm_action(
                    np.asarray(jax.device_get(actions), dtype=np.float32), 1.0
                )
                nxt, reward, terminated, truncated, info = env.step(action)
                info = dict(info)
                if is_intervened(info):
                    intervened = True
                if info.get("succeed"):
                    classifier_success = True
                if info.get("collision") or info.get("servo_faulted") or info.get("is_singular"):
                    collision = True
                    singular = singular or bool(info.get("is_singular"))
                ep_return += float(reward)
                steps += 1
                obs = nxt
                if terminated or truncated:
                    break
            if intervened:
                print(f"EPISODE {ep + 1}: VOID intervened=1 (must re-run this episode)", flush=True)
                human_success = False
                episode_success = False
            else:
                human_success = _ask_human()
                episode_success = bool(classifier_success and human_success and reset_ok and not collision and not singular)
            if collision or singular:
                hard_fail = True
            row = {
                "episode": ep + 1,
                "classifier_success": classifier_success,
                "human_success": human_success,
                "episode_success": episode_success,
                "intervened": intervened,
                "collision": collision,
                "singular": singular,
                "reset_ok": reset_ok,
                "return": ep_return,
                "steps": steps,
            }
            rows.append(row)
            print(
                f"EPISODE {ep + 1}: classifier={classifier_success} human={human_success} "
                f"success={episode_success} intervened={intervened} steps={steps}",
                flush=True,
            )
    except ActorSafetyError as exc:
        print(f"SAFETY: {exc}", flush=True)
        raise
    finally:
        env.close()

    n_success = sum(1 for row in rows if row["episode_success"])
    n_intvn = sum(1 for row in rows if row["intervened"])
    n_cls = sum(1 for row in rows if row["classifier_success"])
    n_human = sum(1 for row in rows if row["human_success"])
    gate = (
        (not hard_fail)
        and n_intvn == 0
        and n_success >= int(args.min_success)
        and len(rows) == int(args.n_episodes)
    )
    summary = {
        "n_episodes": len(rows),
        "n_success": n_success,
        "classifier_success": n_cls,
        "human_success": n_human,
        "intervened": n_intvn,
        "hard_fail": hard_fail,
        "episodes": rows,
        "gate": gate,
    }
    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"CLASSIFIER_SUCCESS={n_cls}/{args.n_episodes}", flush=True)
    print(f"HUMAN_SUCCESS={n_human}/{args.n_episodes}", flush=True)
    print(f"EPISODE_SUCCESS={n_success}/{args.n_episodes}", flush=True)
    print(f"INTERVENED={n_intvn}", flush=True)
    print(f"R13_EVAL: {n_success}/{args.n_episodes}", flush=True)
    if gate:
        print("R13_EVAL: PASS", flush=True)
    else:
        print("R13_EVAL: FAIL", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
