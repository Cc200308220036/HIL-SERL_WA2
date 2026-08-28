#!/usr/bin/env python3
"""R8 Gate: SAC pixel agent init + MemoryEfficientReplayBuffer insert/sample."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# JAX before TensorFlow (Orin XLA protobuf constraint).
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

CATKIN_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CATKIN_SRC))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "examples"))
sys.path.insert(0, str(CATKIN_SRC / "hil-serl-main" / "serl_launcher"))

import jax  # noqa: E402


def _resnet_encoder() -> str:
    pkl = Path.home() / ".serl" / "resnet10_params.pkl"
    if pkl.is_file():
        print(f"RESNET_PKL={pkl}")
        return "resnet-pretrained"
    raise SystemExit(
        "R8_AGENT_INIT: FAIL — missing ~/.serl/resnet10_params.pkl. "
        "encoder_type=resnet is incompatible with this serl EncodingWrapper "
        "(unexpected keyword 'encode'). Download once:\n"
        "  mkdir -p ~/.serl && curl -L --fail -o ~/.serl/resnet10_params.pkl \\\n"
        "    https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
    )


def _make_env(task_id: str):
    from hilserl_wa2.experiments.env_factory import make_wa2_environment_from_id

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_wa2_environment_from_id(
            task_id, fake_env=True, classifier=False
        )


def cmd_agent(task_id: str) -> None:
    from serl_launcher.utils.launcher import make_sac_pixel_agent
    from hilserl_wa2.experiments.task_config import load_task

    task = load_task(task_id)
    env = _make_env(task_id)
    try:
        sample_obs = env.observation_space.sample()
        sample_action = env.action_space.sample()
        encoder = _resnet_encoder()
        agent = make_sac_pixel_agent(
            seed=0,
            sample_obs=sample_obs,
            sample_action=sample_action,
            image_keys=list(task.image_keys),
            encoder_type=encoder,
            discount=float(task.discount),
        )
        print(f"JAX_DEVICE={jax.devices()}")
        print(f"encoder_type={encoder}")
        print(f"setup_mode={task.setup_mode}")
        assert agent is not None
        _ = agent  # keep reference until process exit
    finally:
        env.close()
    print("R8_AGENT_INIT: PASS")


def cmd_buffer(task_id: str, insert_count: int, batch_size: int) -> None:
    from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
    from hilserl_wa2.experiments.task_config import load_task

    task = load_task(task_id)
    env = _make_env(task_id)
    try:
        buf = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=max(256, insert_count + 16),
            image_keys=list(task.image_keys),
        )
        obs, _ = env.reset(seed=0)
        for _ in range(insert_count):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated)
            transition = {
                "observations": obs,
                "actions": np.asarray(action, dtype=np.float32),
                "next_observations": next_obs,
                "rewards": np.asarray(reward, dtype=np.float32),
                "masks": np.asarray(1.0 - float(done), dtype=np.float32),
                "dones": np.asarray(done, dtype=bool),
            }
            for key in ("state", "head", "wrist"):
                if not np.isfinite(np.asarray(obs[key], dtype=np.float32)).all() and key == "state":
                    raise SystemExit("FAIL: NaN/Inf in state")
            buf.insert(transition)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
        print(f"BUFFER_SIZE={len(buf)}")
        if len(buf) < insert_count:
            raise SystemExit(f"FAIL: buffer size {len(buf)} < {insert_count}")
        batch = buf.sample(batch_size)
        actions = np.asarray(batch["actions"])
        print(f"sample_actions_shape={actions.shape}")
        assert actions.shape[0] == batch_size
        assert actions.shape[-1] == 6
        state = np.asarray(batch["observations"]["state"])
        assert np.isfinite(state).all()
    finally:
        env.close()
    print("R8_REPLAY_BUFFER: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="R8 agent / replay-buffer Gate")
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent-only", action="store_true")
    parser.add_argument("--buffer-only", action="store_true")
    parser.add_argument("--insert-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or (not args.agent_only and not args.buffer_only):
        cmd_agent(args.task)
        cmd_buffer(args.task, args.insert_count, args.batch_size)
        print("R8_AGENT_BUFFER: PASS")
        return
    if args.agent_only:
        cmd_agent(args.task)
    if args.buffer_only:
        cmd_buffer(args.task, args.insert_count, args.batch_size)


if __name__ == "__main__":
    main()
