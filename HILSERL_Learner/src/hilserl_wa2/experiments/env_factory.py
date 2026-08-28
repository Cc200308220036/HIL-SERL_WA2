"""Single WA2 Env wrapper factory (R8). Wrapper order is not user-reorderable."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from hilserl_wa2.envs.wa2_env import WA2Env
from hilserl_wa2.experiments.task_config import (
    WA2TaskConfig,
    check_wa2_task_override,
    load_task,
    space_sha256,
)

Role = Literal["actor", "learner"]


def _apply_serl_wrappers(env: Any, task: WA2TaskConfig) -> Any:
    from serl_launcher.wrappers.chunking import ChunkingWrapper
    from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

    if task.serl_obs:
        env = SERLObsWrapper(env, proprio_keys=list(task.proprio_keys))
    env = ChunkingWrapper(
        env,
        obs_horizon=int(task.obs_horizon),
        act_exec_horizon=task.act_exec_horizon,
    )
    return env


def _maybe_wrap_intervention(
    env: Any,
    task: WA2TaskConfig,
    *,
    auto_start_ros: bool,
) -> Any:
    # Lazy import: do not pull interventions package (or Joy ROS) on fake_env path.
    from hilserl_wa2.interventions.wa2_spacemouse_intervention import (
        WA2SpacemouseIntervention,
    )

    return WA2SpacemouseIntervention(
        env,
        config_path=task.spacemouse_path,
        auto_start_ros=auto_start_ros,
    )


def make_wa2_environment(
    task: WA2TaskConfig,
    *,
    fake_env: bool,
    save_video: bool = False,
    classifier: bool = False,
    read_only: Optional[bool] = None,
    classifier_checkpoint: Optional[str] = None,
    classifier_threshold: Optional[float] = None,
    classifier_consecutive_n: Optional[int] = None,
    classifier_infer_mode: Optional[str] = None,
    classifier_infer_every_n: Optional[int] = None,
    classifier_session_infer_every_n: Optional[int] = None,
    end_episode: bool = False,
    grasp_action: bool = False,
    enable_intervention: Optional[bool] = None,
) -> Any:
    """Build Actor/Learner env. ``fake_env=True`` never touches hardware.

    R8–R12 callers leave ``grasp_action=False`` so the Gym action stays 6D.
    R13 passes ``grasp_action=True`` to wrap 7D after intervention, before SERL.

    Classifier stutter controls (optional; env defaults apply when None):
      WA2_CLASSIFIER_INFER_MODE=sync|decimate|async (default decimate)
      WA2_CLASSIFIER_INFER_EVERY_N (default 1)
      WA2_CLASSIFIER_SESSION_EVERY_N (default 2)
    """

    if save_video:
        warnings.warn("save_video is ignored for WA2 R8 factory", stacklevel=2)

    check_wa2_task_override(task.task_id, os.environ.get("WA2_TASK"))

    if fake_env:
        env = WA2Env(
            fake_env=True,
            scene_name=task.scene,
            contract_path=task.contract_path,
            spacemouse_path=task.spacemouse_path,
        )
    else:
        env = WA2Env(
            fake_env=False,
            read_only=False if read_only is None else bool(read_only),
            scene_name=task.scene,
            camera_cfg_path=task.camera_path,
            contract_path=task.contract_path,
            spacemouse_path=task.spacemouse_path,
        )
        wrap_intvn = task.intervention if enable_intervention is None else bool(enable_intervention)
        if wrap_intvn:
            env = _maybe_wrap_intervention(env, task, auto_start_ros=True)

    if grasp_action:
        from hilserl_wa2.wrappers.grasp_action import WA2GraspActionWrapper

        env = WA2GraspActionWrapper(env)

    env = _apply_serl_wrappers(env, task)
    if classifier:
        env = _maybe_wrap_reward_classifier(
            env,
            task,
            checkpoint=classifier_checkpoint,
            threshold=classifier_threshold,
            consecutive_n=classifier_consecutive_n,
            end_episode=end_episode,
            infer_mode=classifier_infer_mode,
            infer_every_n=classifier_infer_every_n,
            session_infer_every_n=classifier_session_infer_every_n,
        )
    return env


def make_wa2_environment_from_id(
    task_id: str,
    *,
    fake_env: bool,
    save_video: bool = False,
    classifier: bool = False,
    read_only: Optional[bool] = None,
    classifier_checkpoint: Optional[str] = None,
    classifier_threshold: Optional[float] = None,
    classifier_consecutive_n: Optional[int] = None,
    classifier_infer_mode: Optional[str] = None,
    classifier_infer_every_n: Optional[int] = None,
    classifier_session_infer_every_n: Optional[int] = None,
    end_episode: bool = False,
    grasp_action: bool = False,
    enable_intervention: Optional[bool] = None,
) -> Any:
    return make_wa2_environment(
        load_task(task_id),
        fake_env=fake_env,
        save_video=save_video,
        classifier=classifier,
        read_only=read_only,
        classifier_checkpoint=classifier_checkpoint,
        classifier_threshold=classifier_threshold,
        classifier_consecutive_n=classifier_consecutive_n,
        classifier_infer_mode=classifier_infer_mode,
        classifier_infer_every_n=classifier_infer_every_n,
        classifier_session_infer_every_n=classifier_session_infer_every_n,
        end_episode=end_episode,
        grasp_action=grasp_action,
        enable_intervention=enable_intervention,
    )


def _maybe_wrap_reward_classifier(
    env: Any,
    task: WA2TaskConfig,
    *,
    checkpoint: Optional[str],
    threshold: Optional[float],
    consecutive_n: Optional[int],
    end_episode: bool,
    infer_mode: Optional[str] = None,
    infer_every_n: Optional[int] = None,
    session_infer_every_n: Optional[int] = None,
) -> Any:
    keys = task.classifier_keys
    if keys is None:
        if task.reward_mode == "constant_zero":
            warnings.warn(
                "classifier=True bypassed: reward.mode=constant_zero and "
                "classifier_keys is null (R8/R9 placeholder)",
                stacklevel=2,
            )
            return env
        raise ValueError(
            "classifier requested but classifier_keys is null and "
            f"reward.mode={task.reward_mode!r}"
        )
    ckpt = checkpoint or os.environ.get("WA2_CLASSIFIER_CKPT") or ""
    if not ckpt:
        raise ValueError(
            "classifier=True and classifier_keys are set; provide "
            "classifier_checkpoint= or WA2_CLASSIFIER_CKPT"
        )
    from hilserl_wa2.wrappers.reward_classifier import (
        WA2RewardClassifierWrapper,
        load_reward_classifier_fn,
        load_threshold_json,
    )

    thresh = threshold
    consec = 3 if consecutive_n is None else int(consecutive_n)
    ckpt_path = Path(os.path.expanduser(str(ckpt))).resolve()
    threshold_json = None
    search_roots = [ckpt_path] if ckpt_path.is_dir() else [ckpt_path.parent]
    # R12 packs keep threshold.json at run root (sibling of classifier_ckpt/),
    # not inside checkpoint_N — walk a few parents.
    for root in search_roots:
        cur = root
        for _ in range(4):
            candidate = cur / "threshold.json"
            if candidate.is_file():
                threshold_json = candidate
                break
            if cur.parent == cur:
                break
            cur = cur.parent
        if threshold_json is not None:
            break
    if thresh is None and threshold_json is not None:
        payload = load_threshold_json(threshold_json)
        thresh = float(payload["threshold"])
        if consecutive_n is None and "consecutive_n" in payload:
            consec = int(payload["consecutive_n"])
        print(f"CLASSIFIER_THRESHOLD from {threshold_json} thr={thresh}", flush=True)
    if thresh is None:
        raise ValueError(
            "classifier wrap needs classifier_threshold= or a threshold.json "
            "near the checkpoint (run root sibling of classifier_ckpt/ is OK)"
        )
    mode = (
        str(infer_mode).strip().lower()
        if infer_mode is not None
        else str(os.environ.get("WA2_CLASSIFIER_INFER_MODE", "decimate")).strip().lower()
    )
    every = (
        int(infer_every_n)
        if infer_every_n is not None
        else int(os.environ.get("WA2_CLASSIFIER_INFER_EVERY_N", "1"))
    )
    session_every = (
        int(session_infer_every_n)
        if session_infer_every_n is not None
        else int(os.environ.get("WA2_CLASSIFIER_SESSION_EVERY_N", "2"))
    )
    predict = load_reward_classifier_fn(
        ckpt,
        env.observation_space.sample(),
        list(keys),
    )
    return WA2RewardClassifierWrapper(
        env,
        predict,
        threshold=float(thresh),
        consecutive_n=consec,
        end_episode=bool(end_episode),
        image_keys=list(keys),
        infer_mode=mode,
        infer_every_n=every,
        session_infer_every_n=session_every,
    )


def _unwrap_base(env: Any) -> Any:
    return env.unwrapped if hasattr(env, "unwrapped") else env


def wrapper_names(env: Any) -> Tuple[str, ...]:
    names = []
    cur = env
    while True:
        names.append(type(cur).__name__)
        inner = getattr(cur, "env", None)
        if inner is None or inner is cur:
            break
        cur = inner
    return tuple(names)


def assert_fake_env_isolated(env: Any) -> Dict[str, Any]:
    """Runtime isolation checks (construction / init_node), not sys.modules."""

    names = wrapper_names(env)
    if "WA2SpacemouseIntervention" in names:
        raise RuntimeError("fake_env must not wrap WA2SpacemouseIntervention")

    base = _unwrap_base(env)
    if not getattr(base, "fake_env", False):
        raise RuntimeError("expected WA2Env.fake_env=True")
    if getattr(base, "_state_monitor", None) is not None:
        raise RuntimeError("fake_env constructed WA2StateMonitor")
    if getattr(base, "_servo", None) is not None:
        raise RuntimeError("fake_env constructed WA2ServoSession")
    cameras = getattr(base, "_cameras", None)
    cam_name = type(cameras).__name__ if cameras is not None else None
    if cam_name not in (None, "MockCameras"):
        raise RuntimeError(f"fake_env cameras should be MockCameras, got {cam_name}")

    rospy_initialized = False
    if "rospy" in __import__("sys").modules:
        rospy = __import__("sys").modules["rospy"]
        core = getattr(rospy, "core", None)
        if core is not None and hasattr(core, "is_initialized"):
            rospy_initialized = bool(core.is_initialized())
    if rospy_initialized:
        raise RuntimeError("rospy.init_node was called during fake_env")

    return {
        "wrappers": list(names),
        "cameras": cam_name,
        "rospy_initialized": rospy_initialized,
        "hardware_touched": False,
    }


def build_space_signature(
    task: WA2TaskConfig,
    role: Role,
    *,
    grasp_action: bool = False,
) -> Dict[str, Any]:
    """Compare Actor/Learner spaces using Mock backend only (no ROS)."""

    env = WA2Env(
        fake_env=True,
        scene_name=task.scene,
        contract_path=task.contract_path,
        spacemouse_path=task.spacemouse_path,
    )
    intervention_before = (env.observation_space, env.action_space)
    wrapped_intervention = False
    if role == "actor" and task.intervention:
        env = _maybe_wrap_intervention(env, task, auto_start_ros=False)
        wrapped_intervention = True
        if serialize_changed(intervention_before, env):
            env.close()
            raise RuntimeError("WA2SpacemouseIntervention must not change spaces")
    if grasp_action:
        from hilserl_wa2.wrappers.grasp_action import WA2GraspActionWrapper

        env = WA2GraspActionWrapper(env)
    env = _apply_serl_wrappers(env, task)
    obs_space = env.observation_space
    act_space = env.action_space
    digest = space_sha256(obs_space, act_space)
    info = {
        "role": role,
        "space_hash": digest,
        "observation": _space_shapes(obs_space),
        "action_shape": tuple(act_space.shape),
        "intervention_wrapped": wrapped_intervention,
        "grasp_action": bool(grasp_action),
    }
    env.close()
    return info


def serialize_changed(before: Tuple[Any, Any], env: Any) -> bool:
    from hilserl_wa2.experiments.task_config import serialize_space

    b_obs, b_act = before
    return serialize_space(b_obs) != serialize_space(env.observation_space) or (
        serialize_space(b_act) != serialize_space(env.action_space)
    )


def _space_shapes(space: Any) -> Dict[str, Any]:
    from gymnasium import spaces

    if isinstance(space, spaces.Dict):
        return {k: _space_shapes(v) for k, v in space.spaces.items()}
    if isinstance(space, spaces.Box):
        return {"shape": list(space.shape), "dtype": str(space.dtype)}
    return {"type": type(space).__name__}
