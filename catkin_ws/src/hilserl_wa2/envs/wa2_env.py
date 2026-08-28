"""WA2 Gymnasium environment (Mock / ROS read-only / ServoL)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, SupportsFloat, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from hilserl_wa2.envs.contracts import WA2EnvContract, resolve_contract_path
from hilserl_wa2.envs.scene_config import WA2SceneConfig, load_scene
from hilserl_wa2.ros_adapters.mock_cameras import MockCameras
from hilserl_wa2.ros_adapters.mock_robot import MockRobot

ObsType = Dict[str, Any]


class WA2Env(gym.Env):
    """Left-arm WA2 env.

    - ``fake_env=True``: R2 Mock backend (no ROS).
    - ``fake_env=False, read_only=True``: R3 ROS state, no ServoL.
    - ``fake_env=False, read_only=False``: R4 ServoL via WA2ServoSession.
    - ``scene_name`` / ``scene_path``: R5 task poses (swap YAML per task).
    - ``camera_cfg_path``: R6 dual-camera YAML (ROS topics); fake_env ignores it.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        fake_env: bool = True,
        read_only: bool = True,
        dry_run: bool = False,
        contract_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
        state_monitor: Optional[Any] = None,
        servo_session: Optional[Any] = None,
        episode_trans_limit_m: Optional[float] = None,
        episode_rot_limit_deg: Optional[float] = None,
        scene: Optional[WA2SceneConfig] = None,
        scene_name: Optional[str] = None,
        scene_path: Optional[Union[str, Path]] = None,
        auto_reset_motion: bool = True,
        camera_cfg_path: Optional[Union[str, Path]] = None,
        cameras: Optional[Any] = None,
        spacemouse_path: Optional[Union[str, Path]] = None,
        grasp_target: Optional[Sequence[float]] = None,
        release_target: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        self.contract = WA2EnvContract.from_yaml(resolve_contract_path(contract_path))
        if scene is not None:
            self.scene = scene
        else:
            self.scene = load_scene(scene_path=scene_path, scene_name=scene_name)
        # Scene may override episode length for the active task.
        self.max_steps = (
            int(self.scene.max_steps)
            if self.scene is not None and self.scene.max_steps is not None
            else int(self.contract.max_steps)
        )
        self.auto_reset_motion = bool(auto_reset_motion)
        self._camera_cfg_path = camera_cfg_path
        self.action_space: spaces.Space = self.contract.build_action_space()
        self.observation_space: spaces.Space = self.contract.build_observation_space()
        self._step_count = 0
        self._closed = False
        self._force_terminated = False
        self._force_truncated = False
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self._last_applied: Optional[Dict[str, Any]] = None
        self._last_reset_info: Dict[str, Any] = {}
        self._action_interrupt_callback: Optional[Callable[[], bool]] = None
        self._action_provider_callback: Optional[Callable[[], Sequence[float]]] = None
        self._state_monitor = None
        self._servo = None
        self._robot = None
        self._navi = None
        self._hand_stable_state = "released"
        self.dry_run = bool(dry_run)
        self.grasp_target, self.release_target = self._resolve_hand_targets(
            spacemouse_path=spacemouse_path,
            grasp_target=grasp_target,
            release_target=release_target,
        )
        self._episode_trans_limit_m = (
            None if episode_trans_limit_m is None else float(episode_trans_limit_m)
        )
        self._episode_rot_limit_deg = (
            None if episode_rot_limit_deg is None else float(episode_rot_limit_deg)
        )
        if self.scene is not None:
            if self.scene.episode_trans_limit_m is not None:
                self._episode_trans_limit_m = float(self.scene.episode_trans_limit_m)
            if self.scene.episode_rot_limit_deg is not None:
                self._episode_rot_limit_deg = float(self.scene.episode_rot_limit_deg)
        self._servo_injected = servo_session is not None

        if fake_env:
            self.fake_env = True
            self.read_only = True
            self._robot = MockRobot(self.contract)
            self._cameras = cameras if cameras is not None else MockCameras(
                self.contract, seed=seed
            )
        else:
            self.fake_env = False
            self.read_only = bool(read_only)
            if state_monitor is not None:
                self._state_monitor = state_monitor
            else:
                from hilserl_wa2.ros_adapters.state_monitor import WA2StateMonitor

                self._state_monitor = WA2StateMonitor(
                    arm=self.contract.arm,
                    state_max_age_s=self.contract.state_max_age_s,
                    joint_names=self.contract.raw["observation"]["state"]["joint_pos"][
                        "names"
                    ],
                    hand_names=self.contract.raw["observation"]["state"]["hand_joints"][
                        "names"
                    ],
                )
            if cameras is not None:
                self._cameras = cameras
            else:
                from hilserl_wa2.ros_adapters.wa2_cameras import WA2Cameras

                self._cameras = WA2Cameras(
                    contract=self.contract,
                    camera_cfg_path=camera_cfg_path,
                )
            if not self.read_only:
                if servo_session is not None:
                    self._servo = servo_session
                else:
                    self._servo = self._new_servo_session()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)
            self._cameras.reset(seed=seed)
        options = options or {}
        self._step_count = 0
        self._closed = False
        self._force_terminated = bool(options.get("force_terminated", False))
        self._force_truncated = bool(options.get("force_truncated", False))
        self._last_applied = None
        self._last_reset_info = {}
        self._hand_stable_state = "released"

        if self.fake_env:
            assert self._robot is not None
            tcp = options.get("tcp_pose")
            joints = options.get("joint_pos")
            hand = options.get("hand_joints")
            if self.scene is not None:
                tcp = tcp if tcp is not None else self.scene.task_reset_tcp
                joints = joints if joints is not None else self.scene.home_joints_left
                hand = hand if hand is not None else self.scene.hand_reset
                self._last_reset_info = {
                    "reset_ok": True,
                    "reset_mode": "fake_scene",
                    "scene_id": self.scene.scene_id,
                }
            self._robot.reset(tcp_pose=tcp, joint_pos=joints, hand_joints=hand)
        else:
            assert self._state_monitor is not None
            if not getattr(self._state_monitor, "_started", False):
                self._state_monitor.start()
            timeout = float(options.get("ready_timeout_s", 5.0))
            self._state_monitor.clear_stale_injection()
            self._state_monitor.wait_ready(timeout_s=timeout)
            # R6: start dual cameras and wait for fresh frames.
            if hasattr(self._cameras, "start"):
                self._cameras.start()
            if hasattr(self._cameras, "clear_stale_injection"):
                self._cameras.clear_stale_injection()
            if hasattr(self._cameras, "wait_ready"):
                cam_timeout = float(options.get("camera_ready_timeout_s", timeout))
                self._cameras.wait_ready(timeout_s=cam_timeout)

            skip_motion = bool(options.get("skip_reset_motion", False))
            do_motion = (
                self.auto_reset_motion
                and self.scene is not None
                and not self.read_only
                and not skip_motion
            )
            if do_motion:
                from hilserl_wa2.ros_adapters.reset_executor import WA2ResetExecutor

                # Close ServoL session before MoveJ/hand (control arbitration).
                if self._servo is not None:
                    try:
                        self._servo.close()
                    except Exception:
                        pass
                executor = WA2ResetExecutor(
                    scene=self.scene,
                    state_monitor=self._state_monitor,
                    dry_run=self.dry_run,
                    confirm_fn=options.get("confirm_fn"),
                )
                result = executor.run()
                self._last_reset_info = result.as_info()
                self._last_reset_info["scene_id"] = self.scene.scene_id
                if not result.ok:
                    raise RuntimeError(
                        f"R5 reset failed (no episode): {result.error}; "
                        f"stages={result.stages}"
                    )
            elif self.scene is not None:
                self._last_reset_info = {
                    "reset_ok": True,
                    "reset_mode": "skipped_motion",
                    "scene_id": self.scene.scene_id,
                }

            if not self.read_only:
                assert self._servo is not None
                # Ensure a fresh session origin each reset (after MoveJ home).
                if self._servo_injected:
                    if self._servo.started or self._servo.faulted or self._servo._closed:
                        try:
                            self._servo.close()
                        except Exception:
                            pass
                        self._servo._closed = False
                        self._servo._faulted = False
                        self._servo._started = False
                        self._servo._publish_count = 0
                else:
                    try:
                        self._servo.close()
                    except Exception:
                        pass
                    self._servo = self._new_servo_session()
                self._servo.start()

        obs = self._get_obs()
        info = self._get_info(applied=None, stale_fields=[])
        info.update(self._last_reset_info)
        return obs, info

    def _new_servo_session(self):
        from hilserl_wa2.ros_adapters.servo_session import WA2ServoSession

        return WA2ServoSession(
            contract=self.contract,
            state_monitor=self._state_monitor,
            dry_run=self.dry_run,
            episode_trans_limit_m=self._episode_trans_limit_m,
            episode_rot_limit_deg=self._episode_rot_limit_deg,
        )

    def hold_servo_latch(self) -> bool:
        """Zero ServoL velocity latch if a live session exists.

        Used by the Actor when the control loop stalls (upload/GIL) so the
        background latch thread cannot keep integrating a stale command.
        """

        if self._servo is None or not hasattr(self._servo, "hold_latched_action"):
            return False
        try:
            return bool(self._servo.hold_latched_action())
        except Exception:
            return False

    def set_action_interrupt_callback(
        self, callback: Optional[Callable[[], bool]]
    ) -> None:
        """Register an Actor-side request that may cancel a policy window.

        The callback is polled by the Servo executor before each 20 ms tick.
        It must be non-blocking and must not publish robot commands.
        """

        self._action_interrupt_callback = callback

    def set_action_provider_callback(
        self, callback: Optional[Callable[[], Sequence[float]]]
    ) -> None:
        """Optional per-tick continuous action source (T3-05 human path).

        When set, ``execute_action_window`` / fake multi-tick steps re-sample
        this callback at Servo rate (~50 Hz). Must be non-blocking and must
        not publish robot commands. Policy steps leave this unset.
        """

        self._action_provider_callback = callback

    def step(
        self, action: Any
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if self._closed:
            raise RuntimeError("step() called after close()")
        action_arr = self._validate_action(action)
        clipped = np.clip(
            action_arr, self.contract.action_low, self.contract.action_high
        ).astype(np.float32)

        applied: Optional[Dict[str, Any]]
        stale_fields = []
        terminated = bool(self._force_terminated)
        truncated = bool(self._force_truncated)
        fault = False

        if self.fake_env:
            assert self._robot is not None
            xyz = np.zeros(3, dtype=np.float64)
            rpy = np.zeros(3, dtype=np.float64)
            executed = 0
            interrupted_by = "none"
            action_sum = np.zeros(6, dtype=np.float64)
            for _ in range(self.contract.servo_ticks_per_action):
                if (
                    self._action_interrupt_callback is not None
                    and bool(self._action_interrupt_callback())
                ):
                    interrupted_by = "intervention"
                    break
                tick_action = clipped
                if self._action_provider_callback is not None:
                    provided = np.asarray(
                        self._action_provider_callback(), dtype=np.float64
                    ).reshape(6)
                    if not np.all(np.isfinite(provided)):
                        interrupted_by = "safety"
                        break
                    tick_action = np.clip(
                        provided, self.contract.action_low, self.contract.action_high
                    ).astype(np.float32)
                action_sum += tick_action.astype(np.float64)
                tick_applied = self._robot.apply_action(tick_action)
                xyz += np.asarray(tick_applied["delta_pos_xyz"], dtype=np.float64)
                rpy += np.asarray(tick_applied["delta_rot_rpy"], dtype=np.float64)
                executed += 1
            mean_action = (
                clipped
                if executed <= 0
                else (action_sum / float(executed)).astype(np.float32)
            )
            applied = {
                "delta_pos_m": float(np.linalg.norm(xyz)),
                "delta_rot_rad": float(np.linalg.norm(rpy)),
                "delta_pos_xyz": xyz,
                "delta_rot_rpy": rpy,
                "action_clipped": mean_action.copy(),
                "servo_ticks_requested": int(self.contract.servo_ticks_per_action),
                "servo_ticks_executed": int(executed),
                "execution_duration_s": float(
                    executed * self.contract.control_dt
                ),
                "interrupted_by": interrupted_by,
                "window_action_mean": mean_action.copy(),
                "window_action_samples": int(executed),
            }
        elif self.read_only:
            applied = {
                "delta_pos_m": 0.0,
                "delta_rot_rad": 0.0,
                "delta_pos_xyz": np.zeros(3, dtype=np.float64),
                "delta_rot_rpy": np.zeros(3, dtype=np.float64),
                "action_ignored_for_motion": True,
                "action_clipped": clipped.copy(),
                "servo_ticks_requested": int(self.contract.servo_ticks_per_action),
                "servo_ticks_executed": 0,
                "execution_duration_s": 0.0,
                "interrupted_by": "read_only",
            }
            assert self._state_monitor is not None
            stale_fields = list(self._state_monitor.stale_fields())
            stale_fields.extend(self._camera_stale_fields())
            if stale_fields:
                truncated = True
        else:
            assert self._servo is not None and self._state_monitor is not None
            from hilserl_wa2.ros_adapters.servo_session import is_firmware_protected

            state_stale = list(self._state_monitor.stale_fields())
            cam_stale = self._camera_stale_fields()
            stale_fields = list(state_stale)
            stale_fields.extend(cam_stale)
            info_pre = self._state_monitor.get_info()
            singular = bool(info_pre.get("is_singular"))
            protected = is_firmware_protected(
                info_pre.get("cmd_name"), info_pre.get("cmd_num")
            )
            # Camera age must not abort ServoL: the 50 Hz latch-velocity thread
            # keeps integrating while env.step copies the latest 128x128 cache.
            if state_stale or singular or protected:
                precheck = (
                    f"precheck stale={state_stale} singular={singular} "
                    f"cmd_name={info_pre.get('cmd_name')} cmd_num={info_pre.get('cmd_num')}"
                )
                try:
                    self._servo._fault_stop(precheck)
                except Exception:
                    pass
                truncated = True
                fault = True
                applied = {
                    "delta_pos_m": 0.0,
                    "delta_rot_rad": 0.0,
                    "delta_pos_xyz": np.zeros(3, dtype=np.float64),
                    "delta_rot_rpy": np.zeros(3, dtype=np.float64),
                    "action_clipped": clipped.copy(),
                    "published": False,
                    "servo_error": precheck,
                    "servo_ticks_requested": int(
                        self.contract.servo_ticks_per_action
                    ),
                    "servo_ticks_executed": 0,
                    "execution_duration_s": 0.0,
                    "interrupted_by": "safety",
                }
            else:
                try:
                    applied = self._servo.execute_action_window(
                        clipped,
                        ticks=self.contract.servo_ticks_per_action,
                        cancel_check=self._action_interrupt_callback,
                        action_provider=self._action_provider_callback,
                    )
                except Exception as exc:
                    truncated = True
                    fault = True
                    applied = {
                        "delta_pos_m": 0.0,
                        "delta_rot_rad": 0.0,
                        "delta_pos_xyz": np.zeros(3, dtype=np.float64),
                        "delta_rot_rpy": np.zeros(3, dtype=np.float64),
                        "action_clipped": clipped.copy(),
                        "published": False,
                        "servo_error": str(exc),
                        "servo_ticks_requested": int(
                            self.contract.servo_ticks_per_action
                        ),
                        "servo_ticks_executed": 0,
                        "execution_duration_s": 0.0,
                        "interrupted_by": "safety",
                    }

        self._step_count += 1
        self._last_applied = applied
        if self._step_count >= self.max_steps:
            truncated = True
        if self._servo is not None and self._servo.faulted:
            truncated = True
            fault = True

        obs = self._get_obs()
        reward = 0.0
        info = self._get_info(applied=applied, stale_fields=stale_fields)
        info["servo_faulted"] = bool(fault or (self._servo.faulted if self._servo else False))
        return obs, reward, terminated, truncated, info

    def request_hand(self, command: str) -> Dict[str, Any]:
        """Open/close the same-side dexterous hand. Action space stays 6D."""

        command = str(command).strip().lower()
        if command not in ("grasp", "release", "toggle"):
            raise ValueError("request_hand command must be grasp, release or toggle")
        if command == "toggle":
            command = "release" if self._hand_stable_state == "grasped" else "grasp"
        target = self.grasp_target if command == "grasp" else self.release_target
        if self.fake_env:
            assert self._robot is not None
            self._robot.hand_joints = np.asarray(target, dtype=np.float32).reshape(6)
            ok = True
        elif self.read_only and not self.dry_run:
            raise RuntimeError("request_hand requires a motion-enabled env")
        elif self.dry_run:
            ok = True
        else:
            ok = bool(self._execute_hand(command, target))
        if ok:
            self._hand_stable_state = "grasped" if command == "grasp" else "released"
        return {
            "ok": bool(ok),
            "command": command,
            "target": [float(value) for value in target],
            "state": self._hand_stable_state,
        }

    def _resolve_hand_targets(
        self,
        *,
        spacemouse_path: Optional[Union[str, Path]],
        grasp_target: Optional[Sequence[float]],
        release_target: Optional[Sequence[float]],
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        yaml_grasp = None
        yaml_release = None
        if spacemouse_path is not None:
            from hilserl_wa2.interventions.spacemouse_config import load_spacemouse_config

            cfg = load_spacemouse_config(spacemouse_path)
            teleop = cfg.teleop_ros_params()
            yaml_grasp = teleop.get("grasp_target")
            yaml_release = teleop.get("release_target")
        elif self.scene is not None:
            yaml_release = list(self.scene.hand_reset)
        grasp = tuple(
            float(value)
            for value in (
                grasp_target
                if grasp_target is not None
                else yaml_grasp
                if yaml_grasp is not None
                else (0.1, 0.9, 0.7, 0.7, 0.4, 0.4)
            )
        )
        release = tuple(
            float(value)
            for value in (
                release_target
                if release_target is not None
                else yaml_release
                if yaml_release is not None
                else (0.1, 0.9, 0.3, 0.3, 0.3, 0.3)
            )
        )
        if len(grasp) != 6 or len(release) != 6:
            raise ValueError("grasp_target and release_target must have length 6")
        return grasp, release

    def _execute_hand(self, command: str, target: Sequence[float]) -> bool:
        from naviai_controller import HandType

        controller = self._ensure_navi()
        joints = list(map(float, target))
        if command == "grasp":
            return bool(controller.grasp_hand(HandType.LEFT, joints))
        if all(abs(value) <= 1e-12 for value in joints):
            return bool(controller.release_hand(HandType.LEFT))
        return bool(controller.grasp_hand(HandType.LEFT, joints))

    def _ensure_navi(self):
        if self._navi is not None:
            return self._navi
        import time

        from naviai_controller import NaviController

        self._navi = NaviController(model="wa2")
        time.sleep(0.3)
        return self._navi

    def close(self) -> None:
        self._closed = True
        if self._servo is not None:
            try:
                self._servo.close()
            except Exception:
                pass
        if self._state_monitor is not None:
            try:
                self._state_monitor.stop()
            except Exception:
                pass
        if self._cameras is not None and hasattr(self._cameras, "stop"):
            try:
                self._cameras.stop()
            except Exception:
                pass

    def inject_success(self) -> None:
        self._force_terminated = True

    def inject_truncate(self) -> None:
        self._force_truncated = True

    def _camera_stale_fields(self) -> list:
        if self.fake_env:
            return []
        if hasattr(self._cameras, "stale_fields"):
            return list(self._cameras.stale_fields())
        return []

    def _validate_action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32)
        if arr.shape != (self.contract.action_dim,):
            raise ValueError(
                f"action shape must be {(self.contract.action_dim,)}, got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("action must not contain NaN/Inf")
        return arr

    def observe(self) -> ObsType:
        """Return a current observation without executing another env step."""

        return self._get_obs()

    def _get_obs(self) -> ObsType:
        if self.fake_env:
            assert self._robot is not None
            state = self._robot.get_state_dict()
        else:
            assert self._state_monitor is not None
            state = self._state_monitor.get_state()
        images = self._cameras.get_images()
        return {
            "state": {
                "tcp_pose": state["tcp_pose"].astype(np.float32, copy=False),
                "tcp_vel": state["tcp_vel"].astype(np.float32, copy=False),
                "joint_pos": state["joint_pos"].astype(np.float32, copy=False),
                "hand_joints": state["hand_joints"].astype(np.float32, copy=False),
            },
            "images": {
                "head": images["head"],
                "wrist": images["wrist"],
            },
        }

    def _get_info(
        self,
        applied: Optional[Dict[str, Any]],
        stale_fields: Optional[list] = None,
    ) -> Dict[str, Any]:
        stale_fields = list(stale_fields or [])
        if self.fake_env:
            assert self._robot is not None
            info = self._robot.get_info_fields()
        else:
            assert self._state_monitor is not None
            info = dict(self._state_monitor.get_info())
        image_ages: Dict[str, Any] = {}
        if hasattr(self._cameras, "get_ages"):
            image_ages = dict(self._cameras.get_ages())
        age_vals = [float(v) for v in image_ages.values() if v is not None]
        image_age = float(max(age_vals)) if age_vals else 0.0
        info.update(
            {
                "image_age": image_age,
                "image_ages": image_ages,
                "step_count": int(self._step_count),
                "max_steps": int(self.max_steps),
                "fake_env": bool(self.fake_env),
                "read_only": bool(self.read_only),
                "would_stop_on_close": True,
                "wrist_enabled": bool(self.contract.wrist_enabled),
                "stale": bool(stale_fields),
                "stale_fields": stale_fields,
                "dry_run": bool(self.dry_run),
                "scene_id": None if self.scene is None else self.scene.scene_id,
                "policy_hz": float(self.contract.policy_hz),
                "servo_hz": float(self.contract.control_hz),
                "servo_ticks_per_action": int(
                    self.contract.servo_ticks_per_action
                ),
            }
        )
        if self._servo is not None:
            info["servo_health"] = self._servo.health()
        if applied is not None:
            info["delta_pos_m"] = float(applied.get("delta_pos_m", 0.0))
            info["delta_rot_rad"] = float(applied.get("delta_rot_rad", 0.0))
            info["delta_pos_xyz"] = np.asarray(
                applied.get("delta_pos_xyz", np.zeros(3)), dtype=np.float64
            )
            info["delta_rot_rpy"] = np.asarray(
                applied.get("delta_rot_rpy", np.zeros(3)), dtype=np.float64
            )
            if "action_ignored_for_motion" in applied:
                info["action_ignored_for_motion"] = True
            if "action_clipped" in applied:
                info["action_clipped"] = np.asarray(
                    applied["action_clipped"], dtype=np.float32
                )
            for key in (
                "cmd_tcp",
                "meas_tcp",
                "tracking_err_m",
                "tracking_err_rad",
                "loop_dt",
                "published",
                "publish_count",
                "integrate_count",
                "interval_ticks",
                "servo_error",
                "servo_ticks_requested",
                "servo_ticks_executed",
                "execution_duration_s",
                "interrupted_by",
                "window_action_mean",
                "window_action_samples",
            ):
                if key in applied:
                    info[key] = applied[key]
        return info
