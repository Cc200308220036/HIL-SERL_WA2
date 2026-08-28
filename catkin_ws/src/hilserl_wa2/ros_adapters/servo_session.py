"""Safe ServoL session: sole motion publisher for WA2Env (R4)."""

from __future__ import annotations

import atexit
import math
import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from hilserl_wa2.envs.contracts import WA2EnvContract

# Firmware UplimbState.cmd_num=15 / cmd_name=PROTECTED is a vendor safety lock
# (not in naviai_controller.CmdState, which only enumerates through SERVOL=14).
PROTECTED_CMD_NAMES = frozenset({"PROTECTED", "PROTECT", "SAFETY_LOCK", "LOCKED"})
PROTECTED_CMD_NUM = 15
# Stop integrating / publishing if commanded TCP has run this far from measured.
TRACKING_ERR_LIMIT_M = 0.015


def is_firmware_protected(cmd_name: Any, cmd_num: Any = None) -> bool:
    name = str(cmd_name or "").strip().upper()
    if name in PROTECTED_CMD_NAMES:
        return True
    try:
        return int(cmd_num) == PROTECTED_CMD_NUM
    except (TypeError, ValueError):
        return False


def integrate_normalized_action(
    pose7: np.ndarray,
    action: Sequence[float],
    contract: WA2EnvContract,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Clip action and integrate one TCP step. Pure function (no ROS)."""

    action_arr = np.asarray(action, dtype=np.float64).reshape(6)
    if not np.all(np.isfinite(action_arr)):
        raise ValueError("action must be finite")
    clipped = np.clip(action_arr, contract.action_low, contract.action_high)

    delta_pos = clipped[:3] * contract.max_pos_delta_m
    delta_rot = clipped[3:] * contract.max_rot_delta_rad

    pos_norm = float(np.linalg.norm(delta_pos))
    if pos_norm > contract.max_pos_delta_m + 1e-12:
        delta_pos *= contract.max_pos_delta_m / pos_norm
    rot_norm = float(np.linalg.norm(delta_rot))
    if rot_norm > contract.max_rot_delta_rad + 1e-12:
        delta_rot *= contract.max_rot_delta_rad / rot_norm

    new_pose = np.asarray(pose7, dtype=np.float64).reshape(7).copy()
    if contract.position_frame != "base":
        raise ValueError("only position_frame=base is supported")
    new_pose[:3] = new_pose[:3] + delta_pos

    current = Rotation.from_quat(new_pose[3:])
    delta = Rotation.from_rotvec(delta_rot)
    if contract.rotation_frame == "tool":
        updated = current * delta
    elif contract.rotation_frame == "base":
        updated = delta * current
    else:
        raise ValueError(f"unknown rotation_frame={contract.rotation_frame}")
    quat = updated.as_quat()
    quat = quat / np.linalg.norm(quat)
    if float(np.dot(quat, new_pose[3:])) < 0.0:
        quat = -quat
    new_pose[3:] = quat
    return new_pose.astype(np.float32), {
        "delta_pos_m": float(np.linalg.norm(delta_pos)),
        "delta_rot_rad": float(np.linalg.norm(delta_rot)),
        "delta_pos_xyz": delta_pos,
        "delta_rot_rpy": delta_rot,
        "action_clipped": clipped.astype(np.float32),
    }


# If the env loop stalls, the ServoL latch would keep integrating the last
# non-zero command ("runs away along the previous axis"). Cap how long a
# latched velocity may live without a fresh apply_normalized_action.
# ~20 Hz Actor ⇒ ~50 ms/step. 0.08s ≈ 1.5 steps / ~4 servo ticks: short enough
# that a GIL/upload stall cannot free-flight for tens of mm, long enough that
# a single slow step does not stutter teleop to zero every frame.
DEFAULT_LATCH_MAX_AGE_S = 0.08


class WA2ServoSession:
    """Exclusive ServoL executor with stale/singular/fault stop+clear.

    Live mode keeps a 50 Hz latch-velocity thread: ``apply_normalized_action``
    only stores the latest 6D command. The thread integrates at most
    1 mm / 0.25° per ``control_dt`` (0.02 s) and is the sole ServoL path.
    If no new apply arrives within ``latch_max_age_s``, the latch is zeroed
    (hold) so a slow control loop cannot free-flight on the previous axis.
    ``dry_run`` never starts that thread: it still integrates one tick
    synchronously so offline tests stay simple.
    """

    def __init__(
        self,
        contract: WA2EnvContract,
        state_monitor: Any,
        dry_run: bool = False,
        episode_trans_limit_m: Optional[float] = None,
        episode_rot_limit_deg: Optional[float] = None,
        arm_ctrl: Any = None,
        require_confirm_env: str = "R4_CONFIRM",
        latch_max_age_s: float = DEFAULT_LATCH_MAX_AGE_S,
    ):
        self.contract = contract
        self.state_monitor = state_monitor
        self.dry_run = bool(dry_run)
        self.episode_trans_limit_m = (
            None if episode_trans_limit_m is None else float(episode_trans_limit_m)
        )
        self.episode_rot_limit_rad = (
            None
            if episode_rot_limit_deg is None
            else math.radians(float(episode_rot_limit_deg))
        )
        age = float(latch_max_age_s)
        if not math.isfinite(age) or age <= 0.0:
            raise ValueError("latch_max_age_s must be finite and > 0")
        self.latch_max_age_s = age
        self._arm_ctrl = arm_ctrl
        self._require_confirm_env = require_confirm_env
        self._started = False
        self._faulted = False
        self._closed = False
        self._origin_pose: Optional[np.ndarray] = None
        self._origin_rot: Optional[Rotation] = None
        self._cmd_tcp: Optional[np.ndarray] = None
        self._latched_action = np.zeros(6, dtype=np.float32)
        self._interval_delta_pos_xyz = np.zeros(3, dtype=np.float64)
        self._interval_delta_rot_rpy = np.zeros(3, dtype=np.float64)
        self._ticks_since_apply = 0
        self._integrate_count = 0
        self._latch_expire_count = 0
        self._box_exceeded: Optional[str] = None
        self._last_publish_t: Optional[float] = None
        self._last_apply_t: Optional[float] = None
        self._stop_ok = False
        self._clear_ok = False
        self._atexit_registered = False
        self._publish_count = 0
        self._cmd_lock = threading.Lock()
        self._pub_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._window_condition = threading.Condition(self._cmd_lock)
        self._window_id = 0
        self._window_active = False
        self._window_requested_ticks = 0
        self._window_remaining_ticks = 0
        self._window_executed_ticks = 0
        self._window_started_t: Optional[float] = None
        self._window_delta_pos_xyz = np.zeros(3, dtype=np.float64)
        self._window_delta_rot_rpy = np.zeros(3, dtype=np.float64)
        self._window_cancel_check: Optional[Callable[[], bool]] = None
        self._window_action_provider: Optional[Callable[[], Sequence[float]]] = None
        self._window_action_sum = np.zeros(6, dtype=np.float64)
        self._window_action_samples = 0
        self._window_interrupted_by = "none"
        self._pub_stop: Optional[threading.Event] = None
        self._pub_thread: Optional[threading.Thread] = None

    @property
    def started(self) -> bool:
        return self._started

    @property
    def faulted(self) -> bool:
        return self._faulted

    @property
    def publish_count(self) -> int:
        return self._publish_count

    @property
    def integrate_count(self) -> int:
        return self._integrate_count

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("cannot start a closed session")
        if self._started:
            return
        if not self.dry_run:
            import os

            if os.environ.get(self._require_confirm_env) != "YES":
                raise RuntimeError(
                    f"refusing real ServoL without {self._require_confirm_env}=YES"
                )
            self._ensure_arm_ctrl()
            ok = self._arm_ctrl.set_servo_params(0.02, 800, self._arm_group_left())
            if not ok:
                raise RuntimeError("set_servo_params failed")

        if not getattr(self.state_monitor, "_started", False):
            self.state_monitor.start()
        self.state_monitor.wait_ready(timeout_s=8.0)
        if not self.state_monitor.is_fresh():
            raise RuntimeError(f"state not fresh: {self.state_monitor.stale_fields()}")
        info = self.state_monitor.get_info()
        if info.get("is_singular"):
            raise RuntimeError("left arm is singular; refuse to start ServoL")
        if is_firmware_protected(info.get("cmd_name"), info.get("cmd_num")):
            raise RuntimeError(
                f"firmware protected; refuse to start ServoL "
                f"cmd_name={info.get('cmd_name')} cmd_num={info.get('cmd_num')}"
            )

        pose = self.state_monitor.get_state()["tcp_pose"].copy()
        with self._cmd_lock:
            self._origin_pose = pose.copy()
            self._origin_rot = Rotation.from_quat(pose[3:])
            self._cmd_tcp = pose.copy()
            self._latched_action = np.zeros(6, dtype=np.float32)
            self._interval_delta_pos_xyz = np.zeros(3, dtype=np.float64)
            self._interval_delta_rot_rpy = np.zeros(3, dtype=np.float64)
            self._ticks_since_apply = 0
            self._integrate_count = 0
            self._latch_expire_count = 0
            self._box_exceeded = None
            self._reset_window_locked()
        self._faulted = False
        self._started = True
        self._last_apply_t = time.monotonic()
        self._stop_ok = False
        self._clear_ok = False
        if not self._atexit_registered:
            atexit.register(self._atexit_close)
            self._atexit_registered = True
        if not self.dry_run:
            self._start_publisher()
            self._publish_pose(pose)

    def apply_normalized_action(
        self, action: Sequence[float], dt: Optional[float] = None
    ) -> Dict[str, Any]:
        if self._closed or self._faulted:
            raise RuntimeError("servo session faulted/closed; refuse apply")
        if not self._started:
            raise RuntimeError("call start() before apply_normalized_action")
        if self._box_exceeded is not None:
            raise RuntimeError(self._box_exceeded)

        t0 = time.monotonic()
        meas = self._precheck_or_fault()
        action_arr = np.asarray(action, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(action_arr)):
            raise ValueError("action must be finite")
        clipped = np.clip(
            action_arr, self.contract.action_low, self.contract.action_high
        ).astype(np.float32)

        loop_dt = None if self._last_apply_t is None else (t0 - self._last_apply_t)
        self._last_apply_t = t0
        loop_dt_out = (
            loop_dt if loop_dt is not None else float(dt or self.contract.control_dt)
        )

        if self.dry_run:
            new_pose, deltas = self._advance_command(
                clipped, meas, raise_on_box=True
            )
            assert new_pose is not None and deltas is not None
            return self._apply_result(
                deltas=deltas,
                cmd_tcp=new_pose,
                meas=meas,
                clipped=deltas["action_clipped"],
                loop_dt=loop_dt_out,
                published=False,
                interval_ticks=1,
            )

        interval = self._latch_action(clipped)
        cmd = interval["cmd_tcp"]
        if cmd is None:
            cmd = meas
        return self._apply_result(
            deltas={
                "delta_pos_m": interval["delta_pos_m"],
                "delta_rot_rad": interval["delta_rot_rad"],
                "delta_pos_xyz": interval["delta_pos_xyz"],
                "delta_rot_rpy": interval["delta_rot_rpy"],
            },
            cmd_tcp=cmd,
            meas=meas,
            clipped=clipped,
            loop_dt=loop_dt_out,
            published=False,
            interval_ticks=int(interval["interval_ticks"]),
        )

    def _clip_norm_action(self, action: Sequence[float]) -> np.ndarray:
        action_arr = np.asarray(action, dtype=np.float64).reshape(6)
        if not np.all(np.isfinite(action_arr)):
            raise ValueError("action must be finite")
        return np.clip(
            action_arr, self.contract.action_low, self.contract.action_high
        ).astype(np.float32)

    def _resolve_tick_action(
        self,
        fallback: np.ndarray,
        action_provider: Optional[Callable[[], Sequence[float]]],
    ) -> np.ndarray:
        """Pick the continuous action for one Servo tick inside a window."""

        if action_provider is None:
            return fallback.astype(np.float32, copy=True)
        try:
            provided = action_provider()
        except Exception as exc:
            raise RuntimeError(f"action_provider failed: {exc}") from exc
        return self._clip_norm_action(provided)

    def _note_window_action(self, action: np.ndarray) -> None:
        self._window_action_sum += np.asarray(action, dtype=np.float64).reshape(6)
        self._window_action_samples += 1

    def _window_action_mean(self, fallback: np.ndarray) -> np.ndarray:
        if self._window_action_samples <= 0:
            return fallback.astype(np.float32, copy=True)
        mean = self._window_action_sum / float(self._window_action_samples)
        return mean.astype(np.float32)

    def execute_action_window(
        self,
        action: Sequence[float],
        *,
        ticks: int,
        cancel_check: Optional[Callable[[], bool]] = None,
        action_provider: Optional[Callable[[], Sequence[float]]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute one finite high-level action window.

        Live mode authorizes the 50 Hz publisher to integrate exactly ``ticks``
        times and blocks until the budget is consumed or cancelled. Dry-run
        performs the same number of integrations synchronously. The latch is
        cleared at every exit, so a high-level action can never leak into a
        sixth Servo tick.

        When ``action_provider`` is set (T3-05 human path), each Servo tick
        re-samples the continuous command at 50 Hz while still emitting one
        high-level transition whose ``action_clipped`` is the mean over
        executed ticks.
        """

        requested = int(ticks)
        if requested < 1:
            raise ValueError("ticks must be >= 1")
        if self._closed or self._faulted:
            raise RuntimeError("servo session faulted/closed; refuse action window")
        if not self._started:
            raise RuntimeError("call start() before execute_action_window")
        if self._box_exceeded is not None:
            raise RuntimeError(self._box_exceeded)

        started = time.monotonic()
        meas = self._precheck_or_fault()
        clipped = self._clip_norm_action(action)
        loop_dt = (
            float(self.contract.control_dt * requested)
            if self._last_apply_t is None
            else float(started - self._last_apply_t)
        )
        self._last_apply_t = started
        self._window_action_sum[:] = 0.0
        self._window_action_samples = 0

        if self.dry_run:
            xyz = np.zeros(3, dtype=np.float64)
            rpy = np.zeros(3, dtype=np.float64)
            cmd = meas.copy()
            executed = 0
            interrupted_by = "none"
            for _ in range(requested):
                if cancel_check is not None and bool(cancel_check()):
                    interrupted_by = "intervention"
                    break
                tick_action = self._resolve_tick_action(clipped, action_provider)
                self._note_window_action(tick_action)
                cmd_next, deltas = self._advance_command(
                    tick_action, meas, raise_on_box=True
                )
                assert cmd_next is not None and deltas is not None
                cmd = cmd_next
                xyz += np.asarray(deltas["delta_pos_xyz"], dtype=np.float64)
                rpy += np.asarray(deltas["delta_rot_rpy"], dtype=np.float64)
                executed += 1
            mean_action = self._window_action_mean(clipped)
            out = self._apply_result(
                deltas={
                    "delta_pos_m": float(np.linalg.norm(xyz)),
                    "delta_rot_rad": float(np.linalg.norm(rpy)),
                    "delta_pos_xyz": xyz,
                    "delta_rot_rpy": rpy,
                },
                cmd_tcp=cmd,
                meas=meas,
                clipped=mean_action,
                loop_dt=loop_dt,
                published=False,
                interval_ticks=executed,
            )
            out.update(
                servo_ticks_requested=requested,
                servo_ticks_executed=executed,
                execution_duration_s=float(time.monotonic() - started),
                interrupted_by=interrupted_by,
                window_action_mean=mean_action,
                window_action_samples=int(self._window_action_samples),
            )
            return out

        with self._window_condition:
            if self._window_active:
                raise RuntimeError("another Servo action window is already active")
            self._window_id += 1
            window_id = self._window_id
            self._latched_action = clipped.copy()
            self._window_active = True
            self._window_requested_ticks = requested
            self._window_remaining_ticks = requested
            self._window_executed_ticks = 0
            self._window_started_t = started
            self._window_delta_pos_xyz[:] = 0.0
            self._window_delta_rot_rpy[:] = 0.0
            self._window_cancel_check = cancel_check
            self._window_action_provider = action_provider
            self._window_interrupted_by = "none"
            self._window_condition.notify_all()

        timeout = (
            float(timeout_s)
            if timeout_s is not None
            else max(0.25, requested * float(self.contract.control_dt) + 0.15)
        )
        deadline = started + timeout
        with self._window_condition:
            while self._window_active and self._window_id == window_id and not self._faulted:
                remain = deadline - time.monotonic()
                if remain <= 0.0:
                    self._cancel_window_locked("timeout")
                    break
                self._window_condition.wait(timeout=min(remain, self.contract.control_dt))
            executed = int(self._window_executed_ticks)
            xyz = self._window_delta_pos_xyz.copy()
            rpy = self._window_delta_rot_rpy.copy()
            interrupted_by = str(self._window_interrupted_by)
            mean_action = self._window_action_mean(clipped)
            samples = int(self._window_action_samples)
            cmd = meas.copy() if self._cmd_tcp is None else self._cmd_tcp.copy()
            self._latched_action = np.zeros(6, dtype=np.float32)
            self._window_cancel_check = None
            self._window_action_provider = None

        if interrupted_by == "timeout":
            self._fault_stop("action_window_timeout")
            raise RuntimeError(
                f"Servo action window timeout: executed {executed}/{requested} ticks"
            )
        if self._faulted:
            raise RuntimeError(
                f"Servo fault during action window: executed {executed}/{requested} ticks"
            )
        out = self._apply_result(
            deltas={
                "delta_pos_m": float(np.linalg.norm(xyz)),
                "delta_rot_rad": float(np.linalg.norm(rpy)),
                "delta_pos_xyz": xyz,
                "delta_rot_rpy": rpy,
            },
            cmd_tcp=cmd,
            meas=meas,
            clipped=mean_action,
            loop_dt=loop_dt,
            published=executed > 0,
            interval_ticks=executed,
        )
        out.update(
            servo_ticks_requested=requested,
            servo_ticks_executed=executed,
            execution_duration_s=float(time.monotonic() - started),
            interrupted_by=interrupted_by,
            window_action_mean=mean_action,
            window_action_samples=samples,
        )
        return out

    def hold_latched_action(self) -> bool:
        """Zero the velocity latch immediately (safe hold).

        Call when the Actor control loop detects a stall or upload backlog so
        the 50 Hz publisher cannot keep integrating a stale non-zero command.
        Returns True if a non-zero latch was cleared.
        """

        with self._cmd_lock:
            was = float(np.linalg.norm(self._latched_action)) > 1e-9
            self._latched_action = np.zeros(6, dtype=np.float32)
            if self._window_active:
                self._cancel_window_locked("hold")
            # Treat as a fresh hold apply so the publisher age clock resets.
            self._last_apply_t = time.monotonic()
        return was

    def stop(self) -> bool:
        if self.dry_run or self._arm_ctrl is None:
            self._stop_ok = True
            return True
        try:
            unlock = getattr(self._arm_ctrl, "unlock", None)
            if unlock is not None:
                unlock()
        except Exception:
            pass
        try:
            self._stop_ok = bool(self._arm_ctrl.stop())
        except Exception:
            self._stop_ok = False
        return self._stop_ok

    def clear(self) -> bool:
        if self.dry_run or self._arm_ctrl is None:
            self._clear_ok = True
            return True
        try:
            self._clear_ok = bool(self._arm_ctrl.clear_servo_params())
        except Exception:
            self._clear_ok = False
        return self._clear_ok

    def health(self) -> Dict[str, Any]:
        thread = self._pub_thread
        with self._cmd_lock:
            latched = self._latched_action.copy()
            ticks = int(self._ticks_since_apply)
            cmd = None if self._cmd_tcp is None else self._cmd_tcp.copy()
            window_active = bool(self._window_active)
            window_remaining = int(self._window_remaining_ticks)
        return {
            "started": self._started,
            "faulted": self._faulted,
            "closed": self._closed,
            "dry_run": self.dry_run,
            "publish_count": self._publish_count,
            "integrate_count": self._integrate_count,
            "latch_expire_count": self._latch_expire_count,
            "latch_max_age_s": self.latch_max_age_s,
            "interval_ticks": ticks,
            "latched_action": latched,
            "cmd_tcp": cmd,
            "window_active": window_active,
            "window_remaining_ticks": window_remaining,
            "box_exceeded": self._box_exceeded,
            "publisher_alive": bool(thread is not None and thread.is_alive()),
            "stop_ok": self._stop_ok,
            "clear_ok": self._clear_ok,
            "stale_fields": self.state_monitor.stale_fields()
            if self.state_monitor is not None
            else [],
            "is_singular": None
            if self.state_monitor is None
            else self.state_monitor.get_info().get("is_singular"),
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            with self._window_condition:
                if self._window_active:
                    self._cancel_window_locked("closed")
            self._stop_publisher()
            self.stop()
            self.clear()
        finally:
            self._started = False
            self._closed = True

    def _atexit_close(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _fault_stop(self, reason: str) -> None:
        with self._stop_lock:
            if self._faulted:
                return
            self._faulted = True
            print(f"SERVO_FAULT_STOP reason={reason}", flush=True)
        with self._window_condition:
            if self._window_active:
                self._cancel_window_locked("safety")
            self._window_condition.notify_all()
        self._stop_publisher()
        self.stop()
        self.clear()
        self._started = False

    def _ensure_arm_ctrl(self) -> None:
        if self._arm_ctrl is not None:
            return
        import rospy
        from naviai_controller import NaviController, RobotModel

        if not rospy.core.is_initialized():
            rospy.init_node("wa2_servo_session", anonymous=True, disable_signals=True)
        # auto_spin keeps ArmController callbacks alive alongside StateMonitor.
        self._arm_ctrl = NaviController(model=RobotModel.WA2, auto_spin=True).arm

    def _start_publisher(self) -> None:
        if self._pub_thread is not None and self._pub_thread.is_alive():
            return
        self._pub_stop = threading.Event()
        self._pub_thread = threading.Thread(
            target=self._publisher_loop,
            name="wa2_servol_latch",
            daemon=True,
        )
        self._pub_thread.start()

    def _stop_publisher(self) -> None:
        if self._pub_stop is not None:
            self._pub_stop.set()
        thread = self._pub_thread
        self._pub_thread = None
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=0.5)

    def _publisher_loop(self) -> None:
        period = float(self.contract.control_dt)
        stop = self._pub_stop
        assert stop is not None
        while not stop.is_set():
            t0 = time.monotonic()
            if self._faulted or self._closed or not self._started:
                break
            reason = self._publisher_safety_reason()
            if reason is not None:
                self._fault_stop(reason)
                break
            try:
                meas = self.state_monitor.get_state()["tcp_pose"].copy()
            except Exception:
                meas = None
            with self._cmd_lock:
                last_apply = self._last_apply_t
                action = self._latched_action.copy()
                pose = None if self._cmd_tcp is None else self._cmd_tcp.copy()
                window_active = bool(self._window_active)
                window_id = int(self._window_id)
                cancel_check = self._window_cancel_check if window_active else None
                action_provider = (
                    self._window_action_provider if window_active else None
                )
                age = None if last_apply is None else float(t0 - last_apply)
                if not window_active and age is not None and age > self.latch_max_age_s:
                    if float(np.linalg.norm(action)) > 1e-9:
                        self._latch_expire_count += 1
                    self._latched_action = np.zeros(6, dtype=np.float32)
                    action = np.zeros(6, dtype=np.float32)
                elif (
                    not window_active
                    and age is not None
                    and age > 0.5 * self.latch_max_age_s
                    and float(np.linalg.norm(action)) > 1e-9
                ):
                    # Fade latch in the second half of the age window so a
                    # brief stall decelerates instead of full-speed free-flight.
                    span = max(1e-6, 0.5 * self.latch_max_age_s)
                    scale = float(
                        np.clip(1.0 - (age - 0.5 * self.latch_max_age_s) / span, 0.0, 1.0)
                    )
                    action = (action * np.float32(scale)).astype(np.float32)
            cancelled = False
            if cancel_check is not None:
                try:
                    cancelled = bool(cancel_check())
                except Exception:
                    cancelled = True
            if cancelled:
                with self._window_condition:
                    if self._window_active and self._window_id == window_id:
                        self._cancel_window_locked("intervention")
            if (
                window_active
                and not cancelled
                and action_provider is not None
            ):
                try:
                    action = self._resolve_tick_action(action, action_provider)
                    with self._cmd_lock:
                        if self._window_active and self._window_id == window_id:
                            self._latched_action = action.copy()
                except Exception:
                    cancelled = True
                    with self._window_condition:
                        if self._window_active and self._window_id == window_id:
                            self._cancel_window_locked("safety")
            deltas = None
            if meas is not None and not cancelled:
                _new_pose, deltas = self._advance_command(action, meas, raise_on_box=False)
                with self._cmd_lock:
                    pose = None if self._cmd_tcp is None else self._cmd_tcp.copy()
            if pose is not None:
                try:
                    self._publish_pose(pose)
                except Exception as exc:
                    self._fault_stop(f"servol_publish:{type(exc).__name__}:{exc}")
                    break
            # A tick is acknowledged only after its ServoL target was
            # successfully published. This prevents Env.step() from returning
            # and sampling next_obs before the fifth command has left the host.
            if window_active and not cancelled:
                with self._window_condition:
                    if self._window_active and self._window_id == window_id:
                        if deltas is None:
                            self._cancel_window_locked("safety")
                        else:
                            self._note_window_action(action)
                            self._window_executed_ticks += 1
                            self._window_remaining_ticks -= 1
                            self._window_delta_pos_xyz += np.asarray(
                                deltas["delta_pos_xyz"], dtype=np.float64
                            ).reshape(3)
                            self._window_delta_rot_rpy += np.asarray(
                                deltas["delta_rot_rpy"], dtype=np.float64
                            ).reshape(3)
                            if self._window_remaining_ticks <= 0:
                                self._latched_action = np.zeros(6, dtype=np.float32)
                                self._window_active = False
                                self._window_cancel_check = None
                                self._window_action_provider = None
                            self._window_condition.notify_all()
            remain = period - (time.monotonic() - t0)
            if remain > 0:
                stop.wait(remain)

    def _reset_window_locked(self) -> None:
        """Reset finite-window state. Caller must hold ``_cmd_lock``."""

        self._window_active = False
        self._window_requested_ticks = 0
        self._window_remaining_ticks = 0
        self._window_executed_ticks = 0
        self._window_started_t = None
        self._window_delta_pos_xyz[:] = 0.0
        self._window_delta_rot_rpy[:] = 0.0
        self._window_cancel_check = None
        self._window_action_provider = None
        self._window_action_sum[:] = 0.0
        self._window_action_samples = 0
        self._window_interrupted_by = "none"

    def _cancel_window_locked(self, reason: str) -> None:
        """Cancel an active window. Caller must hold ``_cmd_lock``."""

        self._latched_action = np.zeros(6, dtype=np.float32)
        self._window_active = False
        self._window_remaining_ticks = 0
        self._window_cancel_check = None
        self._window_action_provider = None
        self._window_interrupted_by = str(reason)
        self._window_condition.notify_all()

    def _publisher_safety_reason(self) -> Optional[str]:
        info = self.state_monitor.get_info()
        if info.get("is_singular"):
            return "singular"
        if is_firmware_protected(info.get("cmd_name"), info.get("cmd_num")):
            return (
                f"firmware_protected cmd_name={info.get('cmd_name')} "
                f"cmd_num={info.get('cmd_num')}"
            )
        try:
            meas = self.state_monitor.get_state()["tcp_pose"]
        except Exception:
            return None
        with self._cmd_lock:
            cmd = None if self._cmd_tcp is None else self._cmd_tcp.copy()
        if cmd is None:
            return None
        err_m = float(np.linalg.norm(meas[:3] - cmd[:3]))
        if err_m > TRACKING_ERR_LIMIT_M:
            return f"tracking_err_m={err_m:.4f}>{TRACKING_ERR_LIMIT_M}"
        return None

    def _precheck_or_fault(self) -> np.ndarray:
        stale = self.state_monitor.stale_fields()
        if stale:
            self._fault_stop(f"stale:{stale}")
            raise RuntimeError(f"stale state fields: {stale}")
        info = self.state_monitor.get_info()
        if info.get("is_singular"):
            self._fault_stop("singular")
            raise RuntimeError("singular during apply")
        if is_firmware_protected(info.get("cmd_name"), info.get("cmd_num")):
            reason = (
                f"firmware_protected cmd_name={info.get('cmd_name')} "
                f"cmd_num={info.get('cmd_num')}"
            )
            self._fault_stop(reason)
            raise RuntimeError(reason)
        meas = self.state_monitor.get_state()["tcp_pose"].copy()
        with self._cmd_lock:
            base = self._cmd_tcp if self._cmd_tcp is not None else meas
            base = np.asarray(base, dtype=np.float64).reshape(7).copy()
        err_before = float(np.linalg.norm(meas[:3] - base[:3]))
        # dry_run has no robot following the command; tracking is live-only.
        if not self.dry_run and err_before > TRACKING_ERR_LIMIT_M:
            reason = f"tracking_err_m={err_before:.4f}>{TRACKING_ERR_LIMIT_M}"
            self._fault_stop(reason)
            raise RuntimeError(reason)
        return meas

    def _latch_action(self, clipped: np.ndarray) -> Dict[str, Any]:
        with self._cmd_lock:
            self._latched_action = np.asarray(clipped, dtype=np.float32).reshape(6).copy()
            xyz = self._interval_delta_pos_xyz.copy()
            rpy = self._interval_delta_rot_rpy.copy()
            ticks = int(self._ticks_since_apply)
            cmd = None if self._cmd_tcp is None else self._cmd_tcp.copy()
            self._interval_delta_pos_xyz[:] = 0.0
            self._interval_delta_rot_rpy[:] = 0.0
            self._ticks_since_apply = 0
        return {
            "delta_pos_xyz": xyz,
            "delta_rot_rpy": rpy,
            "delta_pos_m": float(np.linalg.norm(xyz)),
            "delta_rot_rad": float(np.linalg.norm(rpy)),
            "interval_ticks": ticks,
            "cmd_tcp": cmd,
        }

    def _advance_command(
        self,
        action: Sequence[float],
        meas: np.ndarray,
        *,
        raise_on_box: bool,
    ) -> tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        if self._faulted or self._closed:
            return None, None
        with self._cmd_lock:
            base = self._cmd_tcp if self._cmd_tcp is not None else meas
            base = np.asarray(base, dtype=np.float64).reshape(7).copy()
        new_pose, deltas = integrate_normalized_action(base, action, self.contract)
        try:
            self._enforce_episode_box(new_pose)
        except RuntimeError as exc:
            if raise_on_box:
                raise
            with self._cmd_lock:
                self._latched_action = np.zeros(6, dtype=np.float32)
                self._box_exceeded = str(exc)
            return None, None
        err_m = float(np.linalg.norm(meas[:3] - new_pose[:3]))
        if not self.dry_run and err_m > TRACKING_ERR_LIMIT_M:
            reason = f"tracking_err_m={err_m:.4f}>{TRACKING_ERR_LIMIT_M}"
            self._fault_stop(reason)
            return None, None
        with self._cmd_lock:
            self._cmd_tcp = new_pose.copy()
            self._integrate_count += 1
            self._ticks_since_apply += 1
            self._interval_delta_pos_xyz += np.asarray(
                deltas["delta_pos_xyz"], dtype=np.float64
            ).reshape(3)
            self._interval_delta_rot_rpy += np.asarray(
                deltas["delta_rot_rpy"], dtype=np.float64
            ).reshape(3)
        return new_pose, deltas

    def _apply_result(
        self,
        *,
        deltas: Dict[str, Any],
        cmd_tcp: np.ndarray,
        meas: np.ndarray,
        clipped: np.ndarray,
        loop_dt: float,
        published: bool,
        interval_ticks: int,
    ) -> Dict[str, Any]:
        meas_r = Rotation.from_quat(meas[3:])
        cmd_r = Rotation.from_quat(cmd_tcp[3:])
        err_m = float(np.linalg.norm(meas[:3] - cmd_tcp[:3]))
        err_rad = float(np.linalg.norm((meas_r.inv() * cmd_r).as_rotvec()))
        return {
            "delta_pos_m": float(deltas["delta_pos_m"]),
            "delta_rot_rad": float(deltas["delta_rot_rad"]),
            "delta_pos_xyz": deltas["delta_pos_xyz"],
            "delta_rot_rpy": deltas["delta_rot_rpy"],
            "action_clipped": np.asarray(clipped, dtype=np.float32).reshape(6).copy(),
            "cmd_tcp": np.asarray(cmd_tcp, dtype=np.float32).reshape(7).copy(),
            "meas_tcp": meas,
            "tracking_err_m": err_m,
            "tracking_err_rad": err_rad,
            "loop_dt": float(loop_dt),
            "dry_run": self.dry_run,
            "published": bool(published),
            "publish_count": int(self._publish_count),
            "integrate_count": int(self._integrate_count),
            "interval_ticks": int(interval_ticks),
        }

    def _arm_group_left(self):
        try:
            from naviai_controller import ArmGroup

            return ArmGroup.LEFT
        except ImportError:
            return "LEFT"

    def _publish_pose(self, pose: np.ndarray) -> None:
        if self.dry_run or self._arm_ctrl is None or self._faulted or self._closed:
            return
        with self._pub_lock:
            if self._faulted or self._closed:
                return
            self._arm_ctrl.servol(list(map(float, pose)), self._arm_group_left())
            self._publish_count += 1
            self._last_publish_t = time.monotonic()

    def _enforce_episode_box(self, pose: np.ndarray) -> None:
        if self.episode_trans_limit_m is None and self.episode_rot_limit_rad is None:
            return
        assert self._origin_pose is not None and self._origin_rot is not None
        trans = float(np.linalg.norm(pose[:3] - self._origin_pose[:3]))
        if (
            self.episode_trans_limit_m is not None
            and trans > self.episode_trans_limit_m + 1e-9
        ):
            raise RuntimeError(
                f"episode translation {trans:.4f}m exceeds "
                f"{self.episode_trans_limit_m:.4f}m"
            )
        rel = self._origin_rot.inv() * Rotation.from_quat(pose[3:])
        rot = float(np.linalg.norm(rel.as_rotvec()))
        if (
            self.episode_rot_limit_rad is not None
            and rot > self.episode_rot_limit_rad + 1e-9
        ):
            raise RuntimeError(
                f"episode rotation {math.degrees(rot):.3f}deg exceeds "
                f"{math.degrees(self.episode_rot_limit_rad):.3f}deg"
            )
