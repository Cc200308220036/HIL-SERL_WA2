"""R5 reset/home executor: hand → waist → left arm → TCP check (scene-driven)."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from hilserl_wa2.envs.scene_config import WA2SceneConfig


@dataclass
class ResetResult:
    ok: bool
    attempts: int
    stages: List[str]
    error: Optional[str] = None
    measured: Optional[Dict[str, Any]] = None

    def as_info(self) -> Dict[str, Any]:
        return {
            "reset_ok": bool(self.ok),
            "reset_attempts": int(self.attempts),
            "reset_stages": list(self.stages),
            "reset_error": self.error,
            "reset_measured": self.measured,
        }


def joint_max_abs_err(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size != bb.size:
        raise ValueError("joint length mismatch")
    return float(np.max(np.abs(aa - bb)))


def tcp_errors(meas7: Sequence[float], target7: Sequence[float]) -> Dict[str, float]:
    m = np.asarray(meas7, dtype=np.float64).reshape(7)
    t = np.asarray(target7, dtype=np.float64).reshape(7)
    pos_err = float(np.linalg.norm(m[:3] - t[:3]))
    qm = m[3:] / max(float(np.linalg.norm(m[3:])), 1e-12)
    qt = t[3:] / max(float(np.linalg.norm(t[3:])), 1e-12)
    if float(np.dot(qm, qt)) < 0.0:
        qm = -qm
    # Relative rotation angle
    r = Rotation.from_quat(qt).inv() * Rotation.from_quat(qm)
    rot_err = float(np.linalg.norm(r.as_rotvec()))
    return {"tcp_pos_err_m": pos_err, "tcp_rot_err_rad": rot_err}


def check_tolerances(
    scene: WA2SceneConfig,
    *,
    joint_pos: Optional[Sequence[float]],
    waist_joints: Optional[Sequence[float]],
    hand_joints: Optional[Sequence[float]],
    tcp_pose: Optional[Sequence[float]],
    is_singular: Optional[bool],
    neck_joints: Optional[Sequence[float]] = None,
) -> Optional[str]:
    """Return error string if out of tolerance / singular; else None."""

    if is_singular:
        return "left_arm_is_singular=True"
    if joint_pos is not None:
        err = joint_max_abs_err(joint_pos, scene.home_joints_left)
        if err > scene.joint_tol_rad:
            return f"left joint err {err:.4f} > {scene.joint_tol_rad}"
    if waist_joints is not None and scene.waist_policy != "manual":
        err = joint_max_abs_err(waist_joints, scene.home_joints_waist)
        if err > scene.waist_tol_rad:
            return f"waist joint err {err:.4f} > {scene.waist_tol_rad}"
    if neck_joints is not None and scene.neck_policy != "manual":
        err = joint_max_abs_err(neck_joints, scene.home_joints_neck)
        if err > scene.neck_tol_rad:
            return f"neck joint err {err:.4f} > {scene.neck_tol_rad}"
    if hand_joints is not None:
        err = joint_max_abs_err(hand_joints, scene.hand_reset)
        if err > scene.hand_tol_rad:
            return f"hand joint err {err:.4f} > {scene.hand_tol_rad}"
    if tcp_pose is not None:
        e = tcp_errors(tcp_pose, scene.task_reset_tcp)
        if e["tcp_pos_err_m"] > scene.tcp_pos_tol_m:
            return (
                f"tcp pos err {e['tcp_pos_err_m']:.4f} m > {scene.tcp_pos_tol_m}"
            )
        if e["tcp_rot_err_rad"] > scene.tcp_rot_tol_rad:
            return (
                f"tcp rot err {math.degrees(e['tcp_rot_err_rad']):.2f}° > "
                f"{scene.tcp_rot_tol_deg}°"
            )
    return None


class WA2ResetExecutor:
    """Execute scene reset on real robot via NaviController (+ optional StateMonitor)."""

    def __init__(
        self,
        scene: WA2SceneConfig,
        state_monitor: Any = None,
        controller: Any = None,
        dry_run: bool = False,
        require_confirm_env: str = "R5_CONFIRM",
        scene_ok_env: str = "RESET_SCENE_OK",
        confirm_fn: Optional[Callable[[], bool]] = None,
    ):
        self.scene = scene
        self.state_monitor = state_monitor
        self._controller = controller
        self.dry_run = bool(dry_run)
        self.require_confirm_env = require_confirm_env
        self.scene_ok_env = scene_ok_env
        self.confirm_fn = confirm_fn

    def run(self) -> ResetResult:
        attempts = 0
        last_err: Optional[str] = None
        stages: List[str] = []
        max_attempts = 1 + max(0, int(self.scene.max_retries))
        t0 = time.monotonic()

        while attempts < max_attempts:
            attempts += 1
            stages = []
            try:
                if time.monotonic() - t0 > self.scene.total_timeout_s:
                    raise TimeoutError("reset total timeout before attempt")
                measured = self._run_once(stages, deadline=t0 + self.scene.total_timeout_s)
                return ResetResult(
                    ok=True, attempts=attempts, stages=stages, measured=measured
                )
            except Exception as exc:  # noqa: BLE001 — convert to ResetResult
                last_err = str(exc)
                stages.append(f"fail:{last_err}")
                self._safe_stop_clear()
                if attempts >= max_attempts:
                    break

        return ResetResult(
            ok=False, attempts=attempts, stages=stages, error=last_err or "reset failed"
        )

    def _run_once(self, stages: List[str], deadline: float) -> Dict[str, Any]:
        self._check_motion_confirm()
        self._ensure_controller()

        # 1) stop+clear if leaving ServoL
        stages.append("stop_clear")
        self._safe_stop_clear()

        # 2) hand open
        stages.append("hand_open")
        self._deadline(deadline, "hand")
        self._open_hand()
        self._wait_hand(timeout_s=self.scene.hand_timeout_s)

        # 3) waist
        if self.scene.waist_policy == "auto_movej":
            stages.append("waist_movej")
            self._deadline(deadline, "waist")
            self._movej_waist()
            self._wait_joints(
                get_fn=self._get_waist,
                target=self.scene.home_joints_waist,
                tol=self.scene.waist_tol_rad,
                timeout_s=self.scene.waist_timeout_s,
                label="waist",
            )
        elif self.scene.waist_policy == "check_only":
            stages.append("waist_check")
            w = self._get_waist()
            err = check_tolerances(
                self.scene,
                joint_pos=None,
                waist_joints=w,
                hand_joints=None,
                tcp_pose=None,
                is_singular=None,
            )
            if err:
                raise RuntimeError(err)
        else:
            stages.append("waist_manual_skip")

        # 3b) neck
        if self.scene.neck_policy == "auto_movej":
            stages.append("neck_movej")
            self._deadline(deadline, "neck")
            self._movej_neck()
            self._wait_joints(
                get_fn=self._get_neck,
                target=self.scene.home_joints_neck,
                tol=self.scene.neck_tol_rad,
                timeout_s=self.scene.neck_timeout_s,
                label="neck",
            )
        elif self.scene.neck_policy == "check_only":
            stages.append("neck_check")
            n = self._get_neck()
            err = check_tolerances(
                self.scene,
                joint_pos=None,
                waist_joints=None,
                neck_joints=n,
                hand_joints=None,
                tcp_pose=None,
                is_singular=None,
            )
            if err:
                raise RuntimeError(err)
        else:
            stages.append("neck_manual_skip")

        # 4) left arm MoveJ
        stages.append("arm_movej")
        self._deadline(deadline, "arm")
        self._movej_left()
        self._wait_joints(
            get_fn=self._get_left,
            target=self.scene.home_joints_left,
            tol=self.scene.joint_tol_rad,
            timeout_s=self.scene.arm_timeout_s,
            label="left_arm",
        )

        # 5) optional MoveL
        if self.scene.do_movel_to_tcp:
            stages.append("arm_movel")
            self._deadline(deadline, "movel")
            self._movel_tcp()

        # 6) measure + tolerance
        stages.append("tolerance_check")
        measured = self._measure()
        err = check_tolerances(
            self.scene,
            joint_pos=measured.get("joint_pos"),
            waist_joints=measured.get("waist_joints"),
            neck_joints=measured.get("neck_joints"),
            hand_joints=measured.get("hand_joints"),
            tcp_pose=measured.get("tcp_pose"),
            is_singular=measured.get("is_singular"),
        )
        if err:
            raise RuntimeError(err)

        # 7) human scene confirm
        stages.append("scene_confirm")
        if self.scene.require_human_confirm and not self._scene_confirmed():
            raise RuntimeError(
                f"scene not confirmed; set {self.scene_ok_env}=YES or confirm interactively"
            )

        stages.append("done")
        return measured

    def _check_motion_confirm(self) -> None:
        if self.dry_run:
            return
        if os.environ.get(self.require_confirm_env) == "YES":
            return
        # Allow R4_CONFIRM as alias for convenience during gates.
        if os.environ.get("R4_CONFIRM") == "YES":
            return
        raise RuntimeError(
            f"refusing real reset motion without {self.require_confirm_env}=YES "
            "(or R4_CONFIRM=YES)"
        )

    def _scene_confirmed(self) -> bool:
        if os.environ.get(self.scene_ok_env) == "YES":
            return True
        if self.confirm_fn is not None:
            return bool(self.confirm_fn())
        # Non-interactive default: require env var.
        return False

    def _ensure_controller(self) -> None:
        if self.dry_run:
            return
        if self._controller is not None:
            return
        import rospy
        from naviai_controller import NaviController

        if not rospy.core.is_initialized():
            rospy.init_node("wa2_reset_executor", anonymous=True, disable_signals=True)
        self._controller = NaviController(model="wa2")
        time.sleep(0.5)

    def _safe_stop_clear(self) -> None:
        if self.dry_run or self._controller is None:
            return
        # Safety lock rejects stop/MoveJ. Unlock first, then stop+clear.
        try:
            self._controller.unlock()
        except Exception:
            pass
        try:
            self._controller.stop()
        except Exception:
            pass
        try:
            self._controller.clear_servo_params()
        except Exception:
            pass

    def _open_hand(self) -> None:
        if self.dry_run:
            return
        from naviai_controller import HandType

        target = list(map(float, self.scene.hand_reset))
        if all(abs(value) <= 1e-12 for value in target):
            ok = self._controller.release_hand(HandType.LEFT)
            if not ok:
                ok = self._controller.grasp_hand(HandType.LEFT, target)
        else:
            ok = self._controller.grasp_hand(HandType.LEFT, target)
        if not ok:
            raise RuntimeError("hand reset command failed")

    def _movej_waist(self) -> None:
        if self.dry_run:
            return
        from naviai_controller import ArmGroup

        ok = self._controller.movej(
            list(map(float, self.scene.home_joints_waist)),
            ArmGroup.WAIST,
            v=self.scene.movej_v,
            acc=self.scene.movej_acc,
        )
        if not ok:
            raise RuntimeError("waist movej returned not ok")

    def _movej_neck(self) -> None:
        if self.dry_run:
            return
        from naviai_controller import ArmGroup

        ok = self._controller.movej(
            list(map(float, self.scene.home_joints_neck)),
            ArmGroup.NECK,
            v=self.scene.movej_v,
            acc=self.scene.movej_acc,
        )
        if not ok:
            raise RuntimeError("neck movej returned not ok")

    def _movej_left(self) -> None:
        if self.dry_run:
            return
        from naviai_controller import ArmGroup

        ok = self._controller.movej(
            list(map(float, self.scene.home_joints_left)),
            ArmGroup.LEFT,
            v=self.scene.movej_v,
            acc=self.scene.movej_acc,
        )
        if not ok:
            raise RuntimeError("left arm movej returned not ok")

    def _movel_tcp(self) -> None:
        if self.dry_run:
            return
        from naviai_controller import ArmGroup

        ok = self._controller.movel(
            list(map(float, self.scene.task_reset_tcp)),
            ArmGroup.LEFT,
            v=self.scene.movej_v,
            acc=self.scene.movej_acc,
        )
        if not ok:
            raise RuntimeError("left arm movel returned not ok")

    def _get_left(self) -> np.ndarray:
        if self.state_monitor is not None:
            return np.asarray(
                self.state_monitor.get_state()["joint_pos"], dtype=np.float64
            )
        from naviai_controller import ArmGroup

        j = self._controller.get_joints(ArmGroup.LEFT)
        if j is None:
            raise RuntimeError("left joints unavailable")
        return np.asarray(j, dtype=np.float64)

    def _get_waist(self) -> np.ndarray:
        from naviai_controller import ArmGroup

        if self.dry_run:
            return self.scene.home_joints_waist.copy()
        j = self._controller.get_joints(ArmGroup.WAIST)
        if j is None:
            raise RuntimeError("waist joints unavailable")
        return np.asarray(j, dtype=np.float64)

    def _get_neck(self) -> np.ndarray:
        from naviai_controller import ArmGroup

        if self.dry_run:
            return self.scene.home_joints_neck.copy()
        j = self._controller.get_joints(ArmGroup.NECK)
        if j is None:
            raise RuntimeError("neck joints unavailable")
        return np.asarray(j, dtype=np.float64)

    def _get_hand(self) -> np.ndarray:
        if self.state_monitor is not None:
            return np.asarray(
                self.state_monitor.get_state()["hand_joints"], dtype=np.float64
            )
        from naviai_controller import HandType

        j = self._controller.get_hand_joints(HandType.LEFT)
        if j is None:
            raise RuntimeError("hand joints unavailable")
        return np.asarray(j, dtype=np.float64)

    def _get_tcp(self) -> np.ndarray:
        if self.state_monitor is not None:
            return np.asarray(
                self.state_monitor.get_state()["tcp_pose"], dtype=np.float64
            )
        from naviai_controller import ArmGroup

        # Prefer list pose if available
        if hasattr(self._controller, "get_tcp_rt"):
            tcp = self._controller.get_tcp_rt(ArmGroup.LEFT)
            if tcp is not None:
                return np.asarray(tcp, dtype=np.float64).reshape(7)
        raise RuntimeError("tcp pose unavailable")

    def _wait_hand(self, timeout_s: float) -> None:
        if self.dry_run:
            return
        deadline = time.monotonic() + timeout_s
        last = None
        while time.monotonic() < deadline:
            try:
                last = self._get_hand()
                if joint_max_abs_err(last, self.scene.hand_reset) <= self.scene.hand_tol_rad:
                    return
            except Exception:
                pass
            time.sleep(0.1)
        raise TimeoutError(
            f"hand open timeout; last_err="
            f"{None if last is None else joint_max_abs_err(last, self.scene.hand_reset)}"
        )

    def _wait_joints(
        self,
        *,
        get_fn: Callable[[], np.ndarray],
        target: np.ndarray,
        tol: float,
        timeout_s: float,
        label: str,
    ) -> None:
        if self.dry_run:
            return
        deadline = time.monotonic() + timeout_s
        last_err = None
        while time.monotonic() < deadline:
            cur = get_fn()
            last_err = joint_max_abs_err(cur, target)
            if last_err <= tol:
                return
            time.sleep(0.1)
        raise TimeoutError(f"{label} settle timeout; err={last_err}")

    def _measure(self) -> Dict[str, Any]:
        if self.dry_run:
            return {
                "joint_pos": self.scene.home_joints_left.copy(),
                "neck_joints": self.scene.home_joints_neck.copy(),
                "waist_joints": self.scene.home_joints_waist.copy(),
                "hand_joints": self.scene.hand_reset.copy(),
                "tcp_pose": self.scene.task_reset_tcp.copy(),
                "is_singular": False,
            }
        if self.state_monitor is not None and not getattr(
            self.state_monitor, "_started", False
        ):
            self.state_monitor.start()
            self.state_monitor.wait_ready(timeout_s=5.0)

        joint_pos = self._get_left()
        neck = self._get_neck()
        waist = self._get_waist()
        hand = self._get_hand()
        tcp = self._get_tcp()
        is_singular = False
        if self.state_monitor is not None:
            is_singular = bool(self.state_monitor.get_info().get("is_singular"))
        out = {
            "joint_pos": joint_pos,
            "neck_joints": neck,
            "waist_joints": waist,
            "hand_joints": hand,
            "tcp_pose": tcp,
            "is_singular": is_singular,
        }
        out.update(tcp_errors(tcp, self.scene.task_reset_tcp))
        out["left_joint_err"] = joint_max_abs_err(joint_pos, self.scene.home_joints_left)
        out["neck_joint_err"] = joint_max_abs_err(neck, self.scene.home_joints_neck)
        out["waist_joint_err"] = joint_max_abs_err(waist, self.scene.home_joints_waist)
        out["hand_joint_err"] = joint_max_abs_err(hand, self.scene.hand_reset)
        return out

    def _deadline(self, deadline: float, label: str) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError(f"reset total timeout at stage {label}")
