"""Unit tests for WA2ServoSession (offline / dry-run)."""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.envs.contracts import WA2EnvContract  # noqa: E402
from hilserl_wa2.ros_adapters.servo_session import (  # noqa: E402
    TRACKING_ERR_LIMIT_M,
    WA2ServoSession,
    integrate_normalized_action,
    is_firmware_protected,
)
from hilserl_wa2.ros_adapters.state_monitor import StateCache, WA2StateMonitor  # noqa: E402

CONTRACT = (
    pathlib.Path(__file__).resolve().parents[2] / "configs" / "wa2_env_contract.yaml"
)


def _fake_arm():
    class FakeArm:
        def __init__(self):
            self.n = 0
            self.lock = __import__("threading").Lock()

        def set_servo_params(self, dt, lookahead, group):
            return True

        def servol(self, pose, group):
            with self.lock:
                self.n += 1

        def unlock(self):
            return True

        def stop(self):
            return True

        def clear_servo_params(self):
            return True

    return FakeArm()


def _refresh(mon: WA2StateMonitor, pose=None, cmd_name: str = "SERVOL") -> None:
    pose = pose if pose is not None else [0.3, 0.1, 0.6, 0, 0, 0, 1]
    cache = mon.cache
    cache.update_tcp_pose(pose)
    cache.update_tcp_vel(np.zeros(6))
    cache.update_joint_pos(np.zeros(8))
    cache.update_hand_joints(np.zeros(6))
    cache.update_uplimb_state(
        is_singular=False, cmd_num=14, cmd_name=cmd_name, iddp_status=True
    )


def _monitor() -> WA2StateMonitor:
    cache = StateCache(state_max_age_s=0.2)
    cache.update_tcp_pose([0.3, 0.1, 0.6, 0, 0, 0, 1])
    cache.update_tcp_vel(np.zeros(6))
    cache.update_joint_pos(np.zeros(8))
    cache.update_hand_joints(np.zeros(6))
    cache.update_uplimb_state(
        is_singular=False, cmd_num=0, cmd_name="STOPPED", iddp_status=True
    )
    mon = WA2StateMonitor(arm="left", cache=cache)
    mon._started = True
    return mon


class IntegrateActionTests(unittest.TestCase):
    def test_clip_and_limits(self):
        contract = WA2EnvContract.from_yaml(CONTRACT)
        pose = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        new_pose, info = integrate_normalized_action(pose, [1.5, 0, 0, 0, 0, 0], contract)
        self.assertAlmostEqual(info["delta_pos_m"], contract.max_pos_delta_m, places=6)
        self.assertAlmostEqual(float(new_pose[0]), contract.max_pos_delta_m, places=6)
        _, info_r = integrate_normalized_action(pose, [0, 0, 0, 1, 0, 0], contract)
        self.assertLessEqual(
            info_r["delta_rot_rad"], contract.max_rot_delta_rad + 1e-9
        )


class ServoSessionDryRunTests(unittest.TestCase):
    def test_dry_run_apply_and_box(self):
        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        session = WA2ServoSession(
            contract=contract,
            state_monitor=mon,
            dry_run=True,
            episode_trans_limit_m=0.03,
            episode_rot_limit_deg=5.0,
        )
        session.start()
        out = session.apply_normalized_action([1, 0, 0, 0, 0, 0])
        self.assertTrue(out["dry_run"])
        self.assertFalse(out["published"])
        self.assertAlmostEqual(out["delta_pos_m"], contract.max_pos_delta_m, places=6)
        session.close()
        session.close()

    def test_dry_run_five_tick_action_window(self):
        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        session = WA2ServoSession(contract=contract, state_monitor=mon, dry_run=True)
        session.start()
        out = session.execute_action_window(
            [1, 0, 0, 0, 0, 0], ticks=contract.servo_ticks_per_action
        )
        self.assertEqual(out["servo_ticks_requested"], 5)
        self.assertEqual(out["servo_ticks_executed"], 5)
        self.assertEqual(out["interrupted_by"], "none")
        self.assertAlmostEqual(out["delta_pos_m"], 0.005, places=6)
        np.testing.assert_allclose(session.health()["latched_action"], 0.0, atol=1e-7)
        session.close()

    def test_dry_run_window_interrupts_without_extra_tick(self):
        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        session = WA2ServoSession(contract=contract, state_monitor=mon, dry_run=True)
        session.start()
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] >= 3

        out = session.execute_action_window(
            [1, 0, 0, 0, 0, 0], ticks=5, cancel_check=cancel
        )
        self.assertEqual(out["servo_ticks_executed"], 2)
        self.assertEqual(out["interrupted_by"], "intervention")
        self.assertAlmostEqual(out["delta_pos_m"], 0.002, places=6)
        session.close()

    def test_dry_run_action_provider_resamples_each_tick(self):
        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        session = WA2ServoSession(contract=contract, state_monitor=mon, dry_run=True)
        session.start()
        seq = [
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        idx = {"i": 0}

        def provider():
            a = seq[idx["i"]]
            idx["i"] += 1
            return a

        out = session.execute_action_window(
            [0, 0, 0, 0, 0, 0], ticks=5, action_provider=provider
        )
        self.assertEqual(out["servo_ticks_executed"], 5)
        self.assertEqual(out["window_action_samples"], 5)
        # Net +1+1-1-1+0 mm = 0; mean action x = 0.
        self.assertAlmostEqual(out["delta_pos_m"], 0.0, places=6)
        np.testing.assert_allclose(out["window_action_mean"][0], 0.0, atol=1e-6)
        session.close()

    def test_stale_rejects(self):
        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        session = WA2ServoSession(contract=contract, state_monitor=mon, dry_run=True)
        session.start()
        mon.inject_stale_for_test(["tcp_pose"], age_s=1.0)
        with self.assertRaises(RuntimeError):
            session.apply_normalized_action(np.zeros(6))
        self.assertTrue(session.faulted)
        session.close()

    def test_episode_box(self):
        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        session = WA2ServoSession(
            contract=contract,
            state_monitor=mon,
            dry_run=True,
            episode_trans_limit_m=0.0015,
            episode_rot_limit_deg=5.0,
        )
        session.start()
        session.apply_normalized_action([1, 0, 0, 0, 0, 0])  # 1mm OK
        with self.assertRaises(RuntimeError):
            session.apply_normalized_action([1, 0, 0, 0, 0, 0])  # would be 2mm > 1.5mm
        session.close()

    def test_episode_box_disabled(self):
        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        session = WA2ServoSession(
            contract=contract,
            state_monitor=mon,
            dry_run=True,
            episode_trans_limit_m=None,
            episode_rot_limit_deg=None,
        )
        session.start()
        for _ in range(40):
            session.apply_normalized_action([1, 0, 0, 0, 0, 0])
        self.assertFalse(session.faulted)
        session.close()


class FirmwareGuardTests(unittest.TestCase):
    def test_protected_name_and_num(self):
        self.assertTrue(is_firmware_protected("PROTECTED", 14))
        self.assertTrue(is_firmware_protected("servol", 15))
        self.assertFalse(is_firmware_protected("SERVOL", 14))
        self.assertFalse(is_firmware_protected("STOPPED", 0))

    def test_protected_rejects_apply(self):
        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        session = WA2ServoSession(contract=contract, state_monitor=mon, dry_run=True)
        session.start()
        mon.cache.update_uplimb_state(
            is_singular=False, cmd_num=15, cmd_name="PROTECTED", iddp_status=True
        )
        with self.assertRaises(RuntimeError):
            session.apply_normalized_action(np.zeros(6))
        self.assertTrue(session.faulted)
        session.close()

    def test_tracking_error_rejects_apply(self):
        import os

        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        prev = os.environ.get("R4_CONFIRM")
        os.environ["R4_CONFIRM"] = "YES"
        session = WA2ServoSession(
            contract=contract,
            state_monitor=mon,
            dry_run=False,
            arm_ctrl=_fake_arm(),
        )
        try:
            _refresh(mon, cmd_name="STOPPED")
            session.start()
            # Measured TCP 2 cm away from commanded pose; do not keep integrating.
            _refresh(
                mon,
                pose=[0.3, 0.1, 0.6 - (TRACKING_ERR_LIMIT_M + 0.005), 0, 0, 0, 1],
                cmd_name="SERVOL",
            )
            with self.assertRaises(RuntimeError):
                session.apply_normalized_action(np.zeros(6))
            self.assertTrue(session.faulted)
        finally:
            session.close()
            if prev is None:
                os.environ.pop("R4_CONFIRM", None)
            else:
                os.environ["R4_CONFIRM"] = prev

    def test_dry_run_does_not_start_publisher(self):
        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        session = WA2ServoSession(contract=contract, state_monitor=mon, dry_run=True)
        session.start()
        self.assertIsNone(session._pub_thread)
        self.assertEqual(session.publish_count, 0)
        session.close()


class HoldLastPublisherTests(unittest.TestCase):
    def test_live_window_is_exactly_five_ticks_and_then_holds(self):
        import os
        import time

        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        # No ROS producer refreshes this synthetic cache while the publisher
        # thread runs; use a generous age only for this finite-window test.
        mon.cache.state_max_age_s = 20.0
        arm = _fake_arm()
        prev = os.environ.get("R4_CONFIRM")
        os.environ["R4_CONFIRM"] = "YES"
        session = WA2ServoSession(
            contract=contract, state_monitor=mon, dry_run=False, arm_ctrl=arm
        )
        try:
            _refresh(mon, cmd_name="STOPPED")
            session.start()
            _refresh(mon, cmd_name="SERVOL")
            out = session.execute_action_window([1, 0, 0, 0, 0, 0], ticks=5)
            self.assertEqual(out["servo_ticks_executed"], 5)
            self.assertAlmostEqual(out["delta_pos_m"], 0.005, places=6)
            x_done = float(session.health()["cmd_tcp"][0])
            time.sleep(0.07)
            self.assertAlmostEqual(float(session.health()["cmd_tcp"][0]), x_done, places=7)
            self.assertFalse(session.health()["window_active"])
            np.testing.assert_allclose(session.health()["latched_action"], 0.0, atol=1e-7)
        finally:
            session.close()
            if prev is None:
                os.environ.pop("R4_CONFIRM", None)
            else:
                os.environ["R4_CONFIRM"] = prev

    def test_hold_last_outpaces_slow_apply(self):
        import os
        import time

        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        arm = _fake_arm()
        prev = os.environ.get("R4_CONFIRM")
        os.environ["R4_CONFIRM"] = "YES"
        session = WA2ServoSession(
            contract=contract,
            state_monitor=mon,
            dry_run=False,
            arm_ctrl=arm,
        )
        try:
            _refresh(mon, cmd_name="STOPPED")
            session.start()
            self.assertTrue(session.health()["publisher_alive"])
            n_apply = 3
            for _ in range(n_apply):
                _refresh(mon, cmd_name="SERVOL")
                session.apply_normalized_action(np.zeros(6))
                time.sleep(0.05)
            self.assertGreater(session.publish_count, n_apply)
            self.assertGreaterEqual(arm.n, session.publish_count)
        finally:
            session.close()
            if prev is None:
                os.environ.pop("R4_CONFIRM", None)
            else:
                os.environ["R4_CONFIRM"] = prev
            self.assertFalse(session.health()["publisher_alive"])

    def test_latched_action_integrates_at_control_dt(self):
        import os
        import time

        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        arm = _fake_arm()
        prev = os.environ.get("R4_CONFIRM")
        os.environ["R4_CONFIRM"] = "YES"
        session = WA2ServoSession(
            contract=contract,
            state_monitor=mon,
            dry_run=False,
            arm_ctrl=arm,
            latch_max_age_s=0.08,
        )
        origin = [0.3, 0.1, 0.6, 0, 0, 0, 1]
        origin_x = origin[0]
        try:
            _refresh(mon, pose=origin, cmd_name="STOPPED")
            session.start()
            _refresh(mon, pose=origin, cmd_name="SERVOL")
            t0 = time.monotonic()
            while time.monotonic() - t0 < 0.20:
                # Env-like refresh: latch expires without periodic apply.
                session.apply_normalized_action([1.0, 0, 0, 0, 0, 0])
                with session._cmd_lock:
                    cmd = None if session._cmd_tcp is None else session._cmd_tcp.copy()
                if cmd is not None:
                    _refresh(mon, pose=cmd, cmd_name="SERVOL")
                time.sleep(0.01)
            elapsed = time.monotonic() - t0
            with session._cmd_lock:
                cmd = session._cmd_tcp.copy()
            moved = float(cmd[0] - origin_x)
            expected = elapsed * (contract.max_pos_delta_m / contract.control_dt)
            self.assertGreater(session.integrate_count, 6)
            self.assertGreater(session.publish_count, 6)
            self.assertAlmostEqual(moved, expected, delta=0.006)
            rate = moved / elapsed
            self.assertGreater(rate, 0.030)
            self.assertLess(rate, 0.070)
            _refresh(mon, pose=cmd, cmd_name="SERVOL")
            session.apply_normalized_action(np.zeros(6))
            x_hold = float(session._cmd_tcp[0])
            hold_t0 = time.monotonic()
            while time.monotonic() - hold_t0 < 0.06:
                with session._cmd_lock:
                    hold_pose = session._cmd_tcp.copy()
                _refresh(mon, pose=hold_pose, cmd_name="SERVOL")
                time.sleep(0.01)
            self.assertLess(abs(float(session._cmd_tcp[0]) - x_hold), 0.0025)
        finally:
            session.close()
            if prev is None:
                os.environ.pop("R4_CONFIRM", None)
            else:
                os.environ["R4_CONFIRM"] = prev

    def test_latch_expires_without_fresh_apply(self):
        import os
        import time

        contract = WA2EnvContract.from_yaml(CONTRACT)
        mon = _monitor()
        arm = _fake_arm()
        prev = os.environ.get("R4_CONFIRM")
        os.environ["R4_CONFIRM"] = "YES"
        session = WA2ServoSession(
            contract=contract,
            state_monitor=mon,
            dry_run=False,
            arm_ctrl=arm,
            latch_max_age_s=0.05,
        )
        origin = [0.3, 0.1, 0.6, 0, 0, 0, 1]
        try:
            _refresh(mon, pose=origin, cmd_name="STOPPED")
            session.start()
            _refresh(mon, pose=origin, cmd_name="SERVOL")
            session.apply_normalized_action([1.0, 0, 0, 0, 0, 0])
            t0 = time.monotonic()
            while time.monotonic() - t0 < 0.04:
                with session._cmd_lock:
                    cmd = None if session._cmd_tcp is None else session._cmd_tcp.copy()
                if cmd is not None:
                    _refresh(mon, pose=cmd, cmd_name="SERVOL")
                time.sleep(0.005)
            with session._cmd_lock:
                x_mid = float(session._cmd_tcp[0])
            # Stall the env loop: latch must zero and stop free-flight.
            stall_t0 = time.monotonic()
            while time.monotonic() - stall_t0 < 0.12:
                with session._cmd_lock:
                    hold_pose = session._cmd_tcp.copy()
                _refresh(mon, pose=hold_pose, cmd_name="SERVOL")
                time.sleep(0.01)
            self.assertGreater(session._latch_expire_count, 0)
            self.assertLess(abs(float(session._cmd_tcp[0]) - x_mid), 0.004)
            np.testing.assert_allclose(session._latched_action, 0.0, atol=1e-6)
        finally:
            session.close()
            if prev is None:
                os.environ.pop("R4_CONFIRM", None)
            else:
                os.environ["R4_CONFIRM"] = prev


if __name__ == "__main__":
    unittest.main()
