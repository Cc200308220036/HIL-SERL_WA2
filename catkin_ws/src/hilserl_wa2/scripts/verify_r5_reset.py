#!/usr/bin/env python3
"""R5 Gate: scene load + optional real reset (requires R5_CONFIRM=YES)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]  # .../catkin_ws/src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.envs.scene_config import load_scene  # noqa: E402
from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.ros_adapters.reset_executor import (  # noqa: E402
    WA2ResetExecutor,
    check_tolerances,
)


def _banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


def verify_offline(scene_name: str) -> None:
    _banner(f"offline scene={scene_name}")
    scene = load_scene(scene_name=scene_name)
    assert scene is not None
    print(f"scene_id={scene.scene_id} max_steps={scene.max_steps}")
    print(f"waist_policy={scene.waist_policy} neck_policy={scene.neck_policy} workspace={scene.workspace_policy}")
    env = WA2Env(fake_env=True, scene_name=scene_name, seed=0)
    assert env.max_steps == scene.max_steps
    obs, info = env.reset()
    assert info.get("reset_ok") is True
    assert abs(float(obs["state"]["tcp_pose"][0]) - float(scene.task_reset_tcp[0])) < 1e-5
    env.close()
    ex = WA2ResetExecutor(scene=scene, dry_run=True, confirm_fn=lambda: True)
    result = ex.run()
    assert result.ok, result
    print("OFFLINE: PASS")


def verify_real(scene_name: str, n: int, interactive: bool) -> None:
    if os.environ.get("R5_CONFIRM") != "YES" and os.environ.get("R4_CONFIRM") != "YES":
        raise SystemExit("Set R5_CONFIRM=YES (or R4_CONFIRM=YES) for real reset Gate")

    import rospy

    if not rospy.core.is_initialized():
        rospy.init_node("verify_r5_reset", anonymous=True, disable_signals=True)

    scene = load_scene(scene_name=scene_name)
    assert scene is not None

    def confirm() -> bool:
        if os.environ.get("RESET_SCENE_OK") == "YES":
            return True
        if not interactive:
            return False
        ans = input(
            "场景确认：瓶在头相机可见桌面、无杂物、急停可达？ [y/N] "
        ).strip().lower()
        return ans in ("y", "yes")

    fails = 0
    for i in range(1, n + 1):
        _banner(f"real reset {i}/{n}")
        env = WA2Env(
            fake_env=False,
            read_only=False,
            dry_run=False,
            scene_name=scene_name,
            auto_reset_motion=True,
        )
        # ServoL after reset also needs R4_CONFIRM
        os.environ.setdefault("R4_CONFIRM", os.environ.get("R5_CONFIRM", "YES"))
        try:
            obs, info = env.reset(options={"confirm_fn": confirm})
            print("info:", {k: info.get(k) for k in (
                "reset_ok", "reset_attempts", "reset_stages", "scene_id", "is_singular"
            )})
            if not info.get("reset_ok"):
                fails += 1
                print("FAIL: reset_ok false")
            else:
                err = check_tolerances(
                    scene,
                    joint_pos=obs["state"]["joint_pos"],
                    waist_joints=None,  # measured in reset_measured if present
                    hand_joints=obs["state"]["hand_joints"],
                    tcp_pose=obs["state"]["tcp_pose"],
                    is_singular=info.get("is_singular"),
                )
                measured = info.get("reset_measured") or {}
                if measured.get("waist_joints") is not None:
                    err = check_tolerances(
                        scene,
                        joint_pos=obs["state"]["joint_pos"],
                        waist_joints=measured["waist_joints"],
                        hand_joints=obs["state"]["hand_joints"],
                        tcp_pose=obs["state"]["tcp_pose"],
                        is_singular=info.get("is_singular"),
                    )
                if err:
                    fails += 1
                    print("FAIL tolerance:", err)
                else:
                    print("PASS")
        except Exception as exc:  # noqa: BLE001
            fails += 1
            print("FAIL exception:", exc)
        finally:
            env.close()
            time.sleep(0.5)

    if fails:
        raise SystemExit(f"R5 GATE: FAIL ({fails}/{n})")
    print(f"R5 GATE: PASS ({n}/{n})")


def main() -> None:
    parser = argparse.ArgumentParser(description="R5 reset/home Gate")
    parser.add_argument("--scene", default="bottle_desktop")
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--real", action="store_true", help="run real reset Gate")
    parser.add_argument("--n", type=int, default=10, help="real reset count")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="prompt for scene confirm each reset (else need RESET_SCENE_OK=YES)",
    )
    args = parser.parse_args()

    verify_offline(args.scene)
    if args.offline_only:
        return
    if args.real:
        verify_real(args.scene, n=args.n, interactive=args.interactive)
    else:
        print("Offline done. For real Gate add: --real  (and R5_CONFIRM=YES)")


if __name__ == "__main__":
    main()
