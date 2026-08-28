#!/usr/bin/env python3
"""R7 live Gate: SpaceMouse → intervene_action → WA2Env (optional motion).

Modes:
  readonly  — read_only Env, no ServoL; verify intervene flags with real Joy
  nudge     — small motion via Env (requires R4_CONFIRM=YES); teleop must be off
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT.parent))

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.interventions.wa2_spacemouse_intervention import (  # noqa: E402
    WA2SpacemouseIntervention,
)


def _teleop_running() -> bool:
    try:
        out = subprocess.check_output(["pgrep", "-af", "spacemouse_wa2_teleop"], text=True)
        return "spacemouse_wa2_teleop" in out
    except subprocess.CalledProcessError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("readonly", "nudge"),
        default="readonly",
    )
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--scene", default="bottle_desktop")
    parser.add_argument(
        "--config",
        default=None,
        help="SpaceMouse YAML path or stem (default: configs/spacemouse/default.yaml)",
    )
    parser.add_argument(
        "--with-reset",
        action="store_true",
        help="nudge mode: run R5 scene reset before intervention (default: skip)",
    )
    args = parser.parse_args()

    if _teleop_running():
        raise SystemExit(
            "R7_LIVE: FAIL — spacemouse_wa2_teleop is running; stop it first"
        )

    read_only = args.mode == "readonly"
    if not read_only:
        if os.environ.get("R4_CONFIRM") != "YES":
            raise SystemExit("R7_LIVE nudge mode requires R4_CONFIRM=YES")

    do_reset = (not read_only) and bool(args.with_reset)
    base = WA2Env(
        fake_env=False,
        read_only=read_only,
        scene_name=args.scene if do_reset else None,
        auto_reset_motion=do_reset,
    )
    env = WA2SpacemouseIntervention(
        base, config_path=args.config, auto_start_ros=True
    )
    if read_only or not do_reset:
        opts = {"skip_reset_motion": True}
    else:
        opts = {"confirm_fn": lambda: True}
        os.environ.setdefault("RESET_SCENE_OK", "YES")
        os.environ.setdefault("R5_CONFIRM", "YES")

    print("Waiting for /spacenav/joy ... (start spacenavd + spacenav_node)")
    env.joy.start_ros()
    env.joy.wait_ready(timeout_s=10.0)
    obs, info = env.reset(options=opts)
    print(
        "reset ok; TAP left (buttons[1]) to enter, then MOVE SpaceMouse "
        f"for >=5 intervened steps within {args.seconds:.0f}s"
    )

    intervened_seen = 0
    passthrough_seen = 0
    t0 = time.time()
    last_status = 0.0
    while time.time() - t0 < args.seconds:
        obs, r, term, trunc, info = env.step(np.zeros(6, np.float32))
        if info.get("intervened"):
            intervened_seen += 1
            if intervened_seen == 1:
                print("first intervene_action", info.get("intervene_action"))
        else:
            passthrough_seen += 1
        now = time.time()
        if now - last_status >= 1.0:
            sample = env.joy.get_sample()
            age = env.joy.get_age()
            btns = list(sample.buttons) if sample is not None else []
            axes = (
                [round(float(x), 3) for x in sample.axes[:6]]
                if sample is not None
                else []
            )
            print(
                f"t={now - t0:5.1f}s intervened={intervened_seen} "
                f"joy_age={age:.3f}s buttons={btns} axes={axes}"
            )
            last_status = now
        time.sleep(0.02)

    env.close()
    print(
        f"intervened_steps={intervened_seen} passthrough_steps={passthrough_seen} "
        f"mode={args.mode}"
    )
    if intervened_seen < 5:
        raise SystemExit(
            "R7_LIVE: FAIL — need left-session+motion for >=5 intervened steps "
            f"(got {intervened_seen})"
        )
    print("R7_LIVE: PASS")
    print(
        "Manual checklist: confirm XYZ directions match 0810 teleop "
        "(see docs/solution/R7_方案.md §4.4)"
    )


if __name__ == "__main__":
    main()
