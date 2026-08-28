#!/usr/bin/env python3
"""R7 offline Gate: synthetic Joy intervention (no robot motion)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT.parent))

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.interventions.joy_watchdog import JoyWatchdog  # noqa: E402
from hilserl_wa2.interventions.spacemouse_input import SpaceMouseInputConfig  # noqa: E402
from hilserl_wa2.interventions.wa2_spacemouse_intervention import (  # noqa: E402
    WA2SpacemouseIntervention,
)
from hilserl_wa2.tests.unit.test_spacemouse_input import SAMPLES  # noqa: E402


def main() -> None:
    pkg = Path(__file__).resolve().parents[1]
    src = (pkg / "interventions" / "wa2_spacemouse_intervention.py").read_text(
        encoding="utf-8"
    )
    assert "from franka_env" not in src
    assert "naviai_controller" not in src
    assert "SpaceMouseExpert()" not in src

    joy = JoyWatchdog(max_age_s=0.2)
    env = WA2SpacemouseIntervention(
        WA2Env(fake_env=True, seed=0),
        joy_watchdog=joy,
        auto_start_ros=False,
        input_config=SpaceMouseInputConfig(
            translation_filter_tau=0.0, rotation_filter_tau=0.0
        ),
    )
    env.reset()
    policy = np.zeros(6, np.float32)

    # passthrough
    joy.inject(SAMPLES["forward_translation"], buttons=[0, 0])
    _, _, _, _, info = env.step(policy)
    assert not info.get("intervened") and "intervene_action" not in info

    # intervene
    joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
    env.processor.reset()
    _, _, _, _, info = env.step(policy)
    assert info.get("intervened") and "intervene_action" in info
    assert float(info["intervene_action"][0]) > 0.0

    # stale
    joy.inject(SAMPLES["forward_translation"], buttons=[0, 1])
    joy.inject_stale_for_test(1.0)
    env.processor.reset()
    _, _, _, _, info = env.step(policy)
    assert not info.get("intervened")

    # 12 directions
    dirs = [
        ("forward_translation", 0, 1),
        ("backward_translation", 0, -1),
        ("left_translation", 1, 1),
        ("right_translation", 1, -1),
        ("up_translation", 2, 1),
        ("down_translation", 2, -1),
        ("left_tilt", 3, 1),
        ("right_tilt", 3, -1),
        ("forward_tilt", 4, 1),
        ("backward_tilt", 4, -1),
        ("clockwise_twist", 5, 1),
        ("counterclockwise_twist", 5, -1),
    ]
    for name, axis, sign in dirs:
        joy.clear_stale_injection()
        joy.inject(SAMPLES[name], buttons=[0, 1])
        env.processor.reset()
        _, _, _, _, info = env.step(policy)
        assert info.get("intervened"), name
        assert sign * float(info["intervene_action"][axis]) > 0.0, name

    env.close()
    print("R7_OFFLINE: PASS")


if __name__ == "__main__":
    main()
