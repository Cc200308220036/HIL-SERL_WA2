#!/usr/bin/env python3
"""Launch spacemouse_wa2_teleop using configs/spacemouse/*.yaml (diagnostic only).

Do NOT run together with WA2SpacemouseIntervention / Actor Env control.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT.parent))

import rospy  # noqa: E402

from hilserl_wa2.interventions.spacemouse_config import (  # noqa: E402
    load_spacemouse_config,
)
from hilserl_wa2.interventions.spacemouse_wa2_teleop import (  # noqa: E402
    SpaceMouseWA2Teleop,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help="YAML path or stem under configs/spacemouse/ (default: default.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rospy params that would be set, then exit",
    )
    args = parser.parse_args()

    cfg = load_spacemouse_config(args.config)
    params = cfg.teleop_ros_params()

    if args.dry_run:
        print(f"config={cfg.path}")
        for key, value in sorted(params.items()):
            print(f"  ~{key}: {value}")
        return

    rospy.init_node("spacemouse_wa2_teleop", anonymous=False)
    for key, value in params.items():
        rospy.set_param(f"~{key}", value)
    rospy.loginfo("loaded SpaceMouse config: %s", cfg.path)

    teleop = SpaceMouseWA2Teleop()
    try:
        teleop.run()
    finally:
        teleop.stop_robot()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TypeError, ValueError) as exc:
        print(f"teleop_from_yaml failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
