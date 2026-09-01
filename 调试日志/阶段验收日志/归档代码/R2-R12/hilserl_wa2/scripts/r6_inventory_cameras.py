#!/usr/bin/env python3
"""Inventory head/wrist topics, save raw+128 samples, print encodings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT.parent))

from hilserl_wa2.ros_adapters.image_preprocess import (  # noqa: E402
    decode_ros_image_to_rgb,
    preprocess_rgb,
)

TOPICS = {
    "head": "/zj_humanoid/sensor/realsense_head/color/image_raw",
    "wrist": "/zj_humanoid/sensor/left_wrist/image_raw",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/root/catkin_ws/r6_samples")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not rospy.core.is_initialized():
        rospy.init_node("r6_inventory_cameras", anonymous=True, disable_signals=True)

    meta_lines = []
    for name, topic in TOPICS.items():
        msg = rospy.wait_for_message(topic, Image, timeout=5.0)
        rgb = decode_ros_image_to_rgb(
            bytes(msg.data),
            int(msg.height),
            int(msg.width),
            str(msg.encoding),
            step=int(msg.step) if msg.step else None,
        )
        line = (
            f"{name}: topic={topic} h={msg.height} w={msg.width} "
            f"enc={msg.encoding} shape={rgb.shape}"
        )
        print(line)
        meta_lines.append(line)
        cv2.imwrite(
            str(out / f"{name}_raw_bgr_for_view.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        out128 = preprocess_rgb(rgb, crop=None, out_hw=(128, 128))
        cv2.imwrite(
            str(out / f"{name}_policy_128.png"),
            cv2.cvtColor(out128, cv2.COLOR_RGB2BGR),
        )
        if name == "wrist":
            cv2.imwrite(
                str(out / "classifier_128.png"),
                cv2.cvtColor(out128, cv2.COLOR_RGB2BGR),
            )

    (out / "inventory_meta.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    print(f"wrote samples under {out}")
    print("R6_INVENTORY: PASS")


if __name__ == "__main__":
    main()
