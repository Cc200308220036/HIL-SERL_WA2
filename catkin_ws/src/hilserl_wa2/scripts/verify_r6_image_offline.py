#!/usr/bin/env python3
"""Offline R6 checks: preprocess + fake_env isolation (no live cameras required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_ROOT.parent))

from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402
from hilserl_wa2.ros_adapters.image_preprocess import (  # noqa: E402
    decode_ros_image_to_rgb,
    preprocess_rgb,
)


def main() -> None:
    rgb = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    out = preprocess_rgb(rgb, crop=None, out_hw=(128, 128))
    assert out.shape == (128, 128, 3) and out.dtype == np.uint8
    bgr = rgb[..., ::-1].copy()
    back = decode_ros_image_to_rgb(bgr.tobytes(), 720, 1280, "bgr8")
    assert back.shape == (720, 1280, 3)

    env = WA2Env(fake_env=True, seed=0)
    obs, info = env.reset()
    assert info.get("fake_env") is True
    assert obs["images"]["head"].shape == (128, 128, 3)
    assert obs["images"]["wrist"].shape == (128, 128, 3)
    assert env.contract.wrist_enabled is True
    assert env.contract.version == "0.1.1"
    env.close()
    print("R6_OFFLINE: PASS")


if __name__ == "__main__":
    main()
