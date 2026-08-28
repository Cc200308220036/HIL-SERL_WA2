#!/usr/bin/env python3
"""Validate the pinned HIL-SERL Actor baseline environment."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import platform


EXPECTED = {
    "agentlace": "0.1.3",
    "gym": "0.26.2",
    "gymnasium": "1.2.2",
    "jax": "0.4.35",
    "jax-cuda12-pjrt": "0.4.35",
    "jax-cuda12-plugin": "0.4.35",
    "jaxlib": "0.4.35",
    "numpy": "1.26.4",
    "opencv-python": "4.10.0",
    "scipy": "1.15.3",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Only check imports and versions during docker build.",
    )
    args = parser.parse_args()

    if platform.machine() != "aarch64":
        raise RuntimeError(
            f"Expected aarch64, got {platform.machine()}"
        )

    for package, expected in EXPECTED.items():
        actual = metadata.version(package)
        if actual != expected:
            raise RuntimeError(
                f"{package}: expected {expected}, got {actual}"
            )
        print(f"{package}=={actual}")

    import agentlace  # noqa: F401
    import cv2
    import gym  # noqa: F401
    import gymnasium  # noqa: F401
    import jax
    import lz4  # noqa: F401
    import numpy  # noqa: F401
    import typing_extensions  # noqa: F401
    import zmq  # noqa: F401

    print(f"OpenCV import: {cv2.__version__}")

    if not args.skip_gpu:
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        if not any(device.platform == "gpu" for device in devices):
            raise RuntimeError("No JAX GPU device detected.")

    print("HIL-ACTOR BASELINE: PASS")


if __name__ == "__main__":
    main()
