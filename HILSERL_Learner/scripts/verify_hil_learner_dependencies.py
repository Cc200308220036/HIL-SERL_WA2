#!/usr/bin/env python3
"""Dependency/hash/import checks that do not enumerate or execute GPU devices."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import os
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/home/cyw/orin_hilserl/HILSERL_Learner")
EXPECTED = {
    "jax": "0.4.35",
    "jaxlib": "0.4.35",
    "jax-cuda12-plugin": "0.4.35",
    "jax-cuda12-pjrt": "0.4.35",
    "agentlace": "0.1.3",
    "serl_launcher": "0.1.2",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "flax": "0.10.2",
    "gymnasium": "1.2.2",
    "tensorflow": "2.21.0",
    "tensorflow-probability": "0.25.0",
    "tf_keras": "2.21.0",
    "opencv-python": "4.10.0.84",
}
HASHES = {
    ROOT / "artifacts/wheels/agentlace-0.1.3-py3-none-any.whl": "1a800cc341f03eb6844273571ba26a265920fa1b5a698acc3d954438cbb72d32",
    ROOT / "artifacts/models/resnet10_params.pkl": "175745d43d30233eb01b5369465d1c24c11b8ee71ccb734cc1c1bca13e07f57b",
}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    assert ROOT == EXPECTED_ROOT, (ROOT, EXPECTED_ROOT)
    assert Path(os.environ.get("HILSERL_LEARNER_ROOT", "")).resolve() == ROOT
    assert sys.version_info[:3] == (3, 10, 20), sys.version
    assert os.environ.get("CONDA_DEFAULT_ENV") == "hil-learner"
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert not site.ENABLE_USER_SITE
    bad_paths = [p for p in sys.path if "/opt/ros/" in p or "/ros2_ws/" in p or "/aubo_ros2_ws/" in p]
    assert not bad_paths, bad_paths
    for package, wanted in EXPECTED.items():
        got = metadata.version(package)
        print(f"{package}={got}")
        assert got == wanted, (package, wanted, got)
    for path, wanted in HASHES.items():
        got = file_hash(path)
        print(f"{path.name}_sha256={got}")
        assert got == wanted, (path, wanted, got)
    import agentlace  # noqa: F401
    import cv2
    import gymnasium  # noqa: F401
    import hilserl_wa2
    import serl_launcher  # noqa: F401

    hilserl_path = Path(hilserl_wa2.__file__).resolve()
    assert hilserl_path.is_relative_to(ROOT), hilserl_path
    print(f"HILSERL_LEARNER_ROOT={ROOT}")
    print(f"HILSERL_WA2_IMPORT={hilserl_path}")
    print(f"cv2={cv2.__version__}")
    print("HIL_LEARNER_DEPENDENCIES: PASS")


if __name__ == "__main__":
    main()
