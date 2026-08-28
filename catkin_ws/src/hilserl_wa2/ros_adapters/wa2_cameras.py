"""Real dual-camera facade for WA2Env (ROS Image topics)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from hilserl_wa2.envs.contracts import WA2EnvContract
from hilserl_wa2.ros_adapters.image_monitor import WA2ImageMonitor


class WA2Cameras:
    """Live head/wrist images via :class:`WA2ImageMonitor`."""

    def __init__(
        self,
        contract: WA2EnvContract,
        camera_cfg_path: Optional[Union[str, Path]] = None,
        image_monitor: Optional[WA2ImageMonitor] = None,
        image_max_age_s: Optional[float] = None,
    ):
        self.contract = contract
        self._monitor = image_monitor or WA2ImageMonitor(
            camera_cfg_path=camera_cfg_path,
            image_max_age_s=(
                float(image_max_age_s)
                if image_max_age_s is not None
                else float(contract.image_max_age_s)
            ),
        )
        self._started = False

    @property
    def monitor(self) -> WA2ImageMonitor:
        return self._monitor

    def start(self) -> None:
        if self._started:
            return
        self._monitor.start()
        self._started = True

    def stop(self) -> None:
        self._monitor.stop()
        self._started = False

    def wait_ready(self, timeout_s: float = 5.0) -> None:
        if not self._started:
            self.start()
        self._monitor.wait_ready(timeout_s=timeout_s)

    def reset(self, seed: Optional[int] = None) -> None:
        _ = seed
        self._monitor.clear_stale_injection()

    def get_images(self) -> Dict[str, np.ndarray]:
        images = self._monitor.get_images()
        # Guarantee dual contract keys.
        h, w, c = self.contract.image_shape
        for key in ("head", "wrist"):
            if key not in images:
                images[key] = np.zeros((h, w, c), dtype=np.uint8)
            else:
                img = np.asarray(images[key], dtype=np.uint8)
                if img.shape != (h, w, c):
                    raise RuntimeError(
                        f"camera '{key}' shape {img.shape} != {(h, w, c)}"
                    )
                images[key] = img
        return images

    def get_ages(self) -> Dict[str, Optional[float]]:
        return self._monitor.get_ages()

    def stale_fields(self) -> List[str]:
        return self._monitor.stale_fields()

    def is_fresh(self) -> bool:
        return self._monitor.is_fresh()

    def inject_stale_for_test(
        self, camera: Optional[str] = None, age_s: float = 1.0
    ) -> None:
        self._monitor.inject_stale_for_test(camera=camera, age_s=age_s)

    def clear_stale_injection(self) -> None:
        self._monitor.clear_stale_injection()

    def stats(self) -> Dict[str, Any]:
        return self._monitor.cache.stats()
