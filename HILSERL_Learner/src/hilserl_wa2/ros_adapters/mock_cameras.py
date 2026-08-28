"""Mock cameras for R2: no device open, dual image keys."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from hilserl_wa2.envs.contracts import WA2EnvContract


class MockCameras:
    """Return head/wrist uint8 images without touching hardware."""

    def __init__(self, contract: WA2EnvContract, seed: Optional[int] = None):
        self.contract = contract
        self.shape: Tuple[int, int, int] = contract.image_shape
        self._rng = np.random.default_rng(seed)
        self._pattern_seed = seed

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._pattern_seed = seed

    def get_images(self) -> Dict[str, np.ndarray]:
        h, w, c = self.shape
        # Head: deterministic mild pattern from seed for repro; still valid uint8.
        if self._pattern_seed is None:
            head = np.zeros((h, w, c), dtype=np.uint8)
        else:
            # Low-amplitude pattern so check_env / contains stay happy.
            yy, xx = np.mgrid[0:h, 0:w]
            base = ((xx + yy + int(self._pattern_seed)) % 256).astype(np.uint8)
            head = np.stack([base, base, base], axis=-1)
        if self.contract.missing_policy != "zero_image":
            raise ValueError(
                f"unsupported missing_policy={self.contract.missing_policy}"
            )
        # Wrist disabled in R1/R2: always zeros, never causes truncated.
        wrist = np.zeros((h, w, c), dtype=np.uint8)
        if self.contract.wrist_enabled:
            # Future path: would fill from stream; R2 keeps zeros until R6.
            wrist = wrist
        return {"head": head, "wrist": wrist}
