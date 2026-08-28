"""Image crop/resize/encoding helpers (ROS-free, unit-testable)."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Crop = Optional[Tuple[int, int, int, int]]  # y0, y1, x0, x1


def decode_ros_image_to_rgb(
    data: bytes,
    height: int,
    width: int,
    encoding: str,
    step: Optional[int] = None,
) -> np.ndarray:
    """Decode raw Image message buffer into HxWx3 uint8 RGB."""

    enc = (encoding or "").lower()
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size {height}x{width}")
    arr = np.frombuffer(data, dtype=np.uint8)
    if "rgb8" in enc:
        row = step if step and step >= width * 3 else width * 3
        img = arr.reshape(height, row)[:, : width * 3].reshape(height, width, 3)
        return np.ascontiguousarray(img)
    if "bgr8" in enc:
        row = step if step and step >= width * 3 else width * 3
        img = arr.reshape(height, row)[:, : width * 3].reshape(height, width, 3)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if "mono8" in enc or "8uc1" in enc:
        row = step if step and step >= width else width
        gray = arr.reshape(height, row)[:, :width]
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    raise ValueError(f"unsupported image encoding: {encoding}")


def apply_crop(rgb: np.ndarray, crop: Crop) -> np.ndarray:
    if crop is None:
        return rgb
    y0, y1, x0, x1 = (int(v) for v in crop)
    h, w = rgb.shape[:2]
    if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        raise ValueError(f"crop {crop} out of bounds for shape {(h, w)}")
    return rgb[y0:y1, x0:x1]


def resize_rgb(
    rgb: np.ndarray,
    out_hw: Sequence[int] = (128, 128),
) -> np.ndarray:
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8 RGB, got {rgb.shape} {rgb.dtype}")
    if rgb.shape[0] == out_h and rgb.shape[1] == out_w:
        return np.ascontiguousarray(rgb)
    resized = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)


def preprocess_rgb(
    rgb: np.ndarray,
    *,
    crop: Crop = None,
    out_hw: Sequence[int] = (128, 128),
) -> np.ndarray:
    """Crop then resize to observation image."""

    cropped = apply_crop(rgb, crop)
    return resize_rgb(cropped, out_hw=out_hw)


def parse_crop(value) -> Crop:
    if value is None:
        return None
    arr = list(value)
    if len(arr) != 4:
        raise ValueError(f"crop must be [y0,y1,x0,x1] or null, got {value}")
    return (int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3]))
