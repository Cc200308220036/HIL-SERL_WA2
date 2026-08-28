"""Multi-camera ROS image monitor: subscribe, preprocess, age cache."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from hilserl_wa2.ros_adapters.image_preprocess import (
    Crop,
    decode_ros_image_to_rgb,
    parse_crop,
    preprocess_rgb,
)

DEFAULT_CAMERA_CFG = (
    Path(__file__).resolve().parents[1] / "configs" / "cameras" / "head_wrist.yaml"
)


@dataclass
class CameraStreamConfig:
    name: str
    topic: str
    enabled: bool
    policy_crop: Crop
    out_hw: Tuple[int, int]
    image_max_age_s: float


@dataclass
class FrameSlot:
    image: Optional[np.ndarray] = None
    stamp_mono: Optional[float] = None
    encoding: Optional[str] = None
    raw_shape: Optional[Tuple[int, int, int]] = None
    recv_count: int = 0
    forced_age_s: Optional[float] = None


@dataclass
class RawImage:
    """ROS-free copy of sensor_msgs/Image fields needed for decode."""

    encoding: str
    height: int
    width: int
    step: int
    data: bytes


class ImageCache:
    """Thread-safe latest-frame cache per camera key."""

    def __init__(self, names: Sequence[str], image_max_age_s: float):
        self.image_max_age_s = float(image_max_age_s)
        self._lock = threading.Lock()
        self._slots: Dict[str, FrameSlot] = {n: FrameSlot() for n in names}

    def update(
        self,
        name: str,
        image: np.ndarray,
        *,
        encoding: str,
        raw_shape: Tuple[int, int, int],
        stamp: Optional[float] = None,
    ) -> None:
        with self._lock:
            slot = self._slots[name]
            slot.image = np.ascontiguousarray(image)
            slot.stamp_mono = time.monotonic() if stamp is None else float(stamp)
            slot.encoding = encoding
            slot.raw_shape = raw_shape
            slot.recv_count += 1
            slot.forced_age_s = None

    def get_images(
        self, names: Optional[Sequence[str]] = None
    ) -> Dict[str, np.ndarray]:
        keys = list(names) if names is not None else list(self._slots.keys())
        with self._lock:
            out = {}
            for name in keys:
                slot = self._slots[name]
                if slot.image is None:
                    raise RuntimeError(f"camera '{name}' has no frame yet")
                out[name] = slot.image.copy()
            return out

    def get_ages(self) -> Dict[str, Optional[float]]:
        now = time.monotonic()
        with self._lock:
            ages: Dict[str, Optional[float]] = {}
            for name, slot in self._slots.items():
                if slot.forced_age_s is not None:
                    ages[name] = float(slot.forced_age_s)
                elif slot.stamp_mono is None:
                    ages[name] = None
                else:
                    ages[name] = max(0.0, now - slot.stamp_mono)
            return ages

    def stale_fields(self, enabled: Optional[Sequence[str]] = None) -> List[str]:
        names = list(enabled) if enabled is not None else list(self._slots.keys())
        ages = self.get_ages()
        stale = []
        for name in names:
            age = ages.get(name)
            if age is None or age > self.image_max_age_s:
                stale.append(f"images/{name}")
        return stale

    def is_ready(self, enabled: Optional[Sequence[str]] = None) -> bool:
        names = list(enabled) if enabled is not None else list(self._slots.keys())
        with self._lock:
            return all(self._slots[n].image is not None for n in names)

    def is_fresh(self, enabled: Optional[Sequence[str]] = None) -> bool:
        return len(self.stale_fields(enabled)) == 0

    def inject_stale_for_test(
        self, camera: Optional[str] = None, age_s: float = 1.0
    ) -> None:
        with self._lock:
            targets = [camera] if camera is not None else list(self._slots.keys())
            for name in targets:
                if name not in self._slots:
                    raise ValueError(f"unknown camera {name}")
                self._slots[name].forced_age_s = float(age_s)

    def clear_stale_injection(self) -> None:
        with self._lock:
            for slot in self._slots.values():
                slot.forced_age_s = None

    def stats(self) -> Dict[str, Any]:
        ages = self.get_ages()
        with self._lock:
            return {
                name: {
                    "recv_count": slot.recv_count,
                    "age_s": ages.get(name),
                    "encoding": slot.encoding,
                    "raw_shape": slot.raw_shape,
                    "has_frame": slot.image is not None,
                }
                for name, slot in self._slots.items()
            }


def load_camera_stream_configs(
    path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, CameraStreamConfig], Dict[str, Any]]:
    cfg_path = Path(path) if path is not None else DEFAULT_CAMERA_CFG
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    out_hw = tuple(int(x) for x in raw["output"]["shape"][:2])
    max_age = float(raw.get("freshness", {}).get("image_max_age_s", 0.2))
    streams: Dict[str, CameraStreamConfig] = {}
    for name, cam in (raw.get("cameras") or {}).items():
        streams[name] = CameraStreamConfig(
            name=name,
            topic=str(cam["topic"]),
            enabled=bool(cam.get("enabled", True)),
            policy_crop=parse_crop(cam.get("policy_crop")),
            out_hw=(int(out_hw[0]), int(out_hw[1])),
            image_max_age_s=max_age,
        )
    return streams, raw


class WA2ImageMonitor:
    """Subscribe enabled camera topics; expose latest preprocessed RGB frames."""

    def __init__(
        self,
        camera_cfg_path: Optional[Union[str, Path]] = None,
        image_max_age_s: Optional[float] = None,
        streams: Optional[Mapping[str, CameraStreamConfig]] = None,
        cache: Optional[ImageCache] = None,
    ):
        if streams is not None:
            self.streams = dict(streams)
            self.raw_cfg: Dict[str, Any] = {}
        else:
            self.streams, self.raw_cfg = load_camera_stream_configs(camera_cfg_path)
        if not self.streams:
            raise ValueError("no camera streams configured")
        ages = [
            s.image_max_age_s
            for s in self.streams.values()
            if s.enabled
        ]
        self.image_max_age_s = float(
            image_max_age_s if image_max_age_s is not None else (ages[0] if ages else 0.2)
        )
        self.enabled_names = [n for n, s in self.streams.items() if s.enabled]
        self.cache = cache or ImageCache(
            names=list(self.streams.keys()), image_max_age_s=self.image_max_age_s
        )
        self._started = False
        self._subs = []
        self._bridge = None
        self._rospy = None
        self._raw_lock = threading.Lock()
        self._pending: Dict[str, Tuple[CameraStreamConfig, RawImage]] = {}
        self._raw_event = threading.Event()
        self._worker_stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._started:
            return
        import rospy
        from sensor_msgs.msg import Image

        try:
            from cv_bridge import CvBridge

            self._bridge = CvBridge()
        except Exception:
            self._bridge = None

        self._rospy = rospy
        if not rospy.core.is_initialized():
            rospy.init_node("wa2_image_monitor", anonymous=True, disable_signals=True)

        self._worker_stop.clear()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="wa2_image_preprocess",
            daemon=True,
        )
        self._worker.start()

        for name in self.enabled_names:
            stream = self.streams[name]

            def _cb(msg, camera_name=name, stream_cfg=stream):
                # rospy thread: copy latest raw only. Decode/resize is the worker.
                self.enqueue_raw(camera_name, stream_cfg, msg)

            self._subs.append(
                rospy.Subscriber(stream.topic, Image, _cb, queue_size=1, buff_size=2**24)
            )
        self._started = True

    def stop(self) -> None:
        self._worker_stop.set()
        self._raw_event.set()
        worker = self._worker
        self._worker = None
        if (
            worker is not None
            and worker.is_alive()
            and worker is not threading.current_thread()
        ):
            worker.join(timeout=0.5)
        for sub in self._subs:
            try:
                sub.unregister()
            except Exception:
                pass
        self._subs = []
        self._started = False

    def wait_ready(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._rospy is not None and self._rospy.is_shutdown():
                raise RuntimeError("ROS shutdown while waiting for images")
            if self.cache.is_ready(self.enabled_names) and self.cache.is_fresh(
                self.enabled_names
            ):
                return
            time.sleep(0.01)
        raise TimeoutError(
            f"WA2ImageMonitor not ready within {timeout_s}s; "
            f"stats={self.cache.stats()} stale={self.cache.stale_fields(self.enabled_names)}"
        )

    def get_images(self) -> Dict[str, np.ndarray]:
        images = self.cache.get_images(self.enabled_names)
        h, w = next(iter(self.streams.values())).out_hw
        for name, stream in self.streams.items():
            if not stream.enabled:
                images[name] = np.zeros((h, w, 3), dtype=np.uint8)
        return images

    def get_ages(self) -> Dict[str, Optional[float]]:
        return self.cache.get_ages()

    def stale_fields(self) -> List[str]:
        return self.cache.stale_fields(self.enabled_names)

    def is_fresh(self) -> bool:
        return self.cache.is_fresh(self.enabled_names)

    def inject_stale_for_test(
        self, camera: Optional[str] = None, age_s: float = 1.0
    ) -> None:
        self.cache.inject_stale_for_test(camera=camera, age_s=age_s)

    def clear_stale_injection(self) -> None:
        self.cache.clear_stale_injection()

    def enqueue_raw(self, name: str, stream: CameraStreamConfig, msg) -> None:
        """Keep only the latest raw frame per camera (overwrite)."""

        raw = msg_to_raw_image(msg)
        with self._raw_lock:
            self._pending[name] = (stream, raw)
        self._raw_event.set()

    def drain_pending(self) -> Dict[str, Tuple[CameraStreamConfig, RawImage]]:
        with self._raw_lock:
            batch = self._pending
            self._pending = {}
        return batch

    def process_raw(self, name: str, stream: CameraStreamConfig, raw: RawImage) -> None:
        try:
            rgb = decode_ros_image_to_rgb(
                raw.data,
                int(raw.height),
                int(raw.width),
                str(raw.encoding),
                step=int(raw.step or 0) or None,
            )
            out = preprocess_rgb(
                rgb, crop=stream.policy_crop, out_hw=stream.out_hw
            )
            self.cache.update(
                name,
                out,
                encoding=str(raw.encoding),
                raw_shape=(int(raw.height), int(raw.width), 3),
            )
        except Exception:
            # Drop bad frames; freshness will eventually truncate.
            return

    def _worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            self._raw_event.wait(timeout=0.05)
            self._raw_event.clear()
            if self._worker_stop.is_set():
                break
            batch = self.drain_pending()
            for name, (stream, raw) in batch.items():
                self.process_raw(name, stream, raw)

    def _on_image(self, name: str, stream: CameraStreamConfig, msg) -> None:
        # Kept for tests / offline injection; live ROS path uses the worker.
        self.process_raw(name, stream, msg_to_raw_image(msg))

    def _msg_to_rgb(self, msg) -> np.ndarray:
        raw = msg_to_raw_image(msg)
        enc = (raw.encoding or "").lower()
        if self._bridge is not None:
            try:
                if "bgr8" in enc:
                    bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    import cv2

                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            except Exception:
                pass
        return decode_ros_image_to_rgb(
            raw.data,
            raw.height,
            raw.width,
            raw.encoding,
            step=raw.step or None,
        )


def msg_to_raw_image(msg) -> RawImage:
    if isinstance(msg, RawImage):
        return msg
    data = msg.data
    if isinstance(data, memoryview):
        payload = data.tobytes()
    elif isinstance(data, bytes):
        payload = data
    else:
        payload = bytes(data)
    return RawImage(
        encoding=str(getattr(msg, "encoding", "")),
        height=int(getattr(msg, "height", 0)),
        width=int(getattr(msg, "width", 0)),
        step=int(getattr(msg, "step", 0) or 0),
        data=payload,
    )
