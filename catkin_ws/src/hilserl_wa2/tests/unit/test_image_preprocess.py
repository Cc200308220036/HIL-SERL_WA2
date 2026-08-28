"""Unit tests for image preprocess + ImageCache (no ROS)."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.ros_adapters.image_monitor import (  # noqa: E402
    CameraStreamConfig,
    ImageCache,
    RawImage,
    WA2ImageMonitor,
)
from hilserl_wa2.ros_adapters.image_preprocess import (  # noqa: E402
    decode_ros_image_to_rgb,
    preprocess_rgb,
)


class ImagePreprocessTests(unittest.TestCase):
    def test_rgb8_and_bgr8_decode(self):
        h, w = 4, 6
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = 10
        rgb[..., 2] = 200
        out = decode_ros_image_to_rgb(rgb.tobytes(), h, w, "rgb8")
        self.assertEqual(out.shape, (h, w, 3))
        self.assertEqual(int(out[0, 0, 0]), 10)
        self.assertEqual(int(out[0, 0, 2]), 200)

        bgr = rgb[..., ::-1].copy()
        out2 = decode_ros_image_to_rgb(bgr.tobytes(), h, w, "bgr8")
        np.testing.assert_array_equal(out2, rgb)

    def test_crop_resize(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[100:200, 300:400, :] = 255
        out = preprocess_rgb(img, crop=(100, 200, 300, 400), out_hw=(128, 128))
        self.assertEqual(out.shape, (128, 128, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(int(out.mean()), 200)

    def test_image_cache_stale(self):
        cache = ImageCache(names=["head", "wrist"], image_max_age_s=0.2)
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        cache.update("head", frame, encoding="rgb8", raw_shape=(720, 1280, 3))
        cache.update("wrist", frame, encoding="bgr8", raw_shape=(720, 1280, 3))
        self.assertTrue(cache.is_ready(["head", "wrist"]))
        self.assertEqual(cache.stale_fields(["head", "wrist"]), [])
        cache.inject_stale_for_test(camera="wrist", age_s=1.0)
        self.assertEqual(cache.stale_fields(["head", "wrist"]), ["images/wrist"])
        cache.clear_stale_injection()
        self.assertEqual(cache.stale_fields(["head", "wrist"]), [])

    def test_latest_raw_overwrites_and_worker_resizes(self):
        cache = ImageCache(names=["head"], image_max_age_s=0.2)
        stream = CameraStreamConfig(
            name="head",
            topic="/cam",
            enabled=True,
            policy_crop=None,
            out_hw=(128, 128),
            image_max_age_s=0.2,
        )
        monitor = WA2ImageMonitor(streams={"head": stream}, cache=cache)
        first = np.zeros((8, 8, 3), dtype=np.uint8)
        second = np.full((8, 8, 3), 200, dtype=np.uint8)
        monitor.enqueue_raw(
            "head",
            stream,
            RawImage(
                encoding="rgb8",
                height=8,
                width=8,
                step=24,
                data=first.tobytes(),
            ),
        )
        monitor.enqueue_raw(
            "head",
            stream,
            RawImage(
                encoding="rgb8",
                height=8,
                width=8,
                step=24,
                data=second.tobytes(),
            ),
        )
        pending = monitor.drain_pending()
        self.assertEqual(len(pending), 1)
        name, (cfg, raw) = next(iter(pending.items()))
        self.assertEqual(name, "head")
        monitor.process_raw(name, cfg, raw)
        images = cache.get_images(["head"])
        self.assertEqual(images["head"].shape, (128, 128, 3))
        self.assertGreater(int(images["head"].mean()), 150)


if __name__ == "__main__":
    unittest.main()
