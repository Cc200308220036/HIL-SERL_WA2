"""R13 protocol tests. No ROS / JAX."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.r13_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    TRANSITION_SCHEMA_VERSION,
    compare_handshake,
    scale_arm_action,
    tree_has_nan_or_inf,
    update_info_has_nan,
)


class R13ProtocolTests(unittest.TestCase):
    def manifest(self):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "transition_schema_version": TRANSITION_SCHEMA_VERSION,
            "task_id": "bottle_pick",
            "exp_name": "wa2_bottle_pick",
            "config_bundle_hash": "a",
            "network_config_hash": "b",
            "space_hash": "c",
            "params_tree_signature": "d",
            "agentlace_version": "0.1.3",
            "agentlace_wheel_sha256": "e",
            "source_tree_sha256": "f",
            "demo_pkl_sha256": "g",
            "action_dim": 7,
            "end_episode": True,
            "action_scale": 1.0,
            "policy_hz": 10.0,
            "servo_hz": 50.0,
            "servo_ticks_per_action": 5,
            "discount": 0.98,
            "classifier_consecutive_n": 1,
        }

    def test_accepts_matching_r13_manifest(self):
        expected = self.manifest()
        got = dict(expected, session_id="s1")
        self.assertTrue(compare_handshake(expected, got)["accepted"])

    def test_rejects_r10_protocol(self):
        expected = self.manifest()
        got = dict(expected, session_id="s1", protocol_version="wa2-r10-v1")
        result = compare_handshake(expected, got)
        self.assertFalse(result["accepted"])
        self.assertIn("protocol_version", result["mismatches"])

    def test_rejects_wrong_demo_hash(self):
        expected = self.manifest()
        got = dict(expected, session_id="s1", demo_pkl_sha256="nope")
        result = compare_handshake(expected, got)
        self.assertFalse(result["accepted"])
        self.assertIn("demo_pkl_sha256", result["mismatches"])

    def test_rejects_wrong_time_scale(self):
        expected = self.manifest()
        got = dict(expected, session_id="s1", servo_ticks_per_action=1)
        result = compare_handshake(expected, got)
        self.assertFalse(result["accepted"])
        self.assertIn("servo_ticks_per_action", result["mismatches"])

    def test_scale_only_arm(self):
        action = np.ones(7, dtype=np.float32)
        scaled = scale_arm_action(action, 0.2)
        np.testing.assert_allclose(scaled[:6], 0.2)
        self.assertEqual(float(scaled[6]), 1.0)

    def test_nan_guard(self):
        self.assertTrue(tree_has_nan_or_inf({"a": np.array([1.0, np.nan])}))
        self.assertFalse(tree_has_nan_or_inf({"a": np.array([1.0, 2.0])}))
        self.assertTrue(update_info_has_nan({"critic_loss": np.inf}))


if __name__ == "__main__":
    unittest.main()
