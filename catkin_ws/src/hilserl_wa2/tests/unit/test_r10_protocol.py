from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from hilserl_wa2.experiments.r10_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    compare_handshake,
    ordered_transition_digest,
    transition_sha256,
)


class R10ProtocolTests(unittest.TestCase):
    def manifest(self):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "transition_schema_version": "r9-v1",
            "task_id": "bottle_pick",
            "exp_name": "wa2_bottle_pick",
            "config_bundle_hash": "a",
            "network_config_hash": "b",
            "space_hash": "c",
            "params_tree_signature": "d",
            "agentlace_version": "0.1.3",
            "agentlace_wheel_sha256": "e",
            "source_tree_sha256": "f",
        }

    def test_handshake_accepts_exact_manifest_plus_session(self):
        expected = self.manifest()
        got = dict(expected, session_id="session-1")
        self.assertTrue(compare_handshake(expected, got)["accepted"])

    def test_handshake_rejects_mismatch(self):
        expected = self.manifest()
        got = dict(expected, session_id="session-1", space_hash="wrong")
        result = compare_handshake(expected, got)
        self.assertFalse(result["accepted"])
        self.assertIn("space_hash", result["mismatches"])

    def test_transition_digest_stable_and_ordered(self):
        a = {"x": np.asarray([1, 2], np.int32)}
        b = {"x": np.asarray([3, 4], np.int32)}
        self.assertEqual(transition_sha256(a), transition_sha256(a))
        self.assertNotEqual(ordered_transition_digest([a, b]), ordered_transition_digest([b, a]))


if __name__ == "__main__":
    unittest.main()
