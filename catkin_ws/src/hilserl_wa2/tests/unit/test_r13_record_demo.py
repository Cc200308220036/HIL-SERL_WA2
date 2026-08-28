"""R13 recorded 7D demo schema. No ROS."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.demo_io import (  # noqa: E402
    DemoIOError,
    grasp_action_counts,
    load_bundle,
    validate_r13_grasp_edges,
    write_success_bundle,
)
from hilserl_wa2.experiments.transition import build_actor_transition  # noqa: E402


def _obs(fill: int = 0):
    return {
        "state": np.full((1, 27), fill, dtype=np.float32),
        "head": np.full((1, 128, 128, 3), fill, dtype=np.uint8),
        "wrist": np.full((1, 128, 128, 3), fill + 1, dtype=np.uint8),
    }


def _row(action, terminated=False):
    info = {"intervene_action": np.asarray(action, dtype=np.float32)}
    tr, _ = build_actor_transition(
        _obs(1), action, _obs(2), 0.0, terminated, False, info
    )
    return tr


def _episode_7d():
    rows = []
    for i in range(8):
        action = np.zeros(7, np.float32)
        action[0] = 0.2
        if i == 2:
            action[6] = 1.0
        if i == 6:
            action[6] = -1.0
        rows.append(_row(action, terminated=False))
    rows[-1]["dones"] = np.bool_(True)
    rows[-1]["masks"] = np.float32(0.0)
    return rows


def _sidecar(index: int) -> dict:
    return {
        "episode_index": index,
        "label": "success",
        "operator": "unit",
        "task_id": "bottle_pick",
        "exp_name": "wa2_bottle_pick",
        "config_bundle_hash": "abc",
        "space_hash": "def",
        "transition_schema_version": "r13-v1",
        "started_at": "2026-08-20T00:00:00Z",
        "n_steps": 8,
        "intervened_steps": 8,
        "intervention_count": 1,
        "hand_toggles": 2,
        "reset_ok": True,
        "human_success": True,
        "discard_reason": None,
    }


class R13RecordedDemoTests(unittest.TestCase):
    def test_counts_plus_and_minus(self):
        counts = grasp_action_counts(_episode_7d())
        self.assertEqual(counts["plus"], 1)
        self.assertEqual(counts["minus"], 1)

    def test_rejects_missing_release(self):
        rows = _episode_7d()
        rows[6]["actions"] = np.zeros(7, np.float32)
        rows[6]["actions"][0] = 0.2
        with self.assertRaises(DemoIOError):
            validate_r13_grasp_edges(rows)

    def test_write_bundle_keeps_r13_schema_and_7d(self):
        episode = _episode_7d()
        with tempfile.TemporaryDirectory() as tmp:
            bundle = pathlib.Path(tmp) / "b"
            write_success_bundle(
                bundle,
                bundle_name="wa2_bottle_pick_20_success_7d",
                episodes=[episode],
                sidecars=[_sidecar(0)],
            )
            packed = load_bundle(bundle)
            self.assertEqual(packed["sidecars"][0]["transition_schema_version"], "r13-v1")
            self.assertEqual(int(np.asarray(packed["transitions"][0]["actions"]).shape[-1]), 7)
            self.assertGreaterEqual(int(packed["sidecars"][0]["n_grasp_plus"]), 1)
            self.assertGreaterEqual(int(packed["sidecars"][0]["n_grasp_minus"]), 1)
            last = packed["episodes"][0][-1]
            self.assertEqual(float(last["rewards"]), 1.0)
            self.assertTrue(bool(last["dones"]))
            self.assertEqual(float(last["masks"]), 0.0)


if __name__ == "__main__":
    unittest.main()
