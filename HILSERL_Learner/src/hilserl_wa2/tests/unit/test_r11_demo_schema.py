"""R11 demo schema tests. Must not import rospy, recorder, or request_hand."""

from __future__ import annotations

import inspect
import pathlib
import pickle
import sys
import tempfile
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.demo_io import (  # noqa: E402
    DemoIOError,
    load_bundle,
    load_transitions,
    mark_human_success,
    validate_demo_list,
    validate_success_episode,
    write_failed_episode,
    write_success_bundle,
)
from hilserl_wa2.experiments.transition import (  # noqa: E402
    TRANSITION_KEYS,
    build_actor_transition,
)


def _obs(fill: int = 0):
    return {
        "state": np.full((1, 27), fill, dtype=np.float32),
        "head": np.full((1, 128, 128, 3), fill, dtype=np.uint8),
        "wrist": np.full((1, 128, 128, 3), fill, dtype=np.uint8),
    }


def _step(action, *, intervened: bool, terminated: bool = False):
    info = {}
    if intervened:
        info["intervene_action"] = np.asarray(action, dtype=np.float32)
    tr, _ = build_actor_transition(
        _obs(1),
        action,
        _obs(2),
        0.0,
        terminated,
        False,
        info,
    )
    return tr


def _episode(n: int = 8, intervene: bool = True):
    action = np.asarray([0.4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    rows = []
    for _ in range(n):
        rows.append(_step(action if intervene else np.zeros(6, dtype=np.float32), intervened=intervene))
    return mark_human_success(rows)


def _sidecar(index: int) -> dict:
    return {
        "episode_index": index,
        "label": "success",
        "operator": "unit",
        "task_id": "bottle_pick",
        "exp_name": "wa2_bottle_pick",
        "config_bundle_hash": "abc",
        "space_hash": "def",
        "transition_schema_version": "r9-v1",
        "started_at": "2026-08-15T00:00:00Z",
        "n_steps": 8,
        "intervened_steps": 8,
        "intervention_count": 1,
        "hand_toggles": 1,
        "reset_ok": True,
        "human_success": True,
        "discard_reason": None,
    }


class R11DemoSchemaTests(unittest.TestCase):
    def test_module_stays_offline(self):
        import hilserl_wa2.experiments.demo_io as demo_io

        source = inspect.getsource(demo_io)
        self.assertNotIn("rospy", source)
        self.assertNotIn("record_r11_demos", source)
        self.assertNotIn("request_hand", source)
        self.assertNotIn("naviai_controller", source)

    def test_success_pkl_has_six_keys_and_no_infos(self):
        episode = _episode()
        validate_demo_list(episode)
        self.assertEqual(set(episode[0].keys()), set(TRANSITION_KEYS))
        self.assertNotIn("infos", episode[0])
        self.assertTrue(all(float(row["rewards"]) == 0.0 for row in episode[:-1]))
        self.assertEqual(float(episode[-1]["rewards"]), 1.0)

    def test_last_step_terminated(self):
        episode = _episode()
        stats = validate_success_episode(episode)
        self.assertFalse(bool(episode[0]["dones"]))
        self.assertEqual(float(episode[0]["masks"]), 1.0)
        self.assertTrue(bool(episode[-1]["dones"]))
        self.assertEqual(float(episode[-1]["masks"]), 0.0)
        self.assertEqual(float(episode[-1]["rewards"]), 1.0)
        self.assertEqual(stats["n_steps"], 8)

    def test_intervened_action_matches(self):
        action = np.asarray([0.2, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        row = _step(action, intervened=True)
        np.testing.assert_allclose(row["actions"], action)

    def test_failed_not_in_success_bundle(self):
        success = _episode()
        failed = _episode()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            bundle = root / "success_bundle"
            failed_dir = root / "failed"
            write_success_bundle(
                bundle,
                bundle_name="wa2_bottle_pick_5_fake",
                episodes=[success],
                sidecars=[_sidecar(0)],
            )
            write_failed_episode(
                failed_dir,
                episode_index=1,
                transitions=failed,
                sidecar={**_sidecar(1), "label": "failed", "human_success": False},
            )
            loaded = load_transitions(bundle / "demo.pkl")
            self.assertEqual(len(loaded), len(success))
            packed = load_bundle(bundle)
            self.assertEqual(packed["manifest"]["n_episodes"], 1)
            self.assertTrue((failed_dir / "ep001.pkl").is_file())
            with self.assertRaises(DemoIOError):
                load_bundle(failed_dir)

    def test_rewards_terminal_one_after_roundtrip(self):
        episode = _episode()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "demo.pkl"
            with path.open("wb") as handle:
                pickle.dump(episode, handle, protocol=4)
            loaded = load_transitions(path)
            self.assertTrue(all(float(row["rewards"]) == 0.0 for row in loaded[:-1]))
            self.assertEqual(float(loaded[-1]["rewards"]), 1.0)


if __name__ == "__main__":
    unittest.main()
