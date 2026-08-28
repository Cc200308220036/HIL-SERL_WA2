"""R13 per-episode demo pkls and legacy split. No ROS."""

from __future__ import annotations

import pathlib
import pickle
import sys
import tempfile
import unittest

import numpy as np

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.demo_io import (  # noqa: E402
    DEMO_PKL_FORMAT_STREAM,
    DemoIOError,
    dump_json,
    dump_transitions,
    load_bundle,
    load_transitions,
    require_episode_pkls,
    split_flat_demo_into_episode_pkls,
    write_success_bundle,
)
from hilserl_wa2.experiments.transition import build_actor_transition  # noqa: E402


def _obs(fill: int = 0):
    return {
        "state": np.full((1, 27), fill, dtype=np.float32),
        "head": np.full((1, 8, 8, 3), fill, dtype=np.uint8),
        "wrist": np.full((1, 8, 8, 3), fill + 1, dtype=np.uint8),
    }


def _row(action, terminated=False):
    info = {"intervene_action": np.asarray(action, dtype=np.float32)}
    tr, _ = build_actor_transition(
        _obs(1), action, _obs(2), 0.0, terminated, False, info
    )
    return tr


def _episode_7d():
    rows = []
    for i in range(6):
        action = np.zeros(7, np.float32)
        action[0] = 0.2
        if i == 1:
            action[6] = 1.0
        if i == 4:
            action[6] = -1.0
        rows.append(_row(action, terminated=False))
    rows[-1]["dones"] = np.bool_(True)
    rows[-1]["masks"] = np.float32(0.0)
    return rows


def _sidecar(index: int, n_steps: int = 6) -> dict:
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
        "n_steps": n_steps,
        "intervened_steps": n_steps,
        "intervention_count": 1,
        "hand_toggles": 2,
        "reset_ok": True,
        "human_success": True,
        "discard_reason": None,
    }


class R13DemoStreamTests(unittest.TestCase):
    def test_write_bundle_emits_episode_pkls_and_stream_demo(self):
        ep0, ep1 = _episode_7d(), _episode_7d()
        with tempfile.TemporaryDirectory() as tmp:
            bundle = pathlib.Path(tmp) / "b"
            write_success_bundle(
                bundle,
                bundle_name="wa2_bottle_pick_20_success_7d",
                episodes=[ep0, ep1],
                sidecars=[_sidecar(0), _sidecar(1)],
            )
            manifest = load_bundle(bundle)["manifest"]
            self.assertEqual(manifest["demo_pkl_format"], DEMO_PKL_FORMAT_STREAM)
            self.assertEqual(len(manifest["episode_pkls"]), 2)
            self.assertTrue((bundle / "episodes" / "ep000.pkl").is_file())
            self.assertTrue((bundle / "episodes" / "ep001.pkl").is_file())
            paths = require_episode_pkls(bundle)
            self.assertEqual(len(paths), 2)
            n0 = len(load_transitions(paths[0]))
            n1 = len(load_transitions(paths[1]))
            flat = load_transitions(bundle / "demo.pkl")
            self.assertEqual(len(flat), n0 + n1)

    def test_split_legacy_flat_pkl(self):
        ep0, ep1 = _episode_7d(), _episode_7d()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "legacy"
            (root / "episodes").mkdir(parents=True)
            dump_transitions(root / "demo.pkl", ep0 + ep1)
            dump_json(
                root / "episodes" / "ep000.json",
                {**_sidecar(0, n_steps=len(ep0)), "n_grasp_plus": 1, "n_grasp_minus": 1},
            )
            dump_json(
                root / "episodes" / "ep001.json",
                {**_sidecar(1, n_steps=len(ep1)), "n_grasp_plus": 1, "n_grasp_minus": 1},
            )
            dump_json(
                root / "bundle.json",
                {
                    "bundle_name": "legacy",
                    "pkl": "demo.pkl",
                    "n_episodes": 2,
                    "label": "success",
                    "episode_sidecars": [
                        "episodes/ep000.json",
                        "episodes/ep001.json",
                    ],
                },
            )
            with self.assertRaises(DemoIOError):
                require_episode_pkls(root)
            result = split_flat_demo_into_episode_pkls(root, progress=None)
            self.assertEqual(result["wrote"], 2)
            paths = require_episode_pkls(root)
            self.assertEqual(len(load_transitions(paths[0])), len(ep0))
            self.assertEqual(len(load_transitions(paths[1])), len(ep1))
            again = split_flat_demo_into_episode_pkls(root, progress=None)
            self.assertEqual(again["wrote"], 0)

    def test_legacy_single_pickle_still_loads(self):
        episode = _episode_7d()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "demo.pkl"
            with path.open("wb") as handle:
                pickle.dump(episode, handle, protocol=4)
            loaded = load_transitions(path)
            self.assertEqual(len(loaded), len(episode))


if __name__ == "__main__":
    unittest.main()
