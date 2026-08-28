"""Unit tests for R8 task YAML loading (no ROS / no JAX)."""

from __future__ import annotations

import copy
import pathlib
import sys
import tempfile
import unittest

import yaml

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.task_config import (  # noqa: E402
    TaskConfigError,
    discover_task_ids,
    exp_name_for_task,
    load_task,
    sanitize_task_id,
    task_id_from_exp_name,
)

TASKS = pathlib.Path(__file__).resolve().parents[2] / "configs" / "tasks"


class TaskConfigTests(unittest.TestCase):
    def test_bottle_pick_loads(self):
        cfg = load_task("bottle_pick")
        self.assertEqual(cfg.task_id, "bottle_pick")
        self.assertEqual(cfg.exp_name, "wa2_bottle_pick")
        self.assertEqual(cfg.scene, "bottle_desktop")
        self.assertEqual(cfg.action_mode, "left_arm_6d")
        self.assertEqual(cfg.image_keys, ("head", "wrist"))
        self.assertEqual(cfg.proprio_dim, 27)
        self.assertEqual(cfg.classifier_keys, ("head", "wrist"))
        self.assertEqual(cfg.setup_mode, "single-arm-learned-gripper")
        self.assertTrue(cfg.contract_path.is_file())
        self.assertEqual(len(cfg.config_bundle_hash()), 64)

    def test_discover_includes_both_tasks(self):
        ids = discover_task_ids()
        self.assertIn("bottle_pick", ids)
        self.assertIn("r8_mock_alt", ids)
        self.assertNotIn("_TEMPLATE", ids)

    def test_alt_bundle_hash_differs_space_fields_match(self):
        a = load_task("bottle_pick")
        b = load_task("r8_mock_alt")
        self.assertEqual(a.image_keys, b.image_keys)
        self.assertEqual(a.proprio_keys, b.proprio_keys)
        self.assertEqual(a.obs_horizon, b.obs_horizon)
        self.assertNotEqual(a.config_bundle_hash(), b.config_bundle_hash())
        self.assertNotEqual(a.file_hashes()["task_hash"], b.file_hashes()["task_hash"])

    def test_reject_path_traversal(self):
        with self.assertRaises(TaskConfigError):
            sanitize_task_id("../bad")
        with self.assertRaises(TaskConfigError):
            load_task("../bad")

    def test_exp_name_roundtrip(self):
        self.assertEqual(task_id_from_exp_name("wa2_bottle_pick"), "bottle_pick")
        self.assertEqual(exp_name_for_task("r8_mock_alt"), "wa2_r8_mock_alt")
        with self.assertRaises(TaskConfigError):
            task_id_from_exp_name("wa2")

    def test_task_id_must_match_filename(self):
        raw = yaml.safe_load((TASKS / "bottle_pick.yaml").read_text(encoding="utf-8"))
        raw["task_id"] = "other"
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bottle_pick.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(TaskConfigError):
                load_task("bottle_pick", tasks_dir=tmp)

    def test_unknown_key_rejected(self):
        raw = yaml.safe_load((TASKS / "bottle_pick.yaml").read_text(encoding="utf-8"))
        raw["not_a_field"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bottle_pick.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(TaskConfigError):
                load_task("bottle_pick", tasks_dir=tmp)

    def test_forbidden_serial_rejected(self):
        raw = yaml.safe_load((TASKS / "bottle_pick.yaml").read_text(encoding="utf-8"))
        raw["env"] = copy.deepcopy(raw["env"])
        raw["env"]["serial_number"] = "123"
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bottle_pick.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(TaskConfigError):
                load_task("bottle_pick", tasks_dir=tmp)

    def test_7d_action_rejected(self):
        raw = yaml.safe_load((TASKS / "bottle_pick.yaml").read_text(encoding="utf-8"))
        raw["action"]["mode"] = "left_arm_7d"
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bottle_pick.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(TaskConfigError):
                load_task("bottle_pick", tasks_dir=tmp)

    def test_alt_classifier_keys_remain_null(self):
        self.assertIsNone(load_task("r8_mock_alt").classifier_keys)

    def test_classifier_keys_must_be_image_subset(self):
        raw = yaml.safe_load((TASKS / "bottle_pick.yaml").read_text(encoding="utf-8"))
        raw["reward"]["classifier_keys"] = ["head", "side"]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "bottle_pick.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(TaskConfigError):
                load_task("bottle_pick", tasks_dir=tmp)


if __name__ == "__main__":
    unittest.main()
