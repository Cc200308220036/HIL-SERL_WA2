"""Contract vs Gymnasium space consistency tests."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np
import yaml

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.envs.contracts import WA2EnvContract  # noqa: E402
from hilserl_wa2.envs.wa2_env import WA2Env  # noqa: E402

CONTRACT_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "configs" / "wa2_env_contract.yaml"
)


class WA2ContractTests(unittest.TestCase):
    def test_contract_loads_v011(self):
        contract = WA2EnvContract.from_yaml(CONTRACT_PATH)
        self.assertEqual(contract.version, "0.1.1")
        self.assertEqual(contract.arm, "left")
        self.assertEqual(contract.action_dim, 6)
        self.assertEqual(contract.max_steps, 400)
        self.assertEqual(contract.image_shape, (128, 128, 3))
        self.assertTrue(contract.wrist_enabled)
        self.assertEqual(contract.missing_policy, "zero_image")
        self.assertEqual(contract.position_frame, "base")
        self.assertEqual(contract.rotation_frame, "tool")
        self.assertIn("left_wrist", contract.wrist_topic)

    def test_spaces_match_yaml(self):
        raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
        env = WA2Env(fake_env=True, contract_path=CONTRACT_PATH, seed=0)
        self.assertEqual(env.action_space.shape, (raw["action"]["dim"],))
        self.assertEqual(env.action_space.dtype, np.float32)
        state = env.observation_space["state"]
        self.assertEqual(state["tcp_pose"].shape, (7,))
        self.assertEqual(state["tcp_vel"].shape, (6,))
        self.assertEqual(state["joint_pos"].shape, (8,))
        self.assertEqual(state["hand_joints"].shape, (6,))
        images = env.observation_space["images"]
        self.assertEqual(images["head"].shape, (128, 128, 3))
        self.assertEqual(images["wrist"].shape, (128, 128, 3))
        self.assertEqual(images["head"].dtype, np.uint8)
        env.close()


if __name__ == "__main__":
    unittest.main()
