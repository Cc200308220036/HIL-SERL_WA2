from __future__ import annotations

import importlib.metadata
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hil-serl-main" / "examples"))
sys.path.insert(0, str(ROOT / "hil-serl-main" / "serl_launcher"))

from hilserl_wa2.experiments.env_factory import build_space_signature  # noqa: E402
from hilserl_wa2.experiments.r10_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    TRANSITION_SCHEMA_VERSION,
    network_config_hash,
    sha256_file,
    source_tree_manifest,
)
from hilserl_wa2.experiments.task_config import load_task  # noqa: E402
from hilserl_wa2.scripts.r10_learner_server import (  # noqa: E402
    _validate_runtime_manifest,
)


class LearnerPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.network = pathlib.Path(self.tempdir.name) / "local.yaml"

    def write_network(self, learner="192.168.10.20", actor="192.168.10.30", cidr=None):
        cidr = cidr or f"{actor}/32"
        self.network.write_text(
            "\n".join(
                (
                    "schema_version: 1",
                    "profile_id: test_lan",
                    f"learner_ip: {learner}",
                    f"actor_ip: {actor}",
                    "request_port: 5588",
                    "broadcast_port: 5589",
                    "request_timeout_ms: 3000",
                    "network_max_age_s: 5.0",
                    "upload_every_steps: 10",
                    f"allowed_actor_cidr: {cidr}",
                    "",
                )
            ),
            encoding="utf-8",
        )

    def manifest(self):
        task = load_task("bottle_pick")
        _, source_hash = source_tree_manifest(REPO)
        wheel = REPO / "artifacts/wheels/agentlace-0.1.3-py3-none-any.whl"
        return {
            "protocol_version": PROTOCOL_VERSION,
            "transition_schema_version": TRANSITION_SCHEMA_VERSION,
            "role": "learner",
            "task_id": task.task_id,
            "exp_name": task.exp_name,
            "config_bundle_hash": task.config_bundle_hash(),
            "network_config_hash": network_config_hash(self.network),
            "space_hash": build_space_signature(task, "learner")["space_hash"],
            "params_tree_signature": "08f79859e3ba2f5da5a9b0cb16e63bd10aa355878e3a0f724465b8850aae6920",
            "agentlace_version": importlib.metadata.version("agentlace"),
            "agentlace_wheel_sha256": sha256_file(wheel),
            "source_tree_sha256": source_hash,
        }

    def validate(self, manifest):
        _validate_runtime_manifest(
            manifest,
            task_id="bottle_pick",
            network_config=str(self.network),
            allow_placeholders=False,
        )

    def test_valid_manifest_and_network_pass(self):
        self.write_network()
        self.validate(self.manifest())

    def test_placeholder_network_is_rejected(self):
        self.write_network(learner="REPLACE_WITH_LEARNER_IPV4")
        with self.assertRaises(SystemExit):
            self.validate(self.manifest())

    def test_wrong_actor_cidr_is_rejected(self):
        self.write_network(cidr="192.168.10.99/32")
        with self.assertRaises(SystemExit):
            self.validate(self.manifest())

    def test_stale_source_manifest_is_rejected(self):
        self.write_network()
        manifest = self.manifest()
        manifest["source_tree_sha256"] = "0" * 64
        with self.assertRaises(SystemExit):
            self.validate(manifest)


if __name__ == "__main__":
    unittest.main()
