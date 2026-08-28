"""R10 Actor recovery / session / source-tree unit tests (no ROS, no JAX)."""

from __future__ import annotations

import pathlib
import pickle
import sys
import tempfile
import unittest

import numpy as np
import yaml

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.r10_protocol import (  # noqa: E402
    AUTO_RECONNECT,
    PROTOCOL_VERSION,
    R10ProtocolError,
    R10SessionGuard,
    assert_network_endpoints,
    assert_remote_learner_ip,
    build_handshake_request,
    compare_handshake,
    confirm_r10_server_status,
    count_intervention_segments,
    load_network_config,
    make_session_id,
    normalize_network_fault,
    ordered_transition_digest,
    source_tree_manifest,
    transition_sha256,
)


def _network_yaml(learner_ip="192.168.1.20", actor_ip="192.168.1.30") -> str:
    return yaml.safe_dump(
        {
            "schema_version": 1,
            "profile_id": "lab_lan",
            "learner_ip": learner_ip,
            "actor_ip": actor_ip,
            "request_port": 5588,
            "broadcast_port": 5589,
            "request_timeout_ms": 3000,
            "network_max_age_s": 5.0,
            "upload_every_steps": 10,
            "allowed_actor_cidr": f"{actor_ip}/32",
        }
    )


def _manifest() -> dict:
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


class DummyStatusClient:
    def __init__(self, payload, success=True):
        self.payload = payload
        self.success = success
        self.calls = 0

    def request(self, kind, body):
        self.calls += 1
        if kind != "r10-status":
            return {"success": False, "message": kind}
        return {"success": self.success, "payload": dict(self.payload)}


class R10RecoveryTests(unittest.TestCase):
    def test_auto_reconnect_frozen_false(self):
        self.assertFalse(AUTO_RECONNECT)
        guard = R10SessionGuard(session_id="s1")
        self.assertFalse(guard.auto_reconnect)

    def test_loopback_server_ip_rejected(self):
        for ip in ("127.0.0.1", "localhost", "0.0.0.0", "127.0.0.2", "172.17.0.1"):
            with self.subTest(ip=ip):
                with self.assertRaises(R10ProtocolError):
                    assert_remote_learner_ip(ip)

    def test_server_ip_must_match_profile_and_differ_from_actor(self):
        cfg = yaml.safe_load(_network_yaml())
        self.assertEqual(assert_network_endpoints("192.168.1.20", cfg), "192.168.1.20")
        with self.assertRaises(R10ProtocolError):
            assert_network_endpoints("192.168.1.99", cfg)
        with self.assertRaises(R10ProtocolError):
            assert_network_endpoints("192.168.1.30", {"learner_ip": "192.168.1.30", "actor_ip": "192.168.1.30"})

    def test_handshake_mismatch_blocks_env_steps(self):
        guard = R10SessionGuard(session_id="sess-old")
        expected = _manifest()
        received = dict(expected, session_id="sess-old", space_hash="wrong")
        result = compare_handshake(expected, received)
        self.assertFalse(result["accepted"])
        guard.note_handshake(False)
        self.assertEqual(guard.env_steps, 0)
        with self.assertRaises(R10ProtocolError) as ctx:
            guard.note_env_step()
        self.assertIn("ENV_STEPS=0", str(ctx.exception))
        self.assertFalse(guard.can_step())

    def test_handshake_accept_then_same_session_steps(self):
        guard = R10SessionGuard(session_id="sess-ok")
        expected = _manifest()
        received = dict(expected, session_id="sess-ok")
        self.assertTrue(compare_handshake(expected, received)["accepted"])
        guard.note_handshake(True, server_instance_id="srv-1")
        self.assertEqual(guard.note_env_step(), 1)
        self.assertEqual(guard.note_env_step(), 2)
        self.assertEqual(guard.env_steps, 2)

    def test_server_restart_invalidates_old_session(self):
        guard = R10SessionGuard(session_id="sess-1")
        guard.note_handshake(True, server_instance_id="instance-a")
        guard.note_env_step()
        guard.note_server_instance("instance-b")
        self.assertTrue(guard.invalidated)
        self.assertEqual(guard.fault_reason, "network_loss")
        self.assertEqual(guard.fault_detail, "server_disconnect")
        with self.assertRaises(R10ProtocolError):
            guard.note_env_step()

    def test_new_session_starts_from_zero_without_old_dump(self):
        old = R10SessionGuard(session_id="old")
        old.note_handshake(True, "srv-old")
        old.note_env_step()
        new = R10SessionGuard(session_id="new")
        new.note_handshake(True, "srv-new")
        self.assertEqual(new.env_steps, 0)
        self.assertNotEqual(old.session_id, new.session_id)
        with self.assertRaises(R10ProtocolError) as ctx:
            R10SessionGuard.refuse_old_dump_import(old.session_id, new.session_id)
        self.assertIn("does not auto-import", str(ctx.exception))

    def test_fault_dump_written_and_not_replayed(self):
        guard = R10SessionGuard(session_id="dump-1")
        guard.note_handshake(True, "srv")
        rows = [{"x": np.asarray([1], np.int32)}]
        guard.register_transport("actor_env", 0, transition_sha256(rows[0]))
        guard.trigger_network_loss("server_disconnect")
        with tempfile.TemporaryDirectory() as tmp:
            path = guard.write_fault_dump(tmp, rows)
            self.assertTrue(guard.dump_written)
            dumped = pickle.loads((path / "fault_dump.pkl").read_bytes())
            self.assertEqual(dumped["session_id"], "dump-1")
            self.assertEqual(dumped["auto_reconnect"], False)
            self.assertEqual(len(dumped["ledger"]), 1)
            fresh = R10SessionGuard(session_id="dump-2")
            fresh.note_handshake(True, "srv-2")
            self.assertEqual(fresh.env_steps, 0)
            with self.assertRaises(R10ProtocolError):
                fresh.refuse_old_dump_import(dumped["session_id"], fresh.session_id)

    def test_transport_ids_are_session_stream_sequence_not_content_hash(self):
        guard = R10SessionGuard(session_id="sid")
        payload = {"actions": np.zeros(6, np.float32)}
        digest = transition_sha256(payload)
        rec_a = guard.register_transport("actor_env", 0, digest)
        rec_b = guard.register_transport("actor_env", 1, digest)
        self.assertEqual(rec_a["content_sha256"], rec_b["content_sha256"])
        self.assertNotEqual(
            (rec_a["session_id"], rec_a["stream"], rec_a["sequence_id"]),
            (rec_b["session_id"], rec_b["stream"], rec_b["sequence_id"]),
        )
        self.assertEqual(rec_b["sequence_id"], 1)

    def test_network_faults_normalize_to_network_loss(self):
        self.assertEqual(normalize_network_fault("server_disconnect"), ("network_loss", "server_disconnect"))
        self.assertEqual(normalize_network_fault("network_stale"), ("network_loss", "network_stale"))
        reason, detail = R10SessionGuard("s").trigger_network_loss("network_stale")
        self.assertEqual(reason, "network_loss")
        self.assertEqual(detail, "network_stale")

    def test_same_session_counts_and_last_id(self):
        rows = [{"i": np.asarray([n], np.int32)} for n in range(5)]
        digest = ordered_transition_digest(rows)
        client = DummyStatusClient(
            {
                "actor_env_count": 5,
                "actor_env_intvn_count": 0,
                "last_update_id": {"actor_env": 4, "actor_env_intvn": -1},
                "ordered_digest": digest,
                "ordered_intvn_digest": ordered_transition_digest([]),
                "schema_ok": True,
                "server_instance_id": "srv",
            }
        )
        ok, report = confirm_r10_server_status(
            client,
            local_env=5,
            local_intvn=0,
            local_env_digest=digest,
            local_intvn_digest=ordered_transition_digest([]),
            client_env_id=4,
            client_intvn_id=-1,
        )
        self.assertTrue(ok)
        self.assertTrue(report["last_update_id_match"])
        self.assertTrue(report["ordered_digest_match"])

    def test_status_none_is_disconnect(self):
        class Dead:
            def request(self, kind, body):
                return None

        ok, report = confirm_r10_server_status(
            Dead(),
            local_env=1,
            local_intvn=0,
            local_env_digest="x",
            local_intvn_digest="y",
        )
        self.assertFalse(ok)
        self.assertIn("None", report["error"])

    def test_data_before_handshake_schema_rejected(self):
        client = DummyStatusClient(
            {
                "actor_env_count": 1,
                "actor_env_intvn_count": 0,
                "last_update_id": {"actor_env": 0, "actor_env_intvn": -1},
                "schema_ok": False,
            }
        )
        ok, report = confirm_r10_server_status(
            client,
            local_env=1,
            local_intvn=0,
            local_env_digest="x",
            local_intvn_digest="y",
            client_env_id=0,
        )
        self.assertFalse(ok)
        self.assertIn("schema_ok", report["error"])

    def test_intervention_segments(self):
        flags = [False] * 1000 + [True] * 20
        self.assertEqual(count_intervention_segments(flags), 1)
        self.assertEqual(count_intervention_segments([True, False, True]), 2)

    def test_source_tree_excludes_local_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            layout = [
                root / "src/hilserl_wa2/configs/network",
                root / "src/hilserl_wa2/configs/tasks",
                root / "src/hilserl_wa2/envs",
                root / "src/hilserl_wa2/experiments",
                root / "src/hil-serl-main/serl_launcher/serl_launcher",
                root / "src/hil-serl-main/examples/experiments/wa2",
            ]
            for path in layout:
                path.mkdir(parents=True, exist_ok=True)
            (root / "src/hil-serl-main/examples/experiments/config.py").write_text("X=1\n")
            (root / "src/hil-serl-main/examples/experiments/mappings.py").write_text("Y=1\n")
            (root / "src/hilserl_wa2/configs/tasks/bottle_pick.yaml").write_text("task: bottle_pick\n")
            (root / "src/hilserl_wa2/configs/network/local.yaml").write_text("learner_ip: 10.0.0.1\n")
            (root / "src/hilserl_wa2/configs/network/lab_local.yaml").write_text("actor_ip: 10.0.0.2\n")
            (root / "src/hilserl_wa2/experiments/r10_protocol.py").write_text("Z=1\n")
            rows, digest = source_tree_manifest(root)
            paths = [row["path"] for row in rows]
            self.assertTrue(digest)
            self.assertTrue(any(path.endswith("bottle_pick.yaml") for path in paths))
            self.assertFalse(any("local.yaml" in path or path.endswith("lab_local.yaml") for path in paths))

    def test_source_tree_missing_dir_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(R10ProtocolError):
                source_tree_manifest(tmp)

    def test_actor_repo_whitelist_excludes_local_if_present(self):
        rows, digest = source_tree_manifest(REPO_ROOT)
        paths = [row["path"] for row in rows]
        self.assertTrue(digest)
        self.assertTrue(any(path.endswith("r10_protocol.py") for path in paths))
        self.assertFalse(
            any(
                pathlib.Path(path).suffix in {".yaml", ".yml"} and "local" in pathlib.Path(path).name.lower()
                for path in paths
            )
        )

    def test_handshake_payload_requires_session(self):
        payload = build_handshake_request(_manifest(), "sess-1")
        self.assertEqual(payload["session_id"], "sess-1")
        self.assertEqual(payload["protocol_version"], PROTOCOL_VERSION)
        with self.assertRaises(R10ProtocolError):
            build_handshake_request(_manifest(), "  ")

    def test_make_session_id_unique(self):
        left = make_session_id()
        right = make_session_id()
        self.assertNotEqual(left, right)
        self.assertGreater(len(left), 8)

    def test_load_network_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "local.yaml"
            path.write_text(_network_yaml(), encoding="utf-8")
            cfg = load_network_config(path)
            self.assertEqual(cfg["request_port"], 5588)
            self.assertEqual(cfg["learner_ip"], "192.168.1.20")


if __name__ == "__main__":
    unittest.main()
