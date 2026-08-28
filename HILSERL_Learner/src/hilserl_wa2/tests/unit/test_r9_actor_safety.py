"""R9 fail-closed / policy / upload watchdog unit tests (no ROS, no JAX)."""

from __future__ import annotations

import os
import pathlib
import sys
import unittest

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.actor_safety import (  # noqa: E402
    ActorSafetyError,
    FailClosedController,
    NetworkWatchdog,
    UploadWatchdog,
    assert_live_policy,
    assert_r4_confirm,
    confirm_server_counts,
)


class DummyEnv:
    def __init__(self):
        self.closed = False
        self._servo = None

    def close(self):
        self.closed = True


class DummyClient:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class ActorSafetyTests(unittest.TestCase):
    def test_finally_closes_env_and_client(self):
        env = DummyEnv()
        client = DummyClient()
        ctl = FailClosedController()
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            ctl.trigger("actor_exception")
        finally:
            result = ctl.shutdown(env, client)
        self.assertTrue(env.closed)
        self.assertTrue(client.stopped)
        self.assertTrue(result.env_closed)
        self.assertTrue(result.client_stopped)
        self.assertEqual(result.reason, "actor_exception")

    def test_upload_watchdog_fail_closed_on_first_failure(self):
        wd = UploadWatchdog(max_consecutive_failures=1)
        self.assertIsNone(wd.record(True))
        reason = wd.record(False)
        self.assertEqual(reason, "server_disconnect")
        self.assertEqual(wd.attempts, 2)

    def test_network_stale(self):
        wd = NetworkWatchdog(max_age_s=0.01, enabled=True)
        self.assertEqual(wd.check(), "network_stale")
        wd.note_update("abc")
        self.assertIsNone(wd.check())
        import time

        time.sleep(0.02)
        self.assertEqual(wd.check(), "network_stale")
        self.assertEqual(wd.update_count, 1)

    def test_live_nonzero_policy_rejected(self):
        with self.assertRaises(ActorSafetyError):
            assert_live_policy("live-zero", "sac")
        assert_live_policy("live-zero", "zero")
        assert_live_policy("fake", "sac")

    def test_live_requires_r4_confirm(self):
        env_bak = os.environ.get("R4_CONFIRM")
        try:
            os.environ.pop("R4_CONFIRM", None)
            with self.assertRaises(ActorSafetyError):
                assert_r4_confirm("live-zero")
            os.environ["R4_CONFIRM"] = "YES"
            assert_r4_confirm("live-zero")
            assert_r4_confirm("fake")
        finally:
            if env_bak is None:
                os.environ.pop("R4_CONFIRM", None)
            else:
                os.environ["R4_CONFIRM"] = env_bak

    def test_confirm_server_counts_mismatch(self):
        class Client:
            def request(self, typ, payload):
                return {
                    "success": True,
                    "payload": {
                        "actor_env_count": 9,
                        "actor_env_intvn_count": 2,
                        "last_update_id": {"actor_env": 8, "actor_env_intvn": 1},
                    },
                }

        ok, report = confirm_server_counts(
            Client(), local_env=10, local_intvn=2, client_env_id=9, client_intvn_id=1
        )
        self.assertFalse(ok)
        self.assertIn("mismatch", report["error"])

    def test_confirm_server_counts_match(self):
        class Client:
            def request(self, typ, payload):
                return {
                    "success": True,
                    "payload": {
                        "actor_env_count": 10,
                        "actor_env_intvn_count": 2,
                        "last_update_id": {"actor_env": 9, "actor_env_intvn": 1},
                    },
                }

        ok, report = confirm_server_counts(
            Client(), local_env=10, local_intvn=2, client_env_id=9, client_intvn_id=1
        )
        self.assertTrue(ok)
        self.assertTrue(report.get("last_update_id_match"))

    def test_confirm_status_none_is_failure(self):
        class Client:
            def request(self, typ, payload):
                return None

        ok, report = confirm_server_counts(Client(), local_env=1, local_intvn=0)
        self.assertFalse(ok)
        self.assertIn("None", report["error"])


if __name__ == "__main__":
    unittest.main()
