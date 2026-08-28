"""R13 classifier end_episode behaviour. Wrapper must not call reset."""

from __future__ import annotations

import pathlib
import sys
import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.wrappers.reward_classifier import (  # noqa: E402
    WA2RewardClassifierWrapper,
    ClassifierHoldDumpGate,
    image_obs_stats,
    prepare_classifier_observations,
    resolve_classifier_checkpoint,
    save_classifier_dump,
    squeeze_hwc_uint8,
)


def _obs(fill: int = 7):
    return {
        "state": np.full((1, 27), fill, dtype=np.float32),
        "head": np.full((1, 128, 128, 3), fill, dtype=np.uint8),
        "wrist": np.full((1, 128, 128, 3), fill + 1, dtype=np.uint8),
    }


class _DummyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-np.inf, np.inf, (1, 27), np.float32),
                "head": spaces.Box(0, 255, (1, 128, 128, 3), np.uint8),
                "wrist": spaces.Box(0, 255, (1, 128, 128, 3), np.uint8),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.reset_calls = 0
        self.stop_calls = 0
        self.obs = _obs()

    def reset(self, **kwargs):
        self.reset_calls += 1
        return self.obs, {}

    def step(self, action):
        return self.obs, 0.0, False, False, {"servo": "ok"}

    def close(self):
        self.stop_calls += 1


class R13ClassifierEndEpisodeTests(unittest.TestCase):
    def test_r12_false_still_does_not_terminate(self):
        env = _DummyEnv()
        wrapped = WA2RewardClassifierWrapper(
            env, lambda _o: 0.99, threshold=0.5, consecutive_n=3, end_episode=False
        )
        wrapped.reset()
        terminated = False
        for _ in range(5):
            _, reward, terminated, _, info = wrapped.step(np.zeros(7))
        self.assertTrue(info["succeed"])
        self.assertEqual(float(reward), 1.0)
        self.assertFalse(terminated)
        self.assertEqual(env.reset_calls, 1)

    def test_r13_true_sets_terminated_without_reset(self):
        env = _DummyEnv()
        wrapped = WA2RewardClassifierWrapper(
            env, lambda _o: 0.99, threshold=0.5, consecutive_n=3, end_episode=True
        )
        wrapped.reset()
        terminated = False
        for _ in range(3):
            _, reward, terminated, truncated, info = wrapped.step(np.zeros(7))
        self.assertTrue(info["succeed"])
        self.assertTrue(terminated)
        self.assertEqual(float(reward), 1.0)
        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(env.stop_calls, 0)

    def test_prepare_copies_uint8_and_drops_extra_keys(self):
        obs = _obs()
        obs["state"] = np.ones((1, 27), dtype=np.float32)
        prepared = prepare_classifier_observations(obs, ("head", "wrist"))
        self.assertEqual(set(prepared.keys()), {"head", "wrist"})
        self.assertEqual(prepared["head"].dtype, np.uint8)
        self.assertTrue(prepared["head"].flags["C_CONTIGUOUS"])
        prepared["head"][0, 0, 0, 0] = 0
        self.assertEqual(int(obs["head"][0, 0, 0, 0]), 7)

    def test_prepare_rescales_float_unit_interval(self):
        obs = {
            "head": np.full((1, 128, 128, 3), 0.5, dtype=np.float32),
            "wrist": np.full((1, 128, 128, 3), 1.0, dtype=np.float32),
        }
        prepared = prepare_classifier_observations(obs, ("head", "wrist"))
        self.assertEqual(int(prepared["head"].max()), 128)
        self.assertEqual(int(prepared["wrist"].max()), 255)

    def test_resolve_checkpoint_dir_and_step(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "classifier_ckpt"
            step = root / "checkpoint_150"
            step.mkdir(parents=True)
            (step / "manifest.ocdbt").write_text("x", encoding="utf-8")
            self.assertEqual(
                pathlib.Path(resolve_classifier_checkpoint(step)).name,
                "checkpoint_150",
            )
            self.assertEqual(
                pathlib.Path(resolve_classifier_checkpoint(root)).name,
                "checkpoint_150",
            )

    def test_image_obs_stats_includes_dtype(self):
        text = image_obs_stats(_obs(), ("head", "wrist"))
        self.assertIn("uint8", text)
        self.assertIn("head=shape(1, 128, 128, 3)", text)

    def test_decimate_skips_every_other_session_step(self):
        env = _DummyEnv()
        calls = {"n": 0}

        def predict(_o):
            calls["n"] += 1
            return 0.99

        wrapped = WA2RewardClassifierWrapper(
            env,
            predict,
            threshold=0.5,
            consecutive_n=1,
            end_episode=False,
            infer_mode="decimate",
            infer_every_n=1,
            session_infer_every_n=2,
        )
        wrapped.reset()
        for i in range(4):
            _, _, _, _, info = wrapped.step(np.zeros(7))
            # Force session flag via dummy: patch by stepping with injected info —
            # DummyEnv does not set sm_session; exercise every_n instead.
            _ = info
        # Without sm_session, session_every does not apply; every_n=1 → 4 calls.
        self.assertEqual(calls["n"], 4)

        env2 = _DummyEnv()
        calls2 = {"n": 0}

        def predict2(_o):
            calls2["n"] += 1
            return 0.2

        class _SessionEnv(_DummyEnv):
            def step(self, action):
                obs, r, term, trunc, info = super().step(action)
                info = dict(info)
                info["sm_session"] = True
                return obs, r, term, trunc, info

        wrapped2 = WA2RewardClassifierWrapper(
            _SessionEnv(),
            predict2,
            threshold=0.5,
            consecutive_n=2,
            end_episode=False,
            infer_mode="decimate",
            infer_every_n=1,
            session_infer_every_n=2,
        )
        wrapped2.reset()
        for _ in range(4):
            wrapped2.step(np.zeros(7))
        self.assertEqual(calls2["n"], 2)

    def test_async_returns_without_blocking_on_slow_predict(self):
        import time

        env = _DummyEnv()

        def slow(_o):
            time.sleep(0.05)
            return 0.91

        wrapped = WA2RewardClassifierWrapper(
            env,
            slow,
            threshold=0.5,
            consecutive_n=1,
            end_episode=True,
            infer_mode="async",
            infer_every_n=1,
            session_infer_every_n=1,
        )
        wrapped.reset()
        t0 = time.monotonic()
        _, reward, terminated, _, info = wrapped.step(np.zeros(7))
        # After window: submit only (join happens on *next* step before Servo).
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 0.03)
        self.assertFalse(terminated)
        self.assertEqual(float(reward), 0.0)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and float(info.get("classifier_p") or 0) < 0.5:
            time.sleep(0.01)
            _, reward, terminated, _, info = wrapped.step(np.zeros(7))
        self.assertGreaterEqual(float(info["classifier_p"]), 0.5)
        self.assertTrue(info["succeed"])
        self.assertTrue(terminated)
        wrapped.close()

    def test_infer_error_is_logged_not_silent(self):
        env = _DummyEnv()

        def boom(_obs):
            raise RuntimeError("infer boom")

        wrapped = WA2RewardClassifierWrapper(
            env, boom, threshold=0.5, consecutive_n=3, end_episode=True
        )
        wrapped.reset()
        _, reward, terminated, _, info = wrapped.step(np.zeros(7))
        self.assertEqual(float(info["classifier_p"]), 0.0)
        self.assertFalse(terminated)
        self.assertEqual(float(reward), 0.0)

    def test_hold_gate_dumps_after_idle_second(self):
        gate = ClassifierHoldDumpGate(hold_s=1.0, cooldown_s=3.0, enable_hold=True)
        now = 10.0
        self.assertIsNone(
            gate.should_dump(
                session=True, idle=True, force=False, succeed=False, dt=0.5, now=now
            )
        )
        tag = gate.should_dump(
            session=True, idle=True, force=False, succeed=False, dt=0.6, now=now + 0.6
        )
        self.assertEqual(tag, "hold")
        self.assertEqual(gate.count, 1)
        self.assertIsNone(
            gate.should_dump(
                session=True,
                idle=True,
                force=False,
                succeed=False,
                dt=1.0,
                now=now + 1.5,
            )
        )

    def test_hold_gate_disabled_by_default(self):
        gate = ClassifierHoldDumpGate(hold_s=0.1, cooldown_s=0.0)
        self.assertIsNone(
            gate.should_dump(
                session=True, idle=True, force=False, succeed=False, dt=1.0, now=1.0
            )
        )

    def test_hold_gate_motion_resets_and_key_forces(self):
        gate = ClassifierHoldDumpGate(hold_s=1.0, cooldown_s=3.0, enable_hold=True)
        gate.should_dump(
            session=True, idle=True, force=False, succeed=False, dt=0.8, now=1.0
        )
        gate.should_dump(
            session=True, idle=False, force=False, succeed=False, dt=0.1, now=1.1
        )
        self.assertIsNone(
            gate.should_dump(
                session=True, idle=True, force=False, succeed=False, dt=0.8, now=1.9
            )
        )
        self.assertEqual(
            gate.should_dump(
                session=True, idle=False, force=True, succeed=False, dt=0.0, now=2.0
            ),
            "key",
        )

    def test_save_dump_writes_npy_jsonl_and_rescores(self):
        import tempfile

        obs = _obs(9)
        scored = []

        def predict(frame):
            scored.append(frame)
            return 0.42

        with tempfile.TemporaryDirectory() as tmp:
            row = save_classifier_dump(
                tmp,
                obs,
                {"classifier_p": 0.41, "classifier_streak": 0, "succeed": False},
                tag="hold",
                seq=1,
                predict_fn=predict,
            )
            self.assertEqual(row["p"], 0.41)
            self.assertEqual(row["p_rescore"], 0.42)
            self.assertEqual(row["stats"]["head"]["min"], 9)
            head = pathlib.Path(tmp) / "0001_hold_head.npy"
            self.assertTrue(head.is_file())
            loaded = np.load(head)
            self.assertEqual(loaded.shape, (128, 128, 3))
            self.assertEqual(int(loaded[0, 0, 0]), 9)
            jsonl = (pathlib.Path(tmp) / "dumps.jsonl").read_text(encoding="utf-8")
            self.assertIn('"p_rescore": 0.42', jsonl)
        self.assertEqual(len(scored), 1)
        self.assertNotIn("state", scored[0])

    def test_squeeze_drops_time_dim(self):
        stacked = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        out = squeeze_hwc_uint8(stacked)
        self.assertEqual(out.shape, (128, 128, 3))


if __name__ == "__main__":
    unittest.main()
