"""Unit tests for TransitionPipeline ordering."""

from __future__ import annotations

import pathlib
import sys
import time
import unittest

SRC_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_ROOT))

from hilserl_wa2.experiments.async_transition import TransitionPipeline  # noqa: E402


class TransitionPipelineTests(unittest.TestCase):
    def test_push_returns_previous_in_order(self):
        pipe = TransitionPipeline()
        seen = []

        def job(i):
            def _fn():
                time.sleep(0.01)
                seen.append(i)
                return i

            return _fn

        self.assertIsNone(pipe.push(job(1)))
        self.assertEqual(pipe.push(job(2)), 1)
        self.assertEqual(pipe.push(job(3)), 2)
        self.assertEqual(pipe.flush(), 3)
        self.assertEqual(seen, [1, 2, 3])
        pipe.close()


if __name__ == "__main__":
    unittest.main()
