#!/usr/bin/env python3
"""Formal user-run GPU Gate for the hil-learner environment."""

from __future__ import annotations

import os

if os.environ.get("JAX_PLATFORMS", "").lower() == "cpu":
    raise SystemExit("GPU_GATE: FAIL — JAX_PLATFORMS=cpu is set")

import jax
import jax.numpy as jnp

devices = jax.devices()
print(f"JAX_VERSION={jax.__version__}")
print(f"JAX_DEVICES={devices}")
if not devices or not all(d.platform in ("gpu", "cuda") for d in devices):
    raise SystemExit(f"GPU_GATE: FAIL — expected only GPU devices, got {devices}")

x = jnp.ones((2048, 2048), dtype=jnp.float32)
y = jax.jit(lambda z: z @ z)(x).block_until_ready()
print(f"MATMUL_DEVICE={y.device}")
print(f"MATMUL_VALUE={float(y[0, 0])}")
if y.device.platform not in ("gpu", "cuda"):
    raise SystemExit(f"GPU_GATE: FAIL — result is on {y.device}")

import tensorflow as tf
import tensorflow_probability as tfp

print(f"TENSORFLOW={tf.__version__}")
print(f"TFP={tfp.__version__}")
print("HIL_LEARNER_GPU: PASS")

