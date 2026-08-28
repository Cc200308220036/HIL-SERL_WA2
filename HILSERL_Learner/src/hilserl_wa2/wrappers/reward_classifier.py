"""R12 reward classifier wrapper. Fail-closed: never ends the episode."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import gymnasium as gym
import numpy as np


def sigmoid_scalar(logit: Any) -> float:
    value = float(np.asarray(logit).reshape(-1)[0])
    if value >= 0.0:
        exp_n = math.exp(-value)
        return 1.0 / (1.0 + exp_n)
    exp_p = math.exp(value)
    return exp_p / (1.0 + exp_p)


def load_threshold_json(path: Union[str, Path]) -> Dict[str, Any]:
    import json

    src = Path(path).expanduser().resolve()
    with src.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "threshold" not in payload:
        raise ValueError(f"threshold.json missing threshold: {src}")
    return payload


def resolve_classifier_checkpoint(path: Union[str, Path]) -> str:
    """Accept ``classifier_ckpt`` or ``classifier_ckpt/checkpoint_N``."""

    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"classifier checkpoint not found: {src}")
    if src.is_file():
        return str(src)
    if (src / "manifest.ocdbt").is_file() or (src / "_METADATA").is_file():
        return str(src)
    numbered = sorted(
        child
        for child in src.iterdir()
        if child.is_dir() and child.name.startswith("checkpoint_")
    )
    if numbered:
        return str(numbered[-1])
    return str(src)


def prepare_classifier_observations(
    obs: Dict[str, Any],
    image_keys: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Copy only image keys as contiguous uint8 (training distribution)."""

    prepared: Dict[str, np.ndarray] = {}
    for key in image_keys:
        if key not in obs:
            raise KeyError(f"classifier obs missing {key!r}")
        arr = np.asarray(obs[key])
        if np.issubdtype(arr.dtype, np.floating):
            max_v = float(np.nanmax(arr)) if arr.size else 0.0
            if max_v <= 1.5:
                arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
            else:
                arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
        else:
            arr = np.array(arr, dtype=np.uint8, copy=True)
        prepared[str(key)] = np.ascontiguousarray(arr)
    return prepared


def image_obs_stats(obs: Dict[str, Any], image_keys: Sequence[str]) -> str:
    parts = []
    for key in image_keys:
        if key not in obs:
            parts.append(f"{key}=missing")
            continue
        arr = np.asarray(obs[key])
        if arr.size == 0:
            parts.append(f"{key}=empty")
            continue
        parts.append(
            f"{key}=shape{tuple(arr.shape)}/{arr.dtype}/"
            f"min={float(arr.min()):.1f}/max={float(arr.max()):.1f}/"
            f"mean={float(arr.mean()):.1f}"
        )
    return " ".join(parts)


def squeeze_hwc_uint8(image: Any) -> np.ndarray:
    """Drop a leading T=1 stack dim; return contiguous HxWxC uint8 RGB."""

    arr = np.asarray(image)
    if arr.ndim == 4 and int(arr.shape[0]) == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected HWC or (1,H,W,C), got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        max_v = float(np.nanmax(arr)) if arr.size else 0.0
        if max_v <= 1.5:
            arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
        else:
            arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    else:
        arr = np.asarray(arr, dtype=np.uint8)
    return np.ascontiguousarray(arr)


def _write_rgb_png(path: Path, hwc: np.ndarray) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
    return bool(cv2.imwrite(str(path), bgr))


class ClassifierHoldDumpGate:
    """Optional dump when the operator holds still in an intervention session.

    Training default should keep ``enable_hold=False``: idle during teleop is
    normal and hold dumps (disk + JAX rescore) can stall Servo → window timeout.
    """

    def __init__(
        self,
        *,
        hold_s: float = 1.0,
        cooldown_s: float = 3.0,
        max_dumps: int = 80,
        enable_hold: bool = False,
    ) -> None:
        self.hold_s = float(hold_s)
        self.cooldown_s = float(cooldown_s)
        self.max_dumps = int(max_dumps)
        self.enable_hold = bool(enable_hold)
        self.count = 0
        self._hold_acc = 0.0
        self._last_dump = -1.0e9

    def should_dump(
        self,
        *,
        session: bool,
        idle: bool,
        force: bool,
        succeed: bool,
        dt: float,
        now: float,
    ) -> Optional[str]:
        if self.count >= self.max_dumps:
            return None
        if force:
            self._last_dump = float(now)
            self._hold_acc = 0.0
            self.count += 1
            return "key"
        if bool(succeed) and (float(now) - self._last_dump) >= self.cooldown_s:
            self._last_dump = float(now)
            self.count += 1
            return "succeed"
        if self.enable_hold and session and idle:
            self._hold_acc += max(0.0, float(dt))
            if (
                self._hold_acc >= self.hold_s
                and (float(now) - self._last_dump) >= self.cooldown_s
            ):
                self._last_dump = float(now)
                self._hold_acc = 0.0
                self.count += 1
                return "hold"
            return None
        self._hold_acc = 0.0
        return None


def save_classifier_dump(
    out_dir: Union[str, Path],
    obs: Dict[str, Any],
    info: Dict[str, Any],
    *,
    tag: str,
    seq: int,
    predict_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    image_keys: Sequence[str] = ("head", "wrist"),
) -> Dict[str, Any]:
    """Write live RGB + jsonl. Rescore with the same predict_fn used online."""

    dest = Path(out_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    keys = list(image_keys)
    prepared = prepare_classifier_observations(obs, keys)
    p_live = float(info.get("classifier_p") or 0.0)
    p_rescore = float("nan")
    if predict_fn is not None:
        p_rescore = float(predict_fn(prepared))
    files: Dict[str, str] = {}
    stats: Dict[str, Any] = {}
    prefix = f"{int(seq):04d}_{tag}"
    for key in keys:
        hwc = squeeze_hwc_uint8(prepared[key])
        stats[key] = {
            "shape": list(hwc.shape),
            "dtype": str(hwc.dtype),
            "min": int(hwc.min()) if hwc.size else 0,
            "max": int(hwc.max()) if hwc.size else 0,
            "mean": float(hwc.mean()) if hwc.size else 0.0,
        }
        npy_path = dest / f"{prefix}_{key}.npy"
        np.save(npy_path, hwc)
        files[f"{key}_npy"] = str(npy_path)
        png_path = dest / f"{prefix}_{key}.png"
        if _write_rgb_png(png_path, hwc):
            files[f"{key}_png"] = str(png_path)
    row = {
        "seq": int(seq),
        "tag": str(tag),
        "p": p_live,
        "p_rescore": p_rescore,
        "streak": int(info.get("classifier_streak") or 0),
        "succeed": bool(info.get("succeed")),
        "sm_session": bool(info.get("sm_session")),
        "sm_intent": str(info.get("sm_intent") or ""),
        "step": int(info.get("step_count") or 0),
        "stats": stats,
        "files": files,
    }
    jsonl = dest / "dumps.jsonl"
    with jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        f"CLASSIFIER_DUMP tag={tag} seq={seq} p={p_live:.4f} "
        f"p_rescore={p_rescore:.4f} dir={dest}",
        flush=True,
    )
    return row


def _dense1_digest(params: Any) -> str:
    node = params
    for key in ("Dense_1", "kernel"):
        try:
            node = node[key]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"classifier params missing {key}") from exc
    arr = np.asarray(node, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def load_reward_classifier_fn(
    checkpoint_path: Union[str, Path],
    sample_obs: Dict[str, Any],
    image_keys: Sequence[str],
) -> Callable[[Dict[str, Any]], float]:
    """Restore flax ckpt and return obs -> probability in [0, 1]."""

    # Must be set before importing jax (Orin XLA GPU autotune crash).
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")

    import jax
    from flax.training import checkpoints
    from serl_launcher.networks.reward_classifier import create_classifier

    ckpt = resolve_classifier_checkpoint(checkpoint_path)
    keys = list(image_keys)
    init_sample = prepare_classifier_observations(sample_obs, keys)
    rng = jax.random.PRNGKey(0)
    classifier = create_classifier(rng, init_sample, keys, n_way=2)
    digest_before = _dense1_digest(classifier.params)
    classifier = checkpoints.restore_checkpoint(ckpt, target=classifier)
    digest_after = _dense1_digest(classifier.params)
    if digest_before == digest_after:
        raise RuntimeError(
            f"classifier restore did not load weights from {ckpt} "
            f"(Dense_1 digest still {digest_before})"
        )
    frozen = jax.tree_util.tree_map(
        lambda x: jax.device_put(np.array(jax.device_get(x), copy=True)),
        classifier.params,
    )
    apply = jax.jit(
        lambda obs: classifier.apply_fn({"params": frozen}, obs, train=False)
    )
    print(
        f"CLASSIFIER_RESTORE ok ckpt={ckpt} dense1={digest_before}->{digest_after}",
        flush=True,
    )

    def predict_prob(obs: Dict[str, Any]) -> float:
        prepared = prepare_classifier_observations(obs, keys)
        logits = apply(prepared)
        return sigmoid_scalar(logits)

    return predict_prob


class WA2RewardClassifierWrapper(gym.Wrapper):
    """Annotate info/reward. Must not set terminated or call reset/stop.

    ``infer_mode`` (stutter Phase-0 follow-up):
      - ``sync``: every selected step blocks on ``predict_fn`` (legacy).
      - ``decimate``: only every Nth high-level step runs inference; others
        reuse last ``p`` and freeze streak (no false streak inflation).
      - ``async``: run ``predict_fn`` on a worker thread **after** the Servo
        window returns, but **always join before the next window starts**.
        JAX must not overlap Servo on Orin (GIL/GPU → ``action_window_timeout``).
        Latency win is only overlap with non-servo work (e.g. transition
        pipeline); prefer ``decimate`` for demo/HIL stutter.

    Session stride: when ``sm_session`` and ``session_infer_every_n`` > 1,
    further skip submits (works for all modes).
    """

    def __init__(
        self,
        env: gym.Env,
        predict_fn: Callable[[Dict[str, Any]], float],
        *,
        threshold: float,
        consecutive_n: int = 3,
        end_episode: bool = False,
        image_keys: Sequence[str] = ("head", "wrist"),
        infer_mode: str = "sync",
        infer_every_n: int = 1,
        session_infer_every_n: int = 1,
    ) -> None:
        super().__init__(env)
        if consecutive_n < 1:
            raise ValueError("consecutive_n must be >= 1")
        mode = str(infer_mode or "sync").strip().lower()
        if mode not in ("sync", "decimate", "async"):
            raise ValueError(f"infer_mode must be sync|decimate|async, got {mode!r}")
        if int(infer_every_n) < 1 or int(session_infer_every_n) < 1:
            raise ValueError("infer_every_n / session_infer_every_n must be >= 1")
        self.predict_fn = predict_fn
        self.threshold = float(threshold)
        self.consecutive_n = int(consecutive_n)
        self.end_episode = bool(end_episode)
        self.image_keys = tuple(image_keys)
        self.infer_mode = mode
        self.infer_every_n = int(infer_every_n)
        self.session_infer_every_n = int(session_infer_every_n)
        self._streak = 0
        self._last_prob = 0.0
        self._step_i = 0
        self._session_i = 0
        self.reset_calls = 0
        self._infer_err_logged = False
        self._debug_left = 5
        self._async_q: Optional[queue.Queue] = None
        self._async_thread: Optional[threading.Thread] = None
        self._async_lock = threading.Lock()
        self._async_pending = 0
        self._stop_async = False
        if self.infer_mode == "async":
            self._async_q = queue.Queue(maxsize=1)
            self._async_idle = threading.Event()
            self._async_idle.set()
            self._async_thread = threading.Thread(
                target=self._async_worker,
                name="wa2_classifier_async",
                daemon=True,
            )
            self._async_thread.start()
            print(
                f"CLASSIFIER_INFER mode=async every_n={self.infer_every_n} "
                f"session_every_n={self.session_infer_every_n} "
                f"(serialized: join before Servo; no JAX∩window)",
                flush=True,
            )
        elif self.infer_mode == "decimate":
            print(
                f"CLASSIFIER_INFER mode=decimate every_n={self.infer_every_n} "
                f"session_every_n={self.session_infer_every_n}",
                flush=True,
            )
            self._async_idle = None
        else:
            self._async_idle = None

    def _async_worker(self) -> None:
        assert self._async_q is not None
        while True:
            item = self._async_q.get()
            if item is None:
                return
            prepared = item
            try:
                prob = float(self.predict_fn(prepared))
            except Exception as exc:  # noqa: BLE001
                if not self._infer_err_logged:
                    print(
                        f"CLASSIFIER_INFER_ERR {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    self._infer_err_logged = True
                prob = 0.0
            with self._async_lock:
                self._apply_prob_locked(prob)
                self._async_pending = max(0, self._async_pending - 1)
                if self._async_pending == 0 and self._async_idle is not None:
                    self._async_idle.set()

    def _wait_async_before_servo(self) -> None:
        """Do not start a Servo window while JAX infer is still running."""

        if self._async_idle is None:
            return
        if not self._async_idle.wait(timeout=2.0):
            print(
                "CLASSIFIER_ASYNC_WAIT_TIMEOUT — proceeding; Servo may fault",
                flush=True,
            )

    def _apply_prob_locked(self, prob: float) -> None:
        self._last_prob = float(prob)
        if self._last_prob >= self.threshold:
            self._streak += 1
        else:
            self._streak = 0

    def _should_run_infer(self, *, sm_session: bool) -> bool:
        self._step_i += 1
        if sm_session:
            self._session_i += 1
            if (self._session_i % self.session_infer_every_n) != 0:
                return False
        else:
            self._session_i = 0
        return (self._step_i % self.infer_every_n) == 0

    def _sync_infer(self, obs: Dict[str, Any]) -> None:
        try:
            prob = float(self.predict_fn(obs))
        except Exception as exc:  # noqa: BLE001
            if not self._infer_err_logged:
                print(f"CLASSIFIER_INFER_ERR {type(exc).__name__}: {exc}", flush=True)
                self._infer_err_logged = True
            prob = 0.0
        self._apply_prob_locked(prob)

    def _submit_async(self, obs: Dict[str, Any]) -> None:
        assert self._async_q is not None
        try:
            prepared = prepare_classifier_observations(obs, self.image_keys)
        except Exception as exc:  # noqa: BLE001
            if not self._infer_err_logged:
                print(
                    f"CLASSIFIER_PREPARE_ERR {type(exc).__name__}: {exc}",
                    flush=True,
                )
                self._infer_err_logged = True
            return
        try:
            with self._async_lock:
                if self._async_idle is not None:
                    self._async_idle.clear()
                self._async_pending += 1
            self._async_q.put_nowait(prepared)
        except queue.Full:
            with self._async_lock:
                self._async_pending = max(0, self._async_pending - 1)
                if self._async_pending == 0 and self._async_idle is not None:
                    self._async_idle.set()
            # Drop frame under load; keep last completed result.
            pass

    def reset(self, **kwargs):
        self._wait_async_before_servo()
        self.reset_calls += 1
        self._streak = 0
        self._last_prob = 0.0
        self._step_i = 0
        self._session_i = 0
        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        info["succeed"] = False
        info["classifier_p"] = 0.0
        info["classifier_streak"] = 0
        info["classifier_infer_mode"] = self.infer_mode
        info["classifier_skipped"] = False
        return obs, info

    def step(self, action):
        # Orin: concurrent JAX + Servo window caused action_window_timeout.
        self._wait_async_before_servo()
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        sm = bool(info.get("sm_session"))
        run = self._should_run_infer(sm_session=sm)
        skipped = False
        if run:
            if self.infer_mode == "async":
                self._submit_async(obs)
                skipped = False  # submitted; result may still be stale this step
            else:
                self._sync_infer(obs)
        else:
            skipped = True
        with self._async_lock:
            prob = float(self._last_prob)
            streak = int(self._streak)
        if self._debug_left > 0:
            print(
                f"CLASSIFIER_OBS p={prob:.4f} mode={self.infer_mode} "
                f"skip={int(skipped)} {image_obs_stats(obs, self.image_keys)}",
                flush=True,
            )
            self._debug_left -= 1
        succeed = streak >= self.consecutive_n
        info["succeed"] = bool(succeed)
        info["classifier_p"] = float(prob)
        info["classifier_streak"] = int(streak)
        info["classifier_infer_mode"] = self.infer_mode
        info["classifier_skipped"] = bool(skipped)
        new_reward = 1.0 if succeed else 0.0
        # R12: end_episode=False keeps terminated unchanged (no reset).
        # R13: end_episode=True ORs succeed into terminated; wrapper still
        # never calls reset/stop/unlock — Actor loop does the R5 reset.
        if self.end_episode and succeed:
            terminated = True
        return obs, float(new_reward), bool(terminated), bool(truncated), info

    def close(self):
        if self._async_q is not None and not self._stop_async:
            self._stop_async = True
            try:
                self._async_q.put_nowait(None)
            except queue.Full:
                try:
                    _ = self._async_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._async_q.put_nowait(None)
                except queue.Full:
                    pass
            if self._async_thread is not None:
                self._async_thread.join(timeout=2.0)
        return self.env.close()
