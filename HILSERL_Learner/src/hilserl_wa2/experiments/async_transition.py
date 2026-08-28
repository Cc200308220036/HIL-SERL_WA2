"""Overlap transition build with the next Servo window (stutter Phase-0 follow-up).

Pattern::

    pipe = TransitionPipeline()
    ...
    nxt, reward, term, trunc, info = env.step(action)
    prev = pipe.push(lambda: build_actor_transition(obs, action, nxt, ...))
    if prev is not None:
        tr, meta = prev
        rows.append(tr)  # or route_transition(...)
    obs = nxt
    ...
    last = pipe.flush()
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class TransitionPipeline:
    """Single-worker FIFO: previous job runs during the current ``env.step``."""

    def __init__(self, *, thread_name: str = "wa2_tr_build") -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
        self._fut: Optional[Future[Any]] = None

    def push(self, fn: Callable[[], T]) -> Optional[T]:
        """Schedule ``fn``; wait for the previous job and return its result."""

        prev: Optional[T] = None
        if self._fut is not None:
            prev = self._fut.result()
        self._fut = self._pool.submit(fn)
        return prev

    def flush(self) -> Optional[Any]:
        """Wait for the outstanding job (end of episode / shutdown)."""

        if self._fut is None:
            return None
        out = self._fut.result()
        self._fut = None
        return out

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._pool.shutdown(wait=True)
