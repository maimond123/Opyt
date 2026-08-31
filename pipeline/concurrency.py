"""
pipeline/concurrency.py — process-wide adaptive concurrency gates (ARC-1 Phase 2).

A resizable semaphore with an AIMD (additive-increase / multiplicative-decrease) policy. It caps
how many threads may be inside a provider's network call at once, climbing one step per success
and halving on a 429, so the ceiling is discovered empirically rather than guessed.

Stdlib-only, no pipeline imports: ingestion, processing, and kb layers each own a module-level
singleton gate per provider, so this module must sit below all of them with zero import back-edges.
"""
from __future__ import annotations

import threading


class AdaptiveSemaphore:
    """A concurrency gate whose limit self-tunes by AIMD. Use as a context manager around exactly
    one network call:

        with GATE:
            resp = do_the_call()
        GATE.record_success()          # on a clean 2xx
        # ... or, on a rate-limit:
        GATE.decrease()

    `__enter__` blocks while `in_use >= limit`; the permit is released on `__exit__` regardless of
    success/failure (the SLOT frees either way — only the LIMIT is policy). Acquiring the internal
    condition is brief (increment a counter, then release), so `record_success`/`decrease` — called
    by the same thread while it holds a permit but NOT the condition lock — never deadlock."""

    def __init__(self, start: int, *, min_permits: int = 2,
                 max_permits: int | None = None, increase_after: int | None = None):
        if start < 1:
            raise ValueError("start must be >= 1")
        self._cond = threading.Condition()
        self._limit = start
        self._in_use = 0
        # min can never exceed start, else decrease() would paradoxically RAISE the limit.
        self._min = max(1, min(min_permits, start))
        self._max = max(start, max_permits if max_permits is not None else start * 2)
        # Climb only after a full "wave" of clean calls, so one lucky success doesn't ratchet up.
        self._increase_after = max(1, increase_after if increase_after is not None else start)
        self._success_run = 0

    def __enter__(self) -> "AdaptiveSemaphore":
        with self._cond:
            while self._in_use >= self._limit:
                self._cond.wait()
            self._in_use += 1
        return self

    def __exit__(self, *exc) -> bool:
        with self._cond:
            self._in_use -= 1
            self._cond.notify()          # one freed slot → wake at most one waiter
        return False                     # never suppress the wrapped call's exception

    def record_success(self) -> None:
        """A clean call. After a full wave of them, additive-increase the limit (bounded by max)."""
        with self._cond:
            self._success_run += 1
            if self._success_run >= self._increase_after and self._limit < self._max:
                self._limit += 1
                self._success_run = 0
                self._cond.notify()      # the raised ceiling may admit one more waiter
    def decrease(self) -> None:
        """A rate-limit (429). Multiplicative-decrease: halve the limit (floored at min), reset the
        success run. No notify — LOWERING the limit never opens a slot for a waiter."""
        with self._cond:
            self._success_run = 0
            self._limit = max(self._min, self._limit // 2)

    @property
    def limit(self) -> int:
        with self._cond:
            return self._limit

    @property
    def in_use(self) -> int:
        with self._cond:
            return self._in_use
