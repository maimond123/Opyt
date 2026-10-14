"""
pipeline/worker/quota_loop/pool.py — the web quota's executor (`WebPool`): up to `slots` units at
once, at most one per host.

An `Executor`: it offers a source's ready candidates on free hosts (`units`), and runs the loop
(`execute`): admit, fetch on worker threads, settle on the owning thread. Selection, admission and
settlement stay the loop's, reached through `QuotaLoop`'s public surface.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import local

from pipeline import host_access

from .choose import Picked
from .loop import Fetched, QuotaLoop
from .retries import Wait
from .source import Context, PoolSource


def available(host, key, active):
    """At most one unit per known host, and no duplicate task key across sources.

    Keying on the unit key alone is deliberate: Archives and WebNewPosts offer the same
    archive-task keys, and the same task must never run twice.
    """
    return not any(k[1] == key[1] for k in active) and (host is None or all(h != host for h in active.values()))


def holds(pause: Wait | None, now: float) -> bool:
    """Does the pool's pause still hold? Until its `until`, or, with none, until a wake (as
    `Wait` says)."""
    return pause is not None and (pause.until is None or pause.until > now)


def may_admit(stopping, pause, now, in_flight, slots):
    """Admission is open when running, past any pause, and below the concurrency bound."""
    return not stopping and not holds(pause, now) and in_flight < slots


class WebPool:
    """The `Executor` that runs up to `slots` units at once, one per host."""

    def __init__(self, slots=4):
        self.slots = slots
        self._active = {}
        self._hosts = host_access.Hosts()
        self._local = local()

    def units(self, source: PoolSource, conn, now):
        return (unit for unit in source.candidates(conn, now)
                if available(unit.host, Picked(source, unit).key, self._active))

    def _fetch(self, loop: QuotaLoop, picked):
        """A fetch on a worker thread. Never raises: a connection that will not open is the
        fetch's error, settled with its unit like any other."""
        try:
            with loop.connection() as conn:
                if not hasattr(self._local, "ctx"):
                    self._local.ctx = Context(loop.quota, lease_lost=loop.ctx.lease_lost)
                with host_access.scope(self._hosts):
                    return loop.fetch_unit(conn, picked, self._local.ctx)
        except Exception as exc:
            return Fetched(error=exc)

    def execute(self, loop: QuotaLoop, conn):
        pending = {}
        pause = None
        with ThreadPoolExecutor(max_workers=self.slots, thread_name_prefix="web") as workers:
            while pending or not loop.stopping:
                if loop.take_wake() or not holds(pause, loop.clock()):
                    pause = None
                if (held := self._admit_free(loop, conn, workers, pending, pause)) is not None:
                    pause = held
                if not pending:
                    if loop.stopping:
                        break
                    out = pause or loop.record_idle(conn)
                    loop.sleep_until(out.until)
                    pause = None
                elif (settled := self._settle_finished(loop, conn, pending)) is not None:
                    pause = settled
                loop.beat_if_due()

    def _admit_free(self, loop: QuotaLoop, conn, workers, pending, pause) -> Wait | None:
        """Fill free slots while `may_admit` allows; the wait admission hit, if any, is the
        pool's new pause."""
        while may_admit(loop.stopping, pause, loop.clock(), len(pending), self.slots):
            try:
                picked = loop.admit(conn, loop.clock())
            except Exception as exc:
                return loop.record_backoff(conn, exc)
            if picked is None:
                break
            if isinstance(picked, Wait):
                return picked
            self._active[picked.key] = picked.unit.host
            pending[workers.submit(self._fetch, loop, picked)] = picked
        return None

    def _settle_finished(self, loop: QuotaLoop, conn, pending) -> Wait | None:
        """Settle the fetches that finished within a second; the last wait a settlement
        returned, if any, is the pool's new pause."""
        pause = None
        finished, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
        for future in finished:
            picked = pending.pop(future)
            result = loop.settle(conn, picked, future.result())
            self._active.pop(picked.key, None)
            if isinstance(result, Wait):
                pause = result
        if pending:
            loop.in_flight()
        return pause
