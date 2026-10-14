"""
pipeline/worker/quota_loop/process.py — the loops' process (`LoopProcess`): one loop thread per
quota, from the first request until all are idle.

One per home, launched by the rail worker as the `scrapers` job (`registry.SCRAPERS`) through
`child`. That job is `pipeline/background.py`'s `run`, which hands in oracles' loops and
curation's as one list.
Each loop runs on a thread of its own and holds its quota's lease (`QuotaLoop.run`). Around them
this process does five things:
  • wakes them when a tool asks for work (`request_now("scrapers")`);
  • runs the seeding its owner hands it (`seed`) when it starts and about hourly;
  • wakes idle consumers when another loop produces new work;
  • stops them cleanly on SIGTERM, the worker's restart;
  • exits once every loop has had nothing to run for about a minute, writing when it should
    next start.
It is a process of its own, not threads inside the worker, because the worker serves many homes
on hosted and `OPYT_HOME` is process-wide.

Waking, and the trap in it. The wake-up signal is this job's own `rail_jobs` row. A tool calls
`request_now("scrapers")`, which only ever moves `due_at` earlier (`activate` keeps the MIN). A
row whose `due_at` is already in the past therefore cannot show a request, so the row is kept
armed: `SAFETY_NET_S` ahead, written with a compare-and-set (`RailJobStore.rearm`), and written
again well before that instant passes, or a long run's row would fall into the past and swallow
every request again. This process reads the row about once a second. When `due_at` has moved, it
re-arms first and then wakes every loop. In that order, a request whose data committed before
the re-arm is visible to the loops it wakes, and one that lands after is a new move that the
next read sees.

Exiting. Once every loop has had nothing to run for `IDLE_EXIT_S`, and none means to try
again within `STAY_FOR_S` (a quota reset a minute away is not worth a new process), this process
writes its next start: the earliest moment any loop means to try again, capped at
`SAFETY_NET_S`. That write is the same compare-and-set, so a request that lands during the exit
is never overwritten: the write fails and the process carries on. The worker's `finish` keeps the
written start, being no later than the cadence.

Stopping. `child` turns SIGTERM into a `SystemExit` raised here, in the main thread. The
process makes its row due now, so the restarted worker brings it straight back; stops every
loop; and waits while each finishes its unit and releases its lease.

A loop's thread can end on its own: its setup failed (a locked store), or another process
holds its quota. It counts as idle, and is started again after `RESTART_AFTER_S`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from pipeline.ingestion.utils import log

from .loop import QuotaLoop
from .loop_state import Status

from ..failure_run import FailureRun
from ..job_queue import RailJobStore
from ..registry import QUARTER_HOURLY, SCRAPERS

SAFETY_NET_S = QUARTER_HOURLY  # the latest next start; also how far ahead the row is armed
REARM_MARGIN_S = 5 * 60.0    # re-arm once the armed instant is this close
POLL_S = 1.0                 # how often the row is read
IDLE_EXIT_S = 60.0           # all idle this long → exit
STAY_FOR_S = 120.0           # …unless a loop means to try again sooner than this
SEED_EVERY_S = 3600.0        # seeding, after the one at start
RESTART_AFTER_S = 60.0       # a loop whose thread ended is started again after this


def watch(due_at: float, armed: float | None, now: float) -> tuple[bool, bool]:
    """What one read of this job's row calls for: `(rearm, wake)`. A row that moved from what
    this process armed is a request: re-arm, then wake every loop. An unchanged row is
    re-armed, quietly, only once its armed instant is within `REARM_MARGIN_S`."""
    if due_at != armed:
        return True, True
    return armed - now <= REARM_MARGIN_S, False


def next_start(now: float, idle_since: float | None, tries: list[float], idle_exit_s: float,
               stay_for_s: float, safety_net: float) -> float | None:
    """When the next process should start, once this one may exit; None means stay. Every
    loop must have been idle for `idle_exit_s`, and none may mean to try again within
    `stay_for_s` (a quota reset a minute away is not worth a new process). The start is the
    earliest try, capped at `safety_net` from now."""
    if idle_since is None or now - idle_since < idle_exit_s:
        return None
    soonest = min(tries) if tries else None
    if soonest is not None and soonest - now < stay_for_s:
        return None
    return now + safety_net if soonest is None else min(soonest, now + safety_net)


def idle_since(woken_at: float,
               slots: list[tuple[bool, float | None, float | None]]) -> float | None:
    """Since when no loop has had anything to run, or None while one is running a unit. Each
    slot is `(alive, ended, since)`: whether its thread is alive, when it was first seen ended,
    and its loop's `status.since` (None while running). A wake counts as activity, so a loop
    woken a moment ago gets its look first; a dead thread counts from when it ended."""
    since = woken_at
    for alive, ended, loop_since in slots:
        if not alive:
            since = max(since, ended or since)
            continue
        if loop_since is None:
            return None
        since = max(since, loop_since)
    return since


def to_wake(statuses: list[Status]) -> list[int]:
    """New work wakes idle consumers, including ones with a known future due time."""
    return [i for i, status in enumerate(statuses) if status.state == "idle"]


@dataclass
class _Slot:
    """One loop's thread, and when it was first seen ended."""
    thread: threading.Thread | None = None
    ended: float | None = None

    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


class LoopProcess:
    """The process around the loops. `run()` is the whole job; the tests drive it with fake
    loops, a fake clock and a fake sleep.

    `seed` is run at the start and every `seed_every_s`. Completed units wake idle consumers
    when their writes may have left another lane work."""

    def __init__(self, loops: list[QuotaLoop], *, store: RailJobStore, home_id: str,
                 seed: Callable[[], None],
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, poll_s: float = POLL_S,
                 idle_exit_s: float = IDLE_EXIT_S, seed_every_s: float = SEED_EVERY_S,
                 restart_after_s: float = RESTART_AFTER_S) -> None:
        self.loops = list(loops)
        self.store = store
        self._row = (home_id, SCRAPERS)     # this job's own `rail_jobs` row
        self._seed = seed
        self._clock = clock
        self._sleep = sleep
        self.poll_s = poll_s
        self.idle_exit_s = idle_exit_s
        self.seed_every_s = seed_every_s
        self.restart_after_s = restart_after_s
        self._slots = [_Slot() for _ in self.loops]
        self._armed: float | None = None     # the `due_at` this process last wrote
        self._woken_at = 0.0
        self._seeded_at = 0.0
        self._failures = FailureRun()        # reading this job's row
        self._activity = tuple(s.completed_units for s in self.loops)

    def run(self) -> dict:
        status = "stopped"
        try:
            now = self._clock()
            self._woken_at = now
            self._reseed(now)
            self._rearm_and_wake(now)               # armed before any loop looks at the store
            for i in range(len(self.loops)):
                self._start(i)
            status = self._loop()
        except (KeyboardInterrupt, SystemExit):
            self._due_now()
        finally:
            self._halt()
        return {"status": status}

    def _loop(self) -> str:
        while True:
            self._sleep(self.poll_s)
            now = self._clock()
            if now - self._seeded_at >= self.seed_every_s:
                self._reseed(now)
                self._wake_all(now)
            self._rearm_and_wake(now)
            self._wake_consumers(now)
            # Exit before reviving: a loop whose quota another process holds dies again at
            # once, and reviving it first would hold this process open for good.
            start_at = self._next_start_at(now)
            if start_at is not None and self._write_next_start(start_at):
                return "ok"
            self._revive(now)

    def _wake_consumers(self, now: float) -> None:
        """A completed unit can leave work for another quota. Wake only idle consumers.

        Quota/reset and failure waits retain their deadlines; an idle source re-queries the store.
        """
        activity = tuple(s.completed_units for s in self.loops)
        if activity == self._activity:
            return
        self._activity = activity
        for i in to_wake([loop.status for loop in self.loops]):
            self._woken_at = now
            self.loops[i].wake()

    # ── the row ──────────────────────────────────────────────────────────────────
    def _rearm_and_wake(self, now: float) -> None:
        """Read this job's row, re-arm it when `watch` says, and wake every loop when it moved.
        Never raises: a locked control database is tried again on the next read."""
        try:
            job = self.store.get(*self._row)
            if job is None:
                return                     # run by hand: no row, so nothing can ask
            rearm, moved = watch(job.due_at, self._armed, now)
            if not rearm:
                self._recovered(now)
                return
            armed = now + SAFETY_NET_S
            if not self.store.rearm(*self._row, armed, expected=job.due_at):
                return                     # it moved again since the read; the next read sees it
            self._armed = armed
            self._recovered(now)
        except Exception as exc:
            if self._failures.failed(now):
                log(f"[scrapers] cannot read this job's row, retrying: {exc!r}")
            return
        if moved:
            self._wake_all(now)

    def _recovered(self, now: float) -> None:
        if self._failures.recovered(now) is not None:
            log("[scrapers] this job's row is readable again")

    def _due_now(self) -> None:
        """Ask for this job again at once, as the restart's last word. Never raises."""
        try:
            if self.store.get(*self._row) is not None:
                self.store.activate(*self._row, due_at=self._clock())
        except Exception as exc:
            log(f"[scrapers] could not ask to run again after the stop: {exc!r}")

    # ── exiting ──────────────────────────────────────────────────────────────────
    def _next_start_at(self, now: float) -> float | None:
        """Gather the loops' next tries and decide when the next process should start."""
        tries = [s.status.until for s, slot in zip(self.loops, self._slots)
                 if slot.alive() and s.status.until is not None]
        return next_start(now, self._idle_since(), tries, self.idle_exit_s, STAY_FOR_S,
                          SAFETY_NET_S)

    def _write_next_start(self, start_at: float) -> bool:
        """Write the decided start unless a new request has moved the row."""
        if self._armed is None:
            return True                    # no row to write
        try:
            if not self.store.rearm(*self._row, start_at, expected=self._armed):
                return False               # a request landed: carry on, the next read wakes
        except Exception as exc:
            log(f"[scrapers] could not write the next start, staying: {exc!r}")
            return False
        self._armed = start_at
        log(f"[scrapers] idle; next start in {start_at - self._clock():.0f}s")
        return True

    def _idle_since(self) -> float | None:
        """Gather each loop's thread and status, and decide with `idle_since`."""
        return idle_since(self._woken_at, [(slot.alive(), slot.ended, s.status.since)
                                           for s, slot in zip(self.loops, self._slots,
                                                              strict=True)])

    # ── the threads ──────────────────────────────────────────────────────────────
    def _start(self, i: int) -> None:
        s = self.loops[i]
        s.starting()
        t = threading.Thread(target=s.run, name=f"loop:{s.quota}", daemon=True)
        self._slots[i] = _Slot(t)
        t.start()

    def _revive(self, now: float) -> None:
        for i, slot in enumerate(self._slots):
            if slot.thread is None or slot.alive():
                continue
            if slot.ended is None:
                slot.ended = now
                log(f"[scrapers] {self.loops[i].quota} stopped on its own; trying again "
                    f"in {self.restart_after_s:.0f}s")
            elif now - slot.ended >= self.restart_after_s:
                self._start(i)

    def _wake_all(self, now: float) -> None:
        self._woken_at = now
        for s in self.loops:
            s.wake()

    def _halt(self) -> None:
        """Stop every loop and wait while each finishes its unit and releases its lease."""
        for s in self.loops:
            s.stop()
        for slot in self._slots:
            if slot.thread is not None:
                slot.thread.join()

    def _reseed(self, now: float) -> None:
        self._seeded_at = now
        self._seed()
