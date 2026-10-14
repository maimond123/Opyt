"""
pipeline/worker/quota_loop/loop.py — one quota loop (`QuotaLoop`): the only consumer of one
quota, running its sources' units in class order.

A loop asks its sources what each owes, runs the most urgent unit (one request, one page),
and asks again. Nothing is planned ahead: every source computes what it owes from the store each
time it is asked, so a loop killed at any moment loses at most the page in flight, and the next
one starts from the same rows. It holds a lease named after its quota (`CatchupLock(quota)`) for
as long as it runs, so two loops never spend one quota.

Which unit goes next is `choose.py`'s: the loop holds its `Rotation`, and counts a turn when it
admits a unit.

How units run is its executor's (`Executor`): the serial one (`serial.py`) runs one at a time
through `step`; the web's pool (`pool.py`) runs several at once. Either way the loop admits each
unit and settles it, so selection, failures and state are the same for both.

Pacing is burst: a unit goes as soon as the last returns, bounded only by the meter. When the
meter says the window is spent, the loop sleeps until the reset X announced, plus a few
seconds, without asking any source; X sends the exact instant, so no request is spent finding out.

What a failure does, decided here so that no source repeats it:
  | raised by a unit          | the loop                                   | charged to the item |
  |---------------------------|--------------------------------------------|---------------------|
  | `XRateLimited`            | waits for the reset                        | no                  |
  | `SyncAuthError`           | pauses (cause `x_session`), tries again on | no                  |
  |                           | its next wake                              |                     |
  | `CircuitOpenError`        | pauses until the model provider's circuit  | no                  |
  |                           | may retry (cause `model:<service>`)        |                     |
  | `database is locked`      | backs off for seconds; the unit stays owed | no                  |
  | `Changed`, in fetch or    | settles it as `changed`; its source works  | no, and nothing is  |
  | record, or a `changed`    | out afresh what is owed                    | cleared             |
  | outcome returned          |                                            |                     |
  | anything else, in fetch   | charges the unit's item (`retries.charge`) | yes, by its kind    |
  | anything else otherwise   | skips that unit for 1, 5, 15, then 60 min  | in memory only      |
  | (the lease lost after the | settles it as `lease_lost`, recording      | no                  |
  | fetch)                    | nothing more                               |                     |
A source whose fetch saves content saves it with its completion record in one transaction, and
raises `Changed` instead when the lease is gone by then, so the last row leaves nothing
half-written. A pass (publishing's) runs whole in its fetch, and the last row skips
only its `record` (publishing's watermark), so the lease's next holder may run that pass again.
`retries.quota_wait` holds the first rows. A failure a source reports instead of raising (a page
that would not save, a host that blocked us) comes back in its `Outcome` and is charged the same
way. A unit runs as fetch, the forget cleanup, record, then the charge or the clear
(`run_unit`), so no source charges, retries, gives up or purges on its own.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Protocol

from pipeline import sync_lock
from pipeline.ingestion.utils import log
from pipeline.platforms.x.rate import rate_budget

from . import loop_state
from .choose import Picked, Rotation
from .loop_state import BEAT_S
from .retries import (TRANSIENT, TRANSIENT_BACKOFF_S, Outages, Wait, charge,
                      counts_toward_outage, kind_of, outage_wait, quota_wait, spent_wait,
                      succeeded, transient_wait)
from .source import CHANGED, Changed, Context, Forget, Item, Outcome, Source, Unit

BROKEN_UNIT_BACKOFF_S = (60.0, 300.0, 900.0, 3600.0)
# The backstop under every retry rule: a unit whose failure was charged is not admitted again
# for this long, whatever its row says. Shorter than the shortest retry (a first blip, 48 s with
# jitter), so it delays nothing that works, and a retry rule that leaves a failed item ready at
# once costs two requests a minute, not fifteen a second (seen in Substack samples).
MIN_RETRY_S = 30.0


@dataclass(frozen=True)
class Ran:
    source: str
    unit: Unit
    result: str | None


@dataclass(frozen=True)
class Fetched:
    value: object = None
    error: Exception | None = None
    mark: int | None = None


def _meter_for(quota: str) -> Callable[[], tuple[int, float] | None]:
    """The meter reading for an X quota (`x:<operation>`); other quotas have none."""
    if not quota.startswith("x:"):
        return lambda: None
    op = quota[2:]

    def read() -> tuple[int, float] | None:
        return rate_budget(op)
    return read


def _dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, timezone.utc)


def settlement_of(exc: Exception, kind: str | None, item: Item | None, now: float,
                  transient: int) -> Outcome | Wait | None:
    """What a raise after admission comes to. `kind` is the item's failure kind for a fetch's
    raise (`retries.kind_of`), None for a raise that is not the item's (a record's, or a quota
    stop). `item` is the unit's, None when it has none. `transient` counts this failure in the
    current run of transient ones, from 1.

    A `Changed` settles nothing (`Outcome(CHANGED)`); a fetch's failure on a unit with an item is
    charged to it (its `Outcome`); otherwise the quota's wait (`quota_wait`), and None leaves the
    unit to the backstop's strike."""
    if isinstance(exc, Changed):
        return Outcome(CHANGED)
    if kind is not None and item is not None:
        return Outcome(kind, failure=kind, detail=f"{type(exc).__name__}: {exc}",
                       outage=counts_toward_outage(kind, exc, item), answered=False)
    return quota_wait(exc, now, transient)


def beat_due(stopping: bool, beat_at: float, now: float) -> bool:
    """Is a beat owed? Once `BEAT_S` has passed since the last write, and never once the loop is
    stopping: a stopped loop never beats, and a quota whose lease was lost is another process's."""
    return not stopping and now - beat_at >= BEAT_S


class Executor(Protocol):
    """How a loop's units run: `Serial` (one at a time) or `WebPool` (several, one per host). The
    loop admits and settles every unit; an executor only says which units it may take from a
    source now, and drives admission, fetching and settlement on its own schedule."""

    def units(self, source: Source, conn: sqlite3.Connection,
              now: datetime) -> Iterable[Unit]:
        """The units this executor may run from `source` now, most urgent first."""

    def execute(self, loop: QuotaLoop, conn: sqlite3.Connection) -> None:
        """Run `loop`'s units on `conn`, the owning thread's connection, until it stops."""


class QuotaLoop:
    """The only consumer of one quota. `run()` is its thread's body; `step()` admits one unit and
    runs it, which is what the tests drive."""

    def __init__(self, quota: str, sources: list[Source], *, executor: Executor,
                 outages: Outages, forget: Forget | None = None,
                 meter: Callable[[], tuple[int, float] | None] | None = None,
                 clock: Callable[[], float] = time.time,
                 connect: Callable[[], sqlite3.Connection]) -> None:
        for s in sources:
            if s.quota != quota:
                raise ValueError(f"source {s.name!r} spends {s.quota!r}, not {quota!r}")
        self.completed_units = 0
        self.quota = quota
        self.sources = list(sources)
        self.executor = executor
        self.outages = outages
        self.forget = forget
        self.ctx = Context(quota)
        self._meter = meter if meter is not None else _meter_for(quota)
        self.clock = clock
        self._connect = connect
        self.rotation = Rotation()
        self._transient = 0
        # The backstop's strikes, `(source, unit key)` → `(strikes, until)`; the rotation reads them.
        self.strikes: dict[tuple[str, str], tuple[int, float]] = {}
        self.status = loop_state.Status("running")
        self._beat_at = float("-inf")
        self._wake = threading.Event()
        self._stop = threading.Event()

    # ── control, from other threads ──────────────────────────────────────────────
    def wake(self) -> None:
        """Something may now be owed: look again instead of sleeping on."""
        self._wake.set()

    def stop(self) -> None:
        """Finish the unit in flight, release the lease and return from `run()`."""
        self._stop.set()
        self._wake.set()

    @property
    def stopping(self) -> bool:
        """The one stop signal every executor and every sleep reads: `stop()` was called, or
        another process took this quota's lease."""
        return self._stop.is_set() or self.ctx.lease_lost()

    def starting(self) -> None:
        """Busy until its first look says otherwise: the process says so before it starts the
        thread, so a loop revived after it went idle never reads as idle meanwhile."""
        self.status = loop_state.Status("running")

    def in_flight(self) -> None:
        """A pool still draining fetches is busy even when admission has paused."""
        self.status = self.status.in_flight()

    def take_wake(self) -> bool:
        woken = self._wake.is_set()
        self._wake.clear()
        return woken

    def connection(self):
        """An executor owns its connection and closes it when its work ends."""
        return closing(self._connect())

    # ── admitting and settling ────────────────────────────────────────────────────
    def step(self, conn: sqlite3.Connection) -> Ran | Wait:
        """Run the most urgent unit, or say how long to wait and why."""
        now = self.clock()
        picked = self.admit(conn, now)
        if picked is None:
            return self.record_idle(conn)
        if isinstance(picked, Wait):
            return picked
        return self.run_unit(conn, *picked)

    def admit(self, conn: sqlite3.Connection, now: float) -> Picked | Wait | None:
        """Pick and check one unit, count its class turn, then report running before fetching.
        The turn is counted here, at admission, by every executor: a unit that ends in a wait
        has had its turn, and the pool's completion order cannot distort it."""
        meter = self._reading(now)
        if meter is not None and meter[0] <= 0:
            spent = spent_wait(meter[1], now)
            self._record_wait(now, spent)
            return spent
        picked = self.rotation.pick(self.sources, lambda s: self._free(s, conn, _dt(now)),
                                    meter, self.strikes, now)
        if picked is None:
            return None
        self.rotation.turn(picked.source)
        self._set_state(now, "running", None, None)
        return picked

    def _free(self, source: Source, conn: sqlite3.Connection, now: datetime) -> Iterable[Unit]:
        """A source's ready units, less those on a paused platform. The pause is held here, for
        every source, so one that never asks about it cannot run through it."""
        return (unit for unit in self.executor.units(source, conn, now)
                if unit.item is None or unit.item.platform is None
                or not self.outages.paused(unit.item.platform, now))

    def record_idle(self, conn) -> Wait:
        """Nothing is owed: record the idle wait until the next due time, and return it for the
        executor to sleep on."""
        now = self.clock()
        idle = Wait(self.next_due(conn, now), "idle")
        self._record_wait(now, idle)
        return idle

    def record_backoff(self, conn, exc: Exception) -> Wait:
        """A raise before any unit was picked (a source's read, the store): nothing to charge or
        strike. Roll back, log it, record the backoff wait, and return it for the executor to
        sleep on."""
        now = self.clock()
        self._abandon(conn)
        log(f"[scrapers] {self.quota}: {type(exc).__name__}: {exc}")
        wait = transient_wait(now, len(TRANSIENT_BACKOFF_S))
        self._record_wait(now, wait)
        return wait

    def fetch_unit(self, conn, picked: Picked, ctx: Context) -> Fetched:
        """Fetch on the executing thread's connection; settlement belongs to the owning loop."""
        if ctx.lease_lost():
            return Fetched()
        source, unit = picked
        mark = None
        try:
            mark = self.forget.mark(conn) if unit.footprint and self.forget else None
            return Fetched(source.fetch(conn, unit, ctx), mark=mark)
        except Exception as exc:
            self._abandon(conn)
            return Fetched(error=exc, mark=mark)

    def run_unit(self, conn: sqlite3.Connection, source: Source, unit: Unit) -> Ran | Wait:
        picked = Picked(source, unit)
        return self.settle(conn, picked, self.fetch_unit(conn, picked, self.ctx))

    def settle(self, conn, picked: Picked, fetched: Fetched) -> Ran | Wait:
        """Settle a fetched unit on the owning thread: the forget, lease, completion, failure and
        outage policy for every executor. Never raises: whatever raises after admission is
        settled with its unit."""
        now = self.clock()
        try:
            if (ended := self._purge_forgotten_or_end_lease_lost(conn, picked, fetched.mark)) is not None:
                return ended
            if fetched.error is not None:
                return self._unit_raised(conn, picked, fetched.error, now, kind_of(fetched.error))
            outcome = picked.source.record(conn, picked.unit, fetched.value)
            return self._settle_outcome(conn, picked, outcome, now)
        except Exception as exc:
            self._abandon(conn)
            return self._unit_raised(conn, picked, exc, now, None)

    def _purge_forgotten_or_end_lease_lost(self, conn, picked: Picked, mark: int | None) -> Ran | None:
        """A unit whose Oracle was forgotten while it ran is purged; then one whose lease was
        lost stops. Either way nothing is recorded or settled. The forget comes first, whatever
        the fetch raised: a rate limit mid-walk on a forgotten Oracle must not leave its late
        atoms behind."""
        if self._purge_if_forgotten(conn, picked.unit, mark):
            return self._ran(picked, "forgotten")
        if self.ctx.lease_lost():
            return self._ran(picked, "lease_lost")
        return None

    def _unit_raised(self, conn, picked: Picked, exc: Exception, now: float,
                     kind: str | None) -> Ran | Wait:
        """Settle a raise as `settlement_of` decides: an outcome settles like a returned one, a
        wait is the quota's, and nothing is the backstop's strike, so a unit that always raises
        cannot spin the quota."""
        settled = settlement_of(exc, kind, picked.unit.item, now, self._transient + 1)
        if isinstance(settled, Outcome):
            return self._settle_outcome(conn, picked, settled, now)
        if settled is None:
            self._strike(picked, now, exc)
            return Ran(picked.source.name, picked.unit, "raised")
        if settled.cause == TRANSIENT:
            self._transient += 1
            log(f"[scrapers] {self.quota}: {picked.source.name} {picked.unit.key}: {exc} — the "
                f"unit stays owed; next try in {settled.until - now:.0f}s")
        self._record_wait(now, settled)
        return settled

    def _purge_if_forgotten(self, conn, unit: Unit, mark: int | None) -> bool:
        """If the unit's Oracle left the roster while it ran, purge what arrived since `mark`,
        and say so."""
        if unit.footprint is None or self.forget is None or mark is None:
            return False
        return self.forget.purge(conn, unit.footprint.oracle, mark, unit.footprint.authors)

    def _settle_outcome(self, conn, picked: Picked, outcome: Outcome, now: float) -> Ran | Wait:
        """A `changed` outcome settles nothing; any other is cleared or charged."""
        if outcome.status == CHANGED:
            return self._ran(picked, CHANGED)
        return self._clear_or_charge(conn, picked, outcome, now)

    def _clear_or_charge(self, conn, picked: Picked, outcome: Outcome,
                         now: float) -> Ran | Wait:
        """Clear the unit's item after a success, or charge it a failure. A unit that settles
        without a failure is a completed unit, which may have left other loops work. A pause on
        a platform that is the whole quota stops the loop."""
        item = picked.unit.item
        if outcome.failure is None:
            if item is not None:
                succeeded(conn, item, outcome, self.outages)
            self.completed_units += 1
            return self._ran(picked, outcome.status)
        if item is None:
            return self._ran(picked, outcome.status)
        charged = charge(conn, item, outcome, self.outages, _dt(now))
        if charged.paused_until is not None and self.outages.whole_quota:
            settled = outage_wait(charged.paused_until.timestamp(), item.platform)
            log(f"[scrapers] {self.quota}: {settled.cause}: pausing {settled.until - now:.0f}s")
            self._record_wait(now, settled)
        else:
            settled = self._ran(picked, charged.word)
        self.strikes[picked.key] = (0, now + MIN_RETRY_S)
        return settled

    def _ran(self, picked: Picked, word: str | None) -> Ran:
        """A unit that ran: clear its strikes."""
        self._transient = 0
        self.strikes.pop(picked.key, None)
        return Ran(picked.source.name, picked.unit, word)

    def next_due(self, conn: sqlite3.Connection, now: float) -> float | None:
        """The earliest moment any source says work becomes owed, a skipped unit may be tried
        again, or a paused platform's units are let through. None: nothing known is coming."""
        times = [until for _strikes, until in self.strikes.values() if until > now]
        if (resume := self.outages.next_resume(_dt(now))) is not None:
            times.append(resume.timestamp())
        for s in self.sources:
            try:
                due = s.next_due(conn, _dt(now))
            except Exception as exc:
                log(f"[scrapers] {self.quota}: {s.name}.next_due raised {exc!r}")
                continue
            if due is not None:
                times.append(due.timestamp())
        return min(times) if times else None

    # ── choosing ──────────────────────────────────────────────────────────────────
    def _reading(self, now: float) -> tuple[int, float] | None:
        """The meter's `(remaining, reset)`, or None when nothing is known. A window whose reset
        has passed has refilled to a size only X can state, so it reads as unknown too."""
        try:
            reading = self._meter()
        except Exception:
            return None
        if reading is None or reading[1] <= now:
            return None
        return reading

    # ── failures ──────────────────────────────────────────────────────────────────
    def _abandon(self, conn: sqlite3.Connection) -> None:
        """Roll back whatever the failed unit left open, so the connection stays usable."""
        if conn.in_transaction:
            try:
                conn.rollback()
            except Exception:
                pass

    def _strike(self, picked: Picked, now: float, exc: Exception) -> None:
        strikes = self.strikes.get(picked.key, (0, 0.0))[0] + 1
        delay = BROKEN_UNIT_BACKOFF_S[min(strikes, len(BROKEN_UNIT_BACKOFF_S)) - 1]
        self.strikes[picked.key] = (strikes, now + delay)
        log(f"[scrapers] {self.quota}: {picked.source.name} {picked.unit.key} raised "
            f"{type(exc).__name__}: {exc} — skipped for {delay:.0f}s")

    # ── state ─────────────────────────────────────────────────────────────────────
    def _record_wait(self, now: float, wait: Wait) -> None:
        self._set_state(now, wait.state, wait.cause, wait.until)

    def _set_state(self, now: float, st: str, cause: str | None,
                   until: float | None) -> None:
        """Write `scraper_state` when what it says changes, or to beat. Never raises: the record
        is for `progress`, and failing to write it must not stop the work it describes."""
        status = self.status.transition(st, cause, until, now)
        changed = status != self.status
        self.status = status
        if not changed and not beat_due(self.stopping, self._beat_at, now):
            return
        try:
            loop_state.set_state(self.quota, st, cause=cause, retry_at=until, now=now)
        except Exception as exc:
            self._beat_at = float("-inf")
            log(f"[scrapers] {self.quota}: could not record state {st!r}: {exc}")
            return
        self._beat_at = now

    def _beat(self) -> None:
        now = self.clock()
        try:
            loop_state.beat(self.quota, now=now)
            self._beat_at = now
        except Exception as exc:
            log(f"[scrapers] {self.quota}: heartbeat failed: {exc}")

    def beat_if_due(self) -> None:
        """Beat when `beat_due` says: the one path every beat takes, on either executor."""
        if beat_due(self.stopping, self._beat_at, self.clock()):
            self._beat()

    # ── the thread's body ─────────────────────────────────────────────────────────
    def run(self) -> None:
        """Hold this quota's lease while its executor runs units, until stopped. Returns at once,
        running nothing, when another process holds the lease. A `connect` from the unit that
        owns the sources creates their tables; the lease and the loop's state keep theirs in
        loops.db."""
        with self.connection() as conn:
            with sync_lock.CatchupLock(self.quota) as lease:
                if not lease.acquired:
                    log(f"[scrapers] {self.quota}: another process holds this quota; not running")
                    return
                self.ctx.lease_lost = lease.lost
                self.executor.execute(self, conn)
                if lease.lost():
                    log(f"[scrapers] {self.quota}: lease taken over; stopping")

    def sleep_until(self, until: float | None) -> None:
        """Sleep until `until`, a wake or a stop (a lost lease included), beating when due."""
        while not self.stopping:
            left = None if until is None else until - self.clock()
            if left is not None and left <= 0:
                return
            if self._wake.wait(BEAT_S if left is None else min(left, BEAT_S)):
                return
            self.beat_if_due()
