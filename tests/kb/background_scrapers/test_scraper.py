"""The scraper core, driven with fake sources: which unit runs next, how the quota's meter and
each kind of failure change that, and the thread's lease, wake and stop."""
from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from pipeline.circuit_breaker import CircuitOpenError
from pipeline.ingestion.utils import SyncAuthError
from pipeline.platforms.x.rate import XRateLimited
from pipeline.kb import schema
from pipeline.worker.quota_loop import source
from pipeline.worker.quota_loop import loop as ql
from pipeline.worker.quota_loop import loop_state
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop import process
from pipeline.worker.quota_loop.pool import WebPool
from pipeline.worker.quota_loop.serial import Serial
from pipeline.worker.job_queue import RailJobStore
from pipeline.oracles.sources import scraper_tables

Q = "x:UserOriginalsTimeline"
T0 = 1_800_000_000.0


class Clock:
    def __init__(self, t: float = T0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class Fake:
    """A source whose owed units are a list of keys. `raises` is consumed one entry per run."""

    def __init__(self, name, klass, keys=(), *, floor=0, due=None, raises=(), log=None):
        self.name, self.klass, self.floor, self.quota = name, klass, floor, Q
        self.keys = list(keys)
        self.due = due
        self.raises = list(raises)
        self.log = log if log is not None else []
        self.asked = 0

    def next_unit(self, conn, now):
        self.asked += 1
        return source.Unit(self.keys[0]) if self.keys else None

    def fetch(self, conn, unit, ctx):
        if self.raises:
            exc = self.raises.pop(0)
            if exc is not None:
                raise exc
        return unit

    def record(self, conn, unit, fetched):
        self.keys.remove(unit.key)
        self.log.append(f"{self.name}:{unit.key}")
        return source.Outcome("ok")

    def next_due(self, conn, now):
        return self.due


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    scraper_tables.init_schema(c)
    yield c
    c.close()


def _scraper(*sources, meter=None, clock=None):
    return ql.QuotaLoop(Q, list(sources), executor=Serial(), meter=meter or (lambda: None), clock=clock or Clock(), outages=retries.Outages(retries.PLATFORM_PAUSES_S), connect=schema.connect)


def _drain(s, conn, limit=100) -> list:
    out = []
    for _ in range(limit):
        step = s.step(conn)
        if isinstance(step, retries.Wait):
            return out
        out.append(step)
    raise AssertionError("never went idle")


# ── which unit goes next ──────────────────────────────────────────────────────────

def test_yours_then_new_posts_then_scouting(conn):
    log = []
    s = _scraper(Fake("scouting", source.SCOUTING, ["c1"], log=log),
                 Fake("new_posts", source.NEW_POSTS, ["n1"], log=log),
                 Fake("picks", source.YOURS, ["p1", "p2"], log=log))
    _drain(s, conn)
    assert log == ["picks:p1", "picks:p2", "new_posts:n1", "scouting:c1"]


def test_sources_in_one_class_take_turns(conn):
    log = []
    s = _scraper(Fake("picks", source.YOURS, ["p1", "p2", "p3"], log=log),
                 Fake("check", source.YOURS, ["c1", "c2"], log=log))
    _drain(s, conn)
    assert log == ["picks:p1", "check:c1", "picks:p2", "check:c2", "picks:p3"]


def test_a_unit_that_waits_still_spends_its_sources_turn(conn):
    """The turn is counted at admission, so after a rate-limit wait the next unit comes from
    the class's next source, as the pool already does."""
    log = []
    s = _scraper(Fake("picks", source.YOURS, ["p1", "p2"], log=log,
                      raises=[XRateLimited("spent", reset_at=T0 + 100)]),
                 Fake("check", source.YOURS, ["c1"], log=log))
    assert s.step(conn).state == "waiting"
    assert s.step(conn).source == "check"


def test_new_posts_take_every_tenth_unit_while_owed(conn):
    log = []
    s = _scraper(Fake("picks", source.YOURS, [f"p{i}" for i in range(25)], log=log),
                 Fake("new_posts", source.NEW_POSTS, ["n1", "n2"], log=log))
    _drain(s, conn)
    assert [i for i, entry in enumerate(log, 1) if entry.startswith("new_posts")] == [10, 20]
    assert len(log) == 27


def test_the_tenth_slot_goes_back_to_yours_when_no_new_posts_are_owed(conn):
    log = []
    new_posts = Fake("new_posts", source.NEW_POSTS, log=log)
    s = _scraper(Fake("picks", source.YOURS, [f"p{i}" for i in range(12)], log=log), new_posts)
    _drain(s, conn)
    assert len(log) == 12 and all(e.startswith("picks") for e in log)
    # …and new posts that turn up after a long run of yours go first, not ten units later.
    new_posts.keys = ["n1"]
    s.sources[0].keys = ["p99"]
    assert s.step(conn).source == "new_posts"


def test_scouting_never_runs_while_anything_else_is_owed(conn):
    scouting = Fake("scouting", source.SCOUTING, ["c1"])
    s = _scraper(Fake("picks", source.YOURS, ["p1"]), scouting)
    assert s.step(conn).source == "picks"
    assert scouting.asked == 0                     # not even asked: its list is not computed


@pytest.mark.parametrize("reading, runs", [
    ((50, T0 + 600), False),                       # at the floor: the reserve is kept
    ((51, T0 + 600), True),
    (None, True),                                  # never read: unknown means go
    ((3, T0 - 1), True),                           # the window has reset: unknown again
])
def test_scouting_keeps_its_floor_of_fifty(conn, reading, runs):
    s = _scraper(Fake("scouting", source.SCOUTING, ["c1"], floor=50), meter=lambda: reading)
    assert isinstance(s.step(conn), ql.Ran) is runs


def test_the_floor_is_scoutings_alone(conn):
    s = _scraper(Fake("picks", source.YOURS, ["p1"]), meter=lambda: (1, T0 + 600))
    assert s.step(conn).source == "picks"          # yours may spend the window to zero


# ── waiting ───────────────────────────────────────────────────────────────────────

def test_a_spent_meter_waits_for_the_reset_without_asking_anyone(conn):
    picks = Fake("picks", source.YOURS, ["p1"])
    s = _scraper(picks, meter=lambda: (0, T0 + 300))
    step = s.step(conn)
    assert step == retries.Wait(T0 + 300 + retries.RESET_GRACE_S, "waiting", "quota_spent")
    assert picks.asked == 0
    assert loop_state.read_state(Q)["state"] == "waiting"


def test_a_rate_limited_unit_waits_for_the_reset_x_announced(conn):
    picks = Fake("picks", source.YOURS, ["p1"], raises=[XRateLimited("spent", reset_at=T0 + 100)])
    s = _scraper(picks)
    assert s.step(conn) == retries.Wait(T0 + 100 + retries.RESET_GRACE_S, "waiting", "quota_spent")
    assert picks.keys == ["p1"]                    # still owed


def test_nothing_owed_waits_for_the_earliest_due_source(conn):
    from datetime import datetime, timezone
    soon = datetime.fromtimestamp(T0 + 60, timezone.utc)
    later = datetime.fromtimestamp(T0 + 3600, timezone.utc)
    s = _scraper(Fake("new_posts", source.NEW_POSTS, due=later), Fake("picks", source.YOURS, due=soon))
    step = s.step(conn)
    assert (step.until, step.state) == (T0 + 60, "idle")
    assert s.status.since == T0 and s.status.until == T0 + 60

    assert _scraper(Fake("picks", source.YOURS)).step(conn).until is None


# ── failures ──────────────────────────────────────────────────────────────────────

_LOCK = sqlite3.OperationalError("database is locked")


@pytest.mark.parametrize("exc, kind, has_item, transient, expected", [
    (source.Changed("owner moved"), retries.BLIP, True, 1,     # settles nothing
     source.Outcome(source.CHANGED)),
    (RuntimeError("timeout"), retries.BLIP, True, 1,           # a fetch's blip: charged
     source.Outcome("blip", failure="blip", detail="RuntimeError: timeout", outage=True,
                    answered=False)),
    (RuntimeError("suspended"), retries.ACCOUNT, True, 1,      # an account: charged, no streak
     source.Outcome("account", failure="account", detail="RuntimeError: suspended",
                    answered=False)),
    (RuntimeError("timeout"), retries.BLIP, False, 1, None),   # no item: the backstop
    (RuntimeError("bad row"), None, True, 1, None),            # a record's raise: the backstop
    (XRateLimited("spent", reset_at=T0 + 100), None, True, 1,  # a quota stop: its wait
     retries.Wait(T0 + 100 + retries.RESET_GRACE_S, "waiting", "quota_spent")),
    (_LOCK, None, True, 2,                                     # a lock: the transient ladder
     retries.Wait(T0 + retries.TRANSIENT_BACKOFF_S[1], "backoff", "transient")),
])
def test_what_a_raise_after_admission_comes_to(exc, kind, has_item, transient, expected):
    item = source.Item(None, ("p1",), source.Retry(), "x") if has_item else None
    assert ql.settlement_of(exc, kind, item, T0, transient) == expected


class Timeout(Exception):
    """Named as the transports name theirs (curl_cffi's and requests' `Timeout`)."""


class SSLError(ConnectionError):
    pass


@pytest.mark.parametrize("exc, own_site, counts", [
    (SSLError("TLS connect error"), True, False),     # a dead domain: its own failure
    (Timeout("timed out"), True, False),
    (RuntimeError("HTTP 403"), True, True),            # a refusal is the platform answering
    (SSLError("TLS connect error"), False, True),      # the platform's own host: it counts
    (Timeout("timed out"), False, True),
])
def test_a_site_of_its_owners_that_never_answers_is_not_its_platform_failing(exc, own_site,
                                                                            counts):
    item = source.Item(None, ("p1",), source.Retry(), "substack", own_site=own_site)
    assert ql.settlement_of(exc, retries.BLIP, item, T0, 1).outage is counts


def test_a_dead_x_session_pauses_with_a_cause_the_user_can_act_on(conn):
    s = _scraper(Fake("picks", source.YOURS, ["p1"], raises=[SyncAuthError("401")]))
    step = s.step(conn)
    assert (step.state, step.cause) == ("paused", "x_session")
    assert step.until == T0 + retries.PAUSED_RECHECK_S
    row = loop_state.read_state(Q)
    assert (row["state"], row["cause"]) == ("paused", "x_session")


def test_an_open_model_circuit_pauses_until_it_may_retry(conn):
    s = _scraper(Fake("picks", source.YOURS, ["p1"],
                      raises=[CircuitOpenError("openrouter", retry_after=120)]))
    assert s.step(conn) == retries.Wait(T0 + 120, "paused", "model:openrouter")


@pytest.mark.parametrize("exc", [sqlite3.OperationalError("database is locked")])
def test_a_transient_failure_charges_nothing_and_the_unit_stays_owed(conn, exc):
    clock = Clock()
    picks = Fake("picks", source.YOURS, ["p1"], raises=[exc])
    s = _scraper(picks, clock=clock)
    assert (s.step(conn).state, picks.keys) == ("backoff", ["p1"])
    assert s.strikes == {}                         # no strike, not even in memory
    clock.t += retries.TRANSIENT_BACKOFF_S[0]
    assert s.step(conn).unit.key == "p1"


def test_a_failed_unit_leaves_no_transaction_open(conn):
    class Leaky(Fake):
        def fetch(self, conn, unit, ctx):
            conn.execute("INSERT INTO milestones_shown VALUES ('a', 'reach', 'now')")
            raise sqlite3.OperationalError("database is locked")

    s = _scraper(Leaky("picks", source.YOURS, ["p1"]))
    s.step(conn)
    assert not conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM milestones_shown").fetchone()[0] == 0


class _Raising:
    """`RetryRows` whose charge write raises, stopping the loop so its body returns."""

    def __init__(self, exc, stop):
        self.exc, self.stop = exc, stop

    def read(self, _row):
        return source.Retry()

    def write(self, _conn, _key, _retry):
        self.stop()
        raise self.exc

    def clear(self, _conn, _key):
        pass


class Settles(Fake):
    """One unit, offered to either executor, whose fetch or charge raises `exc`."""

    def __init__(self, at, exc):
        super().__init__("picks", source.YOURS, ["p1"])
        self.at, self.exc, self.loop = at, exc, None

    def unit(self):
        if self.at == "mark":
            return source.Unit("p1", footprint=source.Footprint("x:user:carol"), host="a")
        item = source.Item(_Raising(self.exc, self.loop.stop), ("p1",), source.Retry())
        return source.Unit("p1", item=item, host="a")

    def next_unit(self, _conn, _now):
        return self.unit() if self.keys else None

    def candidates(self, _conn, _now):
        return [self.unit()] if self.keys else []

    def mark(self, _conn):
        self.loop.stop()
        raise self.exc

    def record(self, _conn, _unit, _fetched):
        return source.Outcome("blip", failure="blip")


@pytest.mark.parametrize("executor", ["serial", "pool"])
@pytest.mark.parametrize("at", ["mark", "charge"])
@pytest.mark.parametrize("exc, struck", [(ValueError("bug"), True),
                                         (sqlite3.OperationalError("database is locked"), False)])
def test_a_raise_after_admission_settles_with_its_unit_in_both_executors(
        conn, executor, at, exc, struck):
    """A failed forget mark is the fetch's error, and a raise while settling is the unit's,
    so both executors strike the unit (or back off a lock) the same way."""
    src = Settles(at, exc)
    build = Serial if executor == "serial" else WebPool
    loop = ql.QuotaLoop(Q, [src], executor=build(), meter=lambda: None, clock=Clock(),
                 outages=retries.Outages(retries.PLATFORM_PAUSES_S),
                 forget=source.Forget(mark=src.mark, purge=lambda *_: False),
                 connect=lambda: sqlite3.connect(":memory:"))
    src.loop = loop
    loop.executor.execute(loop, conn)
    if struck:
        assert ("picks", "p1") in loop.strikes
    else:
        assert loop.strikes == {}
        assert (loop.status.state, loop.status.until) == (
            "backoff", T0 + retries.TRANSIENT_BACKOFF_S[0])


def test_a_unit_that_keeps_raising_is_skipped_and_others_go_ahead(conn):
    clock = Clock()
    log = []
    broken = Fake("picks", source.YOURS, ["p1"], raises=[ValueError("bug"), ValueError("bug")],
                  log=log)
    s = _scraper(broken, Fake("check", source.YOURS, ["c1", "c2"], log=log), clock=clock)
    assert s.step(conn).result == "raised"
    _drain(s, conn)
    assert log == ["check:c1", "check:c2"]
    step = s.step(conn)
    assert (step.state, step.until) == ("idle", T0 + 60)   # the skip's end is the next due time

    clock.t += 60
    assert s.step(conn).result == "raised"         # second strike: skipped for longer
    clock.t += 60
    assert isinstance(s.step(conn), retries.Wait)
    clock.t += 240
    assert s.step(conn).result == "ok" and log[-1] == "picks:p1"


def test_every_source_must_spend_the_scrapers_own_quota():
    other = Fake("archives", source.YOURS)
    other.quota = "web"
    with pytest.raises(ValueError):
        ql.QuotaLoop(Q, [other], executor=Serial(), outages=retries.Outages(retries.PLATFORM_PAUSES_S), connect=schema.connect)


# ── the thread: lease, wake, stop ─────────────────────────────────────────────────

def test_a_loop_whose_lease_was_taken_stops_sleeping_within_a_beat(conn, monkeypatch):
    """A lost lease is a stop like any other, so a loop pushed out of its quota
    mid-wait stops within one beat, instead of sleeping to the wait's end while it beats for a
    quota it no longer holds."""
    monkeypatch.setattr(ql, "BEAT_S", 0.001)
    ticks = iter(range(10_000))
    loop = _scraper(clock=lambda: T0 + next(ticks))
    beats = []

    def beat():
        beats.append(True)
        loop.ctx.lease_lost = lambda: True          # another process took the lease meanwhile

    monkeypatch.setattr(loop, "_beat", beat)
    loop.sleep_until(T0 + 50)
    assert len(beats) == 1


def test_a_lease_lost_during_a_wait_gets_no_further_beat(conn, monkeypatch):
    """A lease lost while the loop waits out a beat interval stops the loop
    without one more beat for a quota it no longer holds."""
    monkeypatch.setattr(ql, "BEAT_S", 0.001)
    loop = _scraper(clock=lambda: T0)
    beats = []
    monkeypatch.setattr(loop, "_beat", lambda: beats.append(True))
    real_wait = loop._wake.wait

    def wait(timeout):
        loop.ctx.lease_lost = lambda: True          # another process took the lease meanwhile
        return real_wait(timeout)

    monkeypatch.setattr(loop._wake, "wait", wait)
    loop.sleep_until(T0 + 50)
    assert beats == []


@pytest.mark.parametrize("stopping, beat_at, due", [
    (False, T0 - ql.BEAT_S, True),      # a beat interval since the last write
    (False, T0 - 1, False),             # written lately
    (True, T0 - ql.BEAT_S, False),      # a stopped loop never beats
])
def test_when_a_beat_is_due(stopping, beat_at, due):
    assert ql.beat_due(stopping, beat_at, T0) is due


def test_an_idle_pool_whose_lease_was_lost_writes_no_beat_after(conn, monkeypatch):
    """The pool beats through `beat_if_due`, not only through
    `sleep_until`, so "a stopped loop never beats" must hold there too. An idle pool whose lease
    is lost while it sleeps writes no beat after the loss."""
    monkeypatch.setattr(ql, "BEAT_S", 0.001)

    class Idle:
        name, quota, klass, floor = "documents", "web", source.YOURS, 0

        def candidates(self, _conn, _now):
            return []

        def next_due(self, _conn, _now):
            return None

    ticks = iter(range(10_000))
    pool = ql.QuotaLoop("web", [Idle()], executor=WebPool(), clock=lambda: T0 + next(ticks),
                        outages=retries.Outages(retries.PLATFORM_PAUSES_S), connect=schema.connect)
    lost, beats_after = [], []
    monkeypatch.setattr(pool, "_beat", lambda: lost and beats_after.append(True))
    real_wait = pool._wake.wait

    def wait(timeout):
        lost.append(True)
        pool.ctx.lease_lost = lambda: True          # another process took the lease meanwhile
        return real_wait(timeout)

    monkeypatch.setattr(pool._wake, "wait", wait)
    pool.executor.execute(pool, conn)
    assert lost and beats_after == []


@pytest.fixture()
def fast_beats(monkeypatch):
    monkeypatch.setattr(ql, "BEAT_S", 0.05)


def _start(s):
    t = threading.Thread(target=s.run, daemon=True)
    t.start()
    return t


def _until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.01)


def test_a_wake_cuts_an_idle_sleep_short(kb_home, fast_beats):
    picks = Fake("picks", source.YOURS)
    s = ql.QuotaLoop(Q, [picks], executor=Serial(), meter=lambda: None, outages=retries.Outages(retries.PLATFORM_PAUSES_S), connect=schema.connect)
    t = _start(s)
    try:
        _until(lambda: s.status.since is not None)
        picks.keys.append("p1")
        s.wake()
        _until(lambda: picks.log == ["picks:p1"])
    finally:
        s.stop()
        t.join(2)
    assert not t.is_alive()


def test_stop_finishes_the_unit_in_flight(kb_home, fast_beats):
    entered, release = threading.Event(), threading.Event()

    class Slow(Fake):
        def fetch(self, conn, unit, ctx):
            entered.set()
            release.wait(2)
            return super().fetch(conn, unit, ctx)

    slow = Slow("picks", source.YOURS, ["p1", "p2"])
    s = ql.QuotaLoop(Q, [slow], executor=Serial(), meter=lambda: None, outages=retries.Outages(retries.PLATFORM_PAUSES_S), connect=schema.connect)
    t = _start(s)
    entered.wait(2)
    s.stop()
    release.set()
    t.join(2)
    assert not t.is_alive()
    assert slow.log == ["picks:p1"] and slow.keys == ["p2"]


def test_only_one_scraper_spends_a_quota(kb_home, fast_beats):
    first, second = Fake("picks", source.YOURS), Fake("picks", source.YOURS, ["p1"])
    a = ql.QuotaLoop(Q, [first], executor=Serial(), meter=lambda: None, outages=retries.Outages(retries.PLATFORM_PAUSES_S), connect=schema.connect)
    t = _start(a)
    try:
        _until(lambda: a.status.since is not None)
        b = ql.QuotaLoop(Q, [second], executor=Serial(), meter=lambda: None, outages=retries.Outages(retries.PLATFORM_PAUSES_S), connect=schema.connect)
        b.run()                                    # returns at once: the lease is held
        assert second.asked == 0 and second.keys == ["p1"]
    finally:
        a.stop()
        t.join(2)


def test_the_thread_writes_its_state_to_the_store(kb_home, fast_beats):
    s = ql.QuotaLoop(Q, [Fake("picks", source.YOURS, ["p1"])], executor=Serial(), meter=lambda: None, outages=retries.Outages(retries.PLATFORM_PAUSES_S), connect=schema.connect)
    t = _start(s)
    try:
        _until(lambda: s.status.since is not None)
    finally:
        s.stop()
        t.join(2)
    conn = schema.connect()
    try:
        assert loop_state.read_state(Q)["state"] == "idle"
    finally:
        conn.close()


def test_new_work_wakes_idle_loops_with_due_times_but_no_failure_waits(conn, tmp_path, monkeypatch):
    producer = _scraper(Fake("producer", source.YOURS, ["new"]))
    consumers = [_scraper() for _ in range(4)]
    for consumer, loop_status in zip(consumers, ("idle", "waiting", "paused", "backoff"), strict=True):
        consumer._record_wait(T0, retries.Wait(T0 + 900, loop_status))
    wakes = []
    for i, consumer in enumerate(consumers):
        monkeypatch.setattr(consumer, "wake", lambda i=i: wakes.append(i))
    runner = process.LoopProcess([producer, *consumers],
        store=RailJobStore(tmp_path / "jobs.db"), home_id="local", seed=lambda: None)
    producer.step(conn)
    runner._wake_consumers(T0)
    assert wakes == [0]


def test_a_charged_failure_wakes_no_consumer(conn, tmp_path, monkeypatch):
    failed = Fake("failed", source.YOURS, ["unit"])
    unit = source.Unit("unit", item=source.Item(None, ("unit",), source.Retry()))
    monkeypatch.setattr(failed, "record", lambda *_args: source.Outcome("blip", failure="blip"))
    charges = []
    monkeypatch.setattr(ql, "charge", lambda *args: charges.append(args) or retries.Charged("blip"))
    producer, idle = _scraper(failed), _scraper()
    idle._record_wait(T0, retries.Wait(None, "idle"))
    wakes = []
    monkeypatch.setattr(idle, "wake", lambda: wakes.append(True))
    runner = process.LoopProcess([producer, idle],
        store=RailJobStore(tmp_path / "jobs.db"), home_id="local", seed=lambda: None)
    assert producer.run_unit(conn, failed, unit).result == "blip"
    runner._wake_consumers(T0)
    assert len(charges) == 1
    assert wakes == []


@pytest.mark.parametrize("phase", ["fetch", "record"])
def test_changed_work_is_neither_charged_nor_completed(conn, monkeypatch, phase):
    src = Fake("changed", source.YOURS, ["unit"])
    unit = source.Unit("unit", item=source.Item(None, ("unit",), source.Retry()))
    charges = []

    def changed(*_args):
        raise source.Changed("the input changed")

    monkeypatch.setattr(src, phase, changed)
    monkeypatch.setattr(ql, "charge", lambda *args: charges.append(args) or retries.Charged("blip"))
    loop = _scraper(src)
    assert loop.run_unit(conn, src, unit).result == "changed"
    assert charges == []
    assert src.log == []
    assert loop.completed_units == 0


def test_a_returned_changed_outcome_clears_nothing_and_ends_no_pause(conn):
    """`changed` settles nothing however it is reported; a returned one must not go through
    `succeeded`, which would clear the item's row and end its platform's pause."""
    cleared = []

    class Rows:
        def read(self, _row):
            return source.Retry()

        def write(self, _conn, _key, _retry):
            raise AssertionError("charged")

        def clear(self, _conn, key):
            cleared.append(key)

    src = Fake("archives", source.YOURS, ["unit"])
    src.record = lambda *_args: source.Outcome(source.CHANGED)
    item = source.Item(Rows(), ("unit",), source.Retry("blip", 2, None, "timeout"),
                       platform="substack.com")
    outages = retries.Outages(retries.PLATFORM_PAUSES_S)
    now = ql._dt(T0)
    outages.pause("substack.com", now, 1.0)
    loop = ql.QuotaLoop(Q, [src], executor=Serial(), meter=lambda: None, clock=Clock(), outages=outages,
                        connect=schema.connect)
    assert loop.run_unit(conn, src, source.Unit("unit", item=item)).result == source.CHANGED
    assert cleared == []
    assert outages.paused("substack.com", now)
    assert loop.completed_units == 0


class _Ready:
    """A `RetryRows` that keeps nothing: every item reads as ready, whatever it was charged."""

    def read(self, _row):
        return source.Retry()

    def write(self, _conn, _key, _retry):
        pass

    def clear(self, _conn, _key):
        pass


class Heedless(Fake):
    """A source that never asks whether its platform is paused, and whose items stay ready after
    every failure: three dead Substack domains were hit at fifteen requests a
    second for 48 minutes."""

    def __init__(self, keys, platform="substack", own_site=False, exc=None):
        super().__init__("samples", source.SCOUTING, keys)
        self.platform, self.own_site, self.fetched = platform, own_site, []
        self.exc = exc or RuntimeError("HTTP 403")

    def next_unit(self, _conn, _now):
        self.asked += 1
        return self._unit(self.keys[self.asked % len(self.keys)])

    def _unit(self, key):
        return source.Unit(key, item=source.Item(_Ready(), (key,), source.Retry(), self.platform,
                                                 kind=retries.BLIP, own_site=self.own_site))

    def fetch(self, _conn, unit, _ctx):
        self.fetched.append(unit.key)
        raise self.exc


def test_a_paused_platform_is_held_at_admission_whatever_its_source_asks(conn, no_jitter):
    """The pause is the loop's to keep. Three items fail and their platform pauses; the source
    goes on offering them, and the loop admits none until the pause ends, then one probe."""
    clock = Clock()
    src = Heedless(["a", "b", "c"])
    s = _scraper(src, clock=clock)
    assert [r.result for r in _drain(s, conn)] == ["blip", "blip", "blip"]
    resume = s.outages.until["substack"].timestamp()
    assert resume - T0 == retries.PLATFORM_PAUSES_S[0]
    for _ in range(50):                                   # nothing runs, and the loop knows when
        idle = s.step(conn)
        assert isinstance(idle, retries.Wait) and idle.until <= resume
    assert len(src.fetched) == 3
    clock.t = resume + 1
    assert [r.result for r in _drain(s, conn)] == ["paused"]           # the probe, and only it
    assert len(src.fetched) == 4 and s.outages.pauses["substack"] == 2


def test_three_dead_domains_do_not_pause_the_platform_they_sit_on(conn, no_jitter):
    """Three candidates' Substacks on dead domains must not be read as Substack being down, which
    would make every Oracle's Substack wait. Each is charged its own blip, and the platform runs on."""
    src = Heedless(["a", "b", "c"], own_site=True, exc=SSLError("TLS connect error"))
    s = _scraper(src)
    assert [r.result for r in _drain(s, conn)] == ["blip", "blip", "blip"]
    assert not s.outages.state("substack").in_outage and "substack" not in s.outages.until


def test_a_failed_unit_is_not_admitted_again_at_once_whatever_its_row_says(conn, no_jitter):
    """The backstop under every retry rule: a charged failure holds its unit for `MIN_RETRY_S`
    even when the row reads ready, so a broken rule cannot spin the loop."""
    clock = Clock()
    src = Heedless(["a"], platform=None)                  # no platform: no outage to hold it
    s = _scraper(src, clock=clock)
    assert [r.result for r in _drain(s, conn)] == ["blip"]
    idle = s.step(conn)
    assert isinstance(idle, retries.Wait) and idle.until == T0 + ql.MIN_RETRY_S
    clock.t = T0 + ql.MIN_RETRY_S
    assert [r.result for r in _drain(s, conn)] == ["blip"] and src.fetched == ["a", "a"]


@pytest.mark.parametrize("executor", [Serial, WebPool])
def test_a_failed_admission_backs_off_and_says_so(conn, executor):
    """A raise before any unit is picked backs the loop off through `_wait`, so its status
    and its `scraper_state` row say `backoff`, not `running`."""

    class Unreadable(Fake):
        def next_unit(self, _conn, _now):
            loop.stop()
            raise sqlite3.OperationalError("no such table: oracle_sources")

        candidates = next_unit

    src = Unreadable("picks", source.YOURS, ["p1"])
    loop = ql.QuotaLoop(Q, [src], executor=executor(), meter=lambda: None, clock=Clock(),
                        outages=retries.Outages(retries.PLATFORM_PAUSES_S),
                        connect=lambda: sqlite3.connect(":memory:"))
    loop.executor.execute(loop, conn)
    assert (loop.status.state, loop.status.until) == (
        "backoff", T0 + retries.TRANSIENT_BACKOFF_S[-1])
    row = loop_state.read_state(Q)
    assert (row["state"], row["cause"]) == ("backoff", "transient")


def test_a_fetch_that_raised_after_the_lease_was_lost_charges_nothing(conn, monkeypatch):
    """The lease's next holder redoes the unit, so a fetch that raised once the lease was lost
    settles as `lease_lost`, charging nothing to its item."""
    lost = []

    class Losing(Fake):
        def fetch(self, _conn, _unit, _ctx):
            lost.append(True)
            raise RuntimeError("timeout")

    src = Losing("picks", source.YOURS, ["p1"])
    charges = []
    monkeypatch.setattr(ql, "charge", lambda *args: charges.append(args) or retries.Charged("blip"))
    loop = _scraper(src)
    loop.ctx.lease_lost = lambda: bool(lost)
    unit = source.Unit("p1", item=source.Item(None, ("p1",), source.Retry()))
    assert loop.run_unit(conn, src, unit).result == "lease_lost"
    assert charges == []
