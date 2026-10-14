"""The scrapers' process (`quota_loop/process.py`) against fake scrapers, a fake clock and a real control
database: waking on a request, keeping its row armed, exiting once idle, stopping on SIGTERM,
seeding, and a scraper whose thread ends on its own."""
from __future__ import annotations
from pipeline.platforms import connections
from pipeline.worker.quota_loop.retries import Outages, PLATFORM_PAUSES_S
from pipeline.kb import schema

import os
import threading

import pytest

from pipeline.worker.quota_loop import process
from pipeline.worker.quota_loop.loop_state import Status
from pipeline.worker.quota_loop.serial import Serial
from pipeline.worker.job_queue import CLAIM_HOLD_SECONDS, RailJobStore
from pipeline.worker.registry import RAILS, SCRAPERS
from pipeline.oracles.sources import loops

HOME = "42"
T0 = 1_000_000.0
# How long, in real seconds, a fake scraper blocks before returning as if it ended on its own. Far
# beyond any run: the fake clock's thousands of ticks can take over 10s under a loaded suite, and
# a fake that returned then looked to the runner like a loop that died, so it exited early.
HANG_S = 120


class FakeScraper:
    """What the runner reads and calls on a `QuotaLoop`. `run()` blocks until stopped, as a real
    one does; `dies` makes it return at once, as one does whose lease another process holds.
    `idle_since` is what its first look at the store reports (None: it found work)."""

    def __init__(self, quota: str, *, idle_since=None, waiting_until=None, dies=False):
        self.quota = quota
        state = "running" if idle_since is None else (
            "waiting" if waiting_until is not None else "idle")
        self.first_look = Status(state, until=waiting_until, since=idle_since)
        self.status = Status("running")
        self.completed_units = 0
        self.dies = dies
        self.wakes = 0
        self.runs = 0
        self.looked = threading.Event()
        self.stopped = threading.Event()

    def run(self) -> None:
        self.runs += 1
        self.status = self.first_look
        self.looked.set()
        if not self.dies:
            self.stopped.wait(HANG_S)

    def starting(self) -> None:
        self.status = Status("running")

    def wake(self) -> None:
        self.wakes += 1

    def stop(self) -> None:
        self.stopped.set()


class Clock:
    """Time that moves only when the runner sleeps. Each sleep first lets every fake scraper
    started so far take its first look, so a test never races the threads it starts."""

    def __init__(self):
        self.now = T0
        self.ticks = 0
        self.at: dict[int, callable] = {}     # tick → what happens during that sleep
        self.scrapers: list = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        for s in self.scrapers:
            if isinstance(s, FakeScraper) and s.runs:
                assert s.looked.wait(5)
        self.ticks += 1
        if self.ticks > 10_000:
            raise AssertionError("the runner never exited")
        self.now += seconds
        if self.ticks in self.at:
            self.at[self.ticks]()


@pytest.fixture()
def store(tmp_path):
    """A control database with this home's `scrapers` row claimed, as the worker leaves it."""
    s = RailJobStore(tmp_path / "jobs.db")
    s.activate(HOME, SCRAPERS, due_at=T0 - 5)
    s.claim_next(now=T0)
    return s




def _runner(store, scrapers, clock, **kw):
    kw.setdefault("seed", lambda: None)
    clock.scrapers = scrapers
    return process.LoopProcess(scrapers, store=store, home_id=HOME, clock=clock, sleep=clock.sleep, **kw)


def _row(store) -> float:
    return store.get(HOME, SCRAPERS).due_at


def _sigterm() -> None:
    """What `child`'s handler raises in the main thread when the worker sends SIGTERM."""
    raise SystemExit(143)


# ── waking ────────────────────────────────────────────────────────────────────────
def test_the_row_is_armed_before_any_scraper_looks(store):
    """The claim left it a day out; the runner brings it to the safety net, so a request can be
    told from the runner's own write."""
    clock = Clock()
    seen = {}
    x = FakeScraper("x", idle_since=T0)
    def run():
        x.runs += 1
        seen["due"] = _row(store)
        x.looked.set()
        x.stopped.wait(HANG_S)
    x.run = run
    clock.at[1] = _sigterm

    _runner(store, [x], clock).run()

    assert seen["due"] == T0 + process.SAFETY_NET_S
    assert seen["due"] < T0 + CLAIM_HOLD_SECONDS


def test_a_request_wakes_every_scraper_and_the_row_is_armed_again(store):
    clock = Clock()
    x, web = FakeScraper("x"), FakeScraper("web")       # busy: never idle, so no exit
    clock.at[5] = lambda: store.activate(HOME, SCRAPERS, due_at=clock.now)
    clock.at[8] = _sigterm
    before = {}
    clock.at[4] = lambda: before.update(x=x.wakes, web=web.wakes)

    _runner(store, [x, web], clock).run()

    assert x.wakes == before["x"] + 1 and web.wakes == before["web"] + 1


def test_a_long_run_keeps_its_row_ahead_of_now_so_no_request_is_swallowed(store):
    """A big ask runs for hours. Armed once, the row would fall into the past, and a request
    (`MIN(due_at, now)`) would change nothing again."""
    clock = Clock()
    x = FakeScraper("x")
    seen = []
    for tick in range(60, 3600, 60):
        clock.at[tick] = lambda: seen.append(_row(store) - clock.now)
    clock.at[3600] = _sigterm

    _runner(store, [x], clock).run()

    assert min(seen) >= process.REARM_MARGIN_S - process.POLL_S
    assert x.wakes == 1          # re-arming is not a request: only the start woke it


# ── exiting ───────────────────────────────────────────────────────────────────────
def test_once_all_are_idle_for_a_minute_it_writes_its_next_start_and_exits(store):
    clock = Clock()
    x = FakeScraper("x", idle_since=T0, waiting_until=T0 + 600)
    web = FakeScraper("web", idle_since=T0, waiting_until=None)

    out = _runner(store, [x, web], clock).run()

    assert out == {"status": "ok"}
    assert clock.now - T0 == pytest.approx(process.IDLE_EXIT_S)
    assert _row(store) == T0 + 600                       # the earliest next try
    assert x.stopped.is_set() and web.stopped.is_set()


def test_the_next_start_is_never_later_than_the_safety_net(store):
    clock = Clock()
    x = FakeScraper("x", idle_since=T0, waiting_until=T0 + 5 * 3600)

    _runner(store, [x], clock).run()

    assert _row(store) == clock.now + process.SAFETY_NET_S


def test_a_scraper_running_a_unit_keeps_the_process_up(store):
    clock = Clock()
    x = FakeScraper("x", idle_since=T0)
    web = FakeScraper("web", idle_since=None)            # running units
    clock.at[300] = lambda: setattr(web, "status", Status("idle", since=clock.now))

    _runner(store, [x, web], clock).run()

    assert clock.now - T0 == pytest.approx(300 + process.IDLE_EXIT_S)


def test_a_try_due_within_two_minutes_is_not_worth_a_new_process(store):
    """An X window resetting in a minute: stay for it rather than exit and be started again."""
    clock = Clock()
    x = FakeScraper("x", idle_since=T0, waiting_until=T0 + 90)
    clock.at[90] = lambda: setattr(x, "status", Status("waiting", until=T0 + 900, since=T0))

    _runner(store, [x], clock).run()

    assert clock.now - T0 == pytest.approx(90)
    assert _row(store) == T0 + 900


def test_a_wake_restarts_the_idle_minute(store):
    clock = Clock()
    x = FakeScraper("x", idle_since=T0)
    clock.at[40] = lambda: store.activate(HOME, SCRAPERS, due_at=clock.now)

    _runner(store, [x], clock).run()

    assert clock.now - T0 == pytest.approx(40 + process.IDLE_EXIT_S)


def test_a_request_landing_as_it_exits_keeps_it_running(store, monkeypatch):
    """The exit's write is a compare-and-set: a request that lands between the last read and
    that write makes it fail, and the runner wakes everyone instead of leaving."""
    clock = Clock()
    x = FakeScraper("x", idle_since=T0, waiting_until=T0 + 600)
    real = store.rearm
    raced = []

    def rearm(home, rail, due_at, *, expected):
        if not raced and due_at == T0 + 600:                          # the exit's write
            raced.append(clock.now)
            store.activate(HOME, SCRAPERS, due_at=clock.now)
        return real(home, rail, due_at, expected=expected)

    monkeypatch.setattr(store, "rearm", rearm)
    _runner(store, [x], clock).run()

    assert raced == [T0 + process.IDLE_EXIT_S]
    assert x.wakes == 2                                    # the start, and the raced request
    assert clock.now - T0 == pytest.approx(2 * process.IDLE_EXIT_S + 1)


def test_run_by_hand_without_a_row_it_still_exits(tmp_path):
    clock = Clock()
    x = FakeScraper("x", idle_since=T0)

    out = _runner(RailJobStore(tmp_path / "empty.db"), [x], clock).run()

    assert out == {"status": "ok"}
    assert RailJobStore(tmp_path / "empty.db").list_jobs() == []


# ── stopping ──────────────────────────────────────────────────────────────────────
def test_sigterm_stops_every_scraper_and_makes_the_row_due_now(store):
    """`child` raises SIGTERM as a `SystemExit` in the main thread. The restart should
    bring the runner straight back, not a safety net later."""
    clock = Clock()
    x, web = FakeScraper("x"), FakeScraper("web")
    clock.at[3] = _sigterm

    out = _runner(store, [x, web], clock).run()

    assert out == {"status": "stopped"}
    assert x.stopped.is_set() and web.stopped.is_set()
    assert _row(store) <= clock.now


def test_the_real_scraper_releases_its_lease_on_the_stop(store, kb_home):
    """End to end in this process: a real `QuotaLoop` holding a real lease, stopped by the
    runner's SIGTERM path."""
    import sqlite3

    from pipeline.worker.quota_loop.loop import QuotaLoop
    from tests.kb._rail_children import IdleSource

    clock = Clock()
    s = QuotaLoop("test:quota", [IdleSource()], executor=Serial(), outages=Outages(PLATFORM_PAUSES_S), connect=schema.connect)
    holder = {}

    def held():
        conn = sqlite3.connect(str(kb_home / "loops.db"))
        try:
            row = conn.execute("SELECT holder FROM sync_lock WHERE name='test:quota'").fetchone()
        except sqlite3.OperationalError:
            row = None
        conn.close()
        return row[0] if row else None

    def stop():
        for _ in range(200):
            if held():
                break
            threading.Event().wait(0.02)
        holder["during"] = held()
        raise SystemExit(143)

    clock.at[1] = stop
    _runner(store, [s], clock).run()

    assert holder["during"] is not None
    assert held() is None


# ── seeding ───────────────────────────────────────────────────────────────────────
def test_it_seeds_at_start_and_about_hourly_and_wakes_the_scrapers_after(store):
    clock = Clock()
    seeded = []
    x = FakeScraper("x")
    clock.at[2 * 3600 + 30] = _sigterm

    _runner(store, [x], clock, seed=lambda: seeded.append(clock.now)).run()

    assert seeded == [T0, T0 + 3600, T0 + 7200]
    assert x.wakes == 3


def test_seeding_registers_pairs_with_x_only_while_x_is_connected(kb_home, monkeypatch):
    from pipeline.oracles import pairs

    calls = []
    monkeypatch.setattr(connections, "_connected_x", lambda: False)
    monkeypatch.setattr(pairs, "seed_from_entities",
                        lambda conn, **kw: calls.append(kw))
    loops.seed()
    assert calls == [{"include_x": False}]


def test_a_seeding_that_fails_never_stops_the_process(kb_home, monkeypatch):
    from pipeline.oracles import pairs

    def boom(conn, **kw):
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(pairs, "seed_from_entities", boom)
    loops.seed()


# ── a scraper whose thread ends on its own ────────────────────────────────────────
def test_a_scraper_that_ended_on_its_own_is_started_again(store):
    clock = Clock()
    x = FakeScraper("x", dies=True)
    web = FakeScraper("web")                              # busy throughout
    clock.at[150] = _sigterm

    _runner(store, [x, web], clock).run()

    assert x.runs >= 2


def test_with_every_quota_held_elsewhere_it_exits_instead_of_retrying_forever(store):
    """Another process holds both leases and is doing the work; this one leaves, and the next
    start tries again."""
    clock = Clock()
    x, web = FakeScraper("x", dies=True), FakeScraper("web", dies=True)

    out = _runner(store, [x, web], clock).run()

    assert out == {"status": "ok"}
    assert clock.now - T0 <= process.IDLE_EXIT_S + process.RESTART_AFTER_S + 2


# ── the job ───────────────────────────────────────────────────────────────────────
QUOTAS = ["x:UserOriginalsTimeline", "x:TweetDetail", "x:TweetResultsByRestIds",
              "images:x-oracles", "web",
              "x:Bookmarks", "x:Following", "x:Likes", "x:ListsManagementPageTimeline",
              "x:ListMembers", "saved:x", "images:x", "curation:maintenance", "publishing"]


def test_with_no_routable_model_the_job_ends_before_anything_starts(monkeypatch):
    """The whole-process gate: with no routable model the job ends before anything starts."""
    from pipeline import background
    monkeypatch.setattr(background, "models_unroutable", lambda job: "no provider survives")
    monkeypatch.setattr(background, "scrapers", lambda: pytest.fail("scrapers built"))

    assert background.run() == {"status": "models_unroutable",
                                "message": "no provider survives"}


def test_the_job_runs_one_process_over_every_quota_for_this_home(kb_home, monkeypatch):
    from pipeline import background
    built = {}

    class Recorder:
        def __init__(self, scrapers, *, store, home_id, **kw):
            built.update(quotas=[s.quota for s in scrapers], store=store, home_id=home_id)

        def run(self):
            return {"status": "ok", "ran": True}
    monkeypatch.setattr(background, "LoopProcess", Recorder)
    monkeypatch.setattr(background, "keep_awake", lambda: None)

    assert background.run() == {"status": "ok", "ran": True}
    assert built["quotas"] == QUOTAS
    assert isinstance(built["store"], RailJobStore) and built["home_id"] == "local"


def test_the_job_holds_off_idle_sleep_for_exactly_its_own_lifetime(monkeypatch):
    """`caffeinate -w <this pid>`: the assertion ends when the scrapers process does."""
    from pipeline import background
    spawned = []
    monkeypatch.setattr(background.subprocess, "Popen", lambda argv: spawned.append(argv))

    background.keep_awake()

    assert spawned == [["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())]]


def test_a_failed_spawn_never_stops_the_scrapers(monkeypatch, capsys):
    from pipeline import background

    def fail(argv):
        raise FileNotFoundError("/usr/bin/caffeinate")
    monkeypatch.setattr(background.subprocess, "Popen", fail)

    background.keep_awake()

    assert "could not hold off sleep" in capsys.readouterr().err


def test_one_scraper_per_quota_and_one_web_pool_for_every_domain():
    """Curation's Substack sources and frontier's searches and admissions join oracles' web pool,
    under its outage state, rather than a second pool on the same quota."""
    from pipeline import background
    loops = background.scrapers()
    assert [s.quota for s in loops] == QUOTAS
    assert [s.name for s in loops[0].sources] == ["picks", "check", "new_posts", "scouting"]
    web = next(s for s in loops if s.quota == "web")
    assert [s.name for s in web.sources][-6:] == [
        "substack_follows:SubstackFollows", "substack_subscriptions:SubstackSubscriptions",
        "substack_saved:SubstackSaved", "saved_substack_posts",
        "frontier_searches", "frontier_admissions"]
    assert all(s.outages is web.outages for s in web.sources[-3:])
    assert web.sources[-4].via.outages is web.outages



def test_an_upload_holds_a_serial_quota_of_its_own():
    """It does not join the web pool, so a long upload holds no web slot."""
    from pipeline import background
    from pipeline.worker.quota_loop.loop import QuotaLoop
    loops = {s.quota: s for s in background.scrapers()}
    assert [s.name for s in loops["publishing"].sources] == ["publications"]
    assert type(loops["publishing"]) is QuotaLoop


def test_the_workers_cadence_is_the_runners_safety_net():
    assert RAILS[SCRAPERS].cadence == process.SAFETY_NET_S


# ── the follow-up jobs (`follow_ups.py`) ─────────────────




# ── the process's decisions, one row each ─────────────────────────────────────────
@pytest.mark.parametrize("due_at, armed, now, expected", [
    (T0 + 5, T0 + 900, T0, (True, True)),                # the row moved: re-arm, wake
    (T0 + 900, None, T0, (True, True)),                  # nothing armed yet
    (T0 + 900, T0 + 900, T0, (False, False)),            # unchanged, far ahead: nothing
    (T0 + 200, T0 + 200, T0, (True, False)),             # unchanged, close: re-arm quietly
])
def test_what_one_read_of_the_row_calls_for(due_at, armed, now, expected):
    assert process.watch(due_at, armed, now) == expected


@pytest.mark.parametrize("idle_since, tries, expected", [
    (None, [], None),                                    # a scraper is running a unit
    (T0 - 30, [], None),                                 # not idle long enough
    (T0 - 90, [T0 + 60], None),                          # a try within two minutes: stay
    (T0 - 90, [], T0 + 900),                             # nothing coming: the safety net
    (T0 - 90, [T0 + 600, T0 + 300], T0 + 300),           # the soonest try
    (T0 - 90, [T0 + 5000], T0 + 900),                    # capped at the safety net
])
def test_when_the_next_process_should_start(idle_since, tries, expected):
    assert process.next_start(T0, idle_since, tries, 60.0, 120.0, 900.0) == expected


@pytest.mark.parametrize("slots, expected", [
    ([(True, None, None)], None),                        # a loop running a unit
    ([(True, None, T0 - 50), (True, None, None)], None),  # one idle, one running
    ([(True, None, T0 - 50)], T0 - 50),                  # idle since its look
    ([(True, None, T0 - 200)], T0 - 100),                # a later wake counts as activity
    ([(True, None, T0 - 50), (False, T0 - 20, None)], T0 - 20),   # a thread ended since
    ([(False, None, None)], T0 - 100),                   # ended, not yet seen: the wake
])
def test_since_when_every_loop_has_been_idle(slots, expected):
    assert process.idle_since(T0 - 100, slots) == expected


def test_completed_units_wake_idle_consumers_without_interrupting_quota_waits(store):
    clock = Clock()
    producer = FakeScraper("producer")
    idle = FakeScraper("idle", idle_since=T0)
    waiting = FakeScraper("waiting", idle_since=T0, waiting_until=T0 + 900)
    before = {}
    clock.at[2] = lambda: before.update(idle=idle.wakes, waiting=waiting.wakes)
    clock.at[3] = lambda: setattr(producer, "completed_units", 1)
    clock.at[5] = _sigterm
    _runner(store, [producer, idle, waiting], clock).run()
    assert idle.wakes == before["idle"] + 1
    assert waiting.wakes == before["waiting"]
