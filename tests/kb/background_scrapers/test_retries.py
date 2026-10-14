"""The one retry policy (`pipeline.worker.quota_loop.retries`): kinds, waits, the stuck rule, and how
three different items failing in a row becomes an X-wide pause that charges nobody."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.circuit_breaker import CircuitOpenError
from pipeline.kb.writes import write_tx
from pipeline.ingestion.utils import SyncAuthError
from pipeline.platforms.x.originals import XUserUnavailable
from pipeline.platforms.x.records import XUnrecognizedPage
from pipeline.platforms.x.rate import XRateLimited
from pipeline.worker.quota_loop.source import Item, Outcome, Retry
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.retries import Wait

from ._support import NOW


@pytest.mark.parametrize("exc, kind", [
    (RuntimeError("HTTP 503"), retries.BLIP),
    (XUnrecognizedPage("no timeline"), retries.BLIP),
    (RuntimeError("UserOriginalsTimeline GraphQL 404 — queryId drifted"), retries.BLIP),
    (XUserUnavailable("suspended"), retries.ACCOUNT),
    (sqlite3.OperationalError("no such column: x"), retries.UNSAVED),
    (sqlite3.OperationalError("database is locked"), None),
    (XRateLimited("spent"), None),
    (SyncAuthError("401"), None),
    (CircuitOpenError("openrouter", 60), None),
])
def test_what_kind_of_failure_it_is(exc, kind):
    assert retries.kind_of(exc) == kind


def test_a_blip_waits_longer_each_time_and_never_more_than_an_hour():
    assert [retries.wait_for(retries.BLIP, n) for n in range(1, 9)] == [
        60, 120, 300, 900, 1800, 3600, 3600, 3600]
    assert retries.wait_for(retries.UNSAVED, 9) == 3600     # ours waits as a blip


def test_an_unreadable_account_waits_hours_then_a_day():
    assert [retries.wait_for(retries.ACCOUNT, n) / 3600 for n in range(1, 6)] == [1, 6, 24, 24, 24]


def test_another_kind_of_failure_starts_its_own_count():
    assert retries.failures_after(retries.BLIP, 4, retries.BLIP) == 5
    assert retries.failures_after(retries.ACCOUNT, 4, retries.BLIP) == 1
    assert retries.failures_after(None, None, retries.BLIP) == 1


def test_waits_are_spread_by_a_fifth_either_way():
    waits = [100.0 * retries.jitter() for _ in range(200)]
    assert all(80.0 <= w <= 120.0 for w in waits) and len(set(waits)) > 1


def test_an_item_is_stuck_after_three_pages_that_would_not_save():
    assert not retries.stuck(retries.UNSAVED, 2)
    assert retries.stuck(retries.UNSAVED, 3)
    assert not retries.stuck(retries.BLIP, 9)


# ── X-wide ────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def outages():
    """X's tracker, as its loop holds it: every item is platform `x`."""
    return retries.Outages(retries.OUTAGE_PAUSES_S, whole_quota=True)


class _Ledger:
    """A `RetryRows` keeping a ledger of what is charged and given back: item → charges."""

    def __init__(self, ledger):
        self.ledger = ledger

    def read(self, row):
        return Retry()

    def write(self, conn, key, retry):
        self.ledger[key[0]] = self.ledger.get(key[0], 0) + 1 if retry.failures else 0

    def clear(self, conn, key):
        pass


def _fail(outages, item, exc, ledger, now=NOW):
    """Charge `exc` on `item` as the loop does, keeping a ledger of what is charged and given
    back. What is not the item's is raised, for the loop to wait on."""
    kind = retries.kind_of(exc)
    if kind is None:
        raise exc
    with closing(sqlite3.connect(":memory:")) as conn:
        return retries.charge(conn, Item(_Ledger(ledger), (item,), Retry(), "x"),
                              Outcome(kind, failure=kind, detail=str(exc),
                                      outage=kind == retries.BLIP, answered=False), outages, now)


def _paused_for(charged, now=NOW) -> float:
    return (charged.paused_until - now).total_seconds()


def test_three_different_items_failing_pause_x_and_give_back_the_charges(outages, no_jitter):
    ledger: dict = {}
    _fail(outages, "a", RuntimeError("timeout"), ledger)
    _fail(outages, "b", RuntimeError("timeout"), ledger)
    paused = _fail(outages, "c", RuntimeError("timeout"), ledger)
    assert (_paused_for(paused), outages.whole_quota) == (30, True)
    assert ledger == {"a": 0, "b": 0, "c": 0}


def test_while_x_is_down_each_probe_pauses_longer_and_charges_nobody(outages, no_jitter):
    ledger: dict = {}
    for item in "abc":
        _fail(outages, item, RuntimeError("timeout"), ledger)
    pauses, now = [], outages.until["x"]
    for _ in range(7):
        paused = _fail(outages, "d", RuntimeError("timeout"), ledger, now)    # the probe
        assert paused.word == "paused"
        pauses.append(_paused_for(paused, now))
        now = paused.paused_until
    assert pauses == [60, 120, 300, 600, 900, 900, 900]
    assert "d" not in ledger
    outages.answered("x")                                    # a page came back
    assert _fail(outages, "d", RuntimeError("timeout"), ledger, now).word == retries.BLIP
    assert ledger["d"] == 1


def test_a_failure_while_the_pause_holds_is_no_probe(outages, no_jitter):
    """A unit in flight when the platform paused, or admitted beside the probe, fails too. It
    charges nobody and leaves the pause where it is; otherwise every such failure climbs
    the ladder and logs a new pause (40,000 times in 48 minutes when this went wrong)."""
    ledger: dict = {}
    for item in "abc":
        _fail(outages, item, RuntimeError("timeout"), ledger)
    until = outages.until["x"]
    for item in "defg":
        late = _fail(outages, item, RuntimeError("timeout"), ledger, NOW + timedelta(seconds=5))
        assert (late.word, late.paused_until) == ("paused", None)
    assert outages.until["x"] == until and outages.pauses["x"] == 1
    assert not set("defg") & set(ledger)


def test_the_same_item_failing_again_is_not_x_failing(outages):
    ledger: dict = {}
    for _ in range(5):
        assert _fail(outages, "a", RuntimeError("timeout"), ledger).word == retries.BLIP
    assert not outages.state("x").in_outage


def test_a_page_coming_back_ends_the_streak(outages):
    ledger: dict = {}
    _fail(outages, "a", RuntimeError("timeout"), ledger)
    _fail(outages, "b", RuntimeError("timeout"), ledger)
    outages.answered("x")
    assert _fail(outages, "c", RuntimeError("timeout"), ledger).word == retries.BLIP


def test_only_blips_count_toward_x_failing(outages):
    ledger: dict = {}
    _fail(outages, "a", XUserUnavailable("gone"), ledger)
    _fail(outages, "b", sqlite3.OperationalError("no such column: x"), ledger)
    _fail(outages, "c", RuntimeError("timeout"), ledger)
    assert not outages.state("x").in_outage


def test_what_is_not_the_items_is_raised_and_charged_to_nobody(outages):
    ledger: dict = {}
    with pytest.raises(XRateLimited):
        _fail(outages, "a", XRateLimited("spent"), ledger)
    assert ledger == {}


# ── giving up ─────────────────────────────────────────────────────────────────────

def test_the_tenth_failure_of_a_kind_has_no_next_try():
    assert retries.retry_at(retries.BLIP, retries.GIVE_UP_AFTER - 1, NOW, 1.0) is not None
    assert retries.retry_at(retries.BLIP, retries.GIVE_UP_AFTER, NOW, 1.0) is None
    assert retries.given_up(retries.GIVE_UP_AFTER) and not retries.given_up(None)



class _Charges:
    """A `RetryRows` keeping the `(item, kind)` charged, minus what is given back."""

    def __init__(self, charged):
        self.charged = charged

    def read(self, row):
        return Retry()

    def write(self, conn, key, retry):
        if retry.failures:
            self.charged.append((key[0], retry.status))
        else:
            self.charged.remove((key[0], retries.BLIP))

    def clear(self, conn, key):
        pass


def _web(health, item, key="substack", *, outage=True, charged=None, now=NOW, exc=None):
    """Charge a blip (or what `exc` is) on a web `item` of platform `key`, as the loop does."""
    charged = charged if charged is not None else []
    kind = retries.BLIP
    if exc is not None:
        kind = retries.kind_of(exc)
        if kind is None:
            raise exc
        outage = kind == retries.BLIP
    with closing(sqlite3.connect(":memory:")) as conn:
        return retries.charge(conn, Item(_Charges(charged), (item,), Retry(), key),
                              Outcome(kind, failure=kind, outage=outage, answered=False), health,
                              now)


def test_three_items_on_one_platform_pause_only_that_platform_and_charge_nobody(no_jitter):
    health, charged = retries.Outages(retries.PLATFORM_PAUSES_S), []
    for item in ("a", "b"):
        _web(health, item, charged=charged)
        _web(health, "blog-x", key="blog:willcb.com", charged=charged)
    assert not health.paused("substack", NOW)
    _web(health, "c", charged=charged)
    assert health.paused("substack", NOW)
    assert health.next_resume(NOW) == NOW + timedelta(hours=1)
    assert not health.paused("blog:willcb.com", NOW)       # an independent blog runs on
    assert charged == [("blog-x", retries.BLIP), ("blog-x", retries.BLIP)]


def test_a_probe_that_fails_pauses_longer_and_a_success_clears_the_platform(no_jitter):
    health, charged = retries.Outages(retries.PLATFORM_PAUSES_S), []
    for item in "abc":
        _web(health, item, charged=charged)
    later = NOW + timedelta(hours=2)
    assert not health.paused("substack", later)
    assert _web(health, "d", charged=charged, now=later).word == "paused"
    assert charged == [] and health.paused("substack", later + timedelta(hours=5))
    health.answered("substack")
    assert not health.state("substack").in_outage and health.next_resume(later) is None


def test_a_wrong_input_or_an_unsaved_post_is_the_items_own(no_jitter):
    health, charged = retries.Outages(retries.PLATFORM_PAUSES_S), []
    for item in "abcd":
        _web(health, item, outage=False, charged=charged)
    assert not health.paused("substack", NOW) and len(charged) == 4


def test_a_web_unit_that_raised_is_a_blip_unless_it_is_not_the_items(no_jitter):
    health, charged = retries.Outages(retries.PLATFORM_PAUSES_S), []
    assert _web(health, "a", key="github", charged=charged,
                exc=TimeoutError("read")).word == retries.BLIP
    with pytest.raises(CircuitOpenError):
        _web(health, "b", key="github", charged=charged, exc=CircuitOpenError("openrouter", 60))
    assert charged == [("a", retries.BLIP)]


# ── what a failure that is not the item's costs the quota (`quota_wait`) ───────────
T = 1_800_000_000.0


@pytest.mark.parametrize("exc, expected", [
    (XRateLimited("spent", reset_at=T + 100), Wait(T + 105, "waiting", "quota_spent")),
    (XRateLimited("spent"), Wait(T + 905, "waiting", "quota_spent")),
    (XRateLimited("spent", reset_at=T - 50), Wait(T + 5, "waiting", "quota_spent")),
    (SyncAuthError("401"), Wait(T + 900, "paused", "x_session")),
    (CircuitOpenError("openrouter", 60), Wait(T + 60, "paused", "model:openrouter")),
    (CircuitOpenError("openrouter", 0), Wait(T + 1, "paused", "model:openrouter")),
    (sqlite3.OperationalError("database is locked"), Wait(T + 2, "backoff", "transient")),
    (RuntimeError("HTTP 503"), None),                         # the unit's own: the backstop
    (XUserUnavailable("gone"), None),
])
def test_what_a_stop_that_is_not_the_items_costs_the_quota(exc, expected):
    assert retries.quota_wait(exc, T, 2) == expected


def test_transient_backoffs_climb_and_stay_at_the_top():
    lock = sqlite3.OperationalError("database is locked")
    waits = [retries.quota_wait(lock, T, n).until - T for n in range(1, 8)]
    assert waits == [1.0, 2.0, 5.0, 15.0, 30.0, 30.0, 30.0]


@pytest.mark.parametrize("exc", [
    XRateLimited("spent"), SyncAuthError("401"), CircuitOpenError("openrouter", 60),
    sqlite3.OperationalError("database is locked"),
])
def test_what_waits_the_quota_is_never_an_items_failure(exc):
    assert retries.kind_of(exc) is None and retries.quota_wait(exc, T, 1) is not None


# ── may this item run, and what one more failure costs it ──────────────────────────
@pytest.mark.parametrize("retry_at, failures, expected", [
    (None, None, True),                                          # never failed
    ((NOW - timedelta(minutes=1)).isoformat(), 3, True),         # its wait is over
    (NOW.isoformat(), 3, True),                                  # exactly now counts as over
    ((NOW + timedelta(minutes=1)).isoformat(), 3, False),        # still waiting
    ((NOW + timedelta(minutes=1)).isoformat(), None, False),
    (None, retries.GIVE_UP_AFTER, False),                        # given up
    ((NOW - timedelta(days=9)).isoformat(), retries.GIVE_UP_AFTER, False),
])
def test_may_the_item_run_now(retry_at, failures, expected):
    assert retries.may_run(retry_at, failures, NOW) is expected


@pytest.mark.parametrize("prior_kind, prior_failures, kind, expected", [
    (retries.BLIP, 2, retries.BLIP, (3, NOW + timedelta(minutes=5))),       # same kind: + 1
    (retries.ACCOUNT, 4, retries.BLIP, (1, NOW + timedelta(minutes=1))),    # another kind: 1
    (None, None, retries.BLIP, (1, NOW + timedelta(minutes=1))),
    ("ingested", 0, retries.UNSAVED, (1, NOW + timedelta(minutes=1))),      # unsaved waits as a blip
    (None, None, retries.ACCOUNT, (1, NOW + timedelta(hours=1))),           # the account ladder
    (retries.ACCOUNT, 1, retries.ACCOUNT, (2, NOW + timedelta(hours=6))),
    (retries.BLIP, retries.GIVE_UP_AFTER - 1, retries.BLIP, (retries.GIVE_UP_AFTER, None)),
])
def test_what_one_more_failure_costs(prior_kind, prior_failures, kind, expected, no_jitter):
    assert retries.next_failure(prior_kind, prior_failures, kind, NOW, 1.0) == expected


def test_a_failed_outage_giveback_rolls_back_the_triggering_charge(outages):
    class Rows:
        def write(self, conn, key, retry):
            with write_tx(conn, site="test.retry"):
                if retry.failures is None:
                    raise RuntimeError("giveback failed")
                conn.execute("UPDATE retries SET failures=? WHERE item=?", (retry.failures, key[0]))

    rows = Rows()
    failure = Outcome("blip", failure="blip", outage=True, answered=False)
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE retries (item TEXT, failures INTEGER)")
        conn.executemany("INSERT INTO retries VALUES (?, 0)", [("a",), ("b",), ("c",)])
        conn.commit()
        for key in ("a", "b"):
            retries.charge(conn, Item(rows, (key,), Retry(), "x"), failure, outages, NOW)
        with pytest.raises(RuntimeError, match="giveback failed"):
            retries.charge(conn, Item(rows, ("c",), Retry(), "x"), failure, outages, NOW)
        assert conn.execute("SELECT failures FROM retries ORDER BY item").fetchall() == [(1,), (1,), (0,)]


def test_a_failed_clear_leaves_the_platform_paused(outages):
    """`succeeded` writes before it tells the tracker, the order `charge` keeps, so a clear
    that fails leaves the platform's pause as it was."""
    class Rows:
        def clear(self, _conn, _key):
            raise sqlite3.OperationalError("disk I/O error")

    outages.pause("x", NOW, 1.0)
    item = Item(Rows(), ("a",), Retry("blip", 2, None, "timeout"), "x")
    with closing(sqlite3.connect(":memory:")) as conn, pytest.raises(sqlite3.OperationalError):
        retries.succeeded(conn, item, Outcome("ok"), outages)
    assert outages.paused("x", NOW)


@pytest.mark.usefixtures("no_jitter")
def test_the_charge_that_pauses_logs_the_give_back_not_a_next_try(outages, monkeypatch):
    """The third item's blip gives every charge back and pauses X. Its log says that, and
    not "next try" for a row it has just restored."""
    lines: list[str] = []
    monkeypatch.setattr(retries, "log", lines.append)
    ledger: dict = {}
    _fail(outages, "a", RuntimeError("timeout"), ledger)
    _fail(outages, "b", RuntimeError("timeout"), ledger)
    del lines[:]
    _fail(outages, "c", RuntimeError("timeout"), ledger)
    assert lines == ["[scrapers] x: outage — 3 charges given back; paused until 12:00:30"]


def _earliest_waits(*retry_ats):
    return [Retry("network", 1, at, None) for at in retry_ats]


@pytest.mark.parametrize("waits, earliest", [
    ([], None),
    (_earliest_waits(None, None), None),
    (_earliest_waits("2026-10-05T12:00:00+00:00", None, "2026-10-05T09:00:00+00:00"),
     datetime(2026, 10, 5, 9, tzinfo=timezone.utc)),
])
def test_earliest_retry_table(waits, earliest):
    assert retries.earliest_retry(waits) == earliest
