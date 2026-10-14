"""The three Oracle sources (`background_scrapers.x_timeline.oracle_walks`): what each owes, in
what order, how a page moves its pair's frontier, and how a failure is charged to its pair.

The page itself is faked (`pages.walk_page`); `tests/kb/test_x_pages.py` covers it."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.ingestion.utils import SyncAuthError
from pipeline.platforms.x.originals import XUserUnavailable, id_at
from pipeline.platforms.x.rate import XRateLimited
from pipeline.oracles import windows
from pipeline.timeparse import parse_ts
from pipeline.worker.quota_loop import loop_state, retries
from pipeline.worker.quota_loop.retries import Wait
from pipeline.worker.quota_loop.source import Context
from pipeline.oracles import roster
from pipeline.oracles import asks
from pipeline.oracles import checks
from pipeline.oracles.sources.timeline import pages
from pipeline.oracles.sources.timeline import walks as tl
from pipeline.oracles.sources.timeline import sources

from .._support import NOW, days, page, pair, sample, stored


# ── picks: backward, toward the ask ───────────────────────────────────────────────

def test_a_first_page_stamps_the_moment_it_read_the_top(conn, loop, ctx, walk):
    served, calls = walk
    pair(conn)
    read = NOW - timedelta(seconds=3)
    served.append(page("500", days(5), read_at=read))

    picks = sources.Picks()
    loop.run_unit(conn, picks, picks.next_unit(conn, NOW))

    row = stored(conn)
    assert calls == [("carol", None)]
    assert parse_ts(row.last_pulled_at) == read
    assert parse_ts(row.covered_from) == days(5) and row.back_tail_id == "500"


def test_every_later_page_resumes_below_the_last_post_read(conn, loop, ctx, walk):
    """Twenty posts inside one minute would stall a walk that resumed at a date plus ten
    minutes: it would re-read the same posts forever. Resuming by id cannot."""
    served, calls = walk
    pair(conn)
    minute = days(3)
    served += [page("900", minute), page("880", minute), page("860", minute - timedelta(days=200))]
    picks = sources.Picks()
    for _ in range(3):
        loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    assert [below for _h, below in calls] == [None, "900", "880"]
    assert picks.next_unit(conn, NOW) is None               # back past six months: nothing owed


def test_a_pair_walked_before_the_scrapers_resumes_at_its_date_once(conn, loop, ctx, walk):
    served, calls = walk
    covered = days(30)
    pair(conn, covered_from=covered.isoformat(), last_pulled_at=days(1).isoformat())
    served += [page("700", days(31)), page("650", days(32))]
    picks = sources.Picks()
    loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    expected = str(id_at(covered + windows.X_RESUME_OVERLAP) + 1)
    assert [below for _h, below in calls] == [expected, "700"]
    assert parse_ts(stored(conn).last_pulled_at) == days(1)  # a backward page never moves it


def test_a_timeline_that_ends_is_covered_as_far_as_was_asked(conn, loop, ctx, walk):
    served, _calls = walk
    cid = pair(conn)
    asks.record_ask(conn, cid, x_from=days(365), now=NOW)
    served.append(page("100", days(20), ended=True))
    picks = sources.Picks()
    loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    assert parse_ts(stored(conn).covered_from) <= days(365)
    assert picks.next_unit(conn, NOW) is None


def test_a_timeline_found_later_inherits_its_oracles_ask(conn):
    cid = pair(conn, covered_from=days(200).isoformat())
    picks = sources.Picks()
    assert picks.next_unit(conn, NOW) is None               # six months held, nothing asked
    asks.record_ask(conn, cid, x_from=days(365), now=NOW)
    assert picks.next_unit(conn, NOW) is not None


def test_the_least_covered_pick_goes_next(conn):
    a = pair(conn, "alice", covered_from=days(60).isoformat())
    b = pair(conn, "bob", covered_from=days(10).isoformat())
    c = pair(conn, "cara")
    for cid in (b, a, c):
        asks.record_ask(conn, cid, x_from=days(365), now=NOW)
    picks = sources.Picks()

    def handle_next():
        return picks.next_unit(conn, NOW).payload[0].source_key

    assert handle_next() == "cara"                          # never pulled first
    conn.execute("UPDATE oracle_sources SET covered_from=? WHERE source_key='cara'",
                 (days(100).isoformat(),))
    conn.commit()
    assert handle_next() == "bob"                           # then whoever reaches least far back
    conn.execute("UPDATE oracle_sources SET covered_from=? WHERE source_key IN ('bob','alice')",
                 (days(60).isoformat(),))
    conn.commit()
    assert handle_next() == "bob"                           # a tie goes to the earlier ask


# ── forward: new posts and checks ─────────────────────────────────────────────────

def test_a_forward_walk_stamps_the_moment_it_read_the_top(conn, loop, ctx, walk):
    served, calls = walk
    pair(conn, last_pulled_at=days(10).isoformat(), covered_from=days(400).isoformat())
    top = NOW - timedelta(minutes=5)
    served += [page("900", days(4), read_at=top), page("800", days(11))]
    new_posts = sources.NewPosts()

    assert loop.run_unit(conn, new_posts, new_posts.next_unit(conn, NOW)).result == "walking"
    row = stored(conn)
    assert parse_ts(row.last_pulled_at) == days(10)        # not yet: the walk is mid-way
    assert (parse_ts(row.fwd_started_at), row.fwd_tail_id) == (top, "900")

    loop.run_unit(conn, new_posts, new_posts.next_unit(conn, NOW))
    row = stored(conn)
    assert [below for _h, below in calls] == [None, "900"]
    assert parse_ts(row.last_pulled_at) == top
    assert (row.fwd_started_at, row.fwd_tail_id) == (None, None)


def test_new_posts_keep_every_confirmed_oracle_current_unasked(conn):
    """No consent gate: OPYT never asks. A pair past its refresh time is
    owed new posts; one inside it is not."""
    pair(conn, last_pulled_at=days(10).isoformat(), covered_from=days(400).isoformat())
    assert sources.NewPosts().next_unit(conn, NOW) is not None
    conn.execute("UPDATE oracle_sources SET last_pulled_at=?", (NOW.isoformat(),))
    conn.commit()
    assert sources.NewPosts().next_unit(conn, NOW) is None


def test_a_check_is_satisfied_by_any_forward_walk(conn, loop, ctx, walk, monkeypatch):
    served, _calls = walk
    monkeypatch.setattr("pipeline.worker.job_queue.request_now", lambda *a, **k: True)
    cid = pair(conn, last_pulled_at=days(10).isoformat(), covered_from=days(400).isoformat())
    checks.request_check(conn, cid, now=NOW - timedelta(minutes=1))
    assert sources.Check().next_unit(conn, NOW) is not None

    served.append(page("900", days(11), read_at=NOW))
    new_posts = sources.NewPosts()
    loop.run_unit(conn, new_posts, new_posts.next_unit(conn, NOW))

    assert sources.Check().next_unit(conn, NOW) is None


def test_a_check_on_a_pair_never_pulled_reads_it_from_the_top(conn, loop, ctx, walk, monkeypatch):
    served, calls = walk
    monkeypatch.setattr("pipeline.worker.job_queue.request_now", lambda *a, **k: True)
    cid = pair(conn)
    checks.request_check(conn, cid, now=NOW - timedelta(minutes=1))
    served.append(page("900", days(2), read_at=NOW))
    check = sources.Check()
    loop.run_unit(conn, check, check.next_unit(conn, NOW))
    assert calls == [("carol", None)] and check.next_unit(conn, NOW) is None


# ── failures ──────────────────────────────────────────────────────────────────────

def test_a_failure_is_charged_to_its_pair_by_kind(conn, loop, ctx, walk, no_jitter, monkeypatch):
    served, _calls = walk
    monkeypatch.setattr(tl, "utc_now", lambda: NOW)
    pair(conn)
    picks = sources.Picks()
    served += [RuntimeError("HTTP 503"), RuntimeError("HTTP 503"), XUserUnavailable("gone")]
    seen = []
    for i in range(3):
        loop.run_unit(conn, picks, picks.next_unit(conn, NOW + timedelta(days=i + 1)))
        r = stored(conn)
        seen.append((r.last_status, r.failures,
                     round((parse_ts(r.retry_at) - NOW).total_seconds())))
    assert seen == [("blip", 1, 60), ("blip", 2, 120), ("account", 1, 3600)]
    assert "gone" in stored(conn).last_error


def test_a_success_clears_the_failures(conn, loop, ctx, walk):
    served, _calls = walk
    pair(conn, last_status="blip", failures=4, last_error="HTTP 503",
          retry_at=days(1).isoformat())
    served.append(page("900", days(2)))
    picks = sources.Picks()
    loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    row = stored(conn)
    assert (row.failures, row.retry_at, row.last_error) == (0, None, None)


def test_a_page_that_will_not_save_is_ours_and_after_three_the_pick_is_stuck(conn, loop, ctx, walk,
                                                                             no_jitter):
    from pipeline.circuit_breaker import status
    served, _calls = walk
    pair(conn)
    picks = sources.Picks()
    for _ in range(3):
        assert not retries.stuck(stored(conn).last_status, stored(conn).failures)
        served.append(page("900", days(2), complete=False))
        later = max(parse_ts(stored(conn).retry_at) or NOW, NOW) + timedelta(seconds=1)
        loop.run_unit(conn, picks, picks.next_unit(conn, later))
    row = stored(conn)
    assert row.covered_from is None                         # the frontier never moved
    assert retries.stuck(row.last_status, row.failures) and "not saved" in row.last_error
    assert not [b for b in status() if b["fail_count"]]     # no breaker on this path


def test_a_database_error_that_is_not_a_lock_is_ours(conn, loop, ctx, walk, no_jitter):
    served, _calls = walk
    pair(conn)
    served.append(sqlite3.OperationalError("no such column: frobnicate"))
    picks = sources.Picks()
    assert loop.run_unit(conn, picks, picks.next_unit(conn, NOW)).result == "unsaved"
    assert stored(conn).last_status == "unsaved"


# ── X-wide: three different accounts failing is X failing ─────────────────────────

def test_three_different_accounts_failing_pause_x_and_charge_nobody(conn, loop, ctx, walk, no_jitter):
    served, _calls = walk
    for h in ("alice", "bob", "cara"):
        pair(conn, h)
    picks = sources.Picks()
    served += [RuntimeError("timeout")] * 2
    assert loop.run_unit(conn, picks, picks.next_unit(conn, NOW)).result == "blip"
    assert loop.run_unit(conn, picks, picks.next_unit(conn, NOW)).result == "blip"
    served.append(RuntimeError("timeout"))
    paused = loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    assert paused == Wait(NOW.timestamp() + 30, "paused", "x_outage")
    assert loop_state.read_state(tl.QUOTA)["cause"] == "x_outage"
    for h in ("alice", "bob", "cara"):                      # what the streak charged is given back
        row = stored(conn, h)
        assert (row.failures, row.retry_at, row.last_status) == (None, None, None)

    # The probe after each pause: failing again pauses longer, still charging nobody.
    served.append(RuntimeError("timeout"))
    after = NOW.timestamp() + 31                            # the 30 s pause is over
    loop.clock = lambda: after
    again = loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    assert again == Wait(after + 60, "paused", "x_outage")
    assert all(stored(conn, h).failures is None for h in ("alice", "bob", "cara"))

    # A page that comes back ends it.
    served.append(page("900", days(2)))
    loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    assert not loop.outages.state("x").in_outage


def test_the_same_account_failing_twice_is_not_x_failing(conn, loop, ctx, walk, no_jitter,
                                                        monkeypatch):
    served, _calls = walk
    # Failures are charged on the walk's clock: pinned, or the wall clock overtakes `NOW`.
    monkeypatch.setattr(tl, "utc_now", lambda: NOW)
    pair(conn, "alice")
    pair(conn, "bob")
    picks = sources.Picks()
    for _ in range(3):
        served.append(RuntimeError("timeout"))
        unit = picks.next_unit(conn, NOW + timedelta(days=1))
        if unit.payload[0].source_key == "bob":
            conn.execute("UPDATE oracle_sources SET retry_at=? WHERE source_key='bob'",
                         ((NOW + timedelta(days=9)).isoformat(),))
            conn.commit()
            unit = picks.next_unit(conn, NOW + timedelta(days=1))
        loop.run_unit(conn, picks, unit)                     # alice, alice, alice: no pause
    assert not loop.outages.state("x").in_outage


def test_an_unreadable_account_is_not_x_failing(conn, loop, ctx, walk, no_jitter):
    served, _calls = walk
    for h in ("alice", "bob", "cara"):
        pair(conn, h)
    picks = sources.Picks()
    served += [XUserUnavailable("gone"), RuntimeError("timeout"), RuntimeError("timeout")]
    for _ in range(3):
        loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    assert not loop.outages.state("x").in_outage


@pytest.mark.parametrize("exc", [XRateLimited("spent", reset_at=0.0), SyncAuthError("401"),
                                 sqlite3.OperationalError("database is locked")])
def test_a_session_wide_stop_is_raised_and_charged_to_nobody(conn, ctx, walk, exc):
    from pipeline.circuit_breaker import status
    served, _calls = walk
    pair(conn)
    served.append(exc)
    picks = sources.Picks()
    with pytest.raises(type(exc)):
        picks.fetch(conn, picks.next_unit(conn, NOW), ctx)
    row = stored(conn)
    assert (row.retry_at, row.last_error) == (None, None)
    assert not [b for b in status() if b["fail_count"]]


def test_a_dead_session_is_read_again_next_time(conn, ctx, walk):
    served, _calls = walk
    pair(conn)
    served.append(SyncAuthError("401"))
    picks = sources.Picks()
    with pytest.raises(SyncAuthError):
        picks.fetch(conn, picks.next_unit(conn, NOW), ctx)
    assert ctx.cache["page_kit"].forgot == 1




# ── a discovered account joins its Oracle ─────────────────────────────────────────

def _discovered(conn, handle="hyper_soren"):
    """A Substack-rooted Oracle whose X handle discovery registered, case folded."""
    from pipeline.oracles import registry
    from pipeline.kb import schema
    schema.upsert_entity(conn, "substack:soren", name="Soren",
                         identity_links=["https://soren.substack.com"])
    roster.upsert_oracle(conn, "substack:soren", name="Soren")
    registry.upsert_source(conn, registry.SourceRow("substack:soren", "x", handle))
    conn.commit()


def test_a_discovered_accounts_first_page_folds_it_into_its_oracle(conn, loop, ctx, walk,
                                                                     monkeypatch):
    from pipeline.oracles import registry
    from pipeline.oracles import pairs
    from pipeline.kb import schema
    served, _calls = walk
    _discovered(conn)

    def first_page(conn_, kit, **kw):
        # What the page writer does: the author entity, with X's own spelling and their site.
        schema.upsert_entity(conn_, "x:user:99", name="Soren",
                             identity_links=["https://soren.substack.com"],
                             profile={"handle": "Hyper_Soren"})
        return page("500", days(5))
    monkeypatch.setattr("pipeline.oracles.sources.timeline.pages.walk_page", first_page)

    picks = sources.Picks()
    loop.run_unit(conn, picks, picks.next_unit(conn, NOW))

    head = schema.current_canonical(conn, "substack:soren")
    assert schema.current_canonical(conn, "x:user:99") == head
    (row,) = [r for r in registry.list_sources(conn) if r.source_type == "x"]
    assert (row.canonical_id, row.source_key) == (head, "Hyper_Soren")
    assert [r for r in tl.x_pairs(conn)] and pairs.seed_from_entities(conn)["pairs"] >= 1
    assert len([r for r in registry.list_sources(conn) if r.source_type == "x"]) == 1


def test_a_pair_already_in_its_cluster_is_not_resolved_again(conn, loop, ctx, walk, monkeypatch):
    from pipeline.kb import resolve
    served, _calls = walk
    pair(conn)
    monkeypatch.setattr("pipeline.oracles.sources.timeline.pages.stored_x_user_id", lambda *a: "99")
    monkeypatch.setattr(resolve, "resolve_entities", lambda conn: pytest.fail("resolved"))
    served.append(page("500", days(5)))
    picks = sources.Picks()
    loop.run_unit(conn, picks, picks.next_unit(conn, NOW))
    assert stored(conn).back_tail_id == "500"


def test_nothing_from_a_page_lands_after_its_oracle_is_forgotten(conn, loop, ctx, monkeypatch):
    """The page checks the roster inside its write transaction. A forget that committed
    while the page was read refuses the page, uncharged, and what the Oracle had before the
    forget is left to the forget."""
    from pipeline.oracles import registry
    from tests.kb.test_export import _add
    cid = pair(conn)
    _add(conn, "xprofile:old", "x", "opinion", cid, [], "before")
    wrote = []
    monkeypatch.setattr(pages, "write_page", wrote.append)

    def page_then_forgotten(conn_, kit, **kw):
        conn_.execute("DELETE FROM oracles")
        conn_.execute("DELETE FROM oracle_sources")
        conn_.commit()
        return page("500", days(5))
    monkeypatch.setattr("pipeline.oracles.sources.timeline.pages.walk_page", page_then_forgotten)

    picks = sources.Picks()
    assert loop.run_unit(conn, picks, picks.next_unit(conn, NOW)).result == "changed"
    assert wrote == []
    assert {r[0] for r in conn.execute("SELECT atom_id FROM atoms")} == {"xprofile:old"}
    assert registry.list_sources(conn) == []


# ── a page and its frontier commit together ─────────────────────────────────────────

def test_a_page_rewritten_meanwhile_is_not_written_and_its_frontier_stays(conn, loop, ctx, walk,
                                                                         monkeypatch):
    """Another writer rewrote one of the page's posts while it was read: writing the page would
    write over it. Nothing is written, the pair is not charged, and a later unit reads it again."""
    served, _calls = walk
    pair(conn)
    served.append(page("500", days(5)))
    monkeypatch.setattr(pages, "page_current", lambda *_a: False)

    picks = sources.Picks()
    assert loop.run_unit(conn, picks, picks.next_unit(conn, NOW)).result == "changed"
    row = stored(conn)
    assert (row.back_tail_id, row.covered_from, row.last_pulled_at) == (None, None, None)
    assert row.failures in (None, 0)


def test_a_lease_lost_while_the_page_was_read_writes_neither(conn, loop, ctx, monkeypatch):
    pair(conn)
    wrote = []
    monkeypatch.setattr(pages, "write_page", wrote.append)

    def page_then_lease_lost(*_a, **_kw):
        loop.ctx.lease_lost = lambda: True
        return page("500", days(5))
    monkeypatch.setattr(pages, "walk_page", page_then_lease_lost)

    picks = sources.Picks()
    assert loop.run_unit(conn, picks, picks.next_unit(conn, NOW)).result == "lease_lost"
    assert wrote == [] and stored(conn).back_tail_id is None


# ── the page kit ──────────────────────────────────────────────────────────────────

def test_the_page_kit_is_built_once_per_context(conn):
    """The kit holds the X session and the user ids it looked up, so every unit on the loop
    shares one rather than reading the cookies and resolving handles again."""
    ctx = Context(tl.QUOTA)

    kit = tl.page_kit(conn, ctx)

    assert isinstance(kit, pages.PageKit) and ctx.cache["page_kit"] is kit
    assert tl.page_kit(conn, ctx) is kit
