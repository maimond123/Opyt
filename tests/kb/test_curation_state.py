"""curation_state — the LIST clock. Pure state + pure clock math, no network anywhere.

Every test here defends the one decision the table exists for: `last_attempt_at` and `last_ok_at`
are TWO columns because they answer two different questions, and collapsing them re-creates one of
the two bugs the split prevents — a broken collector retrying forever, or one outage buying a full
floor of false freshness.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.kb import curation_state as cs
from pipeline.kb import schema

T0 = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ── schema ──────────────────────────────────────────────────────────────────────
def test_ddl_layers_onto_a_plain_schema_connect(conn):
    """A caller hands us whatever connection it already has. `schema.connect` has never seen this
    table, so every public entrypoint runs the DDL itself rather than trusting `connect()`."""
    assert cs.list_runs(conn) == []
    cs.record_run(conn, "x_lists", status="ok")
    assert [r.collector for r in cs.list_runs(conn)] == ["x_lists"]


def test_ddl_is_idempotent(conn):
    for _ in range(3):
        cs.init_state_schema(conn)
    cs.record_run(conn, "x_lists", status="ok")
    cs.init_state_schema(conn)
    assert cs.get_run(conn, "x_lists").last_status == "ok"


def test_connect_layers_the_table_on_the_shared_store(kb_home):
    c = cs.connect()
    try:
        assert c.execute("SELECT COUNT(*) FROM collector_runs").fetchone()[0] == 0
        # ...and it is the SAME db as the atom store, not a second file.
        assert c.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    finally:
        c.close()


# ── roundtrip ───────────────────────────────────────────────────────────────────
def test_record_read_roundtrip(conn):
    cs.record_run(conn, "x_following", status="ok", detail="cookie scrape",
                  found=468, stored_after=468, now=_iso(T0))
    row = cs.get_run(conn, "x_following")
    assert row.collector == "x_following"
    assert row.last_attempt_at == row.last_ok_at == _iso(T0)
    assert row.last_status == "ok" and row.last_detail == "cookie scrape"
    assert (row.found, row.stored_after) == (468, 468)
    assert row.ok is True


def test_a_second_run_overwrites_in_place(conn):
    """One row per collector — CURRENT STATE, not an event log (same shape as `probe_pulls`)."""
    cs.record_run(conn, "x_likes", status="ok", found=99, stored_after=99, now=_iso(T0))
    cs.record_run(conn, "x_likes", status="ok", found=101, stored_after=101,
                  now=_iso(T0 + timedelta(hours=6)))
    assert len(cs.list_runs(conn)) == 1
    row = cs.get_run(conn, "x_likes")
    assert row.found == 101 and row.last_ok_at == _iso(T0 + timedelta(hours=6))


# ── never-run ───────────────────────────────────────────────────────────────────
def test_a_never_run_collector_is_due_and_infinitely_stale(conn):
    """The invisible-freeze case. A collector with no row must sort FIRST, not be skipped for
    lacking a timestamp to compare against."""
    row = cs.get_run(conn, "substack_subs")
    assert row is None
    assert cs.hours_since_attempt(row) == float("inf")
    assert cs.hours_since_ok(row) == float("inf")
    assert cs.is_due(row, floor_hours=6.0) is True
    assert cs.is_stale(row) is True


# ── THE split: attempt vs ok ────────────────────────────────────────────────────
def test_an_error_advances_the_attempt_and_leaves_last_ok_alone(conn):
    """THE reason there are two columns. If a failure stamped `last_ok_at`, one dead X session
    would report the candidate list as freshly seen for a full staleness window."""
    cs.record_run(conn, "x_lists", status="ok", found=6, stored_after=6, now=_iso(T0))
    cs.record_run(conn, "x_lists", status="error", detail="RuntimeError: dead session",
                  now=_iso(T0 + timedelta(hours=10)))

    row = cs.get_run(conn, "x_lists")
    assert row.last_attempt_at == _iso(T0 + timedelta(hours=10))   # we DID try
    assert row.last_ok_at == _iso(T0)                              # ...and saw nothing
    assert row.last_status == "error"
    assert row.ok is False


def test_skipped_tier_is_an_attempt_that_saw_nothing_too(conn):
    """A tier skip is not a failure, but for STALENESS it is identical to one: the collector never
    ran, so the list is exactly as old as it was. Only the floor cares about the difference."""
    now = T0 + timedelta(hours=10)
    for status in ("error", "skipped_tier", "no_viewer_id"):
        cs.record_run(conn, status, status="ok", now=_iso(T0))       # one row per status, seeded ok
        cs.record_run(conn, status, status=status, now=_iso(now))
        row = cs.get_run(conn, status)
        assert row.last_ok_at == _iso(T0), status
        assert cs.hours_since_ok(row, now) == pytest.approx(10.0), status
        assert cs.hours_since_attempt(row, now) == pytest.approx(0.0), status


def test_a_first_ever_run_that_fails_leaves_last_ok_null(conn):
    """No prior success to preserve — the row exists (so the floor can throttle the retry) but the
    list has still never been seen."""
    cs.record_run(conn, "x_likes", status="error", detail="boom", now=_iso(T0))
    row = cs.get_run(conn, "x_likes")
    assert row.last_ok_at is None
    assert cs.hours_since_ok(row, T0) == float("inf")
    assert cs.is_stale(row) is True
    assert cs.is_due(row, floor_hours=6.0, now=T0 + timedelta(hours=1)) is False  # ...throttled


# ── the counts ──────────────────────────────────────────────────────────────────
def test_the_counts_survive_a_later_failure(conn):
    """Only an `ok` run has numbers to report, so overwriting would blank the last good reading on
    the first failure. Preserved, they stay unambiguous: always "as of last_ok_at"."""
    cs.record_run(conn, "x_following", status="ok", found=468, stored_after=468, now=_iso(T0))
    cs.record_run(conn, "x_following", status="error", now=_iso(T0 + timedelta(hours=10)))
    row = cs.get_run(conn, "x_following")
    assert (row.found, row.stored_after) == (468, 468)
    assert row.last_ok_at == _iso(T0)


def test_found_and_stored_after_can_disagree(conn):
    """The hot-feed failure shape, recorded rather than smoothed over: the collector truthfully
    reports what it SAW over a write path that landed nothing."""
    cs.record_run(conn, "x_following", status="ok", found=468, stored_after=0, now=_iso(T0))
    row = cs.get_run(conn, "x_following")
    assert (row.found, row.stored_after) == (468, 0)


# ── the floor is pure ───────────────────────────────────────────────────────────
def test_the_floor_is_a_pure_function_of_stored_state(conn):
    """No `random()`, no hidden read of the wall clock when `now` is supplied — the same row and
    the same `now` must give the same verdict in every process, forever."""
    cs.record_run(conn, "x_lists", status="ok", now=_iso(T0))
    row = cs.get_run(conn, "x_lists")

    assert cs.is_due(row, floor_hours=6.0, now=T0 + timedelta(hours=5, minutes=59)) is False
    assert cs.is_due(row, floor_hours=6.0, now=T0 + timedelta(hours=6)) is True
    for _ in range(5):
        assert cs.is_due(row, floor_hours=6.0, now=T0 + timedelta(hours=7)) is True


def test_the_floor_counts_attempts_not_successes(conn):
    """Gating on success would remove the floor from exactly the collector that most needs one: a
    collector that raises every time still costs a request each time it is asked."""
    cs.record_run(conn, "x_likes", status="error", now=_iso(T0))
    row = cs.get_run(conn, "x_likes")
    assert cs.is_stale(row) is True                                        # never seen
    assert cs.is_due(row, floor_hours=6.0, now=T0 + timedelta(hours=1)) is False   # still throttled


def test_an_unparseable_stamp_reads_as_never(conn):
    """Fail-safe: a corrupt timestamp must make the collector DUE (we re-observe) rather than
    permanently fresh (we never look again)."""
    cs.record_run(conn, "x_lists", status="ok", now="not-a-timestamp")
    row = cs.get_run(conn, "x_lists")
    assert cs.hours_since_attempt(row, T0) == float("inf")
    assert cs.is_due(row, floor_hours=6.0, now=T0) is True


# ── status_summary ──────────────────────────────────────────────────────────────
_ALL = ("x_lists", "x_following", "x_likes", "substack_subs")


def test_status_summary_reports_a_collector_that_has_no_row(conn):
    """Driven by the CALLER's list, not by stored rows. A collector that never ran has no row and
    is exactly the one worth reporting — listing only what is stored makes the worst case invisible.
    """
    cs.record_run(conn, "x_lists", status="ok", found=6, stored_after=6, now=_iso(T0))

    out = cs.status_summary(conn, _ALL, now=T0)

    assert out["tracked"] == 4
    assert [e["collector"] for e in out["collectors"]] == list(_ALL)
    never = [e for e in out["collectors"] if e["collector"] == "x_likes"][0]
    assert never["never_ran"] is True and never["stale"] is True
    fresh = [e for e in out["collectors"] if e["collector"] == "x_lists"][0]
    assert fresh["stale"] is False and fresh["hours_since_ok"] == 0.0
    assert out["stale_collectors"] == 3 and out["never_succeeded"] == 3


def test_status_summary_is_quiet_when_everything_is_fresh(conn):
    """`needs_attention` is the whole surfacing rule, computed once here so two call sites cannot
    disagree about what "stale" means."""
    for name in _ALL:
        cs.record_run(conn, name, status="ok", now=_iso(T0))
    out = cs.status_summary(conn, _ALL, now=T0 + timedelta(hours=1))
    assert out["needs_attention"] is False
    assert out["stale_collectors"] == 0 and out["failing_collectors"] == 0


def test_status_summary_flags_a_failing_collector_that_is_not_yet_stale(conn):
    """A collector that failed an hour ago is still well inside the staleness window, and reporting
    only staleness would hide it until two days had passed."""
    for name in _ALL:
        cs.record_run(conn, name, status="ok", now=_iso(T0))
    cs.record_run(conn, "x_likes", status="error", detail="dead session",
                  now=_iso(T0 + timedelta(hours=1)))

    out = cs.status_summary(conn, _ALL, now=T0 + timedelta(hours=2))

    assert out["stale_collectors"] == 0
    assert out["failing_collectors"] == 1
    assert out["needs_attention"] is True
    bad = [e for e in out["collectors"] if e["collector"] == "x_likes"][0]
    assert bad["last_status"] == "error" and bad["last_detail"] == "dead session"


def test_status_summary_goes_stale_after_the_window(conn):
    for name in _ALL:
        cs.record_run(conn, name, status="ok", now=_iso(T0))
    out = cs.status_summary(conn, _ALL, now=T0 + timedelta(hours=cs.STALE_AFTER_HOURS))
    assert out["stale_collectors"] == 4 and out["needs_attention"] is True
    assert out["oldest_ok_at"] == out["newest_ok_at"] == _iso(T0)
