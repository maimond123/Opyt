"""Retiring an unfollowed candidate — the one place an ABSENCE changes what the user sees.

Signals are monotonic by default: unfollow someone and the `follow` row stays forever, so the
candidate list is a high-water mark of everyone ever curated. §7.4 left that deliberately. It was
reopened 2026-08-13 after the list clock measured it: 52 of 470 follow signals on the live store
were people no longer followed, and a list whose entire value is being pre-vetted was carrying 11%
anti-selected names.

Every test here defends one of the three ways acting on an absence goes wrong: retiring on a
truncated walk, retiring everyone the moment the column is added, and retiring silently.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.kb import curation_state as cs
from pipeline.kb import schema, screen

FOLLOW_COLLECTOR = "x_following"


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


def _person(conn, uid, *, name=None):
    eid = f"x:user:{uid}"
    schema.upsert_entity(conn, eid, name=name or f"P{uid}")
    return eid


T0 = datetime(2026, 8, 13, 6, 0, 0, tzinfo=timezone.utc)


def _walk(conn, seen_uids, *, at, found=None, prev_found=None):
    """Simulate one full following walk at instant `at`: confirm everyone seen, stamp the clock.

    The confirmations are REWRITTEN to `at` after the fact. `set_signal` stamps through SQLite's
    own clock, so two walks in a test land in the same second and become indistinguishable — real
    walks are hours apart. Backdating the stored stamp is how the test buys that elapsed time
    without adding a test-only argument to production code."""
    for uid in seen_uids:
        schema.set_signal(conn, f"x:user:{uid}", "follow", "x")
    ids = [f"x:user:{u}" for u in seen_uids]
    ph = ",".join("?" for _ in ids)
    conn.execute(f"UPDATE curation_signals SET last_confirmed_at=? "
                 f" WHERE signal_type='follow' AND entity_id IN ({ph})",
                 [at.strftime("%Y-%m-%d %H:%M:%S"), *ids])
    conn.commit()
    cs.record_run(conn, FOLLOW_COLLECTOR, status="ok",
                  found=len(seen_uids) if found is None else found,
                  stored_after=len(seen_uids), now=at.isoformat(), started_at=at.isoformat())
    if prev_found is not None:
        conn.execute("UPDATE collector_runs SET prev_found=? WHERE collector=?",
                     (prev_found, FOLLOW_COLLECTOR))
        conn.commit()


def _names(cands):
    return sorted(c.canonical_id for c in cands)


# ── the core behaviour ──────────────────────────────────────────────────────────
def test_an_unfollowed_person_drops_out_of_the_candidate_list(conn):
    for uid in ("1", "2", "3"):
        _person(conn, uid)
    _walk(conn, ["1", "2", "3"], at=T0)                          # baseline walk
    _walk(conn, ["1", "2"], at=T0 + timedelta(hours=6))          # 3 is gone

    assert _names(screen.rank_candidates(conn)) == ["x:user:1", "x:user:2"]


def test_the_signal_row_survives_the_retirement(conn):
    """Exclusion is at RANKING time. Nothing is deleted — the evidence that you did follow them is
    real and stays, and their saved posts stay searchable in `atoms` regardless."""
    _person(conn, "3")
    _walk(conn, ["1", "2", "3"], at=T0)
    _walk(conn, ["1", "2"], at=T0 + timedelta(hours=6))

    assert screen.rank_candidates(conn) == [] or "x:user:3" not in _names(
        screen.rank_candidates(conn))
    row = conn.execute("SELECT count, last_confirmed_at FROM curation_signals "
                       " WHERE entity_id='x:user:3' AND signal_type='follow'").fetchone()
    assert row is not None and row["count"] == 1 and row["last_confirmed_at"] is not None


def test_a_bookmark_does_not_rescue_an_unfollowed_person(conn):
    """DECIDED 2026-08-13, reversing an earlier per-signal design. `save` and `like` are append-only
    IN PRACTICE — nobody un-bookmarks someone they lost interest in — so letting them veto the one
    signal the user does maintain makes removal impossible by construction. On the live store 11 of
    the 52 unfollowed accounts carried a bookmark, and keeping them was the wrong call."""
    _person(conn, "3")
    schema.add_signal(conn, "x:user:3", "save", "x", count=12)
    schema.set_signal(conn, "x:user:3", "like", "x", count=4)
    _walk(conn, ["1", "2", "3"], at=T0)
    _walk(conn, ["1", "2"], at=T0 + timedelta(hours=6))

    assert "x:user:3" not in _names(screen.rank_candidates(conn))


def test_a_person_who_was_never_followed_is_untouched(conn):
    """No follow signal means no revocation act ever happened. The 603 bookmark-only candidates on
    the live store must not be swept up by a rule about following."""
    _person(conn, "9")
    schema.add_signal(conn, "x:user:9", "save", "x")
    _walk(conn, ["1", "2"], at=T0)
    _walk(conn, ["1", "2"], at=T0 + timedelta(hours=6))

    assert "x:user:9" in _names(screen.rank_candidates(conn))


# ── the truncated-walk guard ────────────────────────────────────────────────────
def test_a_collapsed_walk_retires_nobody(conn):
    """THE dangerous failure. A session dying mid-page returns a short list that is
    indistinguishable from a mass unfollow, and acting on it retires hundreds of live signals with
    no error anywhere. Absence is only evidence if the walk was complete."""
    for uid in "123456789":
        _person(conn, uid)
    _walk(conn, list("123456789"), at=T0)
    _walk(conn, ["1", "2"], at=T0 + timedelta(hours=6), prev_found=9)           # 9 → 2 is a collapse, not an unfollow spree

    assert len(screen.rank_candidates(conn)) == 9   # everyone survives


def test_a_gentle_decline_still_retires(conn):
    """The guard must not become a rubber stamp. A real unfollow session shrinks the list; only a
    COLLAPSE is suspect."""
    for uid in "12345":
        _person(conn, uid)
    _walk(conn, list("12345"), at=T0)
    _walk(conn, ["1", "2", "3", "4"], at=T0 + timedelta(hours=6), prev_found=5)   # 5 → 4, well above the 0.5 ratio

    assert "x:user:5" not in _names(screen.rank_candidates(conn))


def test_a_failed_run_retires_nobody(conn):
    """An error observed nothing, so nothing is absent from it."""
    for uid in "123":
        _person(conn, uid)
    _walk(conn, list("123"), at=T0)
    _walk(conn, list("123"), at=T0 + timedelta(hours=6))
    cs.record_run(conn, FOLLOW_COLLECTOR, status="error", detail="dead session")

    assert len(screen.rank_candidates(conn)) == 3


def test_the_first_ever_walk_retires_nobody(conn):
    """No prior `found` means no baseline to be suspicious against. Retirement starts on run two,
    which is the right way round for an action no later run undoes."""
    for uid in "123":
        _person(conn, uid)
    _walk(conn, ["1", "2"], at=T0)                   # first walk; 3 has a signal but was not seen
    schema.set_signal(conn, "x:user:3", "follow", "x")

    assert cs.walk_is_trustworthy(cs.get_run(conn, FOLLOW_COLLECTOR)) is False
    assert len(screen.rank_candidates(conn)) == 3


def test_no_clock_at_all_retires_nobody(conn):
    """Fail-safe: a store that has never run the rail must behave exactly as it did before."""
    for uid in "123":
        _person(conn, uid)
        schema.set_signal(conn, f"x:user:{uid}", "follow", "x")
    assert len(screen.rank_candidates(conn)) == 3


# ── the migration hazard ────────────────────────────────────────────────────────
def test_upgrading_an_existing_store_retires_nobody(conn):
    """⚠️ THE hazard. Pre-existing rows have no `last_confirmed_at`. If NULL compared as "older
    than everything", the first read after upgrading would retire EVERY follow signal — 470 on the
    live store — and empty the candidate list in one silent step. The backfill stamps them NOW, so
    the truth arrives one collector run late instead of destructively early."""
    for uid in "123":
        _person(conn, uid)
        # A pre-migration row: written by the summing path, which never confirms.
        schema.add_signal(conn, f"x:user:{uid}", "follow", "x")
    conn.execute("UPDATE curation_signals SET last_confirmed_at = NULL")
    conn.commit()

    schema.init_kb_schema(conn)                      # the upgrade
    _walk(conn, ["1", "2", "3"], at=T0, prev_found=3)  # a healthy walk that DOES see everyone

    assert len(screen.rank_candidates(conn)) == 3
    assert all(r["last_confirmed_at"] for r in
               conn.execute("SELECT last_confirmed_at FROM curation_signals"))


def test_the_save_signal_never_gets_a_confirmation_stamp(conn):
    """`save` is unconfirmable by construction — no path re-reads your bookmark set. It must stay
    NULL rather than collect a meaningless stamp, or half the table carries a column that lies."""
    _person(conn, "1")
    schema.add_signal(conn, "x:user:1", "save", "x")
    schema.set_signal(conn, "x:user:1", "follow", "x")
    rows = {r["signal_type"]: r["last_confirmed_at"] for r in conn.execute(
        "SELECT signal_type, last_confirmed_at FROM curation_signals")}
    assert rows["save"] is None
    assert rows["follow"] is not None


# ── shown, not vanished ─────────────────────────────────────────────────────────
def test_the_retired_count_is_reported_not_hidden(conn, fake_embedder):
    """A list that quietly shrank by 52 is indistinguishable from one that never had them — and the
    reader cannot tell a working retirement from a broken walk without the number."""
    from pipeline.kb import candidate_search

    for uid in ("1", "2", "3"):
        _person(conn, uid)
    _walk(conn, ["1", "2", "3"], at=T0)
    _walk(conn, ["1", "2"], at=T0 + timedelta(hours=6))

    out = candidate_search.candidates_payload(conn, "", None)
    assert out["retired"] == 1
    assert "no longer follow" in out["retired_note"]


def test_nothing_retired_reports_nothing(conn, fake_embedder):
    """Same rule as `signal_reconcile` and `list_freshness`: a line printed on every call stops
    being read."""
    from pipeline.kb import candidate_search

    for uid in ("1", "2"):
        _person(conn, uid)
    _walk(conn, ["1", "2"], at=T0)
    _walk(conn, ["1", "2"], at=T0 + timedelta(hours=6))

    out = candidate_search.candidates_payload(conn, "", None)
    assert "retired" not in out and "retired_note" not in out


def test_include_retired_returns_them_flagged(conn):
    """The caller that REPORTS needs them, so they are available on request rather than erased."""
    for uid in ("1", "2", "3"):
        _person(conn, uid)
    _walk(conn, ["1", "2", "3"], at=T0)
    _walk(conn, ["1", "2"], at=T0 + timedelta(hours=6))

    everyone = screen.rank_candidates(conn, include_retired=True)
    assert len(everyone) == 3
    assert [c.canonical_id for c in everyone if c.retired] == ["x:user:3"]
