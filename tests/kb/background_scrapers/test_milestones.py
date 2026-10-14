"""`oracles.milestones`: where each pick stands, read from the store (ruling 19)."""
from __future__ import annotations

import pytest

from pipeline.worker import liveness
from pipeline.kb import schema
from pipeline.worker.quota_loop import loop_state
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles import asks
from pipeline.oracles import windows
from pipeline.oracles import milestones
from pipeline.oracles import told
from pipeline.oracles.sources.timeline import walks as tl
from pipeline.oracles.sources.web.discovery import QUOTA as WEB

from ._support import NOW, days, pair


@pytest.fixture(autouse=True)
def worker_unknown(monkeypatch):
    monkeypatch.setattr(liveness, "repair_or_report_cached",
                        lambda **kw: {"ok": None, "status": "unknown"})


def _pick(conn, cid=None):
    picks = milestones.for_picks(conn, [cid] if cid else None, now=NOW)
    return picks[0] if cid else picks


def _asked(conn, cid, *, discovered=True, **cols):
    asks.record_ask(conn, cid, x_from=days(30), web_from=windows.ALL_TIME,
                     scholar_from=windows.ALL_TIME, now=days(1))
    if discovered:
        conn.execute("UPDATE oracle_asks SET discovered_at=? WHERE canonical_id=?",
                     (days(1).isoformat(), cid))
    for k, v in cols.items():
        conn.execute(f"UPDATE oracle_asks SET {k}=? WHERE canonical_id=?", (v, cid))
    conn.commit()


def _web_pair(conn, cid, stype="substack", key="https://carol.substack.com", **cols):
    registry.upsert_source(conn, registry.SourceRow(cid, stype, key))
    for k, v in cols.items():
        conn.execute(f"UPDATE oracle_sources SET {k}=? WHERE canonical_id=? AND source_key=?",
                     (v, cid, key))
    conn.commit()


# ── newest posts and reach ────────────────────────────────────────────────────────

def test_newest_posts_is_the_first_page_of_every_x_pair(conn):
    cid = pair(conn)
    assert _pick(conn, cid)["newest_posts"] is False
    conn.execute("UPDATE oracle_sources SET last_pulled_at=?", (days(0).isoformat(),))
    conn.commit()
    assert _pick(conn, cid)["newest_posts"] is True


def test_a_pick_with_no_x_timeline_says_so_rather_than_done(conn):
    """None, not True: "newest posts in" for someone with no X would be announced as news."""
    roster.upsert_oracle(conn, "blog:carol.dev", name="Carol")
    pick = _pick(conn, "blog:carol.dev")
    assert pick["newest_posts"] is None and pick["reach"] is None


def test_reach_is_measured_against_what_the_walk_is_owed(conn):
    cid = pair(conn, last_pulled_at=days(0).isoformat(), covered_from=days(10).isoformat())
    _asked(conn, cid)                                    # X asked back 30 days
    reach = _pick(conn, cid)["reach"]
    assert reach["covered_from"].startswith(days(10).date().isoformat())
    assert reach["asked_from"].startswith(days(183).date().isoformat())   # the 6-month floor
    assert reach["reached"] is False

    conn.execute("UPDATE oracle_sources SET covered_from=?", (days(200).isoformat(),))
    conn.commit()
    assert _pick(conn, cid)["reach"]["reached"] is True


def test_an_oracle_from_before_the_scrapers_is_owed_six_months_and_no_archives(conn):
    cid = pair(conn, last_pulled_at=days(0).isoformat(), covered_from=days(10).isoformat())
    pick = _pick(conn, cid)
    assert pick["asked"] is None and pick["archives"] is None
    assert pick["reach"]["reached"] is False
    assert milestones.owes(pick)


# ── archives ──────────────────────────────────────────────────────────────────────

def test_archives_wait_for_discovery_to_look(conn):
    """Before discovery has looked, nothing beyond X is known, which is not the same as in: five
    X-only picks read `archives: true` 53 s after confirm, then two of them gained sites."""
    cid = pair(conn)
    _asked(conn, cid, discovered=False)
    assert _pick(conn, cid)["archives"] is False
    conn.execute("UPDATE oracle_asks SET discovered_at=?", (days(0).isoformat(),))
    conn.commit()
    assert _pick(conn, cid)["archives"] is True


def test_archives_wait_for_a_rediscovery(conn):
    cid = pair(conn)
    _asked(conn, cid, rediscover=1)
    assert _pick(conn, cid)["archives"] is False


def test_archives_wait_for_every_pair_pulled_for_the_latest_ask(conn):
    cid = pair(conn)
    _asked(conn, cid)
    _web_pair(conn, cid, last_pulled_at=days(2).isoformat())        # before the ask
    assert _pick(conn, cid)["archives"] is False
    conn.execute("UPDATE oracle_sources SET last_pulled_at=? WHERE source_type='substack'",
                 (days(0).isoformat(),))
    conn.commit()
    assert _pick(conn, cid)["archives"] is True


def test_a_pair_that_gave_up_counts_as_settled(conn):
    cid = pair(conn)
    _asked(conn, cid)
    _web_pair(conn, cid, failures=10, last_status="account")
    assert _pick(conn, cid)["archives"] is True


def test_archives_wait_for_each_found_source_first_pull(conn):
    cid = pair(conn)
    _asked(conn, cid)
    conn.execute("INSERT INTO found_sources (canonical_id, source_type, url, found_at) "
                 "VALUES (?, 'blog', 'https://carol.dev', ?)", (cid, days(1).isoformat()))
    conn.commit()
    assert _pick(conn, cid)["archives"] is False
    conn.execute("UPDATE found_sources SET pulled_at=?", (days(0).isoformat(),))
    conn.commit()
    assert _pick(conn, cid)["archives"] is True


# ── stuck ─────────────────────────────────────────────────────────────────────────

def test_x_needing_reconnecting_is_stuck_only_while_x_is_owed(conn):
    cid = pair(conn)
    loop_state.set_state(tl.QUOTA, "paused", cause="x_session", now=NOW.timestamp())
    assert _pick(conn, cid)["stuck"] == {"cause": "x_session", "step": "onboard(source='x')"}

    conn.execute("UPDATE oracle_sources SET last_pulled_at=?, covered_from=?",
                 (days(0).isoformat(), days(400).isoformat()))
    conn.commit()
    assert "stuck" not in _pick(conn, cid)


def test_the_model_provider_refusing_is_stuck(conn):
    cid = pair(conn)
    loop_state.set_state(tl.QUOTA, "paused", cause="model:openrouter", now=NOW.timestamp())
    assert _pick(conn, cid)["stuck"]["cause"] == "model:openrouter"


def test_nothing_claiming_the_work_is_stuck_with_its_cause_and_step(conn, monkeypatch):
    cid = pair(conn)
    monkeypatch.setattr(liveness, "repair_or_report_cached", lambda **kw: {
        "ok": False, "cause": "the helper keeps stopping", "step": "restart OPYT"})
    assert _pick(conn, cid)["stuck"] == {"cause": "not_running",
                                         "detail": "the helper keeps stopping",
                                         "step": "restart OPYT"}
    loop_state.set_state(WEB, "idle", now=NOW.timestamp())          # a scraper beating
    assert "stuck" not in _pick(conn, cid)


def test_an_unknown_worker_never_invents_a_stuck_pick(conn):
    assert "stuck" not in _pick(conn, pair(conn))


def test_an_outage_is_never_stuck(conn):
    """The user can do nothing about it (ruling 19)."""
    cid = pair(conn)
    loop_state.set_state(tl.QUOTA, "paused", cause="x_outage", retry_at=NOW.timestamp() + 60,
                    now=NOW.timestamp())
    assert "stuck" not in _pick(conn, cid)
    assert milestones.waiting_on(conn, [_pick(conn, cid)], now=NOW)["source"] == "x"


def test_a_venue_waiting_for_topics_is_stuck_and_its_archives_are_not_in(conn):
    cid = "openalex:S1"
    roster.upsert_oracle(conn, cid, name="ChemRxiv")
    asks.record_ask(conn, cid, x_from=days(30), web_from=windows.ALL_TIME, now=days(1))
    conn.execute("UPDATE oracle_asks SET discovered_at=?", (days(1).isoformat(),))
    _web_pair(conn, cid, "openalex", "S1")
    pick = _pick(conn, cid)
    assert pick["archives"] is False
    assert pick["stuck"]["cause"] == "needs_topics"


def test_discovery_that_gave_up_is_stuck(conn):
    cid = pair(conn)
    _asked(conn, cid, discovered=False, discovery_failures=10,
           discovery_error="x:carol did not resolve")
    assert _pick(conn, cid)["stuck"]["cause"] == "discovery_failed"


def test_pages_that_will_not_save_are_stuck(conn):
    cid = pair(conn, failures=3, last_status="unsaved", last_error="embed refused")
    assert _pick(conn, cid)["stuck"] == {"cause": "unsaved", "detail": "embed refused"}


# ── since you last looked ─────────────────────────────────────────────────────────

def test_each_milestone_is_told_once(conn):
    cid = pair(conn, last_pulled_at=days(0).isoformat())
    _asked(conn, cid)
    assert [e["milestone"] for e in told.unshown(conn)] == ["newest_posts", "archives"]
    told.mark_shown(conn, milestones.for_picks(conn))
    assert told.unshown(conn) == []


def test_picks_with_no_ask_are_shown_but_never_announced(conn):
    """Otherwise the first call after the cutover announces every Oracle already in."""
    pair(conn, last_pulled_at=days(0).isoformat())
    assert told.unshown(conn) == []


def test_a_stuck_that_clears_is_told_again_next_time(conn):
    cid = pair(conn)
    _asked(conn, cid)
    loop_state.set_state(tl.QUOTA, "paused", cause="x_session", now=NOW.timestamp())
    told.mark_shown(conn, milestones.for_picks(conn))
    assert told.unshown(conn) == []

    loop_state.set_state(tl.QUOTA, "running", now=NOW.timestamp())
    told.mark_shown(conn, milestones.for_picks(conn))
    loop_state.set_state(tl.QUOTA, "paused", cause="x_session", now=NOW.timestamp())
    assert [e["milestone"] for e in told.unshown(conn)] == ["stuck"]


def test_a_store_no_scraper_has_run_on_still_reads_what_is_owed(kb_home, monkeypatch):
    """A fresh install, or a worker that never started: the scrapers' tables do not exist yet.
    That is the post-cutover backfill `outstanding` must report, and nothing to announce."""
    conn = schema.connect()
    roster.upsert_oracle(conn, "x:user:carol", name="Carol")
    registry.upsert_source(conn, registry.SourceRow("x:user:carol", "x", "carol"))

    assert milestones.for_picks(conn, now=NOW)[0]["newest_posts"] is False
    assert told.unshown(conn) == []
    assert milestones.outstanding(conn) == ["newest_posts", "reach"]
    conn.close()


# ── outstanding ───────────────────────────────────────────────────────────────────

def test_outstanding_names_what_is_still_owed(conn, monkeypatch):
    monkeypatch.setattr(milestones.thread_state, "owed",
                        lambda conn, kind: [object(), object()] if kind == "tails" else [])
    cid = pair(conn)
    _asked(conn, cid)
    assert milestones.outstanding(conn) == ["newest_posts", "reach", "thread_tails"]


def test_outstanding_counts_saved_posts_still_owed_their_conversation(conn, monkeypatch):
    conn.execute("CREATE TABLE IF NOT EXISTS sync_dedup (namespace TEXT, item_id TEXT, "
                 "synced_at TEXT)")
    for tid, payload in (("1", '{"is_reply": true}'), ("2", '{"reply_count": 0}'),
                         ("3", '{"reply_count": 4}')):
        conn.execute("INSERT INTO atoms (atom_id, source_type, what_kind, who_id, payload, "
                     "entry_mode) VALUES (?, 'x', 'opinion', 'x:user:1', ?, 'user-saved')",
                     (f"x:{tid}", payload))
    conn.execute("INSERT INTO sync_dedup VALUES ('x_convo_checked', '3', 'x')")
    conn.commit()
    assert milestones.outstanding(conn) == ["bookmark_threads"]


def test_outstanding_folds_in_curations_imports(conn, monkeypatch):
    monkeypatch.setattr(milestones.imports, "imports_outstanding", lambda: ["bookmarks_import"])
    assert milestones.outstanding(conn) == ["bookmarks_import"]
