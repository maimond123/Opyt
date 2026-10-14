"""The rider — how a user who walked away ever learns what landed.

OPYT cannot push, and that is the premise. An MCP server speaks when it is called and never
otherwise. The publishing pass publishes the served copy of the knowledge base and notifies nobody;
`needs_attention` is a flag that needs a tool call to carry it. So what landed while the conversation was closed rides back on whatever the
user does next — including `search`, which has nothing to do with Oracles.

The rider reads the milestones: each pick's newest posts, archives, how
far back X reaches, and what is stuck, each said once. There is no run and no "complete".
"""
from __future__ import annotations

import pytest

from pipeline.worker import liveness
from pipeline.timeparse import utc_now
from pipeline.worker.quota_loop import loop_state
from tests import mcp_session
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles import asks
from pipeline.oracles import windows
from pipeline.oracles import told
from pipeline.oracles import progress
from pipeline.kb import schema
from pipeline.oracles.sources import scraper_tables


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    registry.init_state_schema(c)
    scraper_tables.init_schema(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _worker_unknown(monkeypatch):
    monkeypatch.setattr(liveness, "repair_or_report_cached",
                        lambda **kw: {"ok": None, "status": "unknown"})


def _pick(conn, name, *, pulled=True, asked=True):
    """A confirmed Oracle with one X pair, its first page done when `pulled`, and an ask on
    record (discovery done, nothing beyond X) when `asked`."""
    cid = f"x:user:{name.lower()}"
    roster.upsert_oracle(conn, cid, name=name)
    registry.upsert_source(conn, registry.SourceRow(cid, "x", name.lower()))
    if pulled:
        conn.execute("UPDATE oracle_sources SET last_pulled_at=? WHERE canonical_id=?",
                     (utc_now().isoformat(), cid))
    if asked:
        asks.record_ask(conn, cid, web_from=windows.ALL_TIME)
        conn.execute("UPDATE oracle_asks SET discovered_at=? WHERE canonical_id=?",
                     (utc_now().isoformat(), cid))
    conn.commit()
    return cid


def test_nothing_reached_is_not_announced(conn):
    """There is nothing to tell them yet."""
    cid = _pick(conn, "Andrej", pulled=False)
    registry.upsert_source(conn, registry.SourceRow(cid, "substack", "https://andrej.substack.com"))
    assert told.completion_notice(conn) is None


def test_a_reached_milestone_is_announced_in_plain_words(conn):
    _pick(conn, "Andrej")
    notice = told.completion_notice(conn)
    assert {(e["name"], e["milestone"]) for e in notice["milestones"]} == {
        ("Andrej", "newest_posts"), ("Andrej", "archives")}
    assert "The newest X posts from Andrej are in." in notice["message"]
    assert "status" not in notice and "complete" not in notice["message"]


def test_it_fires_exactly_once(conn):
    """Un-stamped it would ride on every call forever and become furniture — the same argument
    `screen` makes for omitting `omitted: 0`."""
    _pick(conn, "Andrej")
    assert told.completion_notice(conn) is not None
    assert told.completion_notice(conn) is None


def test_a_milestone_is_told_once_briefly_not_made_the_subject(conn):
    """They asked for something else. A writer landing is a clause, not a turn."""
    _pick(conn, "Andrej")
    message = told.completion_notice(conn)["message"]
    assert "don't make it the subject of the turn" in message
    assert "Never explain how it works" in message


def test_what_is_stuck_is_said_with_what_the_user_can_do(conn):
    _pick(conn, "Dean", pulled=False)
    loop_state.set_state("x:UserOriginalsTimeline", "paused", cause="x_session")
    notice = told.completion_notice(conn)
    stuck = next(e for e in notice["milestones"] if e["milestone"] == "stuck")
    assert stuck["step"] == "onboard(source='x')"
    assert "X needs reconnecting before more of Dean's posts can come in." in notice["message"]


def test_oracles_from_before_the_scrapers_are_not_announced_at_the_cutover(conn):
    """Every Oracle already in would otherwise be news at once on the first call."""
    _pick(conn, "Andrej", asked=False)
    assert told.completion_notice(conn) is None


# ── the surfaces that carry it ────────────────────────────────────────────────

def test_search_carries_it(conn):
    """A notice about Oracles on a tool that has nothing to do with them — because it is the
    thing a returning user is most likely to reach for."""
    _pick(conn, "Andrej")
    out = mcp_session.session().call("search", "anything")
    assert "pull_finished" in [n["code"] for n in out["notices"]]


def test_the_screen_surface_carries_it(conn):
    _pick(conn, "Andrej")
    out = mcp_session.session().call("oracle", action="screen")
    assert out["pull_finished"]["milestones"][0]["name"] == "Andrej"


def test_progress_spends_the_notice_so_it_is_not_told_twice(conn, monkeypatch):
    """A host just handed the milestones will tell the user. Letting the rider announce them
    again on their next `search` would have OPYT report one event twice."""
    monkeypatch.setattr(progress, "PROGRESS_WAIT_SECONDS", 0.0)
    _pick(conn, "Andrej")
    assert progress.progress(conn)["picks"][0]["newest_posts"] is True
    assert told.completion_notice(conn) is None


def test_an_unreadable_store_contributes_nothing_and_breaks_nothing(kb_home, monkeypatch):
    """A courtesy on somebody else's answer must never be able to break it: the answer with a
    broken notice is exactly the answer with none."""

    search = mcp_session.session().tools()["search"]
    monkeypatch.setattr(told, "completion_notice", lambda *a, **k: 1 / 0)
    broken = search("anything")

    monkeypatch.setattr(told, "completion_notice", lambda *a, **k: None)
    assert broken == search("anything")

