"""The `oracle` MCP tool's freshness surface — `screen`'s `oracle_freshness` block.

Kept apart from the adapter-seam tests: those prove an ADAPTER skips a call, these prove the TOOL
routes and reports. Modelled on the radar rail's `_FakeMCP` (since deleted).

⚠️ `refresh`, `status` and `sync_follows` left this surface 2026-08-15. Freshness is no longer a
report you request: a report nobody calls is not a mechanism, which is exactly how every Oracle sat
frozen from the day it was added while `status` sat there able to say so. It now rides on `screen`
unasked. These tests pin that inversion, because reverting it would be silent.
"""
from __future__ import annotations

import pytest

from pipeline.kb import oracle_refresh_state as st, schema


class _FakeMCP:
    """Collects the functions `register_oracle_tools` decorates, like tests/radar's does."""

    def __init__(self):
        self.tools = {}

    def tool(self, *a, **kw):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture()
def oracle_tool(kb_home, monkeypatch):
    monkeypatch.setenv("OPYT_ORACLE_REFRESH_CONSENT", str(kb_home / "consent"))
    from mcp_server.oracle_tools import register_oracle_tools
    mcp = _FakeMCP()
    register_oracle_tools(mcp)
    return mcp.tools["oracle"]


@pytest.mark.parametrize("gone", ["refresh", "status", "sync_follows"])
def test_a_removed_action_is_rejected_and_says_where_it_went(oracle_tool, gone):
    """A bare "unknown action" would send whoever remembers these three hunting, so the error
    names where each one went. `refresh` and `status` still RUN and the error names their
    replacement; `sync_follows` was DELETED outright on 2026-08-26, and saying so is the point —
    a deleted action that answers "unknown action" is exactly the hunt this test prevents."""
    out = oracle_tool(action=gone)
    assert "error" in out and gone not in out["error"].split("—")[1]
    assert gone in out["moved"], f"{gone} vanished without saying where it went"


def test_the_four_surviving_actions_are_named_in_the_error(oracle_tool):
    err = oracle_tool(action="bogus")["error"]
    for kept in ("screen", "candidates", "confirm", "ingest"):
        assert f"'{kept}'" in err


# ── `candidates` carries the LIST clock ─────────────────────────────────────────
#
# Two different clocks meet on this payload and neither substitutes for the other. `probe_pulls`
# says whether a candidate's CONTENT is stale; `collector_runs` says whether the candidate LIST is.
# Only the second can tell you that someone you followed last week was never offered as a candidate
# at all, which is the failure that reads as "nothing new to promote".

def _stamp_all(status="ok", **kw):
    from pipeline.kb import curation_state as cs
    from pipeline.kb import ingest_curation as ic
    conn = cs.connect()
    try:
        for name in ic.COLLECTORS:
            cs.record_run(conn, name, status=status, **kw)
    finally:
        conn.close()


def test_candidates_is_quiet_when_the_whole_list_is_fresh(oracle_tool, kb_home):
    """Reported only when something is wrong. A freshness block on every call trains the reader to
    skip the one call where it matters — the same rule `signal_reconcile` follows."""
    _stamp_all()
    out = oracle_tool(action="candidates")
    assert "list_freshness" not in out


def test_candidates_surfaces_a_list_no_collector_has_ever_refreshed(oracle_tool, kb_home):
    """The invisible-freeze case, and the reason the report is driven by the collector list rather
    than by stored rows: a collector that has never run has no row, and it is the worst case."""
    from pipeline.kb import ingest_curation as ic

    out = oracle_tool(action="candidates")

    fresh = out["list_freshness"]
    assert fresh["needs_attention"] is True
    assert fresh["never_succeeded"] == len(ic.COLLECTORS)
    assert {e["collector"] for e in fresh["collectors"]} == set(ic.COLLECTORS)
    assert all(e["never_ran"] for e in fresh["collectors"])


def test_candidates_names_the_one_collector_that_went_stale(oracle_tool, kb_home):
    from datetime import datetime, timedelta, timezone

    from pipeline.kb import curation_state as cs

    _stamp_all()
    old = (datetime.now(timezone.utc) - timedelta(hours=cs.STALE_AFTER_HOURS + 1)).isoformat()
    conn = cs.connect()
    try:
        cs.record_run(conn, "x_following", status="ok", now=old)
    finally:
        conn.close()

    fresh = oracle_tool(action="candidates")["list_freshness"]

    assert fresh["stale_collectors"] == 1
    stale = [e for e in fresh["collectors"] if e["stale"]]
    assert [e["collector"] for e in stale] == ["x_following"]


def test_a_broken_clock_degrades_to_the_payload_without_freshness(oracle_tool, kb_home,
                                                                  monkeypatch):
    """Fail-safe, same shape as `status`'s refresh block: a stale candidate list beats no candidate
    list, so a state-read failure must cost the freshness line and nothing else."""
    from pipeline.kb import curation_state as cs

    monkeypatch.setattr(cs, "status_summary",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no such table")))

    out = oracle_tool(action="candidates")

    assert "list_freshness" not in out
    assert "error" not in out
    assert "candidates" in out


def _one_oracle(handle="willccbb"):
    conn = st.connect()
    try:
        schema.upsert_entity(conn, "x:user:1", name="W", profile={"handle": handle})
        schema.upsert_oracle(conn, "x:user:1", name="W")
        st.seed_from_entities(conn)
    finally:
        conn.close()


def test_screen_carries_per_source_freshness_unasked(oracle_tool, kb_home):
    """UNCONDITIONAL, unlike `candidates`' `list_freshness`, and the difference is call frequency.
    "Do not print on every call" was written about `search`; `screen` is the deliberate, occasional
    "what is my people situation" call, and that is exactly where a roster with no last-pulled
    column hid a frozen loop for months."""
    _one_oracle()
    out = oracle_tool(action="screen")

    fresh = out["oracle_freshness"]
    assert fresh["tracked_pairs"] == 1
    src = fresh["oracles"][0]["sources"][0]
    assert src["source_type"] == "x" and src["never_refreshed"] is True


def test_an_unconsented_roster_needs_attention_and_says_so(oracle_tool, kb_home):
    """THE defect this whole surface exists for: no consent means the loop has no ENTRANCE, so
    nothing ever re-pulls and nothing ever errors. It must not be silent."""
    _one_oracle()
    fresh = oracle_tool(action="screen")["oracle_freshness"]

    assert fresh["consented"] is False
    assert fresh["needs_attention"] is True
    assert "onboard" in fresh["note"], "the note must name the tool that grants consent"


def test_an_empty_roster_stays_silent(oracle_tool, kb_home):
    """A fresh install has no Oracles, so telling it Oracle refresh is off is noise on the first
    surface a new user meets. `needs_attention` is guarded on tracked_pairs for exactly this."""
    fresh = oracle_tool(action="screen")["oracle_freshness"]

    assert fresh["tracked_pairs"] == 0
    assert fresh["needs_attention"] is False
    assert "note" not in fresh


@pytest.mark.parametrize("stale_of_four, attention", [(2, False), (3, True)])
def test_attention_tracks_the_stale_FRACTION_not_a_count(oracle_tool, kb_home, monkeypatch,
                                                         stale_of_four, attention):
    """⚠️ A RATIO, and simulation is why. An absolute count cannot work: overdue pairs grow with
    the roster, so a fixed threshold is generous at 8 Oracles and permanently tripped at 50.
    Half is the knee — solving (cycle-TTL)/cycle > 0.5 gives cycle > 2x TTL, i.e. "your refresh
    cycle has stretched past twice what you asked for". 2 of 4 is a loop working through a queue;
    3 of 4 is a loop losing ground."""
    from datetime import datetime, timedelta, timezone

    from pipeline.kb import oracle_refresh
    monkeypatch.setattr(oracle_refresh, "consented", lambda: True)

    conn = st.connect()
    try:
        for i in range(4):
            schema.upsert_entity(conn, f"x:user:{i}", name=f"P{i}", profile={"handle": f"p{i}"})
            schema.upsert_oracle(conn, f"x:user:{i}", name=f"P{i}")
        st.seed_from_entities(conn)
        # Fresh = pulled just now; stale = well past the 72h X TTL even after jitter.
        for i in range(4):
            hours = 400.0 if i < stale_of_four else 1.0
            when = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            conn.execute("UPDATE oracle_sources SET last_pulled_at=? WHERE canonical_id=?",
                         (when, f"x:user:{i}"))
        conn.commit()
    finally:
        conn.close()

    fresh = oracle_tool(action="screen")["oracle_freshness"]
    assert fresh["tracked_pairs"] == 4 and fresh["stale_pairs"] == stale_of_four
    assert fresh["needs_attention"] is attention
    if attention:
        assert "since_last" in fresh["note"], "the note must offer the cheap targeted top-up"


def test_a_screen_still_returns_when_the_registry_read_fails(oracle_tool, kb_home, monkeypatch):
    """Fail-safe: a screen with no freshness beats no screen."""
    from pipeline.kb import oracle_refresh
    monkeypatch.setattr(oracle_refresh, "status_summary",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("registry gone")))
    out = oracle_tool(action="screen")

    assert "registry gone" in out["oracle_freshness"]["error"]
    assert "candidates" in out
