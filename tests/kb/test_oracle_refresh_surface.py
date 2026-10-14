"""The `oracle` MCP tool's freshness surface — `screen`'s `oracle_freshness` block.

Kept apart from the adapter-seam tests: those prove an adapter skips a call, these prove the tool
routes and reports.

Freshness rides on `screen` unasked, which is the only surface a caller needs to understand.
"""
from __future__ import annotations

import pytest

from pipeline.kb import schema
from pipeline.platforms import connections
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles import pairs
from pipeline.oracles import freshness


class _FakeMCP:
    """Collects the functions `register_oracle_tools` decorates."""

    def __init__(self):
        self.tools = {}

    def tool(self, *a, **kw):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture()
def oracle_tool(kb_home):
    from mcp_server.oracle_tools import register_oracle_tools
    mcp = _FakeMCP()
    register_oracle_tools(mcp)
    return mcp.tools["oracle"]


@pytest.mark.parametrize("gone", ["refresh", "status", "sync_follows"])
def test_a_retired_action_uses_the_ordinary_unknown_action_contract(oracle_tool, gone):
    out = oracle_tool(action=gone)
    assert set(out) == {"error"}


def test_the_surviving_actions_are_named_in_the_error(oracle_tool):
    err = oracle_tool(action="bogus")["error"]
    for kept in ("screen", "candidates", "confirm", "ingest", "progress", "map"):
        assert f"'{kept}'" in err


def test_screen_omits_x_lookback_without_an_x_connection(oracle_tool, monkeypatch):
    """A source that is not connected is not an onboarding question yet."""

    monkeypatch.setattr(connections, "_connected_x", lambda: False)

    out = oracle_tool(action="screen")

    assert "x" not in out["lookback_options"]


def test_screen_asks_the_x_lookback_with_exactly_the_presets_when_x_is_connected(
        oracle_tool, monkeypatch):
    """A separate question, offering exactly the presets — and no "all": X is hard-capped at 2
    years, so an "all" answer would be a promise the pull cannot keep."""

    monkeypatch.setattr(connections, "_connected_x", lambda: True)

    out = oracle_tool(action="screen")

    assert out["lookback_options"] == {"x": {
        "presets": ["6mo", "1yr", "2yr"], "default": "6mo",
        "ask": ("How far back should I pull their X posts: 6 months, 1 year, or 2 years? "
                "(6 months is quickest.)")}}


def test_the_oracle_tool_never_tells_the_host_about_a_pre_tick(oracle_tool):
    """Nothing is selected by default, so nothing the host reads may describe
    a default selection it would then explain to the user."""
    doc = oracle_tool.__doc__.lower()
    for word in ("pre-tick", "pre_tick", "corroborat", "shown_by_default", "recommended_count",
                 "default-yes", "demoted"):
        assert word not in doc


# ── `candidates` carries the list clock ─────────────────────────────────────────
#
# Two different clocks meet on this payload and neither substitutes for the other. `probe_pulls`
# says whether a candidate's content is stale; `collector_runs` says whether the candidate list is.
# Only the second can tell you that someone you followed last week was never offered as a candidate
# at all, which is the failure that reads as "nothing new to promote".

def _stamp_all(status="ok", **kw):
    from pipeline.curation import clock as cs
    from pipeline.curation import collectors
    conn = schema.connect()
    try:
        for name in collectors.COLLECTORS:
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
    from pipeline.curation import collectors

    out = oracle_tool(action="candidates")

    fresh = out["list_freshness"]
    assert fresh["needs_attention"] is True
    assert fresh["never_succeeded"] == len(collectors.COLLECTORS)
    assert {e["collector"] for e in fresh["collectors"]} == set(collectors.COLLECTORS)
    assert all(e["never_ran"] for e in fresh["collectors"])


def test_candidates_names_the_one_collector_that_went_stale(oracle_tool, kb_home):
    from datetime import datetime, timedelta, timezone

    from pipeline.curation import clock as cs

    _stamp_all()
    old = (datetime.now(timezone.utc) - timedelta(hours=cs.STALE_AFTER_HOURS + 1)).isoformat()
    conn = schema.connect()
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
    from pipeline.curation import clock as cs

    monkeypatch.setattr(cs, "status_summary",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no such table")))

    out = oracle_tool(action="candidates")

    assert "list_freshness" not in out
    assert "error" not in out
    assert "candidates" in out


def _one_oracle(handle="willccbb"):
    conn = schema.connect()
    registry.init_state_schema(conn)
    try:
        schema.upsert_entity(conn, "x:user:1", name="W", profile={"handle": handle})
        roster.upsert_oracle(conn, "x:user:1", name="W")
        pairs.seed_from_entities(conn)
    finally:
        conn.close()


def test_screen_carries_per_source_freshness_unasked(oracle_tool, kb_home):
    """Unconditional, unlike `candidates`' `list_freshness`, and the difference is call frequency:
    `screen` is the deliberate, occasional "what is my people situation" call, which is where
    a frozen refresh loop has to show."""
    _one_oracle()
    out = oracle_tool(action="screen")

    fresh = out["oracle_freshness"]
    assert fresh["tracked_pairs"] == 1
    src = fresh["oracles"][0]["sources"][0]
    assert src["source_type"] == "x" and src["never_refreshed"] is True


def test_a_roster_nothing_has_pulled_needs_attention_and_says_so(oracle_tool, kb_home):
    """A roster nothing re-pulls never errors, so it must not be silent. There is no consent
    condition: Opyt never asks."""
    _one_oracle()
    fresh = oracle_tool(action="screen")["oracle_freshness"]

    assert "consented" not in fresh
    assert fresh["needs_attention"] is True
    assert "since_last" in fresh["note"], "the note must offer the cheap targeted top-up"


def test_an_empty_roster_stays_silent(oracle_tool, kb_home):
    """A fresh install has no Oracles, so telling it Oracle refresh is off is noise on the first
    surface a new user meets. `needs_attention` is guarded on tracked_pairs for exactly this."""
    fresh = oracle_tool(action="screen")["oracle_freshness"]

    assert fresh["tracked_pairs"] == 0
    assert fresh["needs_attention"] is False
    assert "note" not in fresh


@pytest.mark.parametrize("stale_of_four, attention", [(2, False), (3, True)])
def test_attention_tracks_the_stale_FRACTION_not_a_count(oracle_tool, kb_home, stale_of_four,
                                                         attention):
    """A ratio, not a count. An absolute count cannot work: overdue pairs grow with
    the roster, so a fixed threshold is generous at 8 Oracles and permanently tripped at 50.
    Half is the knee — solving (cycle-TTL)/cycle > 0.5 gives cycle > 2x TTL, i.e. "your refresh
    cycle has stretched past twice what you asked for". 2 of 4 is a loop working through a queue;
    3 of 4 is a loop losing ground."""
    from datetime import datetime, timedelta, timezone

    conn = schema.connect()
    registry.init_state_schema(conn)
    try:
        for i in range(4):
            schema.upsert_entity(conn, f"x:user:{i}", name=f"P{i}", profile={"handle": f"p{i}"})
            roster.upsert_oracle(conn, f"x:user:{i}", name=f"P{i}")
        pairs.seed_from_entities(conn)
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
    monkeypatch.setattr(freshness, "status_summary",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("registry gone")))
    out = oracle_tool(action="screen")

    assert "registry gone" in out["oracle_freshness"]["error"]
    assert "groups" in out

