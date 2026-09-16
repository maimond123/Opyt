"""The `oracle` MCP tool's freshness surface — `screen`'s `oracle_freshness` block.

Kept apart from the adapter-seam tests: those prove an ADAPTER skips a call, these prove the TOOL
routes and reports. Modelled on the radar rail's `_FakeMCP` (since deleted).

Freshness rides on `screen` unasked, which is the only surface a caller needs to understand.
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
def test_a_retired_action_uses_the_ordinary_unknown_action_contract(oracle_tool, gone):
    out = oracle_tool(action=gone)
    assert set(out) == {"error"}


def test_the_surviving_actions_are_named_in_the_error(oracle_tool):
    err = oracle_tool(action="bogus")["error"]
    for kept in ("screen", "candidates", "confirm", "ingest", "review"):
        assert f"'{kept}'" in err


def test_review_lists_the_user_envelope_and_host_diagnostics(oracle_tool):
    assert oracle_tool(action="review") == {"items": [], "diagnostics": []}


def test_ingest_presentation_hides_diagnostics_and_says_nothing_it_cannot_measure():
    """The host gets a plain-language envelope; the raw adapter detail stays out of it.

    The tour is ABSENT here, and that is the assertion. Until 2026-09-12 this pinned three fixed
    options — search, aggregate, sitting — which is what a menu looks like when it cannot see the
    store: the same three labels for a six-atom corpus and a twelve-hundred-atom one. With no
    `conn` there is nothing to measure, and saying nothing beats offering a reading that
    `sitting(action='preview')` would warn against one call later.
    """
    from mcp_server.oracle_tools import _ingest_presentation

    out = _ingest_presentation([{
        "name": "Will",
        "atoms_added": 3,
        "results": [
            {"type": "substack", "action": "ingested", "detail": "internal adapter data"},
            {"type": "blog", "action": "needs-review", "detail": "internal trust data"},
        ],
        "available_sources": [{"source_type": "x", "url": "https://x.com/will"}],
    }])

    assert out["completed"] == [{"oracle": "Will", "sources": ["substack"], "atoms_added": 3}]
    assert out["possible_sources"]["oracles"] == ["Will"]
    assert out["possible_sources"]["next_step"] == "oracle(action='review')"
    assert out["optional_connections"][0]["source"] == "x"
    assert "tour" not in out
    assert "internal adapter data" not in str(out)


# ── "filling in on its own" is a claim about a running pass (2026-09-13 → 2026-09-14) ──
#
# The claim used to be unconditional, and on a from-source install it was FALSE: the only thing
# that resumed a deferred Oracle was the `oracle_refresh` rail, the only thing that launched a rail
# was the resident worker, and there was none — so the user was told an abandoned Oracle was in
# hand and stopped waiting for a pull that could not start.
#
# ⚠️ THAT LESSON SURVIVES; ONLY THE PROBE CHANGED. `footprint_enrichment` runs in-process and needs
# no worker, so the automatic case is now the normal one — but consent, model routing and a held
# lease can each still stop it, and the copy must read the pass that exists rather than assume it.
# The worker is no longer what decides, which is why it is no longer asked.

def _deferred_result():
    return [{"name": "Will", "atoms_added": 0,
             "results": [{"type": "x", "action": "deferred", "resumes": "next-scheduled-run"}]}]


def _present(monkeypatch, *, running, results=None):
    from mcp_server.oracle_tools import _ingest_presentation
    from pipeline.kb import footprint_enrichment
    monkeypatch.setattr(footprint_enrichment, "is_running", lambda: running)
    return _ingest_presentation(results or _deferred_result())


def test_a_running_pass_says_the_rest_is_filling_in_on_its_own(monkeypatch):
    out = _present(monkeypatch, running=True)

    assert out["in_progress"]["resumes"] is True
    msg = out["in_progress"]["message"]
    assert "filling in on its own" in msg
    assert "Nothing for you to do" in msg


def test_a_pass_that_could_not_start_says_so_instead_of_implying_it_is_in_hand(monkeypatch):
    """It must not merely omit the promise — it must CONTRADICT it, or the host fills the silence
    with the reassurance it used to read here. And it must not offer a countdown: if the pass is
    not running the reason is never the meter, so a clock would be a friendly-looking lie."""
    out = _present(monkeypatch, running=False)

    assert out["in_progress"]["resumes"] is False
    msg = out["in_progress"]["message"]
    assert "isn't moving on its own right now" in msg
    assert "filling in on its own" not in msg
    assert "Tell me when you'd like me to pick it up" in msg


def test_a_start_result_from_this_very_call_counts_as_running(monkeypatch):
    """The thread may not have set its own flag yet when the ingest returns. The start seam's
    answer is the more direct evidence, so it is read first."""
    from mcp_server.oracle_tools import _ingest_presentation
    from pipeline.kb import footprint_enrichment
    monkeypatch.setattr(footprint_enrichment, "is_running", lambda: False)

    out = _ingest_presentation(_deferred_result(), None, {"status": "running"})

    assert out["in_progress"]["resumes"] is True


def test_an_unreadable_probe_degrades_to_the_cautious_message(monkeypatch):
    """Fail-safe, and the direction is the whole point: a broken read must not become a promise."""
    from mcp_server.oracle_tools import _ingest_presentation
    from pipeline.kb import footprint_enrichment
    monkeypatch.setattr(footprint_enrichment, "is_running", lambda: 1 / 0)

    assert _ingest_presentation(_deferred_result())["in_progress"]["resumes"] is False


# ── the vocabulary rule, enforced (§G.0) ──────────────────────────────────────

_BANNED = ("rate limit", "window", "quota", "meter", "requests", "API", "throttled", "backfill",
           "rail", "worker", "foreground", "atoms", "ingest")


@pytest.mark.parametrize("running", [True, False])
def test_neither_variant_ever_explains_the_mechanism(monkeypatch, running):
    """⚠️ The person never learns WHY. The one permitted gesture at cause is "X only lets us read
    so much at a time" — true, needs no vocabulary, invites no follow-up. Everything else this
    message used to say ("x.com's 15-minute request window ran out … this install has no resident
    worker … call `oracle(action='ingest')` again") was five pieces of jargon in three sentences
    ending in a command the reader would never type."""
    msg = _present(monkeypatch, running=running)["in_progress"]["message"]

    for word in _BANNED:
        assert word not in msg.lower() if word.islower() else word not in msg, word
    assert "(" not in msg                       # no tool names, no commands


def test_the_guidance_forbids_elaborating_and_forbids_asking_permission(monkeypatch):
    """Both clauses are load-bearing. Handed a fact about a limit a model will helpfully explain
    the limit; handed unfinished work it will helpfully offer to finish it — and the second
    re-invents the confirmation prompt this revision removed, because offering to help reads as
    helpful."""
    g = _present(monkeypatch, running=True)["in_progress"]["guidance"]

    assert "NEVER explain why X stopped us" in g
    assert "not even if they ask how it works" in g
    assert "Do NOT ask them to authorise finishing it" in g
    assert "AFTER what landed" in g


def test_a_partial_writer_is_introduced_with_the_others_not_hedged(monkeypatch):
    """The failure mode the note heads off: the host reading `partial` as "broken" and burying a
    good writer in caveats. Half of somebody's writing is still a library."""
    out = _present(monkeypatch, running=True, results=[
        {"name": "Andrej", "atoms_added": 41,
         "results": [{"type": "x", "action": "ingested", "partial": True}]}])

    assert out["completed"][0]["partial"] is True
    assert "don't hold them back or hedge" in out["partial_note"]
    assert out["in_progress"]["oracles"] == ["Andrej"]


def test_the_names_read_like_a_sentence_not_a_field_dump(monkeypatch):
    out = _present(monkeypatch, running=True, results=[
        {"name": n, "atoms_added": 0, "results": [{"type": "x", "action": "deferred"}]}
        for n in ("Andrej Karpathy", "Martin Casado")])

    assert "Andrej Karpathy and Martin Casado's writing" in out["in_progress"]["message"]


def test_screen_omits_x_lookback_without_an_x_connection(oracle_tool, monkeypatch):
    """A source that is not connected is not an onboarding question yet."""
    from pipeline.ingestion import x_graphql

    monkeypatch.setattr(x_graphql, "has_managed_x_session", lambda: False)

    out = oracle_tool(action="screen")

    assert "x" not in out["lookback_options"]


# ── `candidates` carries the LIST clock ─────────────────────────────────────────
#
# Two different clocks meet on this payload and neither substitutes for the other. `probe_pulls`
# says whether a candidate's CONTENT is stale; `collector_runs` says whether the candidate LIST is.
# Only the second can tell you that someone you followed last week was never offered as a candidate
# at all, which is the failure that reads as "nothing new to promote".

def _stamp_all(status="ok", **kw):
    from pipeline.kb import curation_state as cs
    from pipeline.kb import ingest_curation as ic
    conn = schema.connect()
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


def test_the_ingest_envelope_leads_with_value_and_ends_with_work():
    """⚠️ ORDER IS THE MESSAGE. A host reads this envelope top-down and leads with what it finds
    first. Until 2026-09-12 that was an unverified profile to adjudicate and a platform to go
    connect — so a user's first moment with a finished library was a chore list, with the one key
    saying what they could now DO ranked below both.

    Nothing is dropped by the reorder. The review queue is durable and `oracle(action='review')`
    is its real home; the source-scoped onboarding plan already called this presentation "only a
    pointer to it", and a pointer need not be the first thing on the page.
    """
    from mcp_server.oracle_tools import _ingest_presentation

    out = _ingest_presentation([{
        "name": "Will", "atoms_added": 3,
        "results": [{"type": "substack", "action": "ingested"},
                    {"type": "blog", "action": "needs-review"},
                    {"type": "x", "action": "needs_reconnect"}],
        "available_sources": [{"source_type": "x", "url": "https://x.com/will"}],
    }])

    keys = list(out)
    assert keys.index("completed") < keys.index("needs_attention")
    assert keys.index("needs_attention") < keys.index("optional_connections")
    assert keys.index("optional_connections") < keys.index("possible_sources")
    assert keys[-1] == "possible_sources"
    # The wording reports a decision already taken, not a classifier's difficulty handed over.
    assert "could not verify" not in out["possible_sources"]["explanation"]
    assert "Never lead with it" in out["possible_sources"]["mention"]
