"""Riders, pinned through the real server: one MCP client session against the built FastMCP.

A rider is a notice one unit produces that rides on another unit's answer (completion, new
material, routing, staleness, what arrived, the
share-back offer, the offer, and the thin-coverage offer). Testing those helpers directly, or
registering the tools on a fake MCP, bypasses the path a host actually sees, so a move of the
attachment would pass while changing the wire. These pin what the host receives instead: the answer as `tools/call` returns it, its key
order, the order of `notices`, which calls consume a notice, and that the wire text is exactly
the structured answer.

Producers are faked at their module attribute, never inside the carrier: what a producer says is
its owner's test; where and when it lands is this file's.
"""
from __future__ import annotations

import pytest

from mcp_server import instructions, offers
from mcp_server.rider_rules import ONCE_PER_CONVERSATION
from pipeline.arrivals import notice as arrivals
from pipeline import model_routing
from pipeline.curation import still_scoring
from pipeline.kb import schema
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.ingest_common import snapshot_hash
from pipeline.platforms import connections
from pipeline.worker import liveness
from pipeline.retrieval import new_material
from pipeline.sharing import page_words
from tests import mcp_session
from pipeline.oracles import freshness
from pipeline.oracles import told

STALE = {"needs_attention": True, "tracked_pairs": 1, "oracles": []}
LANDED = {"milestones": [{"name": "Andrej", "milestone": "newest_posts"}], "message": "landed"}
MATERIAL = {"in_your_library": [{"material": "X bookmarks", "items": 3}], "host_note": "lead"}
ARRIVED = {"arrived": 2, "call": "latest()", "message": "2 new items arrived"}
OFFER = {"say": "s", "if_yes": "y", "host_note": "h"}


@pytest.fixture()
def mcp(kb_home):
    """One MCP session against the real server, closed by `tests/conftest.py`."""
    return mcp_session.session()


def on_calls(value, *fire_on: int):
    """A fake producer: returns `value` on the listed call numbers (1-based), None otherwise, and
    records every call — so a test can assert a notice was not consumed."""
    calls: list[int] = []

    def produce(*_a, **_k):
        calls.append(len(calls) + 1)
        return value if (not fire_on or len(calls) in fire_on) else None

    produce.calls = calls
    return produce


def _corpus():
    conn = schema.connect()
    for atom_id, who, text in (("github:root/agentkit", "github:root", "an agent framework"),
                               ("github:a/b", "github:a", "a framework library for agents"),
                               ("x:1", "x:user:1", "a framework for rollup proofs")):
        raw_hash = snapshot_hash(text)
        store_atom(conn, dict(
            atom_id=atom_id, source_type=atom_id.split(":")[0], what_kind="artifact",
            who_id=who, when_ts="2024-05-01", when_precision="day", about_entities=[],
            source_url=f"https://example/{atom_id}", raw_hash=raw_hash,
            description=f"{atom_id} card", payload={}, entry_mode="user-saved",
        ), text)
    conn.close()


def _begin_import():
    """The Bookmarks walk begun in this home's store: an import promised and not landed."""
    from datetime import datetime, timezone

    from pipeline.curation import bookmark_listing, listings

    conn = schema.connect()
    try:
        listings.init_schema(conn)
        listings.start(conn, bookmark_listing.COLLECTOR, "me", listings.Branch("Bookmarks"),
                       datetime.now(timezone.utc))
    finally:
        conn.close()


def _codes(answer) -> list[str]:
    return [n.get("code") for n in answer["notices"]]


# ── search ──────────────────────────────────────────────────────────────────────

def test_search_notices_keep_their_order_on_the_users_own_store(mcp, monkeypatch):
    """Every rider search carries, in one answer, in a fixed order. The order of `notices` is
    behavior — the host reads them as a sequence of sentences."""
    _corpus()
    monkeypatch.setattr(freshness, "status_summary", lambda conn: STALE)
    monkeypatch.setattr(told, "completion_notice", on_calls(LANDED, 3))
    monkeypatch.setattr(arrivals, "notice", on_calls(ARRIVED, 3))

    mcp.call("search", query="framework", k=1)
    mcp.call("search", query="framework", k=1)
    out = mcp.call("search", query="framework")

    assert _codes(out) == ["oracles_stale", "pull_finished", "thin_coverage"]
    assert list(out) == ["hits", "notices", "insights", "trace", "frontier_atoms", "latest",
                         "readable"]
    assert out["notices"][0]["freshness"] == STALE
    assert out["notices"][1] == {"code": "pull_finished", **LANDED}
    assert out["latest"] == {**ARRIVED, "host_note": ONCE_PER_CONVERSATION}
    assert out["readable"] == page_words.READABLE
    assert "session" not in out["trace"]


def test_staleness_is_not_said_on_an_empty_store(mcp, monkeypatch):
    """Today it is decided inside retrieval's notices, which return `store_empty` alone."""
    monkeypatch.setattr(freshness, "status_summary", lambda conn: STALE)
    out = mcp.call("search", query="framework")
    assert "oracles_stale" not in _codes(out)
    assert _codes(out)[0] == "store_empty"


def test_staleness_is_silent_when_nothing_needs_attention(mcp, monkeypatch):
    _corpus()
    monkeypatch.setattr(freshness, "status_summary", lambda conn: {"tracked_pairs": 1})
    assert "oracles_stale" not in _codes(mcp.call("search", query="framework"))


def test_completion_rides_every_search_and_is_taken_once_per_call(mcp, monkeypatch):
    _corpus()
    produce = on_calls(LANDED, 1)
    monkeypatch.setattr(told, "completion_notice", produce)
    first = mcp.call("search", query="framework")
    second = mcp.call("search", query="framework")
    assert "pull_finished" in _codes(first) and "pull_finished" not in _codes(second)
    assert produce.calls == [1, 2]


def test_the_latest_notice_rides_every_search_it_has_something_for(
        mcp, monkeypatch):
    """No latch. The host keeps "once per conversation", told so by
    the `host_note`; quiet looks contribute nothing."""
    _corpus()
    monkeypatch.setattr(arrivals, "notice", on_calls(ARRIVED, 2, 3))
    answers = [mcp.call("search", query="framework") for _ in range(3)]
    assert ["latest" in a for a in answers] == [False, True, True]
    assert answers[1]["latest"]["host_note"] == ONCE_PER_CONVERSATION


def test_the_latest_notice_does_not_keep_the_offer_off_an_answer(mcp, monkeypatch):
    """It rides nearly every own search, so if it counted as an ask no search would offer."""
    _corpus()
    monkeypatch.setattr(arrivals, "notice", on_calls(ARRIVED))
    monkeypatch.setattr(offers, "offer", on_calls(OFFER))
    out = mcp.call("search", query="framework", k=2)
    assert "latest" in out and list(out)[-1] == "offer"


def test_the_thin_offer_rides_every_thin_search_worded_once_per_conversation(mcp):
    _corpus()
    first = mcp.call("search", query="framework")
    second = mcp.call("search", query="framework")
    note = next(n for n in first["notices"] if n["code"] == "thin_coverage")
    assert (note["ranked"], note["asked"]) == (3, 8)
    assert "once per conversation" in note["message"]
    assert "thin_coverage" in _codes(second)


def test_a_filtered_search_is_never_called_thin(mcp):
    _corpus()
    out = mcp.call("search", query="framework", source_type="github")
    assert "thin_coverage" not in _codes(out)


def test_an_outstanding_import_is_named_instead_of_thin_coverage(mcp):
    _corpus()
    _begin_import()
    out = mcp.call("search", query="framework")
    note = next(n for n in out["notices"] if n["code"] == "import_incomplete")
    assert note["waiting"] == ["bookmarks_import"]
    assert "thin_coverage" not in _codes(out)


def test_the_server_keeps_no_memory_of_the_conversation(mcp):
    """The host holds the conversation. The same search, repeated
    without opening anything, gets the same answer every time: no `nothing_grounded`, no
    `trace.session`, no overlap count."""
    _corpus()
    answers = [mcp.call("search", query="framework", k=2) for _ in range(4)]
    assert all(a == answers[0] for a in answers)
    assert "nothing_grounded" not in _codes(answers[-1])
    assert "session" not in answers[-1]["trace"]


# ── aggregate ───────────────────────────────────────────────────────────────────

def test_aggregate_of_the_own_store_carries_the_offer_last(mcp, monkeypatch):
    _corpus()
    monkeypatch.setattr(offers, "offer", on_calls(OFFER))
    out = mcp.call("aggregate")
    assert list(out)[-1] == "offer" and out["offer"] == OFFER


def test_an_answer_that_already_asks_carries_no_offer_and_asks_for_none(
        mcp, monkeypatch):
    """One ask per answer (R3.3): completion asks the user to look, so the offer stays home."""
    _corpus()
    monkeypatch.setattr(told, "completion_notice", on_calls(LANDED, 1))
    monkeypatch.setattr(offers, "offer", on_calls(OFFER))
    out = mcp.call("aggregate")
    assert "offer" not in out and offers.offer.calls == []
    assert out["notices"][-1] == {"code": "pull_finished", **LANDED}


def test_aggregate_of_an_empty_store_offers_nothing(mcp, monkeypatch):
    monkeypatch.setattr(offers, "offer", on_calls(OFFER))
    assert "offer" not in mcp.call("aggregate") and offers.offer.calls == []


def test_aggregate_of_another_knowledge_base_offers_nothing_but_still_carries_completion(
        mcp, monkeypatch):
    """A foreign aggregate gets counts and nothing else; completion is about the reader's own
    pulls and rides any call, because OPYT cannot push."""
    _corpus()
    monkeypatch.setattr(told, "completion_notice", on_calls(LANDED, 1))
    monkeypatch.setattr(offers, "offer", on_calls(OFFER))
    out = mcp.call("aggregate", kb="nobody")
    assert "offer" not in out
    assert out["notices"][-1]["code"] == "pull_finished"


def test_an_own_search_with_hits_carries_the_offer_last(mcp, monkeypatch):
    _corpus()
    monkeypatch.setattr(offers, "offer", on_calls(OFFER))
    out = mcp.call("search", query="framework", k=2)
    assert list(out)[-1] == "offer" and out["offer"] == OFFER


def test_how_to_make_something_readable_rides_own_searches_and_opens_with_something_in_them(
        mcp):
    """The host may decide how to make a page long after any offer, with reader_view's
    description not loaded, so the line rides every own answer that holds items."""
    _corpus()
    assert mcp.call("search", query="framework")["readable"] == page_words.READABLE
    assert mcp.call("open", atom_id="x:1")["readable"] == page_words.READABLE
    assert "readable" not in mcp.call("search", query="zzzunmatched")
    assert "readable" not in mcp.call("search", query="framework", kb="nobody")
    assert "readable" not in mcp.call("open", atom_id="x:404")


def test_a_thin_search_carries_its_own_offer_and_no_other(mcp, monkeypatch):
    _corpus()
    monkeypatch.setattr(offers, "offer", on_calls(OFFER))
    out = mcp.call("search", query="framework", k=50)
    assert "thin_coverage" in _codes(out) and "offer" not in out


# ── the oracle screen ───────────────────────────────────────────────────────────

@pytest.fixture()
def screen_world(monkeypatch):
    from pipeline.oracles import screen

    monkeypatch.setattr(screen, "build_screen",
                        lambda conn, **kw: {"groups": []})
    monkeypatch.setattr(connections, "_connected_x", lambda: False)
    monkeypatch.setattr(freshness, "status_summary", lambda conn: {"tracked_pairs": 0})
    monkeypatch.setattr(model_routing, "preflight", lambda fetch: {
        "dead": [("m/dead", "deny-listed")], "fragile": [("m/thin", "one org", "acme")]})
    monkeypatch.setattr(still_scoring, "still_scoring_notice", lambda: {"pending": ["x"]})
    monkeypatch.setattr(told, "completion_notice", on_calls(LANDED, 1))
    monkeypatch.setattr(new_material, "new_material_notice", on_calls(MATERIAL, 1))


def test_the_screen_carries_its_riders_in_order_with_no_action_given(mcp, screen_world):
    """`oracle()` defaults to `screen`; a rider decision must see that default."""
    out = mcp.call("oracle")
    assert list(out) == ["groups", "lookback_options", "oracle_freshness", "model_routing",
                         "still_scoring", "pull_finished", "new_material"]
    assert out["model_routing"] == {
        "dead": ["m/dead (deny-listed) — NO provider survives the deny-list"],
        "fragile": ["m/thin (one org) — only acme; one withdrawal from dead"]}
    assert out["pull_finished"] == LANDED and out["new_material"] == MATERIAL


def test_routing_is_absent_when_nothing_is_dead_or_fragile(mcp, screen_world, monkeypatch):
    monkeypatch.setattr(model_routing, "preflight", lambda fetch: {"dead": [], "fragile": []})
    assert "model_routing" not in mcp.call("oracle")


def test_another_oracle_action_consumes_no_notice(mcp, screen_world):
    out = mcp.call("oracle", action="nonsense")
    assert set(out) == {"error"}
    assert told.completion_notice.calls == []
    assert new_material.new_material_notice.calls == []


# ── onboard ─────────────────────────────────────────────────────────────────────

def test_the_onboard_handoff_carries_completion_and_new_material(mcp, monkeypatch):
    """These ride the server's flow, so they land after onboarding's own worker keys."""
    from pipeline.onboarding import worker_check
    from tests.test_onboard_tool import _done_blind

    _done_blind(monkeypatch)
    monkeypatch.setattr(worker_check, "report", lambda worker: {"worker": "down"})
    monkeypatch.setattr(told, "completion_notice", on_calls(LANDED, 1))
    monkeypatch.setattr(new_material, "new_material_notice", on_calls(MATERIAL, 1))
    out = mcp.call("onboard")
    assert list(out) == ["status", "phase", "candidates", "next_tool", "message",
                         "worker", "pull_finished", "new_material"]
    assert out["pull_finished"] == LANDED and out["new_material"] == MATERIAL


def test_an_onboard_answer_that_is_not_the_handoff_consumes_nothing(mcp, monkeypatch):
    from tests.test_onboard_tool import _at_sources

    _at_sources(monkeypatch)
    produce = on_calls(LANDED)
    monkeypatch.setattr(told, "completion_notice", produce)
    out = mcp.call("onboard", source="nowhere")
    assert out["status"] == "error" and "pull_finished" not in out
    assert produce.calls == []


# ── topics ──────────────────────────────────────────────────────────────────────

def test_a_watch_answer_carries_no_rider_and_consumes_none(mcp, monkeypatch):
    """Setting up a topic is mid-decision: nothing rides it (R2)."""
    produce = on_calls(dict(ARRIVED), 1)
    monkeypatch.setattr(arrivals, "notice", produce)
    assert mcp.call("watch", action="list")["topics"] == []
    assert produce.calls == []


# ── setup instructions ──────────────────────────────────────────────────────────

def test_a_session_is_told_the_citation_rule_alone_when_the_frontier_is_quiet(mcp):
    assert mcp.instructions == instructions.CITATION_RULE


# ── the optional-keys question rides the first confirmation ─────────────────────────────────────

@pytest.fixture()
def picked(kb_home, monkeypatch):
    """Two people the store already shows linking work: the first a repo, the second a paper.
    X connected, the worker's word unknown, every wake recorded instead of written."""
    monkeypatch.setattr(connections, "_connected_x", lambda: True)
    monkeypatch.setattr(liveness, "repair_or_report_cached",
                        lambda **kw: {"ok": None, "status": "unknown"})
    monkeypatch.setattr("pipeline.worker.job_queue.request_now", lambda rail, **kw: True)
    for env in ("GITHUB_TOKEN", "OPENALEX_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    conn = schema.connect()
    for uid, (atom_id, kind) in (("1", ("github:o/r", "github")), ("2", ("paper:p", "paper"))):
        schema.upsert_entity(conn, f"x:user:{uid}", name=f"Person {uid}",
                             profile={"handle": f"p{uid}"})
        conn.execute("INSERT INTO atoms(atom_id, source_type, who_id, entry_mode, payload) "
                     "VALUES (?,?,?,?,?)",
                     (atom_id, kind, "someone", "author_referenced",
                      '{"referenced_by": [{"by": "x:user:%s", "post": "x:9"}]}' % uid))
    conn.commit()
    conn.close()
    return ["x:user:1", "x:user:2"]


def test_the_keys_question_rides_the_first_confirmation_and_no_other(mcp, picked):
    first = mcp.call("oracle", action="confirm", canonical_ids=picked[:1])
    again = mcp.call("oracle", action="confirm", canonical_ids=picked)
    assert list(first)[-1] == "optional_keys" and "offer" not in first
    assert first["optional_keys"]["answers"] == ["github"]
    assert "The person you just picked shares code on GitHub." in first["optional_keys"]["message"]
    assert "optional_keys" not in again


def test_the_keys_question_rides_add_oracle_too(mcp, picked):
    out = mcp.call("add_oracle", reference=picked[1], confirm=True)
    assert out["optional_keys"]["answers"] == ["openalex"]
