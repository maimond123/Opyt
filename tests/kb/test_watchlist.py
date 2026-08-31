"""The watchlist — the standing queries as a surface a person can see and change.

RULED 2026-08-25 (David's design): standing queries become a user-visible watchlist, reviewed at
the decision point rather than pinned by a flag. Review-at-the-decision-point beats standing pin
state — no flag, no writer exemption, and strictly more power (add and edit, not just keep).

What these lock, all of them silent failures:

  • PULL-ONLY. The list is shown when the user asks and inside the result of a read they themselves
    triggered. A scheduler read records the same diff and surfaces nothing. Standing queries run
    quietly; announcing them unprompted is the recital the frontier surface's etiquette forbids.
  • NO LANE VOCABULARY. The quota is enforcement-internal. Naming it means teaching the entry_mode
    taxonomy to explain a distinction the user cannot act on.
  • `votable=False` ON A USER QUERY IS LOAD-BEARING. `_sync_speed` takes the MIN miss_count over
    VOTABLE claims, and nothing ever verdicts a user-authored query — so a votable user claim sits
    at 0 forever, pins every query it touches to daily, and erases decay through one shared row.
  • A DROP IS GLOBAL. One list of questions, not a copy per region.
  • THE DIFF IS NAMED, NOT COUNTED. "three new questions" leaves the user unable to judge or drop
    any of them.
"""
from __future__ import annotations

import pytest

from mcp_server import sitting_tools as st
from pipeline.kb import frontier_execute as fe
from pipeline.kb import frontier_queries as fq
from pipeline.kb import schema

REGION, OTHER = "sitting:mlx", "sitting:kiss1r"


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _q(text: str) -> dict:
    return {"text": text, "rationale": "r", "target_sources": ["arxiv"], "atom_ids": []}


def _call(conn, **kw) -> dict:
    return st._watchlist(conn, **{"sitting_id": None, "query": None,
                                  "add": None, "drop": None, **kw})


def _texts(res) -> list[str]:
    return [w["text"] for w in res["watching"]]


# ── the list ────────────────────────────────────────────────────────────────────
def test_the_list_names_the_speed_and_who_asked(conn):
    """Today the user sees only a query COUNT in frontier status. Speed is what answers "how often
    is this running", and `source` is the one distinction they can act on — only a question they
    typed themselves is exempt from decay."""
    fq.upsert_queries(conn, [_q("continuous batching")], generator=REGION)
    fq.add_user_query(conn, "agentic payment rails")

    rows = {w["text"]: w for w in _call(conn)["watching"]}
    assert rows["continuous batching"]["speed"] == "daily"
    assert rows["continuous batching"]["source"] == "a read of your material"
    assert rows["agentic payment rails"]["source"] == "you"


def test_a_slowed_query_reads_as_slower_not_as_gone(conn):
    """The decay tiers exist so no machine path ever needs to remove a query. A quiet thread slows
    to a cheap monthly floor and stays visible — and a list that hid it would make "why did my
    question disappear" unanswerable."""
    fq.upsert_queries(conn, [_q("quiet thread")], generator=REGION)
    for _ in range(fe.DECAY_TIERS[0][0]):
        fq.apply_verdicts(conn, [{"text": "quiet thread", "verdict": "drop", "reason": "r",
                                  "atom_ids": []}], generator=REGION)
    assert [w["speed"] for w in _call(conn)["watching"]] == ["weekly"]


def test_no_lane_vocabulary_reaches_the_surface(conn):
    """The quota is enforcement-internal bookkeeping. Telling a person three of their watched
    questions are 'machine lane' requires teaching the whole entry_mode taxonomy to explain a
    distinction that changes nothing they can do."""
    fq.upsert_queries(conn, [dict(_q("verifiable compute"), lane=fq.LANE_MACHINE)],
                      generator=REGION)
    res = _call(conn)
    assert _texts(res) == ["verifiable compute"], "the query itself must still be shown"
    body = repr(res).lower()
    assert not any(w in body for w in ("lane", "machine", "frontier", "quota"))


def test_the_list_scopes_to_a_region_when_one_is_named(conn):
    fq.upsert_queries(conn, [_q("mlx thread")], generator=REGION)
    fq.upsert_queries(conn, [_q("kisspeptin thread")], generator=OTHER)
    assert _texts(_call(conn, query="mlx")) == ["mlx thread"]
    assert sorted(_texts(_call(conn))) == ["kisspeptin thread", "mlx thread"]


def test_asking_about_an_unwatched_topic_builds_nothing(conn):
    """A watchlist request must not quietly buy an embedding and mint a region as a side effect of
    asking what is being watched."""
    res = _call(conn, query="something nobody has read")
    assert res["status"] == "error" and "read that region first" in res["reason"]
    assert conn.execute("SELECT COUNT(*) FROM sittings").fetchone()[0] == 0


# ── add ─────────────────────────────────────────────────────────────────────────
def test_a_user_added_query_is_not_votable(conn):
    """THE HAZARD. `_sync_speed` MINs over VOTABLE claims only, and nothing ever renders a verdict
    on a user-authored query. Votable, the user's claim sits at miss_count 0 forever and pins every
    query it touches to the daily tier — decay dead across the whole set, counters all healthy."""
    fq.upsert_queries(conn, [_q("shared thread")], generator=REGION)
    for _ in range(4):
        fq.apply_verdicts(conn, [{"text": "shared thread", "verdict": "drop", "reason": "r",
                                  "atom_ids": []}], generator=REGION)
    assert [w["speed"] for w in _call(conn)["watching"]] == ["weekly"]

    _call(conn, add=["shared thread"])                 # the user adopts the same question
    assert conn.execute("SELECT votable FROM frontier_query_generators g "
                        " JOIN frontier_generators fg ON fg.generator = g.generator "
                        " WHERE g.generator = ?", (fq.USER_GENERATOR,)).fetchone()[0] == 0
    assert [w["speed"] for w in _call(conn)["watching"]] == ["weekly"], \
        "the user's claim voted and reset the decay"


def test_an_added_query_says_it_will_not_decay(conn):
    """The pin reborn as the obvious semantics of an add button — and said out loud, because a
    question that silently behaved differently from its neighbours is worse than no exemption."""
    res = _call(conn, add=["agentic payment rails"])
    assert res["added"] == ["agentic payment rails"]
    assert res["added_note"]
    assert "agentic payment rails" in _texts(res)


# ── drop ────────────────────────────────────────────────────────────────────────
def test_a_drop_retires_the_question_everywhere(conn):
    """One list of questions, not a copy per region — so a question two regions both watch is
    retired for both. Stated in the result, because the user asked from inside one region and would
    otherwise have no way to know."""
    fq.upsert_queries(conn, [_q("shared thread")], generator=REGION)
    fq.upsert_queries(conn, [_q("shared thread")], generator=OTHER)

    res = _call(conn, query="mlx", drop=["shared thread"])
    assert res["dropped"] == ["shared thread"]
    assert res["dropped_note"]
    assert _texts(_call(conn)) == []


def test_a_drop_that_matched_nothing_is_reported(conn):
    """SHOW DECIDED, DON'T HIDE. A silent no-op reads as success, and the user walks away believing
    they stopped watching something they did not."""
    res = _call(conn, drop=["never existed"])
    assert res["not_found"] == ["never existed"]
    assert "dropped" not in res


def test_nothing_but_a_human_retires_a_query(conn):
    """No omission-retirement machinery of any kind — and none to exempt user queries FROM. A query
    the reader stops re-emitting only slows; the retired list is therefore always a person's doing,
    which is what makes it a meaningful third bucket in the diff."""
    fq.upsert_queries(conn, [_q("dropped every time")], generator=REGION)
    for _ in range(20):
        fq.apply_verdicts(conn, [{"text": "dropped every time", "verdict": "drop", "reason": "r",
                                  "atom_ids": []}], generator=REGION)
    assert fq.retired_texts(conn, generator=REGION) == []
    assert [w["speed"] for w in _call(conn)["watching"]] == ["monthly"]
