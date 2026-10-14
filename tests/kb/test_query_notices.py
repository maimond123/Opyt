"""`query_notices`: the query notices, decided from plain values — one test per code.

The store's facts are read by `gather_facts` and handed in, so these need no store. The
wording of each sentence is pinned with the whole envelope in `test_retrieval_envelope_pins.py`;
these pin which codes a situation produces, and in what order.
"""
from __future__ import annotations

from pipeline.retrieval.notices import NoticeFacts, query_notices
from pipeline.retrieval.engine import SearchRun

FULL = NoticeFacts(store_empty=False, missing_tags=[], common_tags=[], corpus_newest=None)
ALEX = {"canonical_id": "x:user:1", "name": "Alex", "handle": "alex", "who_ids": ["x:user:1"],
        "matched_on": "handle"}


def _run(candidates=None, hits=0, ranked=0, undated=0):
    """`candidates=0` is a pre-filter that matched nothing, so nothing was ranked."""
    return SearchRun(hits=[object()] * hits, candidates=candidates, ranked=ranked, cutoff=None,
                     fts_query=None, undated_excluded=undated)


def _codes(run=None, facts=FULL, *, tags=None, who=None, resolved=None, cost=None,
           kb_name="me", kb_label=None):
    out = query_notices(run or _run(), facts, tags=tags, who=who, resolved=resolved,
                         filter_cost=cost or {}, applied={}, kb_name=kb_name, kb_label=kb_label)
    return [n["code"] for n in out]


def test_a_healthy_query_says_nothing():
    assert _codes() == []


def test_an_empty_own_store_sends_the_reader_to_onboard_and_says_nothing_else():
    empty = NoticeFacts(True, [], [], None)
    out = query_notices(_run(0), empty, tags=["!!!"], who="@x", resolved=[],
                         filter_cost={}, applied={}, kb_name="me", kb_label=None)
    assert [(n["code"], n.get("next_tool")) for n in out] == [("store_empty", "onboard")]


def test_an_empty_foreign_store_does_not_send_the_reader_to_onboard():
    out = query_notices(_run(0), NoticeFacts(True, [], [], None), tags=None, who=None,
                         resolved=None, filter_cost={}, applied={}, kb_name="alex",
                         kb_label="Alex's KB")
    assert [(n["code"], "next_tool" in n) for n in out] == [("store_empty", False)]


def test_a_foreign_read_is_always_attributed():
    assert _codes(kb_name="alex") == ["foreign_kb"]


def test_a_handle_that_resolves_to_nobody():
    assert _codes(_run(0), who="@nobody", resolved=[]) == ["who_unresolved"]


def test_a_handle_that_resolves_to_two_people():
    assert _codes(who="alex", resolved=[ALEX, {**ALEX, "canonical_id": "x:user:2"}]) == [
        "who_multiple"]


def test_filters_that_matched_nothing():
    assert _codes(_run(0), cost={"source_type": 3}) == ["filters_matched_nothing"]


def test_tags_that_were_dropped_or_rewritten():
    assert _codes(tags=["AI Agents", "!!!"]) == ["tags_normalized"]


def test_a_tag_no_atom_carries():
    facts = NoticeFacts(False, ["nope"], ["ai-agents"], None)
    assert _codes(facts=facts, tags=["nope"]) == ["tag_not_in_store"]


def test_undated_atoms_excluded_by_a_date_bound():
    assert _codes(_run(undated=2)) == ["undated_excluded"]


def test_results_truncated():
    assert _codes(_run(hits=8, ranked=20)) == ["results_truncated"]


def test_codes_keep_their_order():
    facts = NoticeFacts(False, ["nope"], [], None)
    assert _codes(_run(0, undated=1), facts, tags=["Nope"], who="@x", resolved=[],
                  kb_name="alex") == ["foreign_kb", "who_unresolved", "tags_normalized",
                                      "tag_not_in_store", "undated_excluded"]
