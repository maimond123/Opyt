"""Search is keyword-only: stemmed, and an OR-query of the content words."""
from __future__ import annotations

import pytest

from pipeline.kb import schema
from pipeline.kb.ingest_common import snapshot_hash, store_atom
from pipeline.oracles import probe_search, probe_store
from pipeline.retrieval import engine


def test_keyword_search_matches_other_forms_of_a_word(kb_home):
    """"accountable" must find "accountability"."""
    conn = schema.connect()
    text = "a long argument about accountability in model evaluations"
    store_atom(conn, dict(
        atom_id="x:9", source_type="x", what_kind="opinion", who_id="x:user:9",
        when_ts="2024-05-01", when_precision="day", about_entities=[], source_url="u",
        raw_hash=snapshot_hash(text), description="d", payload={},
        entry_mode="user-saved"), text)

    assert [h.atom_id for h in engine.search_atoms(conn, "accountable").hits] == ["x:9"]
    conn.close()


def test_candidate_keyword_search_is_stemmed_too(kb_home):
    conn = schema.connect()
    probe_store.store_probe_atom(conn, {
        "atom_id": "xprobe:1", "source_type": "x", "who_id": "x:user:11",
        "when_ts": "2026-08-01", "when_precision": "day",
        "source_url": "https://x.com/a/status/1", "raw_hash": "h1",
        "description": "post 1", "payload": {"like_count": 3}},
        "notes on accountability in evaluations")
    hits = probe_search.search_candidates(conn, "accountable", k=5, evidence=2)
    assert [h["who_id"] for h in hits] == ["x:user:11"]
    conn.close()


@pytest.mark.parametrize("query,want", [
    ("what is the best agent framework", '"best" OR "agent" OR "framework"'),
    ("the", '"the"'),                       # nothing but stopwords: kept, or nothing matches
    ("", '""'),
])
def test_an_or_query_leaves_out_stopwords(query, want):
    from pipeline.retrieval.rank import fts_query
    assert fts_query(query) == want


def test_a_query_of_only_common_words_has_no_content_words():
    from pipeline.retrieval.rank import content_words
    assert content_words("what is the best agent") == ["best", "agent"]
    assert content_words("The") == [] and content_words("") == []


def test_the_candidate_list_filters_the_matches_and_never_drives_the_join(kb_home):
    """A default search passes most of the library as candidates. If SQLite looks each one up in
    `bodies` for every matching body, a search takes a second instead of milliseconds, so the
    arm's query must reach `bodies` by the index's rowid."""
    conn = schema.connect()
    ran = []
    conn.set_trace_callback(ran.append)
    engine.atom_bm25_search(conn, "word", {"x:1", "x:2", "x:3"}, 8)
    conn.set_trace_callback(None)
    arm = next(sql for sql in ran if "bodies_fts MATCH" in sql)
    plan = [step[3] for step in conn.execute("EXPLAIN QUERY PLAN " + arm)]
    conn.close()
    assert "SEARCH b USING INTEGER PRIMARY KEY (rowid=?)" in plan, plan
