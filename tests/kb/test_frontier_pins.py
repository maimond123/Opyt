"""The frontier unit's entries and durable names, pinned as they answer today, before the moves.

An operator requeues rejected candidates by
hand, and every home's store holds the frontier tables. Those are durable names: no move may
change what any of them answers. (`latest` is pinned in tests/kb/test_arrivals.py.)

The pins reach the unit through its entries (the searches and admissions
sources, `pipeline.frontier.searching` and `admitting`, which replaced the two worker jobs and
their `RAILS` paths) and seed the store with raw SQL, so a move inside the unit leaves them alone.
One import line is the exception and changes with its move, nothing else: `Candidate`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from pipeline.frontier.candidates import Candidate
from pipeline.frontier import admitting
from pipeline.frontier.searching import Searches
from pipeline.kb import schema
from pipeline.worker.quota_loop.retries import PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.source import Context

_NOW = datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _stage(conn, cid, *, status="kept", attempts=0, last_error=None):
    conn.execute(
        """INSERT INTO frontier_candidates
             (candidate_id, source, kind, title, url, published, summary, payload,
              status, first_seen_at, last_seen_at, attempts, last_error)
           VALUES (?,'arxiv','paper',?,?,'2026-08-11','s','{}',?,?,?,?,?)""",
        (cid, f"title {cid}", f"https://example/{cid}", status, "2026-08-11T00:00:00Z",
         "2026-08-11T00:00:00Z", attempts, last_error))
    conn.commit()


# ── The operator's requeue ──────────────────────────────────────────────────────
def _statuses(conn) -> dict:
    return {r["candidate_id"]: (r["status"], r["attempts"], r["last_error"]) for r in
            conn.execute("SELECT candidate_id, status, attempts, last_error "
                         "FROM frontier_candidates")}


def _rejected_and_others(conn):
    _stage(conn, "arxiv:a", status="rejected", attempts=5, last_error="blocked_metadata")
    _stage(conn, "arxiv:b", status="rejected", attempts=1, last_error="no_atom_id")
    _stage(conn, "arxiv:c", status="kept", attempts=2, last_error="error_ValueError")
    _stage(conn, "arxiv:d", status="materialized")


def test_requeue_rejected_resets_every_rejected_row(conn, capsys):
    _rejected_and_others(conn)
    assert admitting.main(["--requeue-rejected"]) == 0
    assert capsys.readouterr().out == "requeued 2 rejected candidate(s)\n"
    assert _statuses(conn) == {
        "arxiv:a": ("kept", 0, None),
        "arxiv:b": ("kept", 0, None),
        "arxiv:c": ("kept", 2, "error_ValueError"),
        "arxiv:d": ("materialized", 0, None),
    }


def test_requeue_rejected_with_a_slug_resets_only_that_reason(conn, capsys):
    _rejected_and_others(conn)
    assert admitting.main(["--requeue-rejected", "blocked_metadata"]) == 0
    assert capsys.readouterr().out == "requeued 1 rejected candidate(s)\n"
    assert _statuses(conn)["arxiv:a"] == ("kept", 0, None)
    assert _statuses(conn)["arxiv:b"] == ("rejected", 1, "no_atom_id")


# ── Stored names ────────────────────────────────────────────────────────────────
def test_the_frontier_tables_and_their_columns(conn):
    """Every home's store holds these. Which topic owns a query is `topic_watches`, topics'."""
    tables = {
        "frontier_queries": ["query_id", "text", "normalized", "target_sources", "created_at"],
        "frontier_query_sources": ["query_id", "source", "last_pulled_at", "last_status",
                                   "tried_at"],
        "frontier_candidates": [
            "candidate_id", "source", "kind", "title", "url", "published", "summary", "payload",
            "status", "first_seen_at", "last_seen_at", "attempts", "last_error",
            "last_attempt_at", "atom_id"],
        "frontier_candidate_queries": ["candidate_id", "query_id", "found_at"],
        "watch_context": ["query_id", "words"],
    }
    assert {t: [r["name"] for r in conn.execute(f"PRAGMA table_info({t})")]
            for t in tables} == tables


def test_a_search_stamps_ok_for_a_page_with_finds_and_empty_for_an_empty_page(conn):
    """The other three `last_status` values are pinned in test_frontier_searching.py: `error`,
    `window_refused`, `no_adapter`. (No `breaker_open`: an open breaker's
    source is not offered, so nothing is stamped.)"""
    conn.execute("INSERT INTO user_topics (topic_id, name, created_at, updated_at) "
                 "VALUES ('t', 'g', 'a', 'a')")
    for qid, text in (("q-finds", "gated deltanet"), ("q-empty", "nothing upstream")):
        conn.execute(
            "INSERT INTO frontier_queries (query_id, text, normalized, target_sources, "
            "created_at) VALUES (?,?,?,?,?)",
            (qid, text, text, json.dumps(["arxiv"]), "2026-08-01T00:00:00Z"))
        conn.execute("INSERT INTO topic_watches (topic_id, query_id, name, added_at) "
                     "VALUES ('t', ?, 'w', 'a')", (qid,))
    conn.commit()

    class Adapter:
        slug, host = "arxiv", "export.arxiv.org"

        def search(self, query, *, since, limit, now, context=()):
            assert now == _NOW
            if query != "gated deltanet":
                return []
            return [Candidate(candidate_id="arxiv:2508.00003", source="arxiv", kind="paper",
                              title="Gated DeltaNet ablations",
                              url="https://arxiv.org/abs/2508.00003")]

    source = Searches(Outages(PLATFORM_PAUSES_S), registry={"arxiv": Adapter()},
                      clock=lambda: _NOW)
    for unit in source.candidates(conn, _NOW):
        source.record(conn, unit, source.fetch(conn, unit, Context("web")))

    assert [tuple(r) for r in conn.execute(
        "SELECT * FROM frontier_query_sources ORDER BY query_id")] == [
        ("q-empty", "arxiv", "2026-08-12T12:00:00+00:00", "empty", "2026-08-12T12:00:00+00:00"),
        ("q-finds", "arxiv", "2026-08-12T12:00:00+00:00", "ok", "2026-08-12T12:00:00+00:00"),
    ]
    assert conn.execute("SELECT COUNT(*) FROM frontier_candidates").fetchone()[0] == 1
