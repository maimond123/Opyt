"""The probe store — candidate content held outside the trusted KB.

The first tests are the trust boundary: writing candidate content into `atoms`/`bodies` would make an unvetted stranger's posts
indistinguishable from Oracle knowledge in every existing search, and the only thing that can
prevent that regression is an assertion that fails when someone reintroduces it.

The rest pin the mechanics the pull and the search depend on: hash-skip idempotency, per-candidate
scoping, a body rewritten in place under the same docid, and an index the triggers keep in step.
"""
from __future__ import annotations

import re
import sqlite3

import pytest

from pipeline.kb import schema, writes
from pipeline.kb.ingest_common import store_atom
from pipeline.oracles import probe_store
from pipeline.retrieval import engine, excerpt
from pipeline.retrieval.rank import fts_query


def _probe_atom(tid: str, who: str = "x:user:11", url: str = "https://x.com/a/status/1") -> dict:
    return {"atom_id": f"xprobe:{tid}", "source_type": "x", "who_id": who,
            "when_ts": "2026-08-01", "when_precision": "day", "source_url": url,
            "raw_hash": f"h{tid}",
            "description": f"post {tid}", "payload": {"like_count": 3}}


def _integrity_check(conn) -> None:
    """FTS5's own check that the external-content index agrees with `probe_bodies`. Raises
    `sqlite3.DatabaseError` ("database disk image is malformed") when a write went unindexed or a
    stale entry was left behind."""
    conn.execute("INSERT INTO probe_bodies_fts(probe_bodies_fts, rank) VALUES ('integrity-check', 1)")


def _body_id(conn, atom_id: str) -> int:
    return conn.execute("SELECT body_id FROM probe_bodies WHERE atom_id=?", (atom_id,)).fetchone()[0]


def _matching(conn, query: str) -> list[str]:
    return [r["atom_id"] for r in probe_store.probe_fts_rows(conn, fts_query(query), limit=10)]


# ── the trust boundary ────────────────────────────────────────────────────────

def test_probe_content_never_lands_in_atoms(kb_home):
    conn = schema.connect()
    probe_store.store_probe_atom(conn, _probe_atom("1"), "an agent framework for autonomous tools")

    assert probe_store.count_probe_atoms(conn) == 1
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM bodies").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM bodies_fts").fetchone()[0] == 0
    conn.close()


def test_store_probe_atom_issues_no_write_to_a_trusted_table(kb_home):
    """Structural, not a row count: every statement the probe writer executes is captured, and
    none may write `atoms`, `bodies` or `bodies_fts`. A count would miss a write that a later
    statement in the same call undid; this cannot."""
    conn = schema.connect()
    probe_store.init_probe_schema(conn)
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    probe_store.store_probe_atom(conn, _probe_atom("1"), "an agent framework")
    probe_store.store_probe_atom(conn, {**_probe_atom("1"), "raw_hash": "h1-v2"}, "a rewrite")
    conn.set_trace_callback(None)

    assert seen, "no SQL captured — the trace callback did not attach, so this proves nothing"
    trusted_write = re.compile(
        r"\b(insert\s+(or\s+\w+\s+)?into|update|delete\s+from)\s+(atoms|bodies|bodies_fts)\b")
    for sql in seen:
        flat = " ".join(sql.split()).lower()
        assert not trusted_write.search(flat), f"the probe writer wrote a TRUSTED table:\n  {sql}"
    conn.close()


def test_existing_search_cannot_see_probe_rows(kb_home):
    """The regression that matters: a probe post about agents must not answer an agent query on the
    trusted rail."""
    conn = schema.connect()
    probe_store.store_probe_atom(conn, _probe_atom("1"), "an agent framework for autonomous tools")

    run = engine.search_atoms(conn, "agent framework")
    assert run.hits == [], "probe row leaked into search_atoms"
    conn.close()


# ── the trusted rail is untouched by the probe store ──────────────────────────

def test_the_trusted_writer_never_creates_the_probe_store(kb_home):
    """`store_atom` is the trusted writer and the probe store is not its business: a store that
    has only ever saved trusted content has no probe tables at all."""
    conn = schema.connect()
    store_atom(conn, {"atom_id": "x:1", "source_type": "x", "who_id": "x:user:11",
                      "description": "d", "raw_hash": "h", "payload": {}}, "an agent framework")
    assert conn.execute("SELECT COUNT(*) FROM atoms").fetchone()[0] == 1
    assert probe_store.count_probe_atoms(conn) == 0
    assert not probe_store.probe_tables_exist(conn)
    conn.close()


def test_both_stores_coexist_in_one_db(kb_home):
    conn = schema.connect()
    store_atom(conn, {"atom_id": "x:1", "source_type": "x", "who_id": "x:user:99",
                      "description": "d", "raw_hash": "h", "payload": {}}, "a crypto rollup proof")
    probe_store.store_probe_atom(conn, _probe_atom("2"), "a crypto rollup proof")

    run = engine.search_atoms(conn, "crypto rollup")
    assert [h.atom_id for h in run.hits] == ["x:1"]     # the trusted one, and only it
    assert _matching(conn, "crypto rollup") == ["xprobe:2"]   # …and the probe one only here
    assert probe_store.count_probe_atoms(conn) == 1
    conn.close()


# ── mechanics the pull depends on ─────────────────────────────────────────────

def test_hash_ledger_scopes_to_one_candidate(kb_home):
    conn = schema.connect()
    probe_store.store_probe_atom(conn, _probe_atom("1", who="x:user:11"), "alpha")
    probe_store.store_probe_atom(conn, _probe_atom("2", who="x:user:22"), "beta")

    assert probe_store.load_probe_hashes(conn) == {"xprobe:1": "h1", "xprobe:2": "h2"}
    assert probe_store.load_probe_hashes(conn, "x:user:11") == {"xprobe:1": "h1"}
    assert probe_store.probed_who_ids(conn) == {"x:user:11", "x:user:22"}
    conn.close()


def test_hash_ledger_is_empty_before_the_first_pull(kb_home):
    """Fail-safe: a store that has never probed answers "nothing seen", not "no such table"."""
    conn = schema.connect()
    assert probe_store.load_probe_hashes(conn) == {}
    assert probe_store.count_probe_atoms(conn) == 0
    conn.close()


def test_rewriting_an_atom_replaces_its_body_in_place_and_bumps_version(kb_home):
    """A rewrite is an update of the one body row, never a second row and never a delete+insert:
    the body keeps its `body_id`, which is the index's docid, and the old text stops matching."""
    conn = schema.connect()
    probe_store.store_probe_atom(conn, _probe_atom("1"), "an agent framework")
    first_id = _body_id(conn, "xprobe:1")

    grown = {**_probe_atom("1"), "raw_hash": "h1-v2"}
    probe_store.store_probe_atom(conn, grown, "now a much longer body about crypto rollups")

    row = conn.execute("SELECT version, raw_hash FROM probe_atoms WHERE atom_id='xprobe:1'"
                       ).fetchone()
    assert row["version"] == 2 and row["raw_hash"] == "h1-v2"
    assert probe_store.count_probe_atoms(conn) == 1          # replaced in place, not duplicated
    assert conn.execute("SELECT COUNT(*) FROM probe_bodies").fetchone()[0] == 1
    assert _body_id(conn, "xprobe:1") == first_id, "the rewrite gave the body a new docid"
    assert _matching(conn, "crypto rollups") == ["xprobe:1"]
    assert _matching(conn, "agent framework") == [], "the old text is still indexed"
    conn.close()


def test_the_index_stays_consistent_through_insert_update_and_delete(kb_home):
    """The triggers are the only thing keeping `probe_bodies_fts` in step with `probe_bodies`
    (external content stores no text of its own). FTS5's integrity-check compares the two, so a
    missing or wrong trigger fails here rather than as a silently stale search."""
    conn = schema.connect()
    probe_store.store_probe_atom(conn, _probe_atom("1"), "an agent framework")
    probe_store.store_probe_atom(conn, _probe_atom("2"), "a crypto rollup proof")
    _integrity_check(conn)

    probe_store.store_probe_atom(conn, {**_probe_atom("1"), "raw_hash": "h1-v2"},
                                 "a protein folding model")
    _integrity_check(conn)

    with writes.write_tx(conn):
        conn.execute("DELETE FROM probe_bodies WHERE atom_id='xprobe:2'")
    _integrity_check(conn)
    assert _matching(conn, "crypto rollup") == []
    assert _matching(conn, "protein folding") == ["xprobe:1"]
    conn.close()


def test_a_failed_body_write_takes_the_probe_atom_row_back_with_it(kb_home, monkeypatch):
    """The probe atom row carries the `raw_hash` the pull's hash-skip reads as done, so it must
    not commit before the text is written; otherwise a body write that fails leaves a candidate's
    post that is never retried, holding its old text."""
    conn = schema.connect()
    probe_store.store_probe_atom(conn, _probe_atom("1"), "an agent framework")

    def locked(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(probe_store, "put_probe_body", locked)
    with pytest.raises(sqlite3.OperationalError):
        probe_store.store_probe_atom(conn, {**_probe_atom("1"), "raw_hash": "h1-v2"},
                                     "a different body about rollups")
    monkeypatch.undo()

    assert probe_store.load_probe_hashes(conn) == {"xprobe:1": "h1"}
    assert conn.execute("SELECT text FROM probe_bodies WHERE atom_id='xprobe:1'"
                        ).fetchone()[0] == "an agent framework"
    assert not conn.in_transaction
    conn.close()


def test_a_probe_atom_joins_the_callers_transaction(kb_home):
    """A page of a candidate's posts commits with its progress, so the writer must not commit."""
    conn = schema.connect()
    probe_store.init_probe_schema(conn)

    with pytest.raises(RuntimeError, match="page refused"), writes.write_tx(conn):
        probe_store.store_probe_atom(conn, _probe_atom("1"), "an agent framework")
        raise RuntimeError("page refused")

    assert probe_store.count_probe_atoms(conn) == 0
    assert conn.execute("SELECT COUNT(*) FROM probe_bodies").fetchone()[0] == 0
    with writes.write_tx(conn):
        probe_store.store_probe_atom(conn, _probe_atom("1"), "an agent framework")
    assert probe_store.count_probe_atoms(conn) == 1
    _integrity_check(conn)
    conn.close()


# ── the reads candidate search and the people map stand on ────────────────────

def test_fts_rows_are_one_per_atom_with_every_match_marked(kb_home):
    """Whole-item: an atom whose body matches many times is one row, and its `marked` text carries
    each match wrapped in `excerpt`'s markers, so the caller can cut its excerpt where they are."""
    conn = schema.connect()
    body = "rollup one. " + "filler words here. " * 30 + "rollup two. " + "more filler. " * 30
    probe_store.store_probe_atom(conn, _probe_atom("1"), body)
    probe_store.store_probe_atom(conn, _probe_atom("2", who="x:user:22"), "nothing relevant")

    rows = probe_store.probe_fts_rows(conn, fts_query("rollup"), limit=10)
    assert [r["atom_id"] for r in rows] == ["xprobe:1"]
    marked = rows[0]["marked"]
    assert marked.count(chr(excerpt.OPEN)) == 2 == marked.count(chr(excerpt.CLOSE))
    assert marked.replace(chr(excerpt.OPEN), "").replace(chr(excerpt.CLOSE), "") == body
    assert (rows[0]["who_id"], rows[0]["source_url"]) == ("x:user:11", "https://x.com/a/status/1")
    conn.close()


def test_fts_rows_scope_to_the_given_people(kb_home):
    conn = schema.connect()
    probe_store.store_probe_atom(conn, _probe_atom("1", who="x:user:11"), "a crypto rollup")
    probe_store.store_probe_atom(conn, _probe_atom("2", who="x:user:22"), "a crypto rollup")
    q = fts_query("crypto rollup")
    assert [r["who_id"] for r in probe_store.probe_fts_rows(conn, q, limit=10,
                                                             who_ids={"x:user:22"})] == ["x:user:22"]
    assert probe_store.probe_fts_rows(conn, q, limit=10, who_ids=set()) == []
    conn.close()


def test_texts_by_person_returns_whole_bodies(kb_home):
    """The people map reads each person's items whole: a long post is one text, never pieces."""
    conn = schema.connect()
    long_body = " ".join(f"word{i}" for i in range(2000))
    probe_store.store_probe_atom(conn, _probe_atom("1"), long_body)
    probe_store.store_probe_atom(conn, _probe_atom("2"), "a short one")
    probe_store.store_probe_atom(conn, _probe_atom("3", who="x:user:22"), "someone else")

    texts = probe_store.texts_by_person(conn)
    assert sorted(texts["x:user:11"]) == sorted([long_body, "a short one"])
    assert texts["x:user:22"] == ["someone else"]
    conn.close()


def test_the_author_rollup_counts_atoms(kb_home):
    conn = schema.connect()
    probe_store.store_probe_atom(conn, {**_probe_atom("1"), "when_ts": "2026-07-01"}, "a")
    probe_store.store_probe_atom(conn, {**_probe_atom("2"), "when_ts": "2026-08-01"}, "b " * 900)
    assert probe_store.probe_author_rollup(conn) == {
        "x:user:11": {"atoms": 2, "first_ts": "2026-07-01", "last_ts": "2026-08-01"}}
    conn.close()
