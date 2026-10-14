"""store_atom — the one writer every KB atom passes through.

An atom row and its body are written in one transaction: the row carries the new `raw_hash` that
every adapter's hash check reads as done, so it must never land without its body. Inside a
caller's `write_tx` the write joins that transaction (a page and its done-state commit together),
and `on_written` waits for the commit. These tests pin those properties. No network, no spend.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from pipeline.kb import schema, writes
from pipeline.kb.ingest_common import store_atom


def _mk_atom(aid: str, **over) -> dict:
    return {**dict(atom_id=aid, source_type="x", what_kind="opinion", who_id="x:u",
                   when_ts="2024-01-01", when_precision="day",
                   about_entities=[], source_url="u", raw_hash="h",
                   description="d", payload={}, entry_mode="user-saved"), **over}


def _payload(conn, aid: str) -> dict:
    return json.loads(conn.execute("SELECT payload FROM atoms WHERE atom_id = ?",
                                   (aid,)).fetchone()[0])


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ── synchronous, durable ─────────────────────────────────────────────────────────────

def test_store_atom_writes_synchronously(kb_home):
    """Written and durable the moment `store_atom` returns: another connection sees the atom and
    its body, and `on_written` has already fired."""
    conn = schema.connect()
    landed: list[str] = []
    store_atom(conn, _mk_atom("x:sa"), "an autonomous agent framework",
               on_written=lambda: landed.append("x:sa"))

    assert landed == ["x:sa"]
    assert not conn.in_transaction
    other = schema.connect(read_only=True)
    try:
        assert other.execute("SELECT COUNT(*) FROM atoms WHERE atom_id='x:sa'").fetchone()[0] == 1
        assert schema.read_body(other, "x:sa") == "an autonomous agent framework"
    finally:
        other.close()
    conn.close()


# ── one atom, one transaction ──────────────────────────────────────────────────────────────

def test_a_failed_body_write_takes_the_atom_row_back_with_it(kb_home, monkeypatch):
    """The atom row carries the new `raw_hash`, and every adapter's hash check reads that as done.
    If it committed before the text was written, a failed text write would leave an atom that
    is never retried, holding its old text."""
    conn = schema.connect()
    store_atom(conn, _mk_atom("x:1", raw_hash="old"), "the old body")
    landed: list[str] = []

    def locked(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(schema, "put_body", locked)
    with pytest.raises(sqlite3.OperationalError):
        store_atom(conn, _mk_atom("x:1", raw_hash="new"), "the new body",
                   on_written=lambda: landed.append("x:1"))
    monkeypatch.undo()

    assert conn.execute("SELECT raw_hash FROM atoms WHERE atom_id='x:1'").fetchone()[0] == "old"
    assert schema.read_body(conn, "x:1") == "the old body"
    assert landed == []                        # never told the caller it landed
    assert not conn.in_transaction
    conn.close()


# ── inside the caller's transaction: a page and its done-state ───────────────────────────

def _page_table(conn):
    conn.execute("CREATE TABLE pages_done (page INTEGER PRIMARY KEY)")
    conn.commit()


def _visible(conn) -> tuple[int, int, int]:
    """(atoms, bodies, pages) as another connection sees them: only what is committed."""
    other = schema.connect(read_only=True)
    try:
        return (_count(other, "atoms"), _count(other, "bodies"), _count(other, "pages_done"))
    finally:
        other.close()


def test_a_page_and_its_done_state_commit_together(kb_home):
    conn = schema.connect()
    _page_table(conn)
    landed: list[str] = []
    with writes.write_tx(conn, site="page"):
        for aid in ("x:1", "x:2", "x:3"):
            store_atom(conn, _mk_atom(aid), f"body of {aid}",
                       on_written=lambda a=aid: landed.append(a))
        conn.execute("INSERT INTO pages_done VALUES (1)")
        assert _visible(conn) == (0, 0, 0)     # the atoms wait for the page
        assert landed == []                    # so their bookkeeping does too
    assert _visible(conn) == (3, 3, 1)
    assert landed == ["x:1", "x:2", "x:3"]
    conn.close()


def test_a_page_that_fails_writes_nothing_and_counts_nothing(kb_home):
    """An exception after the atoms were written inside the caller's transaction takes them all
    back, body and row, and fires none of their `on_written`."""
    conn = schema.connect()
    _page_table(conn)
    landed: list[str] = []
    with pytest.raises(RuntimeError):
        with writes.write_tx(conn, site="page"):
            store_atom(conn, _mk_atom("x:1"), "body of x:1",
                       on_written=lambda: landed.append("x:1"))
            raise RuntimeError("the page's state could not be written")
    assert _visible(conn) == (0, 0, 0)
    assert landed == []
    assert not conn.in_transaction
    conn.close()


# ── owed paper links ─────────────────────────────────────────────────────────────────────

_PAPER_URL = "https://arxiv.org/abs/2401.00001"


def test_a_long_form_post_records_the_paper_links_it_owes(kb_home):
    """Every KB atom passes here, so this is where a blog post first records what it owes."""
    conn = schema.connect()
    store_atom(conn, _mk_atom("blog:b.com/p", source_type="blog", who_id="blog:b.com",
                              source_url="https://b.com/p", entry_mode="oracle-footprint"),
               f"# Essay\n\nSee {_PAPER_URL} and https://example.com/other.")
    payload = _payload(conn, "blog:b.com/p")
    assert [o["url"] for o in payload["links_owed"]] == [_PAPER_URL]
    assert payload["links_owed"][0]["ref"]["post"] == "blog:b.com/p"
    conn.close()


def test_a_post_owes_nothing_this_way(kb_home):
    """An X post's links are queued by the X ingest, not here."""
    conn = schema.connect()
    store_atom(conn, _mk_atom("x:1"), f"new paper {_PAPER_URL}")
    assert "links_owed" not in _payload(conn, "x:1")
    conn.close()
