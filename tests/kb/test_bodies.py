"""bodies — an item's text, stored once, with its search index kept in step by triggers.

`bodies_fts` is external-content over `bodies`, so un-indexing a body reads its old text from
the row. The three triggers in `schema.py` do that for every write, whoever makes it; these tests
check the index against the table with FTS5's own `integrity-check` after each kind of write.
"""
from __future__ import annotations

import sqlite3

import pytest

from pipeline.kb import schema


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _matches(conn, word: str) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT b.atom_id FROM bodies_fts JOIN bodies b ON b.body_id = bodies_fts.rowid "
        "WHERE bodies_fts MATCH ? ORDER BY b.atom_id", (word,))]


def _index_agrees(conn) -> None:
    """Raises `sqlite3.DatabaseError` when the index describes text the table no longer holds."""
    conn.execute("INSERT INTO bodies_fts(bodies_fts, rank) VALUES ('integrity-check', 1)")


def test_a_body_is_indexed_when_it_is_written(conn):
    schema.put_body(conn, "x:1", "an autonomous agent framework")
    conn.commit()
    assert schema.read_body(conn, "x:1") == "an autonomous agent framework"
    assert _matches(conn, "framework") == ["x:1"]
    _index_agrees(conn)


def test_an_update_in_place_keeps_its_id_and_reindexes(conn):
    schema.put_body(conn, "x:1", "an autonomous agent framework")
    before = conn.execute("SELECT body_id FROM bodies WHERE atom_id='x:1'").fetchone()[0]
    schema.put_body(conn, "x:1", "a rollup proof system")
    conn.commit()
    assert conn.execute("SELECT body_id FROM bodies WHERE atom_id='x:1'").fetchone()[0] == before
    assert _matches(conn, "framework") == []
    assert _matches(conn, "rollup") == ["x:1"]
    _index_agrees(conn)


def test_a_deleted_body_leaves_the_index(conn):
    schema.put_body(conn, "x:1", "an autonomous agent framework")
    schema.put_body(conn, "x:2", "another agent")
    conn.execute("DELETE FROM bodies WHERE atom_id='x:1'")
    conn.commit()
    assert schema.read_body(conn, "x:1") is None
    assert _matches(conn, "agent") == ["x:2"]
    _index_agrees(conn)


def test_the_index_is_stemmed(conn):
    schema.put_body(conn, "x:1", "who is accountable here")
    conn.commit()
    assert _matches(conn, "accountability") == ["x:1"]


def test_a_body_written_by_another_connection_is_indexed(conn, kb_home):
    """The triggers live in the store, so a second writer (the worker beside the server) needs no
    code of its own to keep the index right."""
    other = sqlite3.connect(str(kb_home / "opyt.db"))
    other.execute("INSERT INTO bodies (atom_id, text) VALUES ('x:9', 'a second writer')")
    other.commit()
    other.close()
    assert _matches(conn, "writer") == ["x:9"]
    _index_agrees(conn)
