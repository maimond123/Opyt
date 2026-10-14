"""The probe store comes into existence only when something is written to it.

A store that has never probed must stay without probe tables however often it is asked about
candidate content: every probe read degrades to the empty answer instead of creating the schema.
Only the write paths may create it.
"""
from __future__ import annotations

import pytest

from pipeline.kb import schema
from pipeline.oracles import probe_store
from pipeline.retrieval.rank import fts_query


def _probe_tables(conn) -> set[str]:
    return {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'probe%'")}


@pytest.mark.parametrize("read", [
    lambda c: probe_store.count_probe_atoms(c),
    lambda c: probe_store.load_probe_hashes(c),
    lambda c: probe_store.probed_who_ids(c),
    lambda c: probe_store.pull_states(c),
    lambda c: probe_store.next_retry_at(c),
    lambda c: probe_store.probe_fts_rows(c, fts_query("rollup"), limit=5),
    lambda c: probe_store.count_people_matching(c, fts_query("rollup")),
    lambda c: probe_store.texts_by_person(c),
    lambda c: probe_store.probe_author_rollup(c),
])
def test_no_read_ever_creates_the_probe_store(kb_home, read):
    """A reader that called `init_probe_schema` to guarantee the tables it was about to query
    would, on one `count_probe_atoms()` against a never-probed store, create nine tables (three of
    ours plus FTS5's six shadow tables). So merely asking "does this store hold candidate
    content" would write a schema into every store that asked — including an archived store
    someone opened only to survey it, which no re-run undoes.

    Mirrors `schema.connect(read_only=True)`, which skips DDL so retrieval degrades to "no such
    table" rather than creating one. Only the write paths may bring the store into existence."""
    conn = schema.connect()
    assert _probe_tables(conn) == set()

    read(conn)                                   # a question, not a statement

    after = _probe_tables(conn)
    assert after == set(), f"a read created {sorted(after)}"
    conn.close()


def test_a_write_does_create_the_probe_store(kb_home):
    """The other half — the guard above must not make the store uncreatable."""
    conn = schema.connect()
    assert not probe_store.probe_tables_exist(conn)
    probe_store.init_probe_schema(conn)
    probe_store.record_sample(conn, "x:user:11", probe_store.STATUS_EMPTY, atoms=0, pages_done=1,
                              tail_id=None)
    conn.commit()
    assert probe_store.probe_tables_exist(conn)
    assert probe_store.pull_states(conn)["x:user:11"]["status"] == probe_store.STATUS_EMPTY
    conn.close()


def test_storing_a_probe_atom_creates_the_store_on_its_own(kb_home):
    """`store_probe_atom` outside a transaction creates the schema itself, so the first ever
    candidate post needs no setup call from its caller."""
    conn = schema.connect()
    assert not probe_store.probe_tables_exist(conn)
    probe_store.store_probe_atom(conn, {"atom_id": "xprobe:1", "source_type": "x",
                                        "who_id": "x:user:11", "raw_hash": "h1"}, "a rollup")
    assert probe_store.probe_tables_exist(conn)
    assert probe_store.count_probe_atoms(conn) == 1
    conn.close()
