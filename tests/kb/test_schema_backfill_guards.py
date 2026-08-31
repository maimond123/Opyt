"""A converged store must open without ever needing the write lock.

`mcp_server/server.py` forks seven detached rails within milliseconds, and every one calls
`schema.connect()` -> `init_kb_schema`. While the migration backfills wrote unconditionally, those
seven connections raced for the write lock over statements that changed zero rows. The measured
result was 28 `database is locked` events over six days, every one with `elapsed_s: 0.0` — a
read->write upgrade refusal, which no `busy_timeout` can wait out, so shortening the transaction
was never the fix. Read-guarding was.

The property is tested by taking the lock away, not by inspecting statement text. An earlier draft
spied on `conn.execute` and asserted no UPDATE/INSERT string appeared; it flagged three statements
that measurably need no write lock at all — `executemany` with an empty binding list, and
`DROP TABLE IF EXISTS` on an absent table — because a statement's spelling is a proxy for the
thing that actually hurts. Holding `BEGIN IMMEDIATE` on a second connection at `busy_timeout=0`
tests the thing itself: any statement that wants the lock raises immediately.
"""
from __future__ import annotations

import sqlite3

import pytest

from pipeline.kb import schema


@pytest.fixture()
def converged(kb_home, tmp_path):
    """A store that has already been through every migration once, holding a row in each table the
    backfills touch — an empty store would pass even with the guards removed."""
    path = tmp_path / "opyt.db"
    c = schema.connect(path)
    schema.upsert_atom(c, dict(
        atom_id="a1", source_type="x", what_kind="opinion", who_id="x:u",
        when_ts="2024-01-01", when_precision="day", about_entities=[],
        source_url="https://x.com/a/status/1", raw_ref="ref", raw_hash="h",
        description="d", payload={}, entry_mode="user-saved"))
    schema.set_signal(c, "x:user:1", "follow", "x")
    c.execute("INSERT INTO frontier_queries (query_id, text, normalized, generator, created_at, "
              "last_emitted_at) VALUES ('q1', 'T', 't', 'bookmark-reader', '2024-01-01', "
              "'2024-01-01')")
    # The raw INSERT skips `frontier_queries.emit` (which writes the claim alongside the query), so
    # run the migrations once more to converge that one too. This mirrors the real store, where
    # every claim is written by the runtime path and the backfill only ever sees pre-column rows.
    schema.init_kb_schema(c)
    c.commit()
    c.close()
    return path


def _blocked(path):
    """A connection holding the RESERVED write lock, so anyone else who wants it fails instantly."""
    holder = sqlite3.connect(path)
    holder.execute("PRAGMA busy_timeout=0")
    holder.execute("BEGIN IMMEDIATE")
    return holder


def test_a_converged_store_migrates_without_the_write_lock(converged):
    """THE regression. Not scoped to the two named backfills — any future migration that writes
    unconditionally on connect reintroduces exactly this contention, so the assertion covers the
    whole of `init_kb_schema`."""
    holder = _blocked(converged)
    reader = sqlite3.connect(converged)
    reader.execute("PRAGMA busy_timeout=0")
    reader.row_factory = sqlite3.Row
    try:
        schema.init_kb_schema(reader)      # raises OperationalError if anything wants the lock
    finally:
        reader.close()
        holder.rollback()
        holder.close()


def test_a_null_row_still_heals_on_the_next_connect(converged):
    """The guards must not turn a permanent healing path into a one-shot. The MCP server runs from
    the primary checkout, so an older build can still write a NULL row AFTER this ships — the
    reason `_rename_crawled_to_footprint` is read-guarded rather than stamped in `kb_meta`."""
    c = sqlite3.connect(converged)
    c.row_factory = sqlite3.Row
    c.execute("UPDATE curation_signals SET last_confirmed_at = NULL")
    c.execute("UPDATE atoms SET first_seen = NULL")
    c.execute("DELETE FROM frontier_query_generators")
    c.commit()

    schema.init_kb_schema(c)

    assert c.execute("SELECT COUNT(*) FROM curation_signals "
                     "WHERE last_confirmed_at IS NULL").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM atoms WHERE first_seen IS NULL").fetchone()[0] == 0
    assert c.execute("SELECT COUNT(*) FROM frontier_query_generators").fetchone()[0] == 1
    c.close()
