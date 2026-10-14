"""`filters.date_span` — the oldest and newest publication date over the dated atoms. Both
spellings of undated (NULL and '') are in the corpus, and '' sorts before every date, so an
undated atom must never become the oldest."""
from __future__ import annotations

from pipeline.kb import schema
from pipeline.retrieval import filters
from tests.kb.test_export import _corpus


def test_the_span_covers_dated_atoms_only(kb_home):
    conn = schema.connect()
    _corpus(conn)
    try:
        assert conn.execute("SELECT COUNT(*) FROM atoms WHERE when_ts = ''").fetchone()[0] == 1
        assert filters.date_span(conn) == ("2024-01-01", "2025-06-01")
    finally:
        conn.close()


def test_a_store_with_no_dated_atom_has_no_span(kb_home):
    conn = schema.connect()
    try:
        assert filters.date_span(conn) == (None, None)
    finally:
        conn.close()
