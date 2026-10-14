"""excerpt.window — where in a whole body a hit's snippet sits, decided from plain values.

The marked text is what FTS5's `highlight()` returns with `excerpt.OPEN`/`CLOSE` around each
match; here it is built by hand, so no store is needed.
"""
from __future__ import annotations

import pytest

from pipeline.kb import schema
from pipeline.retrieval import engine, excerpt

O, C = chr(excerpt.OPEN), chr(excerpt.CLOSE)


def _mark(text: str, word: str) -> str:
    return text.replace(word, f"{O}{word}{C}")


@pytest.mark.parametrize("case,text,word,width,want_start", [
    # One match deep in a long body: the window opens LEAD characters before it.
    ("one match", "x" * 5000 + "needle" + "y" * 5000, "needle", 1600, 5000 - excerpt.LEAD),
    # A dense cluster late beats a lone match early.
    ("cluster late", "needle" + "x" * 6000 + ("needle " * 5) + "y" * 3000, "needle", 1600,
     6006 - excerpt.LEAD),
    # A match at the very start: the window cannot open before the text.
    ("at the start", "needle" + "x" * 4000, "needle", 1600, 0),
    # A match at the very end: the window still spans `width`, ending at the text's end.
    ("at the end", "x" * 4000 + "needle", "needle", 1600, 4006 - 1600),
    # A text shorter than the window is returned whole.
    ("short text", "a short needle here", "needle", 1600, 0),
    # No marker at all: the body's start.
    ("no match", "x" * 3000, "needle", 1600, 0),
])
def test_the_window_sits_where_the_matches_are(case, text, word, width, want_start):
    snippet, start, end = excerpt.window(_mark(text, word), width)
    assert start == want_start, case
    assert snippet == text[start:end]
    assert end - start == min(width, len(text))
    if word in text:
        assert word in snippet, case


def test_a_hits_snippet_span_slices_the_text_open_returns(kb_home):
    from pipeline.kb.ingest_common import snapshot_hash, store_atom
    from pipeline.retrieval import reads

    conn = schema.connect()
    text = "---\nsource: blog\n---\n\n" + "filler " * 600 + "the zebracorn theorem " + "pad " * 600
    store_atom(conn, dict(
        atom_id="blog:a/p", source_type="blog", what_kind="opinion", who_id="blog:a",
        when_ts="2024-05-01", when_precision="day", about_entities=[], source_url="u",
        raw_hash=snapshot_hash(text), description="d", payload={}, entry_mode="user-saved"),
        text)
    hit = engine.search_atoms(conn, "zebracorn").hits[0]
    conn.close()

    start, end = hit.snippet_span
    assert "zebracorn" in hit.snippet
    assert reads.kb_open("blog:a/p")["raw"][start:end] == hit.snippet
