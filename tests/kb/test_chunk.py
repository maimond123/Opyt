"""chunk — windowing, spans, overlap, the always-≥1 guarantee, and the stitch back."""
from __future__ import annotations

from pipeline.kb.chunk import (CHUNK_CHARS, CHUNK_OVERLAP, split_text, stitch,
                               strip_frontmatter)


def test_strip_frontmatter_removes_yaml_and_reports_offset():
    md = '---\nsource: github\nauthor: "@x"\n---\n\n# Title\n\nreal body here'
    body, offset = strip_frontmatter(md)
    assert body.startswith("# Title")
    assert "source: github" not in body
    assert md[offset:] == body            # offset lets the caller keep spans snapshot-absolute


def test_strip_frontmatter_noop_without_frontmatter():
    md = "# Just a title\n\nbody"
    assert strip_frontmatter(md) == (md, 0)


def test_short_text_is_one_chunk_spanning_everything():
    parts = split_text("hello world")
    assert len(parts) == 1
    seq, text, start, end = parts[0]
    assert (seq, start, end) == (0, 0, len("hello world"))
    assert text == "hello world"


def test_empty_text_still_yields_one_chunk():
    # Every atom must have ≥1 embeddable chunk (the "every atom ≥1 vector" invariant).
    parts = split_text("")
    assert len(parts) == 1
    assert parts[0][1] == ""


def test_long_text_windows_with_overlap_and_contiguous_spans():
    text = "".join("abcde"[i % 5] for i in range(CHUNK_CHARS * 2 + 500))
    parts = split_text(text)
    assert len(parts) >= 3
    # seq is 0,1,2,... and every chunk's text matches the span it claims.
    for i, (seq, chunk, start, end) in enumerate(parts):
        assert seq == i
        assert chunk == text[start:end]
    # Adjacent windows overlap by exactly CHUNK_OVERLAP.
    (_s0, _c0, _a0, e0) = parts[0]
    (_s1, _c1, a1, _e1) = parts[1]
    assert a1 == e0 - CHUNK_OVERLAP
    # The last window reaches the end of the text (no trailing content dropped).
    assert parts[-1][3] == len(text)


def test_no_gap_between_windows():
    text = "x" * (CHUNK_CHARS * 3)
    parts = split_text(text)
    # Union of spans covers [0, len) with no hole.
    covered = set()
    for _s, _c, a, b in parts:
        covered.update(range(a, b))
    assert covered == set(range(len(text)))


def test_a_multi_chunk_split_never_emits_a_chunk_under_the_overlap():
    """The tail window starts one overlap back from the previous window's end, so it always spans
    MORE than `CHUNK_OVERLAP`. Consequence, and the reason this is pinned: a chunk shorter than the
    overlap is always its atom's ONLY chunk, so chunk length carries no "is this scaffolding or
    content" signal. `sitting_builder` deleted a 200-char content filter on exactly that fact, and a
    smaller overlap would quietly make it wrong again.

    Swept across window arithmetic — a tail one character past a boundary is the case that would
    produce a stub if the overlap were not carried back.
    """
    for extra in (1, 2, CHUNK_OVERLAP - 1, CHUNK_OVERLAP, CHUNK_OVERLAP + 1, CHUNK_CHARS - 1):
        text = "x" * (CHUNK_CHARS + extra)
        parts = split_text(text)
        assert len(parts) > 1, f"expected a multi-chunk split at +{extra}"
        shortest = min(len(c) for _s, c, _a, _b in parts)
        assert shortest > CHUNK_OVERLAP, f"+{extra} produced a {shortest}-char chunk"


# ── stitch: the inverse of the split ────────────────────────────────────────────
#
# ⚠️ THESE MOVED HERE FROM `test_bookmark_reader.py` ON 2026-08-16, when that module was retired.
# They were written there because the bookmark window was `stitch`'s first caller, and they drove it
# through `window_atoms`. `stitch` itself survives — `sitting_builder` renders every sitting through
# it — so testing it directly is both more honest and no longer optional.
def _rows(*triples):
    """`(text, char_start, char_end)` tuples in the shape `stitch` reads."""
    return [{"text": t, "char_start": s, "char_end": e} for t, s, e in triples]


def test_stitch_removes_the_embedding_overlap():
    """Adjacent windows overlap by `CHUNK_OVERLAP` characters on purpose. A naive join would
    re-emit that span, so a reader would see the author repeat themselves and pay input tokens for
    it — ~1,800 duplicated characters on a 10-chunk atom."""
    # Snapshot text is "ABCDEFGHIJ"; the two chunks share "EF".
    assert stitch(_rows(("ABCDEF", 0, 6), ("EFGHIJ", 4, 10))) == "ABCDEFGHIJ"


def test_stitch_degrades_to_a_plain_join_without_char_ranges():
    """The store predates those columns in places, and a guessed trim would delete real text."""
    assert stitch(_rows(("one ", None, None), ("two", None, None))) == "one two"
