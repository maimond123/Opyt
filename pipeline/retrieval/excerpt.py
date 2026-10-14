"""
pipeline/retrieval/excerpt.py — which part of a whole body a search hit shows.

A hit's `snippet` is a window of its body, placed where the query's matches are densest. The
engine gathers the match positions from FTS5's `highlight()`, which wraps each match in the two
marker characters below; this module only decides, from that marked text, where the window goes.
FTS5's own `snippet()` stops at 64 tokens, too short to judge a hit by.
"""
from __future__ import annotations

from bisect import bisect_left

WIDTH = 1600          # characters in a search hit's window
LEAD = 100            # characters of context kept before the window's first match

# The markers `highlight()` is asked to wrap each match in, as code points: control characters,
# so they never collide with a body's own text.
OPEN, CLOSE = 1, 2


def _unmark(marked: str) -> tuple[str, list[int]]:
    """The body without markers, and where each match starts in it."""
    out: list[str] = []
    starts: list[int] = []
    at = 0
    for ch in marked:
        if ch == chr(OPEN):
            starts.append(at)
        elif ch != chr(CLOSE):
            out.append(ch)
            at += 1
    return "".join(out), starts


def window(marked: str, width: int = WIDTH) -> tuple[str, int, int]:
    """`(snippet, start, end)`: the `width`-character window of the body that holds the most
    match starts, opened `LEAD` characters before its first match and clamped to the text.
    `start`/`end` are offsets into the unmarked body, the text `open` returns. With no match
    marked, the window is the body's start."""
    text, starts = _unmark(marked)
    best_lo, best_count = 0, 0
    for s in starts:
        lo = max(0, s - LEAD)
        hi = min(len(text), lo + width)
        lo = max(0, hi - width)          # a window near the end still spans `width`
        count = bisect_left(starts, hi) - bisect_left(starts, lo)
        if count > best_count:
            best_lo, best_count = lo, count
    hi = min(len(text), best_lo + width)
    return text[best_lo:hi], best_lo, hi
