"""
pipeline/retrieval/rank.py

Corpus-agnostic ranking primitives: the FTS5 query sanitizer and the reciprocal-rank term.
Nothing here knows what a hit is. Oracles' probe search also uses `fts_query`.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9_]+")


# Words an OR-query matches nearly every text on, so they rank by noise. Dropped from `fts_query`
# unless the query is nothing else. Measured with stemming: narrow lookups by keyword
# went from 0.77 to 0.84 nDCG against 0.86 embedded.
_STOPWORDS = frozenset(
    "a about an and are as at be been but by can could did do does for from had has have how i "
    "if in into is it its me my of on or our so than that the their them then there these they "
    "this to was we were what when where which who why will with would you your".split())


def content_words(text: str) -> list[str]:
    """The query's words that carry meaning: every word but the stopwords. Empty for a query of
    only common words ("the"), which names no subject."""
    return [t for t in _WORD.findall(text or "") if t.lower() not in _STOPWORDS]


def fts_query(text: str) -> str:
    """Lenient FTS5 query: OR of quoted word tokens (safe from MATCH syntax), stopwords left out
    unless the query has nothing else."""
    toks = content_words(text) or _WORD.findall(text or "")
    return " OR ".join(f'"{t}"' for t in toks) or '""'


def fts_phrase(text: str) -> str:
    """Exact-phrase FTS5 query: the words in order, side by side. For counting what a subject
    holds: `fts_query`'s OR counts every text sharing one common word with it, and even requiring
    every word overcounted ("agent memory": 180 people, about a third on topic, against 25 for the
    phrase)."""
    toks = _WORD.findall(text or "")
    return f'"{" ".join(toks)}"'


# Reciprocal rank's constant. Every RRF score in the tree uses it, so ranks summed anywhere mean
# the same thing — oracles' two candidate searches compare their account ranks on it.
RRF_C = 60


def rrf_term(rank: int, weight: float = 1.0) -> float:
    """One list's contribution to an item's RRF score at 0-based `rank`."""
    return weight / (RRF_C + rank)
