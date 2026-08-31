"""
pipeline/rank.py

Corpus-agnostic ranking primitives: the FTS5 query sanitizer, the query-adaptive BM25
weight, and reciprocal rank fusion. Nothing here knows what a hit IS — it knows how to
make a query safe for MATCH, how keyword-shaped a query looks, and how to merge two
ranked lists. Both retrieval rails call all three.

Moved out of the vault rail so the live atom rail (`pipeline/kb/retrieve.py`) would not share a
lifetime with code queued for deletion. That package has since been deleted and these primitives
survived it, which is the whole point of the move: a shared primitive belongs in a neutral module
at the top of `pipeline/`, never borrowed across a rail boundary.
"""

from __future__ import annotations

import re
from typing import Protocol, TypeVar

_WORD = re.compile(r"[A-Za-z0-9_]+")


class Fusable(Protocol):
    """Everything `rrf_fuse` requires of a hit — the entire contract, nothing more. All three
    rank/score members are MUTABLE: `rrf_fuse` writes them back onto the hits it returns."""

    citation_id: str
    bm25_rank: int | None
    sem_rank: int | None
    score: float


# Bound so fusion is identity-preserving in the type: feed it AtomHits, get AtomHits back.
_H = TypeVar("_H", bound=Fusable)


def _fts_query(text: str) -> str:
    """Lenient FTS5 query: OR of quoted word tokens (safe from MATCH syntax)."""
    toks = _WORD.findall(text or "")
    return " OR ".join(f'"{t}"' for t in toks) or '""'


# Query-adaptive fusion: a query naming exact identifiers (snake_case, filenames, ENV vars)
# needs BM25 weight; a pure conceptual question wants semantic only. Weight is graded by how
# many exact tokens appear, so mixed queries get both arms.
_TOKEN_SIGNALS = [
    re.compile(r"\b[a-z]+_[a-z_]+\b"),                              # snake_case
    re.compile(r"\b\w+\.(?:py|db|rs|tsx?|jsx?|json|sock|md|yaml|sh)\b"),  # filename.ext
    re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b"),                         # ALLCAPS_ENV
    re.compile(r"\b[a-z]+[A-Z]\w+\b"),                             # camelCase
    re.compile(r"\b\w+\.\w+\.\w+\b"),                              # dotted.path
    re.compile(r'"[^"]+"|`[^`]+`'),                                # quoted
    re.compile(r"\b\w+\(\)"),                                      # call()
]


def bm25_weight(query: str) -> float:
    """BM25's weight in fusion, inferred from the query (graded gate).

    0 exact tokens -> 0.0 (pure semantic; conceptual question).
    N exact tokens -> min(3, 1+N) (lexical/mixed; BM25 scaled up but semantic
    stays at weight 1.0 so mixed queries keep both arms).
    """
    seen: set[str] = set()
    for rx in _TOKEN_SIGNALS:
        for m in rx.findall(query):
            seen.add(m if isinstance(m, str) else m[0])
    if seen:
        return min(3.0, 1.0 + len(seen))
    # No structural token, but a terse query (<=3 words) is keyword intent (e.g. a bare rare
    # word like "watchdog") where semantic alone whiffs; longer queries stay pure semantic.
    return 1.0 if len(query.split()) <= 3 else 0.0


def rrf_fuse(lists: list[list[_H]], k: int = 8, c: int = 60,
             weights: list[float] | None = None) -> list[_H]:
    """Reciprocal rank fusion — scale-free, no score normalization. `weights`
    (parallel to `lists`, default all 1.0) lets a list contribute more/less; a
    weight of 0 drops that list from ranking entirely."""
    weights = weights or [1.0] * len(lists)
    by_id: dict[str, _H] = {}
    scores: dict[str, float] = {}
    for lst, w in zip(lists, weights):
        for rank, r in enumerate(lst):
            scores[r.citation_id] = scores.get(r.citation_id, 0.0) + w / (c + rank)
            if r.citation_id not in by_id:
                by_id[r.citation_id] = r
            else:  # keep whichever rank info we have
                if r.bm25_rank is not None:
                    by_id[r.citation_id].bm25_rank = r.bm25_rank
                if r.sem_rank is not None:
                    by_id[r.citation_id].sem_rank = r.sem_rank
    ranked = sorted(by_id.values(), key=lambda r: scores[r.citation_id], reverse=True)
    for r in ranked:
        r.score = scores[r.citation_id]
    return ranked[:k]
