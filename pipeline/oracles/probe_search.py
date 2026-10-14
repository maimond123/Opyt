"""pipeline/oracles/probe_search.py — search the candidate probe store and answer with people.

Returns accounts, not posts: this answers "which curated candidate is worth promoting to Oracle,"
not "what was said about X" — so passages ride along as evidence, capped per account, never a feed.
The payload that composes this arm with the trusted-corpus one lives in `candidates.py`, so the
one-store rule below stays true.

This is not a way into the trusted corpus: every row comes from a candidate's own timeline, and
nobody has vouched for the content, so this module never touches `atoms`/`bodies`.

Every read goes through `probe_store`, so `.guards.py`'s one-module rule stays intact. That guard
matches SQL-shaped substrings anywhere, including prose, and cannot see reads of the trusted tables
— so as a human rule, this module reads the probe store and nothing else; never join to `atoms` here.
"""

from __future__ import annotations

import sqlite3

from pipeline.oracles import probe_store
from pipeline.retrieval import excerpt
from pipeline.retrieval.rank import fts_query
from pipeline.retrieval.rank import rrf_term

# Passages shown per account. Raised only if a real screen shows three is too thin.
DEFAULT_EVIDENCE = 3

# Candidates returned by default. Matches oracle(action='screen')'s top_n=30 neighbourhood, so the
# two halves of the promotion decision are the same size list.
DEFAULT_K = 15

# Account score (RRF sum over its matching passages) is left uncapped by design.


# A passage's length in an account's evidence; `candidates` trims the saved arm's to the same.
EVIDENCE_CHARS = 240


def _excerpt(row, limit: int = EVIDENCE_CHARS) -> str:
    """The `limit`-character window of a matching item densest in the query's matches, trimmed
    for a payload."""
    return snippet(excerpt.window(row["marked"] or "", limit)[0], limit)


def snippet(text: str, limit: int = 240) -> str:
    """Whitespace collapsed, and cut on a word boundary with an ellipsis when over `limit`. The
    one trim both candidate arms use, so their evidence reads alike."""
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    cut = t.rfind(" ", 0, limit)
    return t[:cut if cut > 0 else limit].rstrip() + "…"


def search_candidates(conn: sqlite3.Connection, query: str, *,
                      k: int = DEFAULT_K, evidence: int = DEFAULT_EVIDENCE,
                      who_ids: set[str] | None = None) -> list[dict]:
    """Candidates whose probed content best matches `query`, each with its strongest passages.

    Keyword search (BM25), ranked per account rather than per post: an account's score sums the
    reciprocal ranks of its matching passages, so several medium passages can outrank a single
    strong outlier.

    An empty query returns accounts by recency of content with no evidence scored — the natural
    "show me who's been probed" opening call.
    """
    rollup = probe_store.probe_author_rollup(conn)
    if not rollup:
        return []
    if who_ids is not None:
        rollup = {w: v for w, v in rollup.items() if w in who_ids}
        if not rollup:
            return []

    scored: dict[str, float] = {}
    passages: dict[str, list[dict]] = {}

    if (query or "").strip():
        frows = probe_store.probe_fts_rows(conn, fts_query(query),
                                           limit=max(k * evidence * 4, 40),
                                           who_ids=set(rollup))
        for rank, r in enumerate(frows):
            w = r["who_id"]
            scored[w] = scored.get(w, 0.0) + rrf_term(rank)
            passages.setdefault(w, []).append(
                {"when": r["when_ts"], "snippet": _excerpt(r),
                 "score": None, "url": r["source_url"]})

        ranked = sorted(scored, key=lambda w: scored[w], reverse=True)[:k]
    else:
        # No query: most recently active candidates, not most recently pulled.
        ranked = sorted(rollup, key=lambda w: (rollup[w]["last_ts"] or ""), reverse=True)[:k]

    out = []
    for w in ranked:
        seen, ev = set(), []
        for p in passages.get(w, []):
            key = p["snippet"][:80]
            if key in seen:                    # an account's items can repeat a passage
                continue
            seen.add(key)
            ev.append(p)
            if len(ev) >= evidence:
                break
        out.append({"who_id": w, **rollup[w], "match_score": round(scored.get(w, 0.0), 5),
                    "evidence": ev})
    return out
