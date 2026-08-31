"""pipeline/kb/probe_search.py — search the candidate probe store and answer with PEOPLE.

Returns accounts, not posts: this answers "which curated candidate is worth promoting to Oracle,"
not "what was said about X" — so passages ride along as evidence, capped per account, never a feed.
`retrieve.search_atoms`. The PAYLOAD that composes this arm with the trusted-corpus one lives in
`candidate_search.py` — deliberately not here, so the one-store rule below stays true.

This is NOT a way into the trusted corpus: every row comes from a candidate's own timeline, and
nobody has vouched for the content. It must never be cited as knowledge — the MCP payload says so,
and this module never touches `atoms`/`chunks`.

Every read goes through `probe_store`, so `.guards.py`'s one-module rule stays intact. That guard
matches SQL-shaped substrings anywhere, including prose, and cannot see reads of the TRUSTED tables
— so as a human rule, this module reads the probe store and nothing else; never join to `atoms` here.
"""

from __future__ import annotations

import sqlite3

import numpy as np

from . import probe_store, schema
from .embed import stored_dtype

# Passages shown per account. Raised only if a real screen shows three is too thin.
DEFAULT_EVIDENCE = 3

# Candidates returned by default. Matches oracle(action='screen')'s top_n=30 neighbourhood, so the
# two halves of the promotion decision are the same size list.
DEFAULT_K = 15

# The vector arm's relevance floor. Deliberately looser than sitting_builder.FLOOR_DEFAULT, which is
# calibrated for a different corpus (trusted atoms) answering a different question.
EVIDENCE_FLOOR = 0.30

# Account score (RRF sum over its matching passages) is left UNCAPPED by design — the resulting
# volume/topic correlation is confounded, not shown to be wrong. See


def _decode(rows: list[sqlite3.Row], dt: np.dtype) -> np.ndarray:
    """Chunk vectors -> (n, dim) float32, L2-normalized. Width comes from `kb_meta`, never assumed."""
    mat = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=dt).reshape(len(rows), -1)
    mat = mat.astype(np.float32)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return mat


def _snippet(row, limit: int = 240) -> str:
    """A passage trimmed for a payload, cut on a word boundary.

    Reads `embed_text` (OPYT's render template stripped) and falls back to `text` when NULL.
    Showing the stripped surface keeps the character budget on the author's words, not OPYT's markup.
    """
    t = " ".join(((row["embed_text"] or row["text"]) or "").split())
    # A continuation chunk can open mid-word. Marked with a leading ellipsis rather than trimmed,
    # since nothing distinguishes a genuine fragment from a real word.
    #
    # Uses `seq == 0`, not `char_start > 0`, to detect an atom's first chunk: `char_start` indexes
    # the raw source doc, where frontmatter means it's never 0 even for a first chunk.
    lead = "…" if (row["seq"] or 0) > 0 else ""
    if len(t) <= limit:
        return lead + t
    return lead + t[:t.rfind(" ", 0, limit) or limit].rstrip() + "…"


def search_candidates(conn: sqlite3.Connection, query: str, embedder, *,
                      k: int = DEFAULT_K, evidence: int = DEFAULT_EVIDENCE,
                      who_ids: set[str] | None = None) -> list[dict]:
    """Candidates whose probed content best matches `query`, each with its strongest passages.

    Hybrid search (vector + BM25), fused per ACCOUNT rather than per atom, so an account with
    several medium passages can outrank one with a single strong outlier.

    An empty query returns accounts by recency of content with no evidence scored — the natural
    "show me who's been probed" opening call.
    """
    from pipeline.rank import _fts_query

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
        # ── vector arm ─────────────────────────────────────────────────────────
        vrows = probe_store.probe_vector_rows(conn, who_ids=set(rollup))
        if vrows:
            qv = np.asarray(embedder.embed([query], role="query")[0], dtype=np.float32)
            qv /= (np.linalg.norm(qv) + 1e-9)
            sims = _decode(vrows, np.dtype(stored_dtype(conn))) @ qv
            order = np.argsort(-sims)
            for rank, i in enumerate(order):
                r = vrows[int(i)]
                s = float(sims[int(i)])
                if s < EVIDENCE_FLOOR:
                    break                      # sorted: everything after is worse
                w = r["who_id"]
                # RRF over the account, so it accumulates across passages instead of being
                # pinned to its single best one.
                scored[w] = scored.get(w, 0.0) + 1.0 / (60 + rank)
                passages.setdefault(w, []).append(
                    {"when": r["when_ts"], "snippet": _snippet(r),
                     "score": round(s, 3), "url": r["source_url"], "arm": "semantic"})

        # ── lexical arm ────────────────────────────────────────────────────────
        frows = probe_store.probe_fts_rows(conn, _fts_query(query),
                                           limit=max(k * evidence * 4, 40),
                                           who_ids=set(rollup))
        for rank, r in enumerate(frows):
            w = r["who_id"]
            scored[w] = scored.get(w, 0.0) + 1.0 / (60 + rank)
            passages.setdefault(w, []).append(
                {"when": r["when_ts"], "snippet": _snippet(r),
                 "score": None, "url": r["source_url"], "arm": "lexical"})

        ranked = sorted(scored, key=lambda w: scored[w], reverse=True)[:k]
    else:
        # No query: most recently ACTIVE candidates, not most recently pulled.
        ranked = sorted(rollup, key=lambda w: (rollup[w]["last_ts"] or ""), reverse=True)[:k]

    out = []
    for w in ranked:
        seen, ev = set(), []
        for p in passages.get(w, []):
            key = p["snippet"][:80]
            if key in seen:                    # the two arms routinely surface the same passage
                continue
            seen.add(key)
            ev.append(p)
            if len(ev) >= evidence:
                break
        out.append({"who_id": w, **rollup[w], "match_score": round(scored.get(w, 0.0), 5),
                    "evidence": ev})
    return out
