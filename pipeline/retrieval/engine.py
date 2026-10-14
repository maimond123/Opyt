"""
pipeline/retrieval/engine.py — the keyword router over atoms and their whole bodies.

The pipeline (`search_atoms`):
  1. tag pre-filter → the candidate atom set (what_kind/source_type eq + source_tags
     overlap via json_each). The arm then searches only those atoms.
  2. BM25 arm — `bodies_fts MATCH`, one row per atom (its whole body), strongest first.
  3. excerpts — for the returned hits only, the window of each body densest in matches
     (`excerpt.window`, from FTS5's own `highlight()` positions).
  → Ranking is pure relevance; there is no trust re-rank.

This module returns routing only (which atom, which span of its body, the pointer), never a
content claim; asserting what a source says is the host's job after `open()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pipeline.retrieval import excerpt
from pipeline.retrieval.filters import candidate_atom_ids, count_undated, in_clause
from pipeline.retrieval.rank import fts_query


@dataclass
class AtomHit:
    """One retrieved atom: the routing card the host needs to decide whether to `open()` it."""
    atom_id: str
    source_type: str
    what_kind: str | None
    who_id: str | None
    # Display name, LEFT-JOINed from `entities`; `who_id` stays the identity. None when the
    # entity row is missing.
    who_name: str | None
    when_ts: str | None
    when_precision: str | None
    source_url: str | None = None
    description: str | None = None
    snippet: str = ""                            # the excerpt of the body (routing evidence)
    snippet_span: tuple[int, int] | None = None  # (start, end) of `snippet` in the body
    score: float = 0.0
    # `body_state`/`body_basis` are lifted out of `payload` because they qualify the snippet
    # itself (e.g. a paywall stub), not the source.
    body_state: str | None = None
    body_basis: str | None = None
    # Everything else the adapters captured, verbatim and source-shaped (no allowlist).
    payload: dict = field(default_factory=dict)
    # How the atom arrived. Also the value `_filter_clauses` filters on.
    entry_mode: str | None = None
    # The body's docid in `bodies_fts`, which the excerpt reads by. Not on the card.
    body_id: int | None = None


@dataclass
class SearchRun:
    """What `search_atoms` did, not just what it returned: how much was ranked and cut off.

    `candidates` is None when no filter ran, and 0 when a filter ran and matched nothing."""
    hits: list[AtomHit]
    candidates: int | None              # cleared the pre-filter; None = unfiltered whole store
    ranked: int                         # atoms the arm scored (pre-truncation)
    cutoff: dict | None                 # {last_returned, next_best}; None when nothing was cut
    fts_query: str | None               # the rewritten keyword query; None when nothing ran
    pool_saturated: bool = False        # the arm hit the over-fetch ceiling
    # Atoms that cleared every other filter but were dropped for carrying no date; 0 when no date
    # bound was asked for.
    undated_excluded: int = 0


def json_obj(value) -> dict:
    """A stored JSON column that must be a dict → a dict. NULL/empty/malformed → {}, never raises,
    so an atom with a bad `payload` is still returned as a hit."""
    if isinstance(value, dict):
        return dict(value)   # a copy: the caller pops keys off the result
    try:
        obj = json.loads(value) if value else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _atom_hit(row) -> AtomHit:
    """Map a joined atoms row → AtomHit."""
    # Optional columns are read defensively: a bare `atoms` row degrades to a thinner hit
    # instead of raising.
    try:
        payload = json_obj(row["payload"])
    except (IndexError, KeyError):
        payload = {}
    body_state = payload.pop("body_state", None)
    body_basis = payload.pop("body_basis", None)
    try:
        who_name = row["_who_name"]
    except (IndexError, KeyError):
        who_name = None
    try:
        entry_mode = row["entry_mode"]      # in the row already; the arm `SELECT a.*`
    except (IndexError, KeyError):
        entry_mode = None
    return AtomHit(
        atom_id=row["atom_id"],
        source_type=row["source_type"],
        what_kind=row["what_kind"],
        who_id=row["who_id"],
        who_name=who_name,
        when_ts=row["when_ts"],
        when_precision=row["when_precision"],
        source_url=row["source_url"],
        description=row["description"],
        body_state=body_state,
        body_basis=body_basis,
        payload=payload,
        entry_mode=entry_mode,
        body_id=row["_bid"],
    )


def atom_bm25_search(conn, query: str, candidates: set[str] | None, k: int) -> list[AtomHit]:
    """The arm: the `k` atoms whose whole bodies best match `query`, strongest first. Lower
    `bm25()` is better. The hits carry no excerpt yet (`fill_excerpts`)."""
    # `+a`: the candidate list only filters. Without the `+` SQLite carries the list across the
    # join onto `bodies(atom_id)` and probes it once per candidate for every matching body, about
    # a second on a 4,300-item library where the filtered scan takes 8 ms.
    frag, cand_params = in_clause(candidates, "+a")
    rows = conn.execute(
        "SELECT a.*, e.name AS _who_name, b.body_id AS _bid, bm25(bodies_fts) AS _rank "
        "FROM bodies_fts "
        "JOIN bodies b ON b.body_id = bodies_fts.rowid "
        "JOIN atoms a  ON a.atom_id = b.atom_id "
        # LEFT so an atom with no author entity row is still returned, unnamed.
        "LEFT JOIN entities e ON e.entity_id = a.who_id "
        "WHERE bodies_fts MATCH ?" + frag + " ORDER BY _rank LIMIT ?",
        [fts_query(query), *cand_params, int(k)]).fetchall()
    out = []
    for r in rows:
        hit = _atom_hit(r)
        hit.score = 1.0 / (1.0 + len(out))   # reciprocal rank: `score` keeps one scale
        out.append(hit)
    return out


def fill_excerpts(conn, query: str, hits: list[AtomHit], *,
                  width: int = excerpt.WIDTH) -> None:
    """Set each hit's `snippet` and `snippet_span` to the `width`-character window of its body
    densest in matches. Reads only these hits' bodies, through the index's own `highlight()`,
    so a match here is a match as the search counted it (stemming included)."""
    ids = [h.body_id for h in hits if h.body_id is not None]
    if not ids:
        return
    ph = ",".join("?" for _ in ids)
    marked = dict(conn.execute(
        f"SELECT rowid, highlight(bodies_fts, 0, char({excerpt.OPEN}), char({excerpt.CLOSE})) "
        f"FROM bodies_fts WHERE bodies_fts MATCH ? AND rowid IN ({ph})",
        [fts_query(query), *ids]).fetchall())
    for h in hits:
        if h.body_id in marked:
            h.snippet, start, end = excerpt.window(marked[h.body_id], width)
            h.snippet_span = (start, end)


def search_atoms(conn, query: str, *, tags: list[str] | None = None,
                 what_kind: str | None = None, source_type: str | None = None,
                 who_id: str | list[str] | None = None, date_from: str | None = None,
                 date_to: str | None = None, entry_mode: str | list[str] | None = None,
                 k: int = 8) -> SearchRun:
    """The search entry: filter → the keyword arm → top-k with excerpts, on pure relevance.

    `date_from`/`date_to` are already-normalized `YYYY-MM-DD` bounds (see `date_bound`).
    `entry_mode` is an allow-list on how the atom arrived; `list(schema.HUMAN_ATTESTED)` is the
    human-attested pile.
    """
    cand = candidate_atom_ids(conn, tags, what_kind, source_type, who_id, date_from, date_to,
                              entry_mode=entry_mode)
    # Before the early return: an empty date-filtered result is exactly when the undated count
    # explains it.
    undated = (count_undated(conn, tags, what_kind, source_type, who_id, entry_mode)
               if (date_from or date_to) else 0)
    if cand is not None and not cand:
        # Nothing runs, so the keyword rewrite is absent rather than empty.
        return SearchRun(hits=[], candidates=0, ranked=0, cutoff=None, fts_query=None,
                         undated_excluded=undated)

    pool = max(k * 5, 40)   # over-fetch, so `ranked` and the cutoff say what lay past k
    ranked = atom_bm25_search(conn, query, cand, pool)
    hits = ranked[:k]
    fill_excerpts(conn, query, hits)

    # The scores either side of the `[:k]` cut: whether it landed mid-cluster or on a cliff.
    cutoff = None
    if k > 0 and len(ranked) > k:
        cutoff = {"last_returned": round(ranked[k - 1].score, 4),
                  "next_best": round(ranked[k].score, 4)}
    return SearchRun(
        hits=hits, candidates=None if cand is None else len(cand), ranked=len(ranked),
        cutoff=cutoff, fts_query=fts_query(query), pool_saturated=len(ranked) >= pool,
        undated_excluded=undated,
    )
