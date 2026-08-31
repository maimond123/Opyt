"""
pipeline/kb/retrieve.py — the enforced-hybrid router over atoms/chunks.

A BM25 arm and a max-pooled cosine arm over the atom-KB tables, fused by
`pipeline/rank.py` (`rrf_fuse`, `bm25_weight`, `_fts_query`). `AtomHit` implements
`pipeline.rank.Fusable`.

The pipeline (`search_atoms`):
  1. tag pre-filter → the candidate atom set (what_kind/source_type eq + source_tags
     overlap via json_each). Both arms then search ONLY those atoms.
  2. BM25 arm — chunks_fts MATCH, grouped to the atom by its best chunk.
  3. vector arm — cosine of the query vector over candidate chunk vectors, MAX-POOLED
     per atom; the argmax chunk names the matched span. A long atom draws that maximum
     from more chunks and so scores higher for its length — a known, measured, UNFIXED
     residual; see the note on `atom_semantic_search`.
  4. rrf_fuse([bm, sem], weights=[bm25_weight(query), 1.0]) — the query decides the mix.
  → Ranking is pure relevance. There is deliberately NO trust re-rank here; see

This module NEVER returns a content claim — only routing (which atom, which chunk-span,
the pointer). Asserting what a source says is the host's job after `open()`.
"""

from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass, field
from datetime import date as _date

import numpy as np

from pipeline.rank import _fts_query, bm25_weight, rrf_fuse

from . import derive, schema


@dataclass
class AtomHit:
    """One retrieved atom + the chunk that matched. `citation_id` (= atom_id),
    `bm25_rank`, `sem_rank`, `score` satisfy `pipeline.rank.Fusable`; the rest is the
    routing card the host needs to decide whether to `open()` it."""
    citation_id: str          # = atom_id — the rrf fusion key
    atom_id: str
    source_type: str
    what_kind: str | None
    who_id: str | None
    # The author's DISPLAY name, LEFT-JOINed from `entities`. Display only — it does NOT make
    # authorship queryable. `who_id` stays the identity; a name is not unique and is not a key.
    # None when the entity row is missing (the join is LEFT so a hit is never dropped for it).
    who_name: str | None
    when_ts: str | None
    when_precision: str | None
    source_url: str | None = None
    raw_ref: str | None = None
    description: str | None = None
    snippet: str = ""                          # the matched chunk's text (routing evidence)
    chunk_seq: int | None = None
    chunk_span: tuple[int, int] | None = None  # (char_start, char_end) in the snapshot
    bm25_rank: int | None = None
    sem_rank: int | None = None
    score: float = 0.0
    # `body_state`/`body_basis` are lifted out of `payload` because they qualify the snippet
    # itself (e.g. a paywall stub), not the source.
    body_state: str | None = None
    body_basis: str | None = None
    # Everything else the adapters captured, verbatim and source-shaped (no allowlist).
    payload: dict = field(default_factory=dict)
    # How the atom arrived: 'user-saved' vs 'crawled'. Also the value `_filter_clauses` filters on.
    entry_mode: str | None = None


@dataclass
class SearchRun:
    """What `search_atoms` DID, not just what it returned: which arms ran, what units `score`
    is in, and how much was cut off — all already computed during a normal search.

    `candidates` is None when no filter ran (the whole store was searched), which is NOT the
    same as 0 (a filter ran and matched nothing)."""
    hits: list[AtomHit]
    effective_mode: str                 # hybrid | semantic | bm25 | none — the arms that RAN
    why: str                            # one clause naming the cause of that choice
    score_scale: str | None             # cosine | rrf | reciprocal_rank — the units of `score`
    candidates: int | None              # cleared the pre-filter; None = unfiltered whole store
    ranked: int                         # atoms an arm actually scored (pre-truncation)
    cutoff: dict | None                 # {last_returned, next_best}; None when nothing was cut
    bm25_weight: float | None           # the inferred weight; None outside hybrid
    fts_query: str | None               # the REWRITTEN keyword query; None when BM25 didn't run
    arm_sizes: dict = field(default_factory=dict)
    pool_saturated: bool = False        # an arm hit the over-fetch ceiling before fusion
    # Atoms cleared every other filter but were dropped for carrying no date. 0 when no date
    # bound was asked for. Not the same as `filter_cost["date_from"]` (which also counts
    # dated-but-out-of-window atoms) — this is only the subset the filter couldn't evaluate.
    undated_excluded: int = 0


def _json_obj(value) -> dict:
    """A stored JSON column that must be a dict → a dict. NULL/empty/malformed → {}, never raises.

    Fail-safe: a row whose `payload` never parsed must still be RETURNED as a hit (minus its
    extras), because a decode error is our problem and dropping the atom makes it the user's."""
    if isinstance(value, dict):
        return dict(value)   # a COPY — the caller pops off the result, and that must not reach back
    try:
        obj = json.loads(value) if value else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _atom_hit(row, *, chunk_text, chunk_seq, chunk_span) -> AtomHit:
    """Map a joined atoms+chunk row → AtomHit. payload is JSON in the column."""
    # Read defensively for the same reason as `_who_name` below — a caller handing over a bare
    # `atoms` row, or an older row predating the column, degrades instead of raising.
    try:
        payload = _json_obj(row["payload"])
    except (IndexError, KeyError):
        payload = {}
    body_state = payload.pop("body_state", None)
    body_basis = payload.pop("body_basis", None)
    # `_who_name` is supplied by both arms' LEFT JOIN. Read defensively so a caller that
    # hands over a bare `atoms` row degrades to an unnamed hit instead of raising — the
    # name is a label, and losing a label must never cost the result.
    try:
        who_name = row["_who_name"]
    except (IndexError, KeyError):
        who_name = None
    try:
        entry_mode = row["entry_mode"]      # in the row already; both arms `SELECT a.*`
    except (IndexError, KeyError):
        entry_mode = None
    return AtomHit(
        citation_id=row["atom_id"],
        atom_id=row["atom_id"],
        source_type=row["source_type"],
        what_kind=row["what_kind"],
        who_id=row["who_id"],
        who_name=who_name,
        when_ts=row["when_ts"],
        when_precision=row["when_precision"],
        source_url=row["source_url"],
        raw_ref=row["raw_ref"],
        description=row["description"],
        snippet=chunk_text or "",
        chunk_seq=chunk_seq,
        chunk_span=chunk_span,
        body_state=body_state,
        body_basis=body_basis,
        payload=payload,
        entry_mode=entry_mode,
    )


def _in_clause(candidates: set[str] | None, alias: str = "a") -> tuple[str, list]:
    """A candidate-restriction fragment. None → no restriction (empty fragment)."""
    if candidates is None:
        return "", []
    ids = list(candidates)
    ph = ",".join("?" for _ in ids)
    return f" AND {alias}.atom_id IN ({ph})", ids


# Chunk vectors held in RAM at once during the ranking scan. 512 x 4096 float32 = 8.4 MB — small
# enough to stay in L3 while the multiply runs, and the reason peak RSS stops being a function of
# corpus size. Derivation: docs/lessons/query-cost-is-residency-not-arithmetic.md.
VEC_BATCH = 512


def _hit_columns(conn, chunk_ids: list[int]) -> dict[int, object]:
    """`{chunk_id: row}` carrying every column a hit CARD needs — for the winning chunks only.

    The second pass of the vector arm's split. The projection is the one the arm used to run
    across the whole store, so `_atom_hit` reads exactly the columns it always did; what changed
    is that it is now paid per returned hit rather than per chunk scanned."""
    if not chunk_ids:
        return {}
    ph = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        "SELECT a.*, e.name AS _who_name, "
        "c.chunk_id AS _cid, c.seq AS _seq, c.char_start AS _cs, c.char_end AS _ce, "
        "c.text AS _chunk "
        "FROM chunks c JOIN atoms a ON a.atom_id = c.atom_id "
        # LEFT, never INNER: an atom whose author entity row is missing must still be
        # RETURNED (unnamed), not silently dropped from retrieval.
        "LEFT JOIN entities e ON e.entity_id = a.who_id "
        f"WHERE c.chunk_id IN ({ph})",
        chunk_ids,
    ).fetchall()
    return {r["_cid"]: r for r in rows}


# ── handle → who_id (the read-side lookup) ────────────────────────────────────
# Not shared with `oracles._resolve_handle`: that is a write path (pays twitterapi.io, mints
# entity rows). This is read-only local SQL — a search must never cost money or invent an entity.

# Entity-id namespaces (`derive.py` mints all of them). A caller who already HAS an id gets to
# pass it straight through, which is what makes this function total over its input.
_ID_PREFIXES = ("x:user:", "github:", "substack:", "blog:", "org:", "scholar:", "paper-authors:")


def resolve_who(conn, who: str) -> list[dict]:
    """A handle / URL / id → the cluster(s) of entities it names, each with the `who_ids` to
    hand back as a `who_id=` filter. Local SQL only. Unknown input → [] (never everything).

    A handle is stored in three shapes depending on adapter: GitHub/Substack key the id suffix
    directly (`github:karpathy`, matched case-insensitively); X ids are numeric so the handle
    lives in `entities.profile.handle`; Substack/blog homes key on the host and resolve via
    `derive.*_entity_id`, tried only when the input looks like a URL.

    Each match expands to its whole resolved cluster: an atom carries the per-platform `who_id`
    of its author, so one person's work spans up to five ids. `who_ids` is the full cluster —
    pass it all to `who_id=`, narrow with `source_type=` if one platform is wanted. The cluster
    is only as complete as Stage-3 resolution: an unlinked platform stays its own cluster, which
    under-returns but never returns the wrong person's atoms. See

    Returns [{canonical_id, name, handle, who_ids, matched_on}], one entry per cluster."""
    raw = (who or "").strip()
    if not raw:
        return []

    seeds: dict[str, str] = {}          # lower(entity_id) → why it was tried
    if raw.startswith(_ID_PREFIXES):
        seeds[raw.lower()] = "entity_id"

    handle = raw.lstrip("@").lower()
    if handle and "/" not in handle and not raw.startswith(_ID_PREFIXES):
        seeds.setdefault(f"github:{handle}", "handle")
        seeds.setdefault(f"substack:{handle}", "handle")

    if raw.startswith("http") or "." in raw:
        for eid in (derive.blog_entity_id(raw), derive.substack_entity_id(None, raw)):
            if not eid.endswith(":unknown"):
                seeds.setdefault(eid.lower(), "home_url")

    # ONE query for all three shapes. The `profile` arm is what covers X; `json_extract` over a
    # null/handle-less profile yields NULL, and `lower(NULL) = ?` is never true, so a store with
    # no X entities simply fails that arm instead of erroring.
    ph = ",".join("?" for _ in seeds) or "NULL"
    rows = conn.execute(
        f"SELECT entity_id, canonical_id FROM entities "
        f"WHERE lower(entity_id) IN ({ph}) "
        f"   OR lower(json_extract(profile, '$.handle')) = ?",
        [*seeds.keys(), handle],
    ).fetchall()
    if not rows:
        return []

    from .screen import _best_name        # one name-picking rule for the whole KB, not a second

    out, seen_heads = [], set()
    for r in rows:
        head = schema.current_canonical(conn, r["canonical_id"] or r["entity_id"])
        if head in seen_heads:            # two seeds landing in one cluster = ONE person
            continue
        seen_heads.add(head)
        members = [{"entity_id": m["entity_id"], "name": m["name"], "profile": m["profile"]}
                   for m in schema.entities_for_canonical(conn, head)]
        name, x_handle = _best_name(members)
        out.append({
            "canonical_id": head,
            "name": name,
            "handle": x_handle,
            "who_ids": sorted(m["entity_id"] for m in members),
            "matched_on": seeds.get(r["entity_id"].lower(), "x_profile_handle"),
        })
    return sorted(out, key=lambda c: c["canonical_id"])


# ── date bounds ───────────────────────────────────────────────────────────────
# Atoms carry a real `when_precision` column instead of inferring year-precision by sniffing the
# date string, which is what the deleted vault-rail clause this replaced did — do not collapse
# this back into a LIKE. `when_ts` is always `YYYY-MM-DD` shaped or empty; see
_WHEN = "a.when_ts"
_PREC = "a.when_precision"

# How far an atom's TRUE date range extends past `when_ts`, keyed by `when_precision`. A `year`
# atom stores Jan 1 as a floor, so its range runs to Dec 31 of that year; everything else is the
# single day it names. An unrecognized precision falls through to exact-day; `tests/kb/test_date_
# filter.py` asserts every deriver-written literal has a key here so a new one can't inherit a
# wrong default.
_PRECISION_SPAN: dict[str, str] = {
    "day": _WHEN,       # a real publication date
    "push": _WHEN,      # GitHub's last-push day — activity, not publication; filtered exactly
    "year": f"substr({_WHEN},1,4) || '-12-31'",   # Jan 1 is a floor — the atom spans its year
    "unknown": _WHEN,   # always paired with an empty `when_ts` — the undated clause drops it first
    "": _WHEN,
}

_DATE_INPUT = re.compile(r"^(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?$")


def _range_end_sql() -> str:
    """The atom's range END, as a SQL expression over the `atoms a` row.

    GENERATED from `_PRECISION_SPAN` rather than hardcoded, so a precision's meaning is written
    in exactly one place. When no precision widens the range the CASE disappears entirely, which
    is what keeps the clause sargable in that degenerate case."""
    wide = [(p, expr) for p, expr in _PRECISION_SPAN.items() if expr != _WHEN]
    if not wide:
        return _WHEN
    arms = " ".join(f"WHEN '{p}' THEN {expr}" for p, expr in wide)
    return f"(CASE {_PREC} {arms} ELSE {_WHEN} END)"


def _date_bound(value: str | None, *, end: bool) -> str | None:
    """A caller's date → `YYYY-MM-DD`. None/empty → None, meaning no bound was asked for.

    Accepts `YYYY`, `YYYY-MM`, `YYYY-MM-DD`, and WIDENS a partial to its natural edge — which
    differs by which end it is. `date_from="2026"` means from the first instant of 2026
    (`2026-01-01`); `date_to="2026"` means through the last (`2026-12-31`). Widening both to the
    same floor would make `date_to="2026"` quietly mean "before January 2nd".

    RAISES `ValueError` on anything else, including the shape that motivated this whole feature
    (`"5/11/2026"`). A malformed date filter that is silently ignored is exactly the silent-wrong
    being removed here: the date gets dropped, results come back looking filtered, and nothing
    anywhere says otherwise. An MCP host can correct an error; it cannot see a filter that never
    ran."""
    raw = (value or "").strip()
    if not raw:
        return None
    m = _DATE_INPUT.match(raw)
    if m:
        try:
            year = int(m.group(1))
            month = int(m.group(2)) if m.group(2) else (12 if end else 1)
            if m.group(3):
                day = int(m.group(3))
            else:
                day = calendar.monthrange(year, month)[1] if end else 1
            return _date(year, month, day).isoformat()
        except (ValueError, calendar.IllegalMonthError):
            pass          # a well-SHAPED but impossible date ("2026-13", "2026-02-30")
    raise ValueError(
        f"date bound {value!r} is not a date. Use YYYY, YYYY-MM or YYYY-MM-DD; a partial widens "
        f"to its natural edge (date_from='2026' is 2026-01-01, date_to='2026' is 2026-12-31)."
    )


def candidate_atom_ids(conn, tags: list[str] | None, what_kind: str | None,
                       source_type: str | None, who_id: str | list[str] | None = None,
                       date_from: str | None = None, date_to: str | None = None,
                       *, entry_mode: str | list[str] | None = None,
                       ) -> set[str] | None:
    """The pre-filter (step 1). Returns the atom_id set matching ALL given
    constraints, or None when no filter is requested (search the whole store).

    `who_id` is EXACT id matching — one id or a LIST of them, never a name; an atom carries its
    author's per-platform id, so `resolve_who` hands back the whole cluster. Name resolution is
    deliberately not supported — display names collide and would make the filter guess who it
    returns. HANDLE→id resolution lives in `resolve_who` (above).

    `tags` overlap `payload.source_tags` via `json_each` (an atom's own declared labels);
    `what_kind`/`source_type` are exact equals. `source_tags` is the only tag space —
    `atoms.about_topics` was dropped 2026-08-17, never populated.

    `date_from`/`date_to` are already-normalized `YYYY-MM-DD` bounds (run a caller's input
    through `_date_bound` first) and are inclusive at both edges. See `_PRECISION_SPAN` for what
    a coarse date means at a boundary, and `_filter_clauses` for why an undated atom is excluded.

    `entry_mode` is HOW the atom arrived, and it is an ALLOW-LIST: name the modes you want. The
    human-attested pile is `entry_mode=list(schema.HUMAN_ATTESTED)`. Keyword-only so the existing
    positional callers are untouched.
    """
    built = _filter_clauses(tags, what_kind, source_type, who_id, date_from, date_to, entry_mode)
    if built is None:
        return set()                    # an explicitly empty author/tag set → match nothing
    clauses, params = built
    if not clauses:
        return None                     # nothing was asked for → search the whole store
    sql = "SELECT a.atom_id FROM atoms a WHERE " + " AND ".join(clauses)
    return {r["atom_id"] for r in conn.execute(sql, params)}


def _filter_clauses(tags: list[str] | None, what_kind: str | None, source_type: str | None,
                    who_id: str | list[str] | None, date_from: str | None = None,
                    date_to: str | None = None,
                    entry_mode: str | list[str] | None = None,
                    ) -> tuple[list[str], list] | None:
    """The pre-filter's WHERE clauses, in ONE spelling shared by the filter itself and the
    counterfactual that prices it (`filter_costs`). Two spellings of the tag subquery is
    how a reported cost silently stops describing the query that was actually run.

    Returns `(clauses, params)`; an EMPTY clause list means no filter was asked for. Returns
    None for match nothing — an explicitly empty author set or tag set, which no WHERE clause
    can express because there is no value left to compare against."""
    clauses: list[str] = []
    params: list = []
    if what_kind:
        clauses.append("a.what_kind = ?")
        params.append(what_kind)
    if source_type:
        clauses.append("a.source_type = ?")
        params.append(source_type)
    if entry_mode is not None:
        # Allow-list by design (see `entry_mode` in schema.py): a mode added later is excluded by
        # default, never silently swept in by a deny-list. Empty list matches nothing, like `who_id`
        # and `tags` below.
        modes = [entry_mode] if isinstance(entry_mode, str) else list(entry_mode)
        modes = [m for m in modes if m]
        if not modes and not isinstance(entry_mode, str):
            return None
        if modes:
            mph = ",".join("?" for _ in modes)
            clauses.append(f"a.entry_mode IN ({mph})")
            params.extend(modes)
    if who_id is not None:
        # A LIST is an explicit author set, so an empty one means "no such author" and must match
        # nothing (a failed `resolve_who` must not widen back to the whole store). A bare "" means
        # "no filter" (absent).
        ids = [who_id] if isinstance(who_id, str) else list(who_id)
        ids = [i for i in ids if i]
        if not ids and not isinstance(who_id, str):
            return None
        if ids:
            wph = ",".join("?" for _ in ids)
            clauses.append(f"a.who_id IN ({wph})")   # served by idx_atoms_who_id
            params.extend(ids)
    if tags is not None:
        # Empty list = "match nothing", same as `who_id` above — None is the only "no filter" input.
        if not tags:
            return None
        ph = ",".join("?" for _ in tags)
        clauses.append(
            f"a.atom_id IN ("
            f"SELECT a3.atom_id FROM atoms a3, "
            f"json_each(json_extract(a3.payload, '$.source_tags')) js "
            f"WHERE js.value IN ({ph}))"
        )
        params.extend(tags)
    if date_from or date_to:
        # An undated atom satisfies no date claim in either direction (explicit because SQL's
        # own `''` comparisons disagree with each other across `>=` and `<=`). See
        clauses.append(f"({_WHEN} IS NOT NULL AND {_WHEN} != '')")
    if date_from:
        # Range-aware: the atom's range END must reach the window, so a `year` atom matches a
        # later `date_from` in the same year instead of vanishing at its floor. Deliberately
        # asymmetric.
        clauses.append(f"{_range_end_sql()} >= ?")
        params.append(date_from)
    if date_to:
        # The atom's range START must not pass the window. Sargable, unlike the date_from side.
        clauses.append(f"{_WHEN} <= ?")
        params.append(date_to)
    return clauses, params


def _count_matching(conn, tags, what_kind, source_type, who_id,
                    date_from=None, date_to=None, entry_mode=None) -> int:
    """How many atoms clear this exact filter combination. Match-nothing counts as 0."""
    built = _filter_clauses(tags, what_kind, source_type, who_id, date_from, date_to, entry_mode)
    if built is None:
        return 0
    clauses, params = built
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return conn.execute(f"SELECT COUNT(*) FROM atoms a{where}", params).fetchone()[0]


def _count_undated(conn, tags, what_kind, source_type, who_id, entry_mode=None) -> int:
    """How many atoms clear every NON-DATE filter but carry no date at all.

    The number behind the `undated_excluded` notice, and the reason it is computed separately
    from `filter_costs`: this is the group the date filter could not EVALUATE, which is a
    different fact from the group it evaluated and rejected."""
    built = _filter_clauses(tags, what_kind, source_type, who_id, entry_mode=entry_mode)
    if built is None:
        return 0
    clauses, params = built
    clauses = [*clauses, f"({_WHEN} IS NULL OR {_WHEN} = '')"]
    return conn.execute("SELECT COUNT(*) FROM atoms a WHERE " + " AND ".join(clauses),
                        params).fetchone()[0]


def filter_costs(conn, tags: list[str] | None = None, what_kind: str | None = None,
                 source_type: str | None = None, who_id: str | list[str] | None = None,
                 date_from: str | None = None, date_to: str | None = None,
                 entry_mode: str | list[str] | None = None) -> dict[str, int]:
    """For each ACTIVE filter, how many MORE atoms the query would have reached with that one
    filter dropped and all the others kept. `{}` when no filter costs anything (self-pruning,
    always computed rather than gated on thin results). One `COUNT(*)` per active filter plus
    one baseline. Every filter `_filter_clauses` knows about must appear in `args` below, or the
    baseline — and every other filter's reported cost — comes out wrong. Full rationale:
"""
    args = {"tags": tags, "what_kind": what_kind, "source_type": source_type, "who_id": who_id,
            "date_from": date_from, "date_to": date_to, "entry_mode": entry_mode}
    # `""` is the "unused optional slot" spelling and is NOT a filter; `[]` IS one (it is the
    # explicit empty set, and pricing it is exactly how a caller learns their `who=` resolved
    # to nobody).
    active = [name for name, v in args.items() if v is not None and v != ""]
    if not active:
        return {}
    baseline = _count_matching(conn, **args)
    out: dict[str, int] = {}
    for name in active:
        gain = _count_matching(conn, **{**args, name: None}) - baseline
        if gain > 0:
            out[name] = gain
    return out


def atom_bm25_search(conn, query: str, candidates: set[str] | None, k: int) -> list[AtomHit]:
    """BM25 arm: best-matching chunk per atom via chunks_fts, grouped to the atom.

    Lower `bm25()` = better; we scan chunk hits in that order and keep the FIRST (best)
    chunk seen per atom, so an atom is ranked by its single strongest passage — the same
    max-pool intuition as the vector arm, done by FTS ordering."""
    frag, cand_params = _in_clause(candidates, "a")
    sql = (
        "SELECT a.*, e.name AS _who_name, "
        "c.seq AS _seq, c.char_start AS _cs, c.char_end AS _ce, "
        "c.text AS _chunk, bm25(chunks_fts) AS _rank "
        "FROM chunks_fts "
        "JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id "
        "JOIN atoms a  ON a.atom_id = c.atom_id "
        # LEFT, never INNER: an atom whose author entity row is missing must still be
        # RETURNED (unnamed), not silently dropped from retrieval.
        "LEFT JOIN entities e ON e.entity_id = a.who_id "
        "WHERE chunks_fts MATCH ?" + frag + " ORDER BY _rank"
    )
    # ITERATED, not `.fetchall()`. The `break` below already stops at `k` unique atoms; a
    # `fetchall` in front of it built a full Python row — with the whole wide `a.*` projection —
    # for every FTS match first, so a broad query paid for the entire match set to return eight
    # cards. SQLite still does the `ORDER BY _rank` sort internally, in C; only the rows this
    # loop actually consumes are ever materialized.
    cur = conn.execute(sql, [_fts_query(query), *cand_params])
    out, seen = [], set()
    for r in cur:
        aid = r["atom_id"]
        if aid in seen:
            continue
        seen.add(aid)
        hit = _atom_hit(r, chunk_text=r["_chunk"], chunk_seq=r["_seq"],
                        chunk_span=(r["_cs"], r["_ce"]))
        hit.bm25_rank = len(out)
        hit.score = 1.0 / (1.0 + len(out))   # meaningful for the bm25-only trust re-rank
        out.append(hit)
        if len(out) >= k:
            break
    return out


def atom_semantic_search(conn, query: str, embedder, candidates: set[str] | None,
                         k: int) -> list[AtomHit]:
    """Vector arm: cosine of the query vector over candidate chunk vectors, max-pooled
    per atom. The query is embedded with `role="query"` (the Qwen instruction prefix);
    chunks were stored raw — same subspace by construction (one embedder, guarded).

    TWO PASSES over two different column sets, and the split is the whole reason this function
    costs a constant. Ranking reads `atom_id` and `vector`; a hit CARD reads eighteen more
    columns. Fusing both into one `SELECT a.*` across a one-to-many join materialized every
    atom's card columns once per chunk — 1,930 ms of a 1,930 ms call, against ~10 ms of actual
    arithmetic. So pass 1 projects what cosine reads and pass 2 pays the wide projection once
    per WINNER. Measured 1,930 → 296 ms.

    And pass 1 STREAMS rather than materializing: peak memory is one batch plus one float per
    atom, flat in corpus size, where `.fetchall()` + `.astype(float32)` held every row object,
    every float16 blob and a double-width copy of all of them simultaneously (339 → 79 MB).
    Safe because a chunk's score depends only on that chunk and the query — nothing is
    normalized across the candidate set — and a maximum taken in pieces is the maximum.
    Full derivation: docs/lessons/query-cost-is-residency-not-arithmetic.md.

    A KNOWN, MEASURED, DELIBERATELY UNFIXED BIAS lives in the max-pool: an atom with 52 chunks
    draws its maximum from 52 samples and one with 2 draws from 2, so the longer atom scores
    higher for its length. Measured at +0.066 mean cosine for the SAME content chunked both ways
    (2026-08-26). A percentile-within-chunk-band correction was built for it and REVERSED on
    2026-08-27: the between-band cosine gap is +0.28, so ~0.21 of what a rank statistic removes
    is genuine topical coverage rather than sample size, and thinning the high bands turned one
    Oracle-cited paper from rank 0 into a miss. What contains the bias now is provenance
    SECTIONING (`opyt_core.kb.run_kb_search`), which puts the long crawl documents in their own
    list, leaving a spread inside each list too narrow for the correction to be worth its cost.
    Full record, and every instrument rejected:
    docs/plans/2026-08-26-frontier-crowding-in-search.md
    """
    qvec = np.asarray(embedder.embed([query], role="query")[0], dtype=np.float32)
    qn = qvec / (np.linalg.norm(qvec) + 1e-9)

    frag, cand_params = _in_clause(candidates, "a")
    # The `atoms` join stays even though no `a.` column is selected: it is what the candidate
    # restriction filters on, and dropping it would also admit a chunk whose atom row is gone.
    cur = conn.execute(
        "SELECT c.chunk_id AS _cid, c.atom_id AS _aid, c.vector AS _vec "
        "FROM chunks c JOIN atoms a ON a.atom_id = c.atom_id "
        "WHERE c.vector IS NOT NULL" + frag,
        cand_params,
    )

    # Width from `kb_meta`, never assumed: this reshape is exactly the operation that turns a
    # width disagreement into silent garbage rather than an exception, so the reader must read
    # the store's own record of what it wrote.
    from .embed import stored_dtype
    dt = np.dtype(stored_dtype(conn))

    # MAX-POOL per atom, accumulated across batches: an atom scores as its single strongest
    # chunk; keep that chunk's id so pass 2 can name the matched span. This dict is the ONLY
    # thing that grows with the corpus — one float and one int per atom, not per chunk.
    best: dict[str, tuple[float, int]] = {}
    while batch := cur.fetchmany(VEC_BATCH):
        # `.astype(float32)` both restores full-precision arithmetic (the narrowing is a STORAGE
        # choice) and hands us an OWNED array, so the normalize runs in-place — per batch, which
        # is what keeps the double-width copy from being corpus-sized.
        mat = np.frombuffer(b"".join(r["_vec"] for r in batch), dtype=dt).reshape(len(batch), -1)
        mat = mat.astype(np.float32)
        mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = mat @ qn
        for r, sim in zip(batch, sims):
            aid = r["_aid"]
            if aid not in best or sim > best[aid][0]:
                best[aid] = (float(sim), r["_cid"])

    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:k]
    cards = _hit_columns(conn, [cid for _aid, (_score, cid) in ranked])
    out = []
    for rank, (_aid, (score, cid)) in enumerate(ranked):
        r = cards[cid]
        hit = _atom_hit(r, chunk_text=r["_chunk"], chunk_seq=r["_seq"],
                        chunk_span=(r["_cs"], r["_ce"]))
        hit.sem_rank = rank
        hit.score = score
        out.append(hit)
    return out


def search_atoms(conn, query: str, embedder, *, tags: list[str] | None = None,
                 what_kind: str | None = None, source_type: str | None = None,
                 who_id: str | list[str] | None = None, date_from: str | None = None,
                 date_to: str | None = None, entry_mode: str | list[str] | None = None,
                 k: int = 8, mode: str = "hybrid") -> SearchRun:
    """The enforced-hybrid entry: filter → arm(s) → fuse → top-k, ranked on pure relevance.
    No trust re-rank (see module docstring). mode ∈ {hybrid, bm25, semantic}.

    `date_from`/`date_to` are already-normalized `YYYY-MM-DD` bounds — normalize a caller's raw
    input with `_date_bound` first, so a malformed one raises before any work is done.

    `entry_mode` filters on HOW the atom arrived and is an allow-list by design (see
    `_filter_clauses`); `entry_mode=list(schema.HUMAN_ATTESTED)` is the human-attested pile.

    Returns a `SearchRun`, not a bare list: `.hits` plus the mode/scale/cutoff metadata a caller
    needs, since `score` means a different thing per branch and `mode="hybrid"` often runs only
    one arm.
    """
    cand = candidate_atom_ids(conn, tags, what_kind, source_type, who_id, date_from, date_to,
                              entry_mode=entry_mode)
    # Computed BEFORE the early return, and only when a date bound is active. A date filter that
    # empties the candidate set entirely is precisely when "…and N of them had no date" is the
    # fact that explains the empty result.
    undated = (_count_undated(conn, tags, what_kind, source_type, who_id, entry_mode)
               if (date_from or date_to) else 0)
    if cand is not None and not cand:
        # A filter was requested and nothing matched it. NO arm runs, so there is no score
        # scale and no keyword rewrite to report — absent, rather than zeroed.
        return SearchRun(hits=[], effective_mode="none",
                         why="pre-filter emptied the candidate set", score_scale=None,
                         candidates=0, ranked=0, cutoff=None, bm25_weight=None, fts_query=None,
                         undated_excluded=undated)

    pool = max(k * 5, 40)   # over-fetch so fusion sees enough of each arm before top-k
    arm_sizes: dict[str, int] = {}
    w_bm25: float | None = None
    fts: str | None = None
    if mode == "bm25":
        fused = atom_bm25_search(conn, query, cand, pool)
        arm_sizes["bm25"] = len(fused)
        fts = _fts_query(query)
        ran, scale, why = "bm25", "reciprocal_rank", "mode=bm25 requested"
    elif mode == "semantic":
        fused = atom_semantic_search(conn, query, embedder, cand, pool)
        arm_sizes["semantic"] = len(fused)
        ran, scale, why = "semantic", "cosine", "mode=semantic requested"
    else:  # hybrid
        w_bm25 = bm25_weight(query)
        sem = atom_semantic_search(conn, query, embedder, cand, pool)
        arm_sizes["semantic"] = len(sem)
        if w_bm25 == 0.0:
            # Pure-conceptual query → semantic only. "hybrid" was asked for; one arm RAN, and
            # the scores that come back are therefore raw cosines, not fused ranks.
            fused = sem
            ran, scale = "semantic", "cosine"
            why = f"no literal tokens in query (bm25_weight {w_bm25})"
        else:
            bm = atom_bm25_search(conn, query, cand, pool)
            arm_sizes["bm25"] = len(bm)
            fts = _fts_query(query)
            fused = rrf_fuse([bm, sem], k=pool, weights=[w_bm25, 1.0])
            ran, scale = "hybrid", "rrf"
            why = f"literal tokens in query (bm25_weight {w_bm25})"

    # The truncation boundary, from data this function already had and threw away at `[:k]`.
    # `candidates` says how many atoms cleared the FILTER; these two say whether the cut landed
    # mid-cluster (near-identical scores) or on a cliff (a large drop). Reported, never judged.
    cutoff = None
    if k > 0 and len(fused) > k:
        cutoff = {"last_returned": round(fused[k - 1].score, 4),
                  "next_best": round(fused[k].score, 4)}
    return SearchRun(
        hits=fused[:k], effective_mode=ran, why=why, score_scale=scale,
        candidates=None if cand is None else len(cand), ranked=len(fused), cutoff=cutoff,
        bm25_weight=w_bm25, fts_query=fts, arm_sizes=arm_sizes,
        pool_saturated=any(n >= pool for n in arm_sizes.values()),
        undated_excluded=undated,
    )
