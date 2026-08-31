"""
pipeline/kb/frontier_surface.py — Frontier stage 4 (SURFACE): rank what stage 2 staged, record
what the surface did, and say when there is something worth saying.

Stage 4 is the ONLY component in this rail licensed to hide anything. Stage 1 may not retire a
query (it decays the cadence instead), stage 2 writes to staging and stops before judgement, and
stage 3 admits autonomously. Every one of those was deliberately forbidden from pruning, which
concentrates the entire risk here. So this module is built around one rule:

    Every term demotes. Nothing gates.

A bad candidate sinks; it never vanishes — a dropped candidate can never be noticed, a demoted
one still can. Truncation is PAGINATION: callers are told what they did not receive.

No score is ever persisted; ranking is recomputed from verifiable inputs on every call.
`frontier_candidate_events` stores only what happened ("shown at 14:02"), never a computed score.

Substance is normalized WITHIN a source (percentile, not raw value) since raw signals are not
comparable across sources — see doc for the measured distributions.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pipeline.timeparse import utc_iso, utc_now

# ── Weights ─────────────────────────────────────────────────────────────────────
# Additive and each bounded, so no term can drive a score to infinity and turn a demotion into a
# de-facto exclusion.
W_AGREEMENT = 0.40      # per doubling of distinct queries that found it
W_DEMAND = 0.30         # per doubling of distinct generators that asked for those queries
W_RECENCY = 0.25
W_SUBSTANCE = 0.20
W_ATTENTION = 0.30      # per doubling of (1 + times shown) — SUBTRACTED

RECENCY_HALF_LIFE_DAYS = 30.0

# No saturation constant here — a prior per-source cap (e.g. a star ceiling) was a measured bug:
# `_percentile_within_source` reads only ORDER, so clamping ties off the top of a distribution
# instead of helping. See doc for the incident. Return the raw signal, unclamped.


def _parse_day(stamp: str | None) -> datetime | None:
    """Parse a candidate's own publication date. Tolerant on purpose: a date we cannot read is
    UNKNOWN, and unknown must not read as 'ancient' — that would demote by parser accident."""
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(str(stamp)[:19].replace("Z", "")).replace(
            tzinfo=timezone.utc)
    except ValueError:
        return None


# ── The terms ───────────────────────────────────────────────────────────────────
def _recency(published: str | None, *, now: datetime) -> float:
    """1.0 today, 0.5 at one half-life, asymptotically 0 — never negative, so age alone can never
    push a candidate below a dismissed one. An unreadable or missing date scores the NEUTRAL 0.5
    rather than 0: we do not know when it appeared, and guessing 'old' is a silent demotion."""
    day = _parse_day(published)
    if day is None:
        return 0.5
    age_days = max(0.0, (now - day).total_seconds() / 86400.0)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def _raw_substance(kind: str | None, summary: str | None, payload: dict) -> float:
    """The orthogonal signal: is this the deep thing or the shallow thing, holding topic fixed?
    (Rejected: cosine to the nearest human-attested atom — can't distinguish junk from a genuinely
    new direction.) Returned raw and unclamped; `_percentile_within_source` makes it comparable.

    Keyed on `kind`, not `source`, and the two are used deliberately on either side of this:
    KIND says what the signal IS (a paper has an abstract, a repo has stars), SOURCE says what it
    is comparable TO. Keyed on source, a second paper finder would be a second name in the same
    arm and a third would be a third — the same duplication the finder/minter split removed from
    stage 3. An unrecognized kind is constant across its group, so the percentile resolves it
    to the neutral 0.5.
    """
    if kind == "repo":
        stars = payload.get("stars")
        return float(stars) if isinstance(stars, (int, float)) else 0.0
    if kind == "paper":
        return float(len(summary or ""))
    return 0.0


def _artifact_key(d: dict) -> tuple | None:
    """What makes two staged candidates the SAME ARTIFACT, computed OFFLINE. None = never merge.

    One artifact reaches the queue under several `candidate_id`s routinely, because a finder's id
    is a fact about the FINDER and not about the thing. Measured over 68 live OpenAlex results on
    2026-08-26: 12 groups covering 24 of the 68 works, in three shapes —

      * the same arXiv preprint returned twice, once with `doi: null` and an `/abs/` landing page
        and once with its `10.48550` DOI (7 groups);
      * a DOI carrying a `.v1` version suffix alongside the bare one (1 group);
      * two Zenodo deposits whose record ids differ by exactly one — Zenodo's concept/version DOI
        pair (4 groups).

    Keyed on CONTENT and not on the resolved atom id, deliberately. Only the first shape shares an
    atom id; the other two mint two atoms, so an atom-id key would leave 5 of the 12 uncollapsed
    while costing a `paper_from_url` parse per row.

    Title + first author + date, all three required. Title alone is not enough (a series of
    identically-titled weekly reports is real), and a MISSING title returns None rather than
    merging every untitled row into one heap — a key that groups on absence is how a collapse
    turns into a silent mass delete.
    """
    title = " ".join((d.get("title") or "").lower().split())
    published = (d.get("published") or "").strip()
    if not title or not published:
        return None
    authors = d["payload"].get("authors") or []
    first = str(authors[0] or "").lower().strip() if authors else ""
    return (title, first, published)


def _collapse_duplicates(rows: list[dict]) -> list[dict]:
    """One artifact, one card. MERGES the group's evidence — it never hides a row's signal.

    Runs BEFORE `_percentile_within_source` so a source's substance distribution is not skewed by
    counting the same abstract twice.

    The survivor is the most informative card (longest summary, `candidate_id` breaking the tie so
    the choice is deterministic and a paginated read cannot reshuffle). Evidence carried over is
    the MAX across the group, never the sum: two candidate rows found by the SAME standing query
    would double-count under a sum, and a max can only under-state. `dismissed_n` maxes too —
    saying stop to one form of an artifact means stop to the artifact.
    """
    groups: dict[tuple, list[dict]] = {}
    out: list[dict] = []
    for d in rows:
        key = _artifact_key(d)
        if key is None:
            out.append(d)                                # unkeyable → always its own card
        else:
            groups.setdefault(key, []).append(d)
    for group in groups.values():
        keep = sorted(group, key=lambda r: (-len(r.get("summary") or ""), r["candidate_id"]))[0]
        if len(group) > 1:
            for k in ("n_queries", "n_generators", "shown_n", "dismissed_n"):
                keep[k] = max(int(r.get(k) or 0) for r in group)
            # A group can straddle an admit run (stage 3 takes ADMIT_MAX_PER_RUN at a time), so
            # one row can be materialized while its twin is still `new`. The artifact is in the KB
            # either way, and the card must not say otherwise.
            if any((r.get("status") or "new") == "materialized" for r in group):
                keep["status"] = "materialized"
            # Reported, never silent: the card says how many staged rows it stands for, and which.
            keep["duplicate_of"] = sorted(r["candidate_id"] for r in group
                                          if r["candidate_id"] != keep["candidate_id"])
        out.append(keep)
    return out


def _percentile_within_source(rows: list[dict]) -> None:
    """Replace each row's raw substance with its MIDRANK percentile among rows of the same source.

    Midrank rather than "count strictly below" because ties are common (many repos sit at zero
    stars); midrank puts a tied group at its shared middle instead of pinning it to the floor. A
    source with one row gets the neutral 0.5 — a single observation has no distribution.
    """
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"] or "", []).append(r)
    for group in by_source.values():
        n = len(group)
        if n == 1:
            group[0]["substance"] = 0.5
            continue
        ordered = sorted(group, key=lambda r: r["_raw_substance"])
        i = 0
        while i < n:
            j = i
            while j + 1 < n and ordered[j + 1]["_raw_substance"] == ordered[i]["_raw_substance"]:
                j += 1
            midrank = (i + j) / 2.0                      # shared position of the tied block
            for r in ordered[i:j + 1]:
                r["substance"] = midrank / (n - 1)
            i = j + 1


def _bonus(count: int | None, weight: float) -> float:
    """A per-doubling bonus that VANISHES at a count of 1 — written as an additive bonus (not a
    multiplier/divisor) so a row with count=1 degrades to recency-plus-substance instead of a
    ranking-breaking constant.
    """
    n = max(1, int(count or 1))
    return weight * math.log2(n)


def _penalty(shown_n: int | None, weight: float) -> float:
    """Attention already spent. Bounded by construction and always subtracted from a bounded
    score, so a much-shown candidate SINKS and is never removed — the difference constraint 6
    turns on. Zero shows costs nothing."""
    return weight * math.log2(1 + max(0, int(shown_n or 0)))


# ── Ranking ─────────────────────────────────────────────────────────────────────
# The demand subquery joins `frontier_query_generators` and NEVER `frontier_generators`, so
# `votable` is not consulted here on purpose — `votable` gates whether a channel can still vote on
# a query's cadence, not whether its past claim counts as demand evidence. Filtering on it would
# discard real demand from write-once generators. Pinned by a test. Likewise `status='retired'`
# is not filtered: retiring a query stops it running, it doesn't un-find what it already found.
_SELECT = """
SELECT c.candidate_id, c.source, c.kind, c.title, c.url, c.published, c.summary, c.payload,
       c.status,
       c.first_seen_at, c.last_seen_at,
       (SELECT COUNT(DISTINCT q.query_id)
          FROM frontier_candidate_queries q
         WHERE q.candidate_id = c.candidate_id)                        AS n_queries,
       (SELECT COUNT(DISTINCT g.generator)
          FROM frontier_candidate_queries q
          JOIN frontier_query_generators g ON g.query_id = q.query_id
         WHERE q.candidate_id = c.candidate_id)                        AS n_generators,
       (SELECT COUNT(*) FROM frontier_candidate_events e
         WHERE e.candidate_id = c.candidate_id AND e.event = 'shown')  AS shown_n,
       (SELECT MAX(e.at) FROM frontier_candidate_events e
         WHERE e.candidate_id = c.candidate_id AND e.event = 'shown')  AS last_shown_at,
       (SELECT COUNT(*) FROM frontier_candidate_events e
         WHERE e.candidate_id = c.candidate_id AND e.event='dismissed') AS dismissed_n
  FROM frontier_candidates c
"""


def rank_candidates(conn: sqlite3.Connection, *, limit: int | None = None,
                    include_dismissed: bool = True,
                    now: datetime | None = None) -> list[dict]:
    """Rank every staged candidate. Returns cards, each carrying its own state.

    `include_dismissed=True` is the default, and that is a deliberate reading of constraint 6 —
    a surface shows acted-on items WITH their state rather than silently removing them. A
    dismissed candidate ranks below every live one, so keeping it costs the reader nothing but
    keeps "I already said no to this" visible. Passing False is an explicit opt-out; callers that
    do should report how many rows it suppressed, so even the opt-out is not silent.

    `limit` is PAGINATION, not a filter. It truncates the ranked list; the caller is responsible
    for saying what it held back (see `frontier_tools.deliver`).

    One artifact staged under several `candidate_id`s becomes ONE card (`_collapse_duplicates`),
    carrying the group's merged evidence and a `duplicate_of` list of the ids it stands for. That
    is a merge and not a filter: no row's signal is discarded and the card names what it absorbed.
    """
    ref = now or utc_now()
    try:
        raw = conn.execute(_SELECT).fetchall()
    except sqlite3.OperationalError:
        return []                                    # no such table → empty, never a crash

    rows: list[dict] = []
    for r in raw:
        d = dict(r)
        try:
            payload = json.loads(d.get("payload") or "{}")
        except (TypeError, ValueError):
            payload = {}
        d["payload"] = payload if isinstance(payload, dict) else {}
        d["_raw_substance"] = _raw_substance(d.get("kind"), d.get("summary"), d["payload"])
        rows.append(d)

    rows = _collapse_duplicates(rows)
    _percentile_within_source(rows)

    for d in rows:
        d.pop("_raw_substance", None)
        d["dismissed"] = bool(d.get("dismissed_n") or 0)
        d["shown_n"] = int(d.get("shown_n") or 0)
        d["n_queries"] = int(d.get("n_queries") or 0)
        d["n_generators"] = int(d.get("n_generators") or 0)
        d["recency"] = _recency(d.get("published"), now=ref)
        d["score"] = round(
            _bonus(d["n_queries"], W_AGREEMENT)
            + _bonus(d["n_generators"], W_DEMAND)
            + W_RECENCY * d["recency"]
            + W_SUBSTANCE * d["substance"]
            - _penalty(d["shown_n"], W_ATTENTION), 6)
        d["state"] = _state_of(d)
        d["why"] = _why(d)

    if not include_dismissed:
        rows = [d for d in rows if not d["dismissed"]]

    # Two tiers, then score, then id. Dismissal is a tier and not a big negative number, so
    # "below every live candidate" is guaranteed by the sort rather than by hoping no weight ever
    # out-grows the constant. `candidate_id` last makes the order TOTAL: equal scores must not
    # reshuffle between two calls, or a paginated read silently skips and repeats rows.
    rows.sort(key=lambda d: (1 if d["dismissed"] else 0, -d["score"], d["candidate_id"]))
    return rows[:limit] if limit is not None else rows


def _state_of(d: dict) -> str:
    """What the user has already done with this row. Never removes it — names it."""
    if d["dismissed"]:
        return "dismissed"
    if (d.get("status") or "new") != "new":
        return d["status"]                            # stage 3's verdict, shown as-is
    return "seen" if d["shown_n"] else "new"


def _why(d: dict) -> list[str]:
    """Plain-language reasons, for a host that has to explain an ordering to a human. Built from
    the same facts the score reads, so the explanation cannot drift from the ranking."""
    out = []
    if d["n_queries"] > 1:
        out.append(f"{d['n_queries']} standing queries found it")
    if d["n_generators"] > 1:
        out.append(f"{d['n_generators']} regions of your KB asked for it")
    if d["shown_n"]:
        out.append(f"shown {d['shown_n']}x already")
    # No source or kind test: the PAYLOAD KEY is the signal. `_raw_substance` above must
    # dispatch on kind because it returns a number for every row; this only appends a line when
    # there is one to append, so the dispatch was never load-bearing — it was the finder/minter
    # coincidence left behind when `_raw_substance` moved off `source`. Dropping it rather than
    # switching it to `kind` also keeps the line on the pre-split rows, whose `kind` is NULL by
    # decision (no backfill — see `schema.init_kb_schema`).
    if d["payload"].get("stars"):
        out.append(f"{d['payload']['stars']} stars")
    if d.get("duplicate_of"):
        out.append(f"staged {len(d['duplicate_of']) + 1}x under different ids")
    return out


# ── The event log ───────────────────────────────────────────────────────────────
def record_event(conn: sqlite3.Connection, candidate_ids, event: str, *,
                 surface: str | None = None, at: str | None = None) -> int:
    """APPEND one row per (candidate, event). Never an UPDATE and never an upsert: showing the
    same candidate twice is two facts, and the second must not overwrite the first."""
    stamp = at or utc_iso()
    ids = [c for c in dict.fromkeys(candidate_ids) if c]
    if not ids:
        return 0
    conn.executemany(
        "INSERT INTO frontier_candidate_events (candidate_id, event, surface, at) VALUES (?,?,?,?)",
        [(cid, event, surface, stamp) for cid in ids])
    conn.commit()
    return len(ids)


def record_shown(conn, candidate_ids, *, surface: str | None = None) -> int:
    return record_event(conn, candidate_ids, "shown", surface=surface)


def record_dismissed(conn, candidate_ids, *, surface: str | None = None) -> int:
    """The user said stop. Recorded, and the row keeps being returned — demoted below everything
    live and labelled `dismissed`, per constraint 6."""
    return record_event(conn, candidate_ids, "dismissed", surface=surface)


# ── The push notice ─────────────────────────────────────────────────────────────
def notice(conn: sqlite3.Connection, *, now: datetime | None = None) -> dict | None:
    """A one-line count plus the single strongest unshown candidate — or None.

    Returning None on a quiet frontier keeps the notice's footprint at zero on an unrelated
    conversation. The quiet case costs one `COUNT`; full ranking only runs once there is
    something to rank, since this fires on every ordinary search.
    """
    try:
        unshown_n = conn.execute(
            """SELECT COUNT(*) FROM frontier_candidates c
                WHERE NOT EXISTS (SELECT 1 FROM frontier_candidate_events e
                                   WHERE e.candidate_id = c.candidate_id)""").fetchone()[0]
    except Exception:
        return None                                   # a notice must never break its carrier
    if not unshown_n:
        # No events at all means never shown AND never dismissed — the same set the full pass
        # computes below, reached without ranking anything.
        return None
    try:
        ranked = rank_candidates(conn, now=now)
    except Exception:
        return None
    unshown = [d for d in ranked if not d["shown_n"] and not d["dismissed"]]
    if not unshown:
        return None
    top = unshown[0]
    return {
        "unshown": len(unshown),
        "total": len(ranked),
        "top": {k: top[k] for k in ("candidate_id", "source", "title", "url", "published")},
        "call": "frontier()",
        "message": (f"{len(unshown)} unseen artifact"
                    f"{'s' if len(unshown) != 1 else ''} on the frontier; the strongest is "
                    f"{top['title'][:110]!r} ({top['source']}). Call frontier() for the ranked "
                    f"list — mention this to the user only if it is relevant to what they asked."),
    }
