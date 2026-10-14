"""
pipeline/frontier/cadence.py — when a (query, source) pair is due, and what window it asks for.

Decisions only: every function here takes values (a watermark row, a source, a clock reading) and
returns an answer. The searches source (`pipeline/frontier/searching.py`) reads the watermarks
and the clock and hands them in.

Watermarks are keyed on `query_id`, which requires stable query strings across runs (pinned by a
test).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from pipeline.timeparse import parse_ts

# ── Cadence ─────────────────────────────────────────────────────────────────────
# Every watch runs once a day on every source. Re-asking OpenAlex re-reads its
# 30-day window; the repeats are already-seen candidates and change nothing.
TTL_HOURS = 24.0
JITTER = 0.10               # ±10%, hash-derived and stable per pair
OVERLAP_HOURS = 6.0         # re-ask a sliver behind the resume point; dedup absorbs it
MAX_WINDOW_DAYS = 60        # the window assertion's ceiling, on the daily pulls' path only
LANDSCAPE_OVER_DAYS = 90    # past this a window is read as a landscape, not as news
FIRST_PULL_DAYS = 14        # a never-pulled pair looks back this far, not to the beginning of time
# A pair tried since its last pull (a failed request, a refused window) is due again this long
# after the try. Never given up.
RETRY_AFTER = timedelta(hours=1)

def jitter(query_id: str, source: str) -> float:
    """Stable ±JITTER from a hash, never random, so queries seeded on one day don't all fall due
    in the same second forever."""
    h = int(hashlib.sha256(f"{query_id}:{source}".encode()).hexdigest()[:8], 16)
    return 1.0 + ((h % 2001) / 1000.0 - 1.0) * JITTER


def due_at(row, source: str, *, query_id: str) -> datetime | None:
    """When this pair is next due. `row` is the pair's watermark, or None. None: due now and
    always (never pulled, or no readable last pull).

    TTL = `TTL_HOURS` × jitter: every watch runs once a day on every source, with no decay. A
    pair tried since its last pull waits `RETRY_AFTER` from the try instead.
    """
    last = parse_ts(row["last_pulled_at"]) if row is not None else None
    tried = parse_ts(row["tried_at"]) if row is not None else None
    if tried is not None and (last is None or tried > last):
        return tried + RETRY_AFTER
    if last is None:
        return None                                    # never pulled → infinitely stale
    ttl = TTL_HOURS * jitter(query_id, source)
    return last + timedelta(hours=ttl)


def is_due(row, source: str, *, query_id: str, now: datetime) -> bool:
    """Whether this pair should be searched now (`due_at` has passed)."""
    at = due_at(row, source, query_id=query_id)
    return at is None or now >= at


def since_for(row, *, now: datetime) -> datetime:
    """Where this pair should resume, minus an overlap sliver.

    A hard boundary at the last pull would silently drop what an upstream published while the
    previous request was in flight; dedup on `candidate_id` absorbs the re-ask.
    """
    base = parse_ts(row["last_pulled_at"]) if row and row["last_pulled_at"] else None
    if base is None:
        return now - timedelta(days=FIRST_PULL_DAYS)   # bounded first look-back, not all of time
    return base - timedelta(hours=OVERLAP_HOURS)


def lookback_floor(adapter, since: datetime, now: datetime) -> datetime:
    """Reach further back than `since_for` when a source windows on publication date but indexes
    late, which would otherwise silently miss work published before the window and indexed after.

    Declared on the adapter (`min_lookback_days`, optional) and applied here rather than in the
    adapter, so `window_ok` validates the window actually sent.
    """
    days = float(getattr(adapter, "min_lookback_days", 0.0) or 0.0)
    return since if days <= 0 else min(since, now - timedelta(days=days))


def landscape(days: float | None) -> bool:
    """Whether a window that long is a landscape rather than news.

    One line, read in two places that must not disagree: which order OpenAlex is asked in
    (`openalex_candidates.sort_for`) and which order the answer shows (`look.by_name`). Inside it
    every find is recent, so date order is the useful one and citation counts are all about zero —
    sorted by them, a page comes back tied in an order OpenAlex does not define. Beyond it the
    newest page of a five-year window is a recency slice of minor work, not the field, and
    re-sorting a ranked page by date would cut the most-cited work out of what the host sees.
    """
    return days is not None and days > LANDSCAPE_OVER_DAYS


def window_ok(since: datetime, now: datetime) -> bool:
    """Refuse an absurd window before making the request.

    Catches a threading bug where `since` fails to reach the adapter and every pull silently
    becomes a full-history re-read.
    """
    delta = (now - since).total_seconds() / 86400.0
    return 0 <= delta <= MAX_WINDOW_DAYS


@dataclass(frozen=True)
class DuePair:
    """One (standing query, source) pair, with the watermark it was planned from."""
    query: Mapping                     # the `frontier_queries` row
    source: str
    mark: Mapping | None               # its `frontier_query_sources` row; None = never pulled


@dataclass(frozen=True)
class Plan:
    """A pass's pairs: the ones due, stalest first, and the ones no adapter can run."""
    due: list[DuePair]
    no_adapter: list[DuePair]


def plan(queries, marks: Mapping, sources, *, now: datetime) -> Plan:
    """Which of these queries' pairs a pass pulls, and in what order.

    `marks` is every watermark keyed `(query_id, source)`; `sources` the slugs that have an
    adapter. Built in full before anything is pulled, so `pairs_due` counts what wanted to run,
    not what fit the budget. A pair with no adapter is returned in `no_adapter`, never pulled.

    Stalest first (never-pulled first), so the budget delays everyone in turn instead of starving
    the pairs that sort last.
    """
    due: list[DuePair] = []
    no_adapter: list[DuePair] = []
    for q in queries:
        for source in json.loads(q["target_sources"] or "[]"):
            pair = DuePair(q, source, marks.get((q["query_id"], source)))
            if source not in sources:
                no_adapter.append(pair)
            elif is_due(pair.mark, source, query_id=q["query_id"], now=now):
                due.append(pair)
    due.sort(key=lambda p: (p.mark["last_pulled_at"] if p.mark else None) or "")
    return Plan(due, no_adapter)
