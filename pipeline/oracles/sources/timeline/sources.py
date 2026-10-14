"""
pipeline/oracles/sources/timeline/sources.py — the three Oracle sources on X's
`UserOriginalsTimeline` quota: `Picks` and `Check` (yours) and `NewPosts`. Each picks its next
pair and hands the page to the walks (`walks.py`).
"""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta

from pipeline.oracles import registry
from pipeline.oracles import windows
from pipeline.oracles.asks import ask_for_head
from pipeline.oracles.asks import asks_by_head
from pipeline.oracles.sources.timeline.walks import (QUOTA, Fetched, below_back, below_forward,
                                                    fetch_page, record_back, record_forward,
                                                    save_back, save_forward, unit_for, x_pairs)
from pipeline.timeparse import parse_ts
from pipeline.worker.quota_loop.source import Context
from pipeline.worker.quota_loop.source import NEW_POSTS
from pipeline.worker.quota_loop.source import Outcome
from pipeline.worker.quota_loop.source import Unit
from pipeline.worker.quota_loop.source import YOURS
from pipeline.worker.quota_loop.source import Source
from pipeline.oracles.sources.pair_rows import pair_key
from pipeline.oracles.sources.pair_rows import next_free
from pipeline.oracles.sources.pair_rows import free


class Picks(Source):
    """YOURS: the history each pick is owed, one page at a time, least-covered first:
    a pair never pulled, then the one whose coverage reaches least far back, ties by ask order."""
    name, quota, klass, floor = "picks", QUOTA, YOURS, 0

    def _owed(self, conn, now):
        asks = asks_by_head(conn)
        rows = [r for r in x_pairs(conn) if windows.owes_history(r, asks.get(r.canonical_id), now)]
        return rows, asks

    def next_unit(self, conn, now: datetime) -> Unit | None:
        rows, asks = self._owed(conn, now)
        rows = free(rows, now)
        if not rows:
            return None

        def order(r: registry.SourceRow) -> tuple:
            covered = parse_ts(r.covered_from)
            ask = asks.get(r.canonical_id)
            return (covered is not None, -(covered.timestamp() if covered else 0.0),
                    ask.position if ask else float("inf"), pair_key(r))
        row = min(rows, key=order)
        return unit_for(row, f"{row.canonical_id}|{row.source_key}|back",
                        (row, asks.get(row.canonical_id)))

    def fetch(self, conn, unit: Unit, ctx: Context) -> Fetched:
        row, ask = unit.payload
        return fetch_page(conn, row, ctx, unit, below_id=below_back(row),
                          save=lambda c, f: save_back(c, row, ask, f))

    def record(self, conn, unit: Unit, fetched: Fetched) -> Outcome:
        row, _ask = unit.payload
        return record_back(conn, row, fetched)

    def next_due(self, conn, now: datetime) -> datetime | None:
        rows, _asks = self._owed(conn, now)
        return next_free(rows, now)


def check_owed(row: registry.SourceRow) -> bool:
    """Someone asked for this pair's newest posts after the last walk from the top."""
    asked = parse_ts(row.check_requested_at)
    if asked is None:
        return False
    last = parse_ts(row.last_pulled_at)
    return last is None or asked > last


class Check(Source):
    """YOURS: the newest posts of a pair someone asked about (`checks.request_check`), oldest
    request first. A pair never pulled gets its first page from the top, which is also a check."""
    name, quota, klass, floor = "check", QUOTA, YOURS, 0

    def next_unit(self, conn, now: datetime) -> Unit | None:
        rows = free([r for r in x_pairs(conn) if check_owed(r)], now)
        if not rows:
            return None
        row = min(rows, key=lambda r: (parse_ts(r.check_requested_at), pair_key(r)))
        return unit_for(row, f"{row.canonical_id}|{row.source_key}|check", row)

    def fetch(self, conn, unit: Unit, ctx: Context) -> Fetched:
        row = unit.payload
        if row.last_pulled_at is None:
            ask = ask_for_head(conn, row.canonical_id)
            return fetch_page(conn, row, ctx, unit, below_id=below_back(row),
                              save=lambda c, f: save_back(c, row, ask, f))
        return fetch_page(conn, row, ctx, unit, below_id=below_forward(row),
                          save=lambda c, f: save_forward(c, row, f))

    def record(self, conn, unit: Unit, fetched: Fetched) -> Outcome:
        row = unit.payload
        if row.last_pulled_at is None:
            return record_back(conn, row, fetched)
        return record_forward(row, fetched)

    def next_due(self, conn, now: datetime) -> datetime | None:
        return next_free([r for r in x_pairs(conn) if check_owed(r)], now)


class NewPosts(Source):
    """NEW POSTS: a pair past its refresh time, worst lag first; a forward walk in progress keeps
    going. A pair never pulled is the picks' to start."""
    name, quota, klass, floor = "new_posts", QUOTA, NEW_POSTS, 0

    def _owed(self, conn, now):
        return [r for r in x_pairs(conn)
                if r.last_pulled_at and (r.fwd_started_at or windows.is_stale(r, now))]

    def next_unit(self, conn, now: datetime) -> Unit | None:
        rows = free(self._owed(conn, now), now)
        if not rows:
            return None
        row = max(rows, key=lambda r: (r.fwd_started_at is not None,
                                       windows.staleness_hours(r, now)))
        return unit_for(row, f"{row.canonical_id}|{row.source_key}|forward", row)

    def fetch(self, conn, unit: Unit, ctx: Context) -> Fetched:
        row = unit.payload
        return fetch_page(conn, row, ctx, unit, below_id=below_forward(row),
                          save=lambda c, f: save_forward(c, row, f))

    def record(self, conn, unit: Unit, fetched: Fetched) -> Outcome:
        return record_forward(unit.payload, fetched)

    def next_due(self, conn, now: datetime) -> datetime | None:
        times = []
        for r in x_pairs(conn):
            last = parse_ts(r.last_pulled_at)
            if last is not None:
                times.append(last + timedelta(hours=windows.pair_ttl_hours(r)))
        waiting = next_free(self._owed(conn, now), now)
        if waiting is not None:
            times.append(waiting)
        later = [t for t in times if t > now]
        return min(later) if later else None
