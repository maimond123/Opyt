"""
pipeline/oracles/sources/pair_rows.py — the Oracle pairs the scrapers walk, and the retry columns
on a pair's row. `checks.py` reads the same definitions, so a check and the walk that answers it
cannot disagree about which pairs exist or which may run.

A pair is one `oracle_sources` row: one Oracle × one source. The quota loop reads and writes its
retry state through `PAIR_ROWS`.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from pipeline.kb import writes
from pipeline.timeparse import parse_ts
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import Item, Retry
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.worker.quota_loop.source import RetryRows



def walked(conn: sqlite3.Connection, source_types, canonical_id: str | None = None
           ) -> list[registry.SourceRow]:
    """Every pair of `source_types` the scrapers walk: a confirmed Oracle's, paused pairs
    excluded. Scoped to one Oracle when `canonical_id` is given."""
    oracles = roster.heads(conn)
    rows = registry.list_sources(conn, [canonical_id] if canonical_id else None)
    return [r for r in rows
            if r.source_type in source_types and r.status != "paused"
            and r.canonical_id in oracles]


def free(rows: list[registry.SourceRow], now: datetime) -> list[registry.SourceRow]:
    """The pairs that may run now: no `retry_at` ahead, and not given up on."""
    return [r for r in rows if retries.may_run(r.retry_at, r.failures, now)]


def gave_up(rows: list[registry.SourceRow]) -> list[registry.SourceRow]:
    """The pairs no longer tried until the user asks for their Oracle again."""
    return [r for r in rows if retries.given_up(r.failures)]


def next_free(rows: list[registry.SourceRow], now: datetime) -> datetime | None:
    """When the earliest of `rows` that is waiting may run again."""
    times = [t for t in (parse_ts(r.retry_at) for r in rows) if t is not None and t > now]
    return min(times) if times else None


def pair_key(row: registry.SourceRow) -> tuple:
    return (row.canonical_id, row.source_type, row.source_key)


def unit_key(row: registry.SourceRow) -> str:
    """The key of a pair's units, and of the archive walk its pull begins under them."""
    return f"pair|{'|'.join(pair_key(row))}"


# ── the retry columns, on the pair's row ──────────────────────────────────────────
class _PairRows(RetryRows):
    """`oracle_sources`' retry columns, keyed by `pair_key`. `last_status` names the kind while
    the pair is failing, and the record step's outcome otherwise."""

    def read(self, row: registry.SourceRow) -> Retry:
        return Retry(row.last_status, row.failures, row.retry_at, row.last_error)

    def write(self, conn: sqlite3.Connection, key: tuple, retry: Retry) -> None:
        with writes.write_tx(conn, site="pairs.failure"):
            registry.charge(conn, key, retry.failures, retry.status, retry.retry_at,
                            retry.last_error)

    def clear(self, conn: sqlite3.Connection, key: tuple) -> None:
        with writes.write_tx(conn, site="pairs.clear"):
            registry.clear_retry(conn, key)


PAIR_ROWS = _PairRows()


def item(row: registry.SourceRow, platform: str) -> Item:
    """What a unit on this pair charges a failure to: its own row."""
    return Item(PAIR_ROWS, pair_key(row), PAIR_ROWS.read(row), platform)
