"""
pipeline/kb/id_batches.py — one query over a list of ids too long for one statement.

SQLite refuses a statement with more bound parameters than its ceiling (999 by default), and a
region's atoms or a sample of bodies runs to thousands. A lookup by an id list runs here, once per
slice of at most `MAX_IDS`, so the ceiling is handled in one place rather than by a loop copied
into every caller.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator

MAX_IDS = 900               # under SQLite's default 999-parameter ceiling


def placeholders(n: int) -> str:
    """`?,?,…`, one bound parameter per value, for an `IN (…)` over `n` values."""
    return ",".join("?" * n)


def rows_for(conn: sqlite3.Connection, sql: str, ids: Iterable) -> Iterator[sqlite3.Row]:
    """Every row `sql` returns over `ids`, run once per slice of at most `MAX_IDS`. `sql` holds
    `{ids}` where a slice's placeholders go. Each slice is fetched whole before it is handed on, so
    no cursor stays open while the caller works through the rows. An `ORDER BY` orders within a
    slice."""
    ids = list(ids)
    for i in range(0, len(ids), MAX_IDS):
        part = ids[i:i + MAX_IDS]
        yield from conn.execute(sql.format(ids=placeholders(len(part))), part).fetchall()
