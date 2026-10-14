"""
pipeline/oracles/found.py — what discovery found for an Oracle beyond X, awaiting a first pull.

The `found_sources` table: a Substack, a blog, a GitHub, …, and whether its first pull has
happened. A row and not an `oracle_sources` pair, because a website pair's key comes from the
entity its first pull creates. Additive, like `oracle_asks`.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from pipeline.oracles.asks import read_rows

_DDL = """
CREATE TABLE IF NOT EXISTS found_sources (
  canonical_id TEXT NOT NULL,        -- the Oracle, as `oracle_asks` names it
  source_type  TEXT NOT NULL,        -- substack | blog | github | … (never x: that is a pair)
  url          TEXT NOT NULL,
  metadata     TEXT,                 -- the rest of discovery's source dict, JSON (trust, shape)
  found_at     TEXT NOT NULL,
  pulled_at    TEXT,                 -- its first pull is done, whatever it found; NULL = owed
  last_status  TEXT,                 -- that pull's outcome, or the kind of its failures
  failures     INTEGER,              -- failed first pulls in a row, of that kind
  retry_at     TEXT,                 -- not retried before this; NULL once it gave up
  last_error   TEXT,
  PRIMARY KEY (canonical_id, source_type, url)
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL, safe on every writable open."""
    conn.executescript(_DDL)
    conn.commit()


@dataclass(frozen=True)
class FoundSource:
    canonical_id: str
    source_type: str
    url: str
    metadata: str | None
    found_at: str
    pulled_at: str | None
    last_status: str | None
    failures: int | None
    retry_at: str | None
    last_error: str | None

    def source(self) -> dict:
        """Back to the shape discovery returned it in, which is what `onboard_footprint` takes."""
        try:
            rest = json.loads(self.metadata) if self.metadata else {}
        except ValueError:
            rest = {}
        return {**rest, "source_type": self.source_type, "url": self.url}


def list_found(conn: sqlite3.Connection, canonical_id: str | None = None) -> list[FoundSource]:
    sql, params = "SELECT * FROM found_sources", ()
    if canonical_id is not None:
        sql, params = sql + " WHERE canonical_id=?", (canonical_id,)
    return [FoundSource(**dict(r))
            for r in read_rows(conn, sql + " ORDER BY canonical_id, source_type, url", params)]


def owe_first_pull(conn: sqlite3.Connection, f: FoundSource) -> None:
    """Record `f` as owed its first pull, clearing a finished or failing one before it. The
    caller holds the write transaction."""
    conn.execute(
        "INSERT INTO found_sources (canonical_id, source_type, url, metadata, found_at) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(canonical_id, source_type, url) DO UPDATE SET "
        "metadata=excluded.metadata, pulled_at=NULL, last_status=NULL, failures=0, "
        "retry_at=NULL, last_error=NULL",
        (f.canonical_id, f.source_type, f.url, f.metadata, f.found_at))
