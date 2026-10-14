"""
pipeline/oracles/asks.py — what each Oracle is owed: the `oracle_asks` table.

One row per Oracle: how far back on X, on the web and in papers the user asked for, when, and in
what order, when each of discovery's two stages last ran for them, and its retry state. Written before anything runs, so a pick that is never reached still has
its window on record; the scrapers read what is owed from here. `wake` asks for the scrapers now.

Additive, like every table the scrapers keep: a server and a worker on different versions share
one store for hours, so a column added later is added by `init_schema`. Until the table exists
every reader answers as if it were empty. Times are ISO text compared as text in SQL, so every
writer stores the one format `utc_iso` produces.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from pipeline.ingestion.utils import log
from pipeline.kb import schema, writes
from pipeline.timeparse import parse_ts, utc_iso
from pipeline.worker import job_queue, registry
from pipeline.oracles import registry as pair_registry

_DDL = """
CREATE TABLE IF NOT EXISTS oracle_asks (
  canonical_id   TEXT PRIMARY KEY,
  x_from         TEXT,               -- oldest instant asked for on X; NULL = nothing asked
  web_from       TEXT,               -- the same for blogs, Substacks and GitHub (ALL_TIME = all)
  scholar_from   TEXT,               -- the same for papers (ALL_TIME = all)
  scholar_topics TEXT,               -- unwritten: topics live on the OpenAlex pair (topic_filter)
  asked_at       TEXT NOT NULL,      -- the latest ask; an archive is owed until pulled after it
  position       INTEGER NOT NULL,   -- order of the first ask; breaks ties between picks
  root_probed_at TEXT,               -- when discovery's root stage read their root profile
  discovered_at  TEXT,               -- when discovery's links stage last found their sources
  rediscover     INTEGER NOT NULL DEFAULT 0, -- 1 = run discovery again (`ingest(force=True)`)
  discovery_failures INTEGER,        -- discovery runs that failed in a row (`quota_loop/retries.py`)
  discovery_retry_at TEXT,           -- not retried before this; NULL once it gave up
  discovery_error    TEXT            -- what the last failed run saw
);
"""

def read_rows(conn: sqlite3.Connection, sql: str, params=()) -> list[sqlite3.Row]:
    """Rows from one of the scrapers' tables, or none on a store that has not made them yet."""
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return []
        raise


def init_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL, safe on every writable open."""
    conn.executescript(_DDL)
    conn.commit()


@dataclass(frozen=True)
class Ask:
    canonical_id: str
    x_from: str | None
    web_from: str | None
    scholar_from: str | None
    scholar_topics: str | None
    asked_at: str
    position: int
    discovered_at: str | None
    rediscover: bool
    discovery_failures: int | None = None
    discovery_retry_at: str | None = None
    discovery_error: str | None = None
    root_probed_at: str | None = None


def _instant(value) -> str | None:
    """A datetime or ISO string → the stored `utc_iso` form. Raises on a string that isn't one:
    an ask is the user's own words, and a window silently dropped would be a window lost."""
    if value is None:
        return None
    parsed = value if isinstance(value, datetime) else parse_ts(value)
    if parsed is None:
        raise ValueError(f"not an instant: {value!r}")
    return utc_iso(parsed)


def _widen(col: str) -> str:
    # The older of the stored and the new instant, either of which may be NULL. SQLite's
    # two-argument MIN is NULL when either side is, hence the COALESCE on both.
    return (f"{col}=MIN(COALESCE(oracle_asks.{col}, excluded.{col}), "
            f"COALESCE(excluded.{col}, oracle_asks.{col}))")


def record_ask(conn: sqlite3.Connection, canonical_id: str, *, x_from=None, web_from=None,
               scholar_from=None, rediscover: bool = False,
               now: datetime | None = None) -> None:
    """Record what the user asked for one Oracle. Starts nothing.

    Windows only widen, as `registry.record_request` does for a pair: a later,
    narrower ask never takes back an earlier, wider one. `asked_at` moves to now on every ask,
    which is what makes each archive owed again. `position` is set by the first ask and kept, so
    picks that tie keep the order they were asked in. `rediscover` stays set until
    discovery clears it. Subjects are not part of the ask: they live on the Oracle's OpenAlex pair
    (`topic_filter`), the one place archives reads them.

    An ask also re-arms everything of this Oracle's the scrapers gave up on, or are waiting to
    retry: its pairs, its found sources and its discovery start counting failures afresh. Only an
    ask does this; a check never does."""
    with writes.write_tx(conn, site="record_ask"):
        conn.execute(
            "INSERT INTO oracle_asks (canonical_id, x_from, web_from, scholar_from, "
            "  asked_at, position, rediscover) "
            "VALUES (?, ?, ?, ?, ?, (SELECT COALESCE(MAX(position), 0) + 1 FROM oracle_asks), "
            "  ?) "
            "ON CONFLICT(canonical_id) DO UPDATE SET "
            f"  {_widen('x_from')}, {_widen('web_from')}, {_widen('scholar_from')}, "
            "  asked_at=excluded.asked_at, "
            "  rediscover=MAX(oracle_asks.rediscover, excluded.rediscover), "
            # A rediscovery runs both of discovery's stages again.
            "  root_probed_at=CASE WHEN excluded.rediscover=1 THEN NULL "
            "                      ELSE oracle_asks.root_probed_at END",
            (canonical_id, _instant(x_from), _instant(web_from), _instant(scholar_from),
             utc_iso(now), 1 if rediscover else 0))
        _rearm(conn, canonical_id)


def _rearm(conn: sqlite3.Connection, canonical_id: str) -> None:
    """Reset the retry counts of everything the scrapers walk for this Oracle."""
    conn.execute("UPDATE oracle_asks SET discovery_failures=0, discovery_retry_at=NULL "
                 "WHERE canonical_id=?", (canonical_id,))
    conn.execute("UPDATE found_sources SET failures=0, retry_at=NULL WHERE canonical_id=?",
                 (canonical_id,))
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='oracle_sources'").fetchone():
        pair_registry.rearm(conn, {canonical_id, schema.current_canonical(conn, canonical_id)})

    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='atoms'").fetchone():
        return
    head = schema.current_canonical(conn, canonical_id)
    members = {head, canonical_id, *(e["entity_id"] for e in schema.entities_for_canonical(conn, head))}
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='archive_units'").fetchone():
        conn.execute("UPDATE archive_units SET failures=0,retry_at=NULL,last_error=NULL "
                     "WHERE done=0 AND walk IN (SELECT key FROM archive_walks WHERE canonical_id IN "
                     f"({', '.join('?' for _ in members)}))", tuple(members))
    conn.execute(f"UPDATE atoms SET payload=json_remove(payload, '$.thread_retry') "
                 f"WHERE who_id IN ({', '.join('?' for _ in members)}) "
                 "AND source_type='x' AND entry_mode='oracle-footprint'",
                 tuple(members))

    for row in conn.execute(f"SELECT atom_id, payload FROM atoms WHERE who_id IN "
                            f"({', '.join('?' for _ in members)}) AND source_type='x' "
                            "AND entry_mode='oracle-footprint'", tuple(members)).fetchall():
        payload = json.loads(row["payload"] or "{}")
        entries = [*payload.get("links_owed", []), *payload.pop("links_failed", [])]
        if not entries:
            continue
        payload["links_owed"] = [{k: v for k, v in e.items() if k not in ("retry", "tries")}
                                 for e in entries]
        conn.execute("UPDATE atoms SET payload=? WHERE atom_id=?", (json.dumps(payload), row["atom_id"]))


def _ask(row: sqlite3.Row) -> Ask:
    # Only `Ask`'s own fields: a store made before a column was dropped (the host-found URLs)
    # still carries it, and the schema rule is edit-the-DDL with no migration.
    fields = {k: row[k] for k in row.keys() if k in Ask.__dataclass_fields__}
    return Ask(**{**fields, "rediscover": bool(row["rediscover"])})


def ask_for(conn: sqlite3.Connection, canonical_id: str) -> Ask | None:
    rows = read_rows(conn, "SELECT * FROM oracle_asks WHERE canonical_id=?", (canonical_id,))
    return _ask(rows[0]) if rows else None


def list_asks(conn: sqlite3.Connection) -> list[Ask]:
    """Every ask, in the order the picks were first asked for."""
    return [_ask(r) for r in read_rows(conn, "SELECT * FROM oracle_asks ORDER BY position")]


def asks_by_head(conn: sqlite3.Connection) -> dict[str, Ask]:
    """Every ask, keyed by its Oracle's current cluster head. Pairs are registered under the
    head, and a resolve that moves it leaves the ask under the anchor it was written with."""
    return {schema.current_canonical(conn, a.canonical_id): a for a in list_asks(conn)}


def ask_for_head(conn: sqlite3.Connection, canonical_id: str) -> Ask | None:
    """The ask of whichever Oracle `canonical_id` now belongs to."""
    return asks_by_head(conn).get(schema.current_canonical(conn, canonical_id))


def wake() -> bool:
    """Ask for the scrapers now, ahead of routine jobs: someone is waiting. Never raises."""
    try:
        return job_queue.request_now(registry.SCRAPERS, priority=job_queue.URGENT_PRIORITY)
    except Exception as exc:
        log(f"[asks] could not wake the scrapers: {type(exc).__name__}: {exc}")
        return False
