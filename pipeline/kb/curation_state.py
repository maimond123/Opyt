"""
pipeline/kb/curation_state.py — per-collector run clock for the curation candidate list.

One row per curation collector. `last_attempt_at` advances on any outcome and drives the retry
floor; `last_ok_at` advances only on success and drives staleness reporting. Mirrors
`oracle_refresh_state.py` in shape (same DB, own DDL, `schema.connect` layering) but keeps no
breaker state and no adaptive cadence. Nothing here touches the network or the wall clock except
via the `now` argument, so callers can test without sleeping. Full design rationale:
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from pipeline.timeparse import parse_ts, utc_now

from . import schema

# Hours a collector may go without a successful run before it is reported stale. Well above the
# 6h retry floor so a collector mid-cadence isn't flagged — only one that is failing or never ran.
STALE_AFTER_HOURS = 48.0

# The one status string that advances `last_ok_at`. Every other value — 'error', 'skipped_tier',
# 'no_viewer_id' — is an attempt that observed nothing.
STATUS_OK = "ok"

_DDL = """
CREATE TABLE IF NOT EXISTS collector_runs (
  collector       TEXT PRIMARY KEY,  -- 'x_lists' | 'x_following' | 'x_likes' | 'substack_subs'
  last_attempt_at TEXT,              -- ANY outcome  → drives the FLOOR
  last_ok_at      TEXT,              -- SUCCESS only → drives STALENESS
  last_status     TEXT,              -- 'ok' | 'skipped_tier' | 'error' | 'no_viewer_id'
  last_detail     TEXT,
  found           INTEGER,           -- what the COLLECTOR said it saw, as of `last_ok_at`
  stored_after    INTEGER,           -- rows the STORE holds for this signal_type+platform, ditto
  prev_found      INTEGER,           -- the PREVIOUS ok run's `found` — the collapse guard's input
  started_at      TEXT               -- when the walk BEGAN — see `record_run`
);
"""

# Ratio of found vs. the previous run's found below which a walk is treated as broken rather than
# the list having shrunk (absence from a walk retires a signal). Compared run-over-run, not against
# `stored_after`, so the check doesn't decay as unfollows accumulate over time.
WALK_COLLAPSE_RATIO = 0.5


def _now() -> str:
    # Microsecond precision, kept deliberately: `collector_runs` already stores stamps at this
    # width, and narrowing would break sort order against existing rows.
    return utc_now().isoformat()


# ── the row ─────────────────────────────────────────────────────────────────────
@dataclass
class CollectorRun:
    collector: str
    last_attempt_at: str | None = None
    last_ok_at: str | None = None
    last_status: str | None = None
    last_detail: str | None = None
    found: int | None = None
    stored_after: int | None = None
    prev_found: int | None = None
    started_at: str | None = None

    @property
    def ok(self) -> bool:
        return self.last_status == STATUS_OK


def _row_to_run(row: sqlite3.Row) -> CollectorRun:
    return CollectorRun(
        collector=row["collector"],
        last_attempt_at=row["last_attempt_at"],
        last_ok_at=row["last_ok_at"],
        last_status=row["last_status"],
        last_detail=row["last_detail"],
        found=row["found"],
        stored_after=row["stored_after"],
        prev_found=row["prev_found"],
        started_at=row["started_at"],
    )


# ── connection + schema ─────────────────────────────────────────────────────────
def init_state_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL. Safe on every writable open, and called by every public writer here — a
    caller may hand us a plain `schema.connect()` that has never seen this table.

    `CREATE TABLE IF NOT EXISTS` does NOT add a column to a table that already exists, so the two
    columns added after the first release go through `_ensure_column` as well."""
    conn.executescript(_DDL)
    schema._ensure_column(conn, "collector_runs", "prev_found", "INTEGER")
    schema._ensure_column(conn, "collector_runs", "started_at", "TEXT")
    conn.commit()


def connect(db_path=None, *, read_only: bool = False) -> sqlite3.Connection:
    """The atom-KB store with `collector_runs` guaranteed present. Reuses `schema.connect`
    (WAL + busy_timeout + row_factory + `$OPYT_HOME`) and layers this table on top. Read-only
    opens skip DDL, matching `schema.connect`'s contract."""
    conn = schema.connect(db_path, read_only=read_only)
    if not read_only:
        init_state_schema(conn)
    return conn


# ── persistence ─────────────────────────────────────────────────────────────────
def record_run(conn: sqlite3.Connection, collector: str, *, status: str,
               detail: str | None = None, found: int | None = None,
               stored_after: int | None = None, now: str | None = None,
               started_at: str | None = None) -> None:
    """Persist ONE collector's outcome. The only writer of this table.

    `last_attempt_at` advances on every outcome so the floor counts failed attempts too.
    `last_ok_at` advances only on `ok`. `started_at` marks when the walk began; retirement compares
    against that instead of the finish stamp. Counts (`found`/`stored_after`) coalesce rather than
    overwrite, so a failure doesn't blank the last good reading. Full rationale:
    """
    init_state_schema(conn)
    stamp = now or _now()
    # `prev_found` carries the value `found` is about to overwrite — one step of history, which is
    # all the collapse guard needs. Only an `ok` run shifts it.
    conn.execute(
        "INSERT INTO collector_runs "
        "(collector, last_attempt_at, last_ok_at, last_status, last_detail, found, stored_after, "
        " started_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(collector) DO UPDATE SET "
        "  last_attempt_at=excluded.last_attempt_at, "
        "  started_at=COALESCE(excluded.started_at, collector_runs.started_at), "
        "  last_ok_at=COALESCE(excluded.last_ok_at, collector_runs.last_ok_at), "
        "  last_status=excluded.last_status, "
        "  last_detail=excluded.last_detail, "
        "  prev_found=CASE WHEN excluded.found IS NOT NULL THEN collector_runs.found "
        "                  ELSE collector_runs.prev_found END, "
        "  found=COALESCE(excluded.found, collector_runs.found), "
        "  stored_after=COALESCE(excluded.stored_after, collector_runs.stored_after)",
        (collector, stamp, stamp if status == STATUS_OK else None, status, detail,
         None if found is None else int(found),
         None if stored_after is None else int(stored_after),
         started_at if status == STATUS_OK else None),
    )
    conn.commit()


def list_runs(conn: sqlite3.Connection) -> list[CollectorRun]:
    init_state_schema(conn)
    return [_row_to_run(r) for r in
            conn.execute("SELECT * FROM collector_runs ORDER BY collector")]


def get_run(conn: sqlite3.Connection, collector: str) -> CollectorRun | None:
    """This collector's row, or None if it has never run. None is a real state, not an error —
    `is_due` treats it as due and `hours_since_ok` treats it as infinitely stale."""
    init_state_schema(conn)
    row = conn.execute("SELECT * FROM collector_runs WHERE collector=?", (collector,)).fetchone()
    return _row_to_run(row) if row else None


# ── pure clock math ─────────────────────────────────────────────────────────────
def _hours_since(stamp: str | None, now: datetime | None) -> float:
    """Hours since an ISO stamp. A missing or unparseable stamp is INFINITE, so it sorts first and
    reads as "we have no evidence this ever happened" — matching
    `oracle_refresh_state.staleness_hours`."""
    parsed = parse_ts(stamp)
    if parsed is None:
        return float("inf")
    return ((now or utc_now()) - parsed).total_seconds() / 3600.0


def hours_since_attempt(row: CollectorRun | None, now: datetime | None = None) -> float:
    """How long since we last TRIED this collector, whatever the outcome. Drives the floor."""
    return float("inf") if row is None else _hours_since(row.last_attempt_at, now)


def hours_since_ok(row: CollectorRun | None, now: datetime | None = None) -> float:
    """How long since this collector last actually SAW its list. Drives staleness."""
    return float("inf") if row is None else _hours_since(row.last_ok_at, now)


def is_due(row: CollectorRun | None, *, floor_hours: float,
           now: datetime | None = None) -> bool:
    """Is this collector allowed to run again? A never-run collector is always due.

    Reads `last_attempt_at`, NOT `last_ok_at` — the floor's job is to stop a collector from being
    re-run every few minutes, and a collector that fails every time still costs a request each time
    it is asked. Gating on success would remove the floor from exactly the collector that most
    needs one."""
    return hours_since_attempt(row, now) >= floor_hours


def is_stale(row: CollectorRun | None, *, stale_after_hours: float = STALE_AFTER_HOURS,
             now: datetime | None = None) -> bool:
    """Has this collector's slice of the candidate list gone unrefreshed too long? Never-run counts
    as stale — no evidence is the worst evidence, and it is the invisible-freeze case."""
    return hours_since_ok(row, now) >= stale_after_hours


# ── read-only report ────────────────────────────────────────────────────────────
def status_summary(conn: sqlite3.Connection, collectors,
                   *, stale_after_hours: float = STALE_AFTER_HOURS,
                   now: datetime | None = None) -> dict:
    """A read-only freshness snapshot over the NAMED collectors. All derived; writes nothing.

    Driven by the caller's collector list rather than the stored rows, so a collector that has
    NEVER run (and has no row) still gets reported. `needs_attention` is computed here so callers
    don't re-derive what "stale" means differently at each call site.
    """
    now = now or utc_now()
    stored = {r.collector: r for r in list_runs(conn)}
    names = list(collectors)

    entries = []
    stale = failing = never = 0
    for name in names:
        row = stored.get(name)
        since_ok = hours_since_ok(row, now)
        row_stale = since_ok >= stale_after_hours
        row_failing = row is not None and row.last_status not in (None, STATUS_OK)
        stale += 1 if row_stale else 0
        failing += 1 if row_failing else 0
        never += 1 if row is None or row.last_ok_at is None else 0
        entries.append({
            "collector": name,
            "last_ok_at": row.last_ok_at if row else None,
            "last_attempt_at": row.last_attempt_at if row else None,
            "last_status": row.last_status if row else None,
            "last_detail": row.last_detail if row else None,
            "hours_since_ok": None if since_ok == float("inf") else round(since_ok, 1),
            "never_ran": row is None,
            "stale": row_stale,
            "found": row.found if row else None,
            "stored_after": row.stored_after if row else None,
        })

    ok_stamps = [e["last_ok_at"] for e in entries if e["last_ok_at"]]
    return {
        "collectors": entries,
        "tracked": len(names),
        "stale_collectors": stale,
        "failing_collectors": failing,
        "never_succeeded": never,
        "stale_after_hours": stale_after_hours,
        "oldest_ok_at": min(ok_stamps) if ok_stamps else None,
        "newest_ok_at": max(ok_stamps) if ok_stamps else None,
        "needs_attention": bool(stale or failing),
    }


def walk_is_trustworthy(row: CollectorRun | None, *,
                        collapse_ratio: float = WALK_COLLAPSE_RATIO) -> bool:
    """May we treat ABSENCE from this collector's last walk as evidence?

    Requires the run to have reported `ok` (a skip/error observed nothing) and `found` to not have
    collapsed against the previous run's `found` — a truncated walk looks like a mass unfollow.
    False is the fail-safe answer for every uncertain case: never run, no success, no prior reading.
    """
    if row is None or not row.ok or not row.last_ok_at:
        return False
    if row.found is None or row.prev_found is None:
        # No baseline yet — refuse so retirement begins on the second successful run, not the first.
        return False
    if row.prev_found <= 0:
        return True
    return row.found >= row.prev_found * collapse_ratio
