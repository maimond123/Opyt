"""
pipeline/curation/clock.py — per-collector run clock for the curation candidate list.

One row per curation collector. `last_attempt_at` advances on any outcome and drives the retry
floor; `last_ok_at` advances only on success and drives staleness reporting. Mirrors
`registry.py` in shape (same DB, own DDL, `schema.connect` layering) but keeps no
breaker state and no adaptive cadence. Nothing here touches the network. The clock math takes
`now` as an argument, so callers can test without sleeping; a function called without it
falls back to the wall clock.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from pipeline.kb.writes import write_tx
from pipeline.timeparse import parse_ts, utc_now


# How long a collector waits after its last attempt before its next walk starts (`due_at`), and
# how often the saved lists may publish counts (`is_due`). Gates on `last_attempt_at`, not
# `last_ok_at`, so a collector that keeps failing does not walk on every turn.
FLOOR_HOURS = 6.0

# Hours a collector may go without a successful run before it is reported stale. Well above the
# 6h retry floor so a collector mid-cadence isn't flagged — only one that is failing or never ran.
STALE_AFTER_HOURS = 48.0

# The one status string that advances `last_ok_at`. Every other value (the recommendations
# read's 'partial'; 'error' and 'no_viewer_id' on older stores) is an attempt that did not.
STATUS_OK = "ok"

_DDL = """
CREATE TABLE IF NOT EXISTS collector_runs (
  collector       TEXT PRIMARY KEY,  -- 'x_lists' | 'x_following' | 'x_likes' | 'x_bookmark_signals'
                                     --   | 'substack_follows' | 'substack_subscriptions'
                                     --   | 'substack_saved_signals' | 'substack_recommendations'
  last_attempt_at TEXT,              -- ANY outcome  → drives the FLOOR
  last_ok_at      TEXT,              -- SUCCESS only → drives STALENESS
  last_status     TEXT,              -- 'ok' | 'partial' (free-form; older stores also hold
                                     --   'error' and 'no_viewer_id'). Only 'ok' has meaning
  last_detail     TEXT,
  found           INTEGER,           -- what the COLLECTOR said it saw, as of `last_ok_at`
  stored_after    INTEGER,           -- rows the STORE holds for this signal_type+platform, ditto
  prev_found      INTEGER,           -- the last ACCEPTED `found` — the collapse guard's baseline
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
    caller may hand us a plain `schema.connect()` that has never seen this table."""
    conn.executescript(_DDL)
    conn.commit()


# ── persistence ─────────────────────────────────────────────────────────────────
def record_run(conn: sqlite3.Connection, collector: str, *, status: str,
               detail: str | None = None, found: int | None = None,
               stored_after: int | None = None, now: str | None = None,
               started_at: str | None = None) -> None:
    """Persist one collector's outcome. The only writer of this table.

    `last_attempt_at` advances on every outcome so the floor counts failed attempts too.
    `last_ok_at` advances only on `ok`. `started_at` marks when the walk began; retirement compares
    against that instead of the finish stamp. Counts (`found`/`stored_after`) coalesce rather than
    overwrite, so a failure doesn't blank the last good reading.
    """
    init_state_schema(conn)
    stamp = now or _now()
    with write_tx(conn, site="clock.record_run"):
        prior = conn.execute("SELECT found, prev_found FROM collector_runs WHERE collector=?",
                             (collector,)).fetchone()
        conn.execute(
            "INSERT INTO collector_runs "
            "(collector, last_attempt_at, last_ok_at, last_status, last_detail, found, "
            " stored_after, started_at, prev_found) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(collector) DO UPDATE SET "
            "  last_attempt_at=excluded.last_attempt_at, "
            "  started_at=COALESCE(excluded.started_at, collector_runs.started_at), "
            "  last_ok_at=COALESCE(excluded.last_ok_at, collector_runs.last_ok_at), "
            "  last_status=excluded.last_status, "
            "  last_detail=excluded.last_detail, "
            "  prev_found=excluded.prev_found, "
            "  found=COALESCE(excluded.found, collector_runs.found), "
            "  stored_after=COALESCE(excluded.stored_after, collector_runs.stored_after)",
            (collector, stamp, stamp if status == STATUS_OK else None, status, detail,
             None if found is None else int(found),
             None if stored_after is None else int(stored_after),
             started_at if status == STATUS_OK else None,
             next_baseline(prior, found)),
        )


def held(found: int | None, prev_found: int | None, *,
         collapse_ratio: float = WALK_COLLAPSE_RATIO) -> bool:
    """Did a walk's `found` hold against the accepted baseline, rather than collapse (deciding)?
    No baseline holds; a baseline with no reading beside it does not."""
    if prev_found is None:
        return True
    if found is None:
        return False
    return found >= prev_found * collapse_ratio


def next_baseline(prior: tuple | None, found: int | None) -> int | None:
    """The `prev_found` a run leaves (deciding). `prior` is the row's `(found, prev_found)` before
    this run. `prev_found` is the last accepted baseline: the previous reading becomes it only if
    it held, so a second truncated walk cannot make itself look trustworthy. A run with no reading
    leaves the baseline alone."""
    if prior is None:
        return None
    last_found, prev_found = prior
    if found is None:
        return prev_found
    return last_found if held(last_found, prev_found) else prev_found


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
    """Hours since an ISO stamp; a missing or unparseable stamp is infinite, so it sorts first."""
    parsed = parse_ts(stamp)
    if parsed is None:
        return float("inf")
    return ((now or utc_now()) - parsed).total_seconds() / 3600.0


def hours_since_ok(row: CollectorRun | None, now: datetime | None = None) -> float:
    """How long since this collector last actually saw its list. Drives staleness."""
    return float("inf") if row is None else _hours_since(row.last_ok_at, now)


def due_at(row: CollectorRun | None, *, floor_hours: float) -> datetime | None:
    """When this collector may run again; None for one that never ran or has no readable
    attempt, which is due now."""
    last = parse_ts(row.last_attempt_at) if row is not None else None
    return None if last is None else last + timedelta(hours=floor_hours)


def is_due(row: CollectorRun | None, *, floor_hours: float,
           now: datetime | None = None) -> bool:
    """Is this collector allowed to run again? A never-run collector is always due. Gates on
    `last_attempt_at`, so a collector that keeps failing still waits out the floor."""
    at = due_at(row, floor_hours=floor_hours)
    return at is None or (now or utc_now()) >= at


# ── read-only report ────────────────────────────────────────────────────────────
def status_summary(conn: sqlite3.Connection, collectors,
                   *, stale_after_hours: float = STALE_AFTER_HOURS,
                   now: datetime | None = None) -> dict:
    """A read-only freshness snapshot over the named collectors. Driven by the caller's list, so a
    collector that never ran (and has no row) is still reported."""
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
    """May we treat absence from this collector's last walk as evidence?

    Requires the run to have reported `ok` (a skip/error observed nothing) and `found` to not have
    collapsed against the previous run's `found` — a truncated walk looks like a mass unfollow.
    False is the fail-safe answer for every uncertain case: never run, no success, no prior reading.
    """
    if row is None or not row.ok or not row.last_ok_at:
        return False
    if row.found is None or row.prev_found is None:
        # No baseline yet — refuse so retirement begins on the second successful run, not the first.
        return False
    return held(row.found, row.prev_found, collapse_ratio=collapse_ratio)
