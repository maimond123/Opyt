"""
pipeline/worker/job_queue.py — every read and write of the worker's control database,
``rail_jobs.db``: the job rows (queued, claimed, finished, recovered), the worker's heartbeat, and
what producers call: ``request_now`` to ask for a job, ``enroll`` to give a home one it lacks.

Rail cursors, freshness, budgets, and source state stay in each home's ``opyt.db``.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pipeline.ingestion import utils

from opyt_core.paths import owner_only, validate_home_id

from .home import current_home_id, worker_db_path

DEFAULT_PRIORITY = 0
URGENT_PRIORITY = 100

# How far `claim_next` pushes a claimed row's `due_at` out, so a request made while the job runs
# moves it earlier (`activate` keeps `MIN(due_at)`) and `finish` can see and keep it. Nothing
# schedules from a claimed row, and `recover_interrupted_claims` clamps it back to now.
CLAIM_HOLD_SECONDS = 24 * 3600.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rail_jobs (
    home_id     TEXT NOT NULL,
    rail        TEXT NOT NULL,
    priority    INTEGER NOT NULL,
    due_at      REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    exit_code   INTEGER,
    PRIMARY KEY (home_id, rail)
);
CREATE INDEX IF NOT EXISTS rail_jobs_due
    ON rail_jobs (priority DESC, due_at ASC);
CREATE TABLE IF NOT EXISTS rail_worker_beat (
    scope      TEXT PRIMARY KEY,           -- LOCAL_HOME_ID, or SHARED_BEAT_SCOPE for a hosted pool
    worker_id  TEXT NOT NULL,              -- host:pid:uuid, same shape as sync_lock's holder id
    started_at REAL NOT NULL,              -- this lifetime's birth
    beat_at    REAL NOT NULL,
    lifetimes  INTEGER NOT NULL DEFAULT 1  -- ++ on each new worker_id: restart churn, in-product
);
"""

# The heartbeat write is throttled to BEAT_SECONDS (the loop polls every second). The stale
# window is generous because this is not a mutex: a false "stale" only under-promises briefly.
BEAT_SECONDS = 30.0
BEAT_STALE_AFTER = 180.0

# One row per worker scope: the flock in supervisor.WorkerLock guarantees one worker per
# database, so a local store holds one LOCAL_HOME_ID row and the hosted pool's shared store
# one row under this marker. The table never grows.
SHARED_BEAT_SCOPE = "shared"


@dataclass(frozen=True)
class RailJob:
    home_id: str
    rail: str
    priority: int
    due_at: float
    started_at: float | None
    finished_at: float | None
    exit_code: int | None


def _among(column: str, names: Iterable[str] | None) -> tuple[str, tuple[str, ...]]:
    """The SQL condition `column IN (?, …)` and its parameters. None limits nothing ("1"); an
    empty list matches nothing ("0")."""
    if names is None:
        return "1", ()
    names = tuple(names)
    if not names:
        return "0", ()
    return f"{column} IN ({', '.join('?' * len(names))})", names


class RailJobStore:
    """The one-row-per-home-and-rail scheduling store."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else worker_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        owner_only(self.db_path)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            # `journal_mode=WAL` needs a brief EXCLUSIVE lock, and `request_now` constructs a
            # fresh store on every producer call — on the hosted box that is every home and
            # every rail child re-taking that lock against one shared file. The mode is
            # persistent in the database, so read it first and set it only on a store that is
            # not WAL yet. The DDL below stays: IF NOT EXISTS needs only the schema table.
            if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _job(row: sqlite3.Row) -> RailJob:
        return RailJob(**dict(row))

    def activate(self, home_id: str, rail: str, *, due_at: float | None = None,
                 priority: int = DEFAULT_PRIORITY) -> None:
        """Create a durable job, or bring its existing schedule forward.

        Activation never creates a second row and never postpones already-due work.
        """
        validate_home_id(home_id)
        due = time.time() if due_at is None else due_at
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO rail_jobs(home_id, rail, priority, due_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(home_id, rail) DO UPDATE SET "
                "priority=MAX(rail_jobs.priority, excluded.priority), "
                "due_at=MIN(rail_jobs.due_at, excluded.due_at)",
                (home_id, rail, priority, due),
            )
            conn.commit()
        finally:
            conn.close()

    def enroll(self, home_id: str, rail: str, *, due_at: float | None = None) -> None:
        """Create a durable job, due now, unless the home already has it. An existing row keeps
        its schedule, so this is safe to repeat: it never brings a job forward."""
        validate_home_id(home_id)
        due = time.time() if due_at is None else due_at
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO rail_jobs(home_id, rail, priority, due_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(home_id, rail) DO NOTHING",
                (home_id, rail, DEFAULT_PRIORITY, due),
            )
            conn.commit()
        finally:
            conn.close()

    def claim_next(self, *, now: float | None = None, home_id: str | None = None,
                   rails: Iterable[str] | None = None,
                   beside: Iterable[str] = ()) -> RailJob | None:
        """Atomically claim the highest-priority earliest-due eligible job.

        ``home_id`` pins a local worker to its sole fixed home.  Hosted workers use the whole
        control database.  A claimed row for a home excludes every other rail for that home,
        except those ``beside`` (the registry's `RailSpec.beside` jobs), which neither wait for
        another job nor make one wait.

        ``rails`` limits the claim to those names. The worker passes the jobs it knows, so a
        row it could not launch is never claimed.

        The claimed row's ``due_at`` moves `CLAIM_HOLD_SECONDS` out, and the returned job
        carries that value, which is what `finish` compares against.
        """
        claimed_at = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            where_home = "AND candidate.home_id = ?" if home_id is not None else ""
            in_rails, rail_params = _among("candidate.rail", rails)
            beside = tuple(beside)
            runs_beside, _ = _among("candidate.rail", beside)
            active_beside, _ = _among("active.rail", beside)
            params: tuple[object, ...] = (
                claimed_at, *((home_id,) if home_id is not None else ()), *rail_params,
                *beside, *beside)
            row = conn.execute(
                f"""
                SELECT candidate.*
                  FROM rail_jobs AS candidate
                 WHERE candidate.due_at <= ?
                   AND (candidate.started_at IS NULL OR candidate.finished_at IS NOT NULL)
                   {where_home}
                   AND {in_rails}
                   AND ({runs_beside} OR NOT EXISTS (
                       SELECT 1 FROM rail_jobs AS active
                        WHERE active.home_id = candidate.home_id
                          AND NOT ({active_beside})
                          AND active.started_at IS NOT NULL
                          AND active.finished_at IS NULL
                   ))
                 ORDER BY candidate.priority DESC, candidate.due_at ASC,
                          candidate.home_id ASC, candidate.rail ASC
                 LIMIT 1
                """,
                params,
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            held = claimed_at + CLAIM_HOLD_SECONDS
            conn.execute(
                "UPDATE rail_jobs SET started_at=?, due_at=?, finished_at=NULL, exit_code=NULL "
                "WHERE home_id=? AND rail=?",
                (claimed_at, held, row["home_id"], row["rail"]),
            )
            conn.commit()
            return RailJob(
                home_id=row["home_id"], rail=row["rail"], priority=row["priority"],
                due_at=held, started_at=claimed_at, finished_at=None, exit_code=None,
            )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def finish(self, job: RailJob, exit_code: int, *, cadence: float,
               now: float | None = None) -> None:
        """Record the child result and schedule that activated rail's next pass.

        The next pass is ``cadence`` from now, unless the row's ``due_at`` moved while the job
        ran: then it is the earlier of the two. A move is a request made during the run (the
        claim pushed ``due_at`` out so a request can show), or a wake the job wrote for itself
        (`SCRAPERS` does)."""
        finished_at = time.time() if now is None else now
        conn = self._connect()
        try:
            changed = conn.execute(
                "UPDATE rail_jobs SET priority=?, finished_at=?, exit_code=?, "
                "due_at=CASE WHEN due_at != ? THEN MIN(due_at, ?) ELSE ? END "
                "WHERE home_id=? AND rail=? AND started_at=? AND finished_at IS NULL",
                (DEFAULT_PRIORITY, finished_at, exit_code,
                 job.due_at, finished_at + cadence, finished_at + cadence,
                 job.home_id, job.rail, job.started_at),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"rail claim is no longer active: {job.home_id}/{job.rail}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def drop_retired_jobs(self, *, rails: Iterable[str]) -> None:
        """Delete every row whose job is not among ``rails`` (the registry's), even a claimed one:
        left claimed, it would block every other job for its home. A retired job's row holds no
        work, only a wake-up time."""
        in_rails, params = _among("rail", rails)
        conn = self._connect()
        try:
            conn.execute(f"DELETE FROM rail_jobs WHERE NOT ({in_rails})", params)
            conn.commit()
        finally:
            conn.close()

    def recover_interrupted_claims(self, *, now: float | None = None) -> None:
        """Make claims left by the previous worker lifetime eligible again."""
        recovered_at = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE rail_jobs SET started_at=NULL, due_at=MIN(due_at, ?), "
                "finished_at=NULL, exit_code=NULL "
                "WHERE started_at IS NOT NULL AND finished_at IS NULL",
                (recovered_at,),
            )
            conn.commit()
        finally:
            conn.close()

    def next_due_at(self, *, home_id: str | None = None,
                    rails: Iterable[str] | None = None) -> float | None:
        """Return the next unclaimed schedule time for loop sleep calculation."""
        in_rails, rail_params = _among("rail", rails)
        where_home, home_params = ("AND home_id=?", (home_id,)) if home_id is not None else ("", ())
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT MIN(due_at) FROM rail_jobs "
                f"WHERE (started_at IS NULL OR finished_at IS NOT NULL) {where_home} "
                f"AND {in_rails}",
                (*home_params, *rail_params),
            ).fetchone()
            return None if row[0] is None else float(row[0])
        finally:
            conn.close()

    def rearm(self, home_id: str, rail: str, due_at: float, *, expected: float) -> bool:
        """Set a running job's own ``due_at`` to ``due_at``, only if it still reads ``expected``.

        For `SCRAPERS`, which reads its row about once a second to see requests made while it
        runs. A plain `activate` cannot re-arm it, because `MIN(due_at)` only ever moves earlier;
        and an unguarded write could land just after a request and erase it. False means the
        row moved since the caller read it: someone asked, and the caller should look again."""
        conn = self._connect()
        try:
            changed = conn.execute(
                "UPDATE rail_jobs SET due_at=? WHERE home_id=? AND rail=? AND due_at=?",
                (due_at, home_id, rail, expected),
            ).rowcount
            conn.commit()
            return changed == 1
        finally:
            conn.close()

    def get(self, home_id: str, rail: str) -> RailJob | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM rail_jobs WHERE home_id=? AND rail=?", (home_id, rail)
            ).fetchone()
            return None if row is None else self._job(row)
        finally:
            conn.close()

    def beat(self, scope: str, worker_id: str, *, started_at: float,
             now: float | None = None) -> None:
        """Stamp this worker lifetime's liveness row.

        A new ``worker_id`` on an existing scope is a new lifetime: ``lifetimes`` increments
        and ``started_at`` resets, so restart churn is readable from the store.
        """
        beat_at = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO rail_worker_beat(scope, worker_id, started_at, beat_at, lifetimes) "
                "VALUES (?, ?, ?, ?, 1) "
                "ON CONFLICT(scope) DO UPDATE SET "
                "lifetimes=rail_worker_beat.lifetimes"
                "  + (rail_worker_beat.worker_id != excluded.worker_id), "
                "started_at=CASE WHEN rail_worker_beat.worker_id != excluded.worker_id "
                "  THEN excluded.started_at ELSE rail_worker_beat.started_at END, "
                "worker_id=excluded.worker_id, "
                "beat_at=excluded.beat_at",
                (scope, worker_id, started_at, beat_at),
            )
            conn.commit()
        finally:
            conn.close()

    def list_jobs(self) -> list[RailJob]:
        """Every home's rows, for the operator's checks and the tests."""
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM rail_jobs ORDER BY home_id, rail")
            return [self._job(row) for row in rows.fetchall()]
        finally:
            conn.close()


def worker_is_alive(*, now: float | None = None) -> bool | None:
    """Has anything been claiming from this process's control database recently? Works on
    local and hosted alike.

    True: a beat inside ``BEAT_STALE_AFTER``. False: stale or absent, including a freshly
    deployed hosted box before the worker's first beat (a fail-safe under-promise). None: the
    store exists but cannot be read.
    """
    checked_at = time.time() if now is None else now
    try:
        path = worker_db_path()
        if not path.exists():
            return False
        # Read-only on purpose: a probe must not create the database or take write locks.
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            row = conn.execute("SELECT MAX(beat_at) FROM rail_worker_beat").fetchone()
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return False  # a store no worker lifetime has ever touched
        return None
    except Exception:
        return None
    if row is None or row[0] is None:
        return False
    return (checked_at - float(row[0])) <= BEAT_STALE_AFTER


def request_now(rail: str, *, priority: int = DEFAULT_PRIORITY) -> bool:
    """Make ``rail`` due immediately for this process's own home. True iff the job is durable.

    The producer half of the store: a tool calls this instead of forking the rail itself, so the
    request outlives the MCP session and one resident worker owns every launch. Callers pass a
    name from `registry.RAILS`.

    Fail-safe: a failed queue (an unconfigured hosted control database, an unwritable one) means
    the rail runs on its own cadence instead of now. The reason is logged and False returned.
    """
    try:
        RailJobStore().activate(current_home_id(), rail, due_at=time.time(), priority=priority)
        return True
    except Exception as exc:
        utils.log(f"[rail_jobs] could not queue {rail}: {exc}")
        return False


def enroll(rail: str) -> bool:
    """Give this process's own home ``rail``, due now, unless it already has it. True iff the job
    is durable.

    For a job that keeps its own cadence once it has a row (`registry.LINE`), asked for by
    something that recurs: a repeat leaves the schedule alone, where `request_now` would run the
    job again every time. Fail-safe like `request_now`: a failed write is logged, and the next
    repeat tries again.
    """
    try:
        RailJobStore().enroll(current_home_id(), rail)
        return True
    except Exception as exc:
        utils.log(f"[rail_jobs] could not enroll {rail}: {exc}")
        return False
