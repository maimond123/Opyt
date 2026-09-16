"""Durable scheduling state for the resident rail worker.

This database answers one question: which bounded rail pass should run next?  Rail cursors,
consent, freshness, budgets, and source state stay in each home's ``opyt.db``.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from opyt_core.paths import opyt_path

LOCAL_HOME_ID = "local"
DEFAULT_PRIORITY = 0
URGENT_PRIORITY = 100

# The two non-secret facts a hosted MCP child is given about the worker, and the only way it
# learns them. `OPYT_WORKER_HOME_ID` is the Google subject the gateway already validated;
# `OPYT_WORKER_DB` is the operator's one shared control database. No tool argument may set
# either, which is what stops one user's request from scheduling work against another's home.
WORKER_HOME_ID_ENV = "OPYT_WORKER_HOME_ID"
WORKER_DB_ENV = "OPYT_WORKER_DB"

_HOME_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
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
"""


@dataclass(frozen=True)
class RailJob:
    home_id: str
    rail: str
    priority: int
    due_at: float
    started_at: float | None
    finished_at: float | None
    exit_code: int | None


class WorkerDbNotConfigured(RuntimeError):
    """A hosted process has a home id but no shared control database to write it into."""


def worker_db_path() -> Path:
    """The configured control database, or the local home's control database.

    A hosted process must be told the shared database explicitly. Falling back to
    ``<OPYT_HOME>/rail_jobs.db`` there would write every queued job into the user's own home,
    where the one worker never looks: no rail would ever run, and nothing would say so. That is
    the silent total failure this whole migration exists to remove, so it is raised instead.
    """
    configured = os.environ.get(WORKER_DB_ENV)
    if configured:
        return Path(configured).expanduser()
    if os.environ.get(WORKER_HOME_ID_ENV):
        raise WorkerDbNotConfigured(
            f"{WORKER_HOME_ID_ENV} is set without {WORKER_DB_ENV}; a hosted process cannot "
            f"queue rail work into a per-home database the worker never opens")
    return opyt_path("rail_jobs.db")


def queue_is_shared() -> bool:
    """Does this process queue into a SHARED control database that a SEPARATE resident worker
    claims from, rather than into its own home's?

    This is the question "will anything ever act on a row I queue", and it is NOT the question
    "can this platform install a LaunchAgent". The two coincide only on a local mac, and
    `onboard_tools._follow_consent_with_a_worker` read the second to answer the first: on the
    hosted box — Ubuntu, where `install_worker.status()["supported"]` is False by ruling F1 —
    every remote user was told "nothing on this machine will act on it on its own" while
    `gateway/deploy/opyt-worker.service` sat beside them claiming exactly those rows.

    `WORKER_HOME_ID_ENV` is the marker because only the gateway sets it, `current_home_id` already
    reads it as "this is a hosted process", and `worker_db_path` REFUSES to run when it is set
    without the shared database beside it. So its presence means both halves are configured, and
    no local home can answer True by accident — a local user who relocates their database sets
    `WORKER_DB_ENV` alone and still gets their LaunchAgent.

    It does not prove the worker PROCESS is up: an operator who stops the unit leaves this True.
    That residual is accepted deliberately. The alternative it replaces was not an uncertainty
    but a certainty in the wrong direction, on every hosted run.
    """
    return os.environ.get(WORKER_HOME_ID_ENV) is not None


def validate_home_id(home_id: str) -> str:
    """Validate the identifier before it can become durable path-selection input."""
    if not _HOME_ID_RE.fullmatch(home_id):
        raise ValueError(f"home id is not a usable directory name: {home_id!r}")
    return home_id


def current_home_id() -> str:
    """The home id this process may schedule work for: its own, and only its own.

    Hosted, that is the subject the gateway validated before it named a directory. Locally
    there is one home and one fixed id. The value is validated again here because the local
    producer is an operator's service file rather than the gateway, so this read is a second
    trust boundary and not a re-check of the gateway's work.
    """
    configured = os.environ.get(WORKER_HOME_ID_ENV)
    return LOCAL_HOME_ID if configured is None else validate_home_id(configured)


class RailJobStore:
    """The one-row-per-home-and-rail scheduling store."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else worker_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA busy_timeout=5000")
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
                 priority: int = DEFAULT_PRIORITY) -> RailJob:
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
            row = conn.execute(
                "SELECT * FROM rail_jobs WHERE home_id=? AND rail=?", (home_id, rail)
            ).fetchone()
            return self._job(row)
        finally:
            conn.close()

    def claim_next(self, *, now: float | None = None,
                   home_id: str | None = None) -> RailJob | None:
        """Atomically claim the highest-priority earliest-due eligible job.

        ``home_id`` pins a local worker to its sole fixed home.  Hosted workers use the whole
        control database.  A claimed row for a home excludes every other rail for that home.
        """
        claimed_at = time.time() if now is None else now
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            where_home = "AND candidate.home_id = ?" if home_id is not None else ""
            params: tuple[object, ...] = ((claimed_at, home_id) if home_id is not None
                                          else (claimed_at,))
            row = conn.execute(
                f"""
                SELECT candidate.*
                  FROM rail_jobs AS candidate
                 WHERE candidate.due_at <= ?
                   AND (candidate.started_at IS NULL OR candidate.finished_at IS NOT NULL)
                   {where_home}
                   AND NOT EXISTS (
                       SELECT 1 FROM rail_jobs AS active
                        WHERE active.home_id = candidate.home_id
                          AND active.started_at IS NOT NULL
                          AND active.finished_at IS NULL
                   )
                 ORDER BY candidate.priority DESC, candidate.due_at ASC,
                          candidate.home_id ASC, candidate.rail ASC
                 LIMIT 1
                """,
                params,
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            conn.execute(
                "UPDATE rail_jobs SET started_at=?, finished_at=NULL, exit_code=NULL "
                "WHERE home_id=? AND rail=?",
                (claimed_at, row["home_id"], row["rail"]),
            )
            conn.commit()
            return RailJob(
                home_id=row["home_id"], rail=row["rail"], priority=row["priority"],
                due_at=row["due_at"], started_at=claimed_at, finished_at=None, exit_code=None,
            )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def finish(self, job: RailJob, exit_code: int, *, cadence: float,
               now: float | None = None) -> RailJob:
        """Record the child result and schedule that activated rail's next pass."""
        finished_at = time.time() if now is None else now
        conn = self._connect()
        try:
            changed = conn.execute(
                "UPDATE rail_jobs SET priority=?, due_at=?, finished_at=?, exit_code=? "
                "WHERE home_id=? AND rail=? AND started_at=? AND finished_at IS NULL",
                (DEFAULT_PRIORITY, finished_at + cadence, finished_at, exit_code,
                 job.home_id, job.rail, job.started_at),
            ).rowcount
            if changed != 1:
                raise RuntimeError(f"rail claim is no longer active: {job.home_id}/{job.rail}")
            conn.commit()
            row = conn.execute(
                "SELECT * FROM rail_jobs WHERE home_id=? AND rail=?",
                (job.home_id, job.rail),
            ).fetchone()
            return self._job(row)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def recover_interrupted_claims(self, *, now: float | None = None) -> int:
        """Make claims left by the previous worker lifetime eligible again."""
        recovered_at = time.time() if now is None else now
        conn = self._connect()
        try:
            changed = conn.execute(
                "UPDATE rail_jobs SET started_at=NULL, due_at=MIN(due_at, ?), "
                "finished_at=NULL, exit_code=NULL "
                "WHERE started_at IS NOT NULL AND finished_at IS NULL",
                (recovered_at,),
            ).rowcount
            conn.commit()
            return changed
        finally:
            conn.close()

    def next_due_at(self, *, home_id: str | None = None) -> float | None:
        """Return the next unclaimed schedule time for loop sleep calculation."""
        conn = self._connect()
        try:
            if home_id is None:
                row = conn.execute(
                    "SELECT MIN(due_at) FROM rail_jobs "
                    "WHERE started_at IS NULL OR finished_at IS NOT NULL"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT MIN(due_at) FROM rail_jobs WHERE home_id=? "
                    "AND (started_at IS NULL OR finished_at IS NOT NULL)",
                    (home_id,),
                ).fetchone()
            return None if row[0] is None else float(row[0])
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

    def list_jobs(self) -> list[RailJob]:
        conn = self._connect()
        try:
            return [self._job(row) for row in conn.execute(
                "SELECT * FROM rail_jobs ORDER BY home_id, rail"
            ).fetchall()]
        finally:
            conn.close()


def request_now(rail: str, *, priority: int = DEFAULT_PRIORITY) -> bool:
    """Make ``rail`` due immediately for this process's own home. True iff the job is durable.

    The producer half of the store. A tool that has just recorded a consent, a grant, or a claim
    calls this instead of forking the rail itself, so the request outlives the MCP session that
    made it and one resident worker owns every launch.

    Callers pass the rail's registry name from `rail_worker.RAILS`; that agreement is proven by
    each call site's own test rather than by an import, because the worker imports this module
    and the reverse direction would be a cycle.

    Fail-safe, and the log line is what makes that safe: a failed queue means the rail runs on
    its own cadence instead of now — a slower product, not a wrong one. Both real failures are
    operator mistakes no tool caller can fix mid-call: `WorkerDbNotConfigured` on a hosted deploy
    that set the home id without the control database, and an unwritable database. Swallowing
    them silently would rebuild the invisible failure `worker_db_path()` raises to prevent, so
    the reason goes to stderr and the False goes into the tool's response.
    """
    return _activate(rail, time.time(), priority)


def request_in(rail: str, seconds: float, *, priority: int = DEFAULT_PRIORITY) -> bool:
    """Make ``rail`` due ``seconds`` from now. True iff the job is durable.

    THE OTHER SENTENCE A PRODUCER CAN SAY. `request_now` means "there is work waiting"; this one
    means "this work has a REPEAT — schedule it", and the difference matters wherever the caller
    is ALREADY DOING the pass itself in-process. `sitting_tools._watchlist` is the case it was
    written for: a hand-added watch runs its own scoped first pull on a thread, so a due-now row
    would have the worker's child walk those same (query, source) pairs concurrently — and
    `frontier_execute` has no single-flight lock to stop it. One cadence out, the pairs the first
    pull stamped are no longer due and the child costs one exit code.

    It can only ever bring a schedule FORWARD or create one: `activate` keeps `MIN(due_at)`, so
    this never postpones work that is already waiting.

    Fail-safe on the same terms as `request_now`, and for the same reasons — see its docstring.
    """
    return _activate(rail, time.time() + seconds, priority)


def _activate(rail: str, due_at: float, priority: int) -> bool:
    from pipeline.ingestion.utils import log
    try:
        RailJobStore().activate(current_home_id(), rail, due_at=due_at, priority=priority)
        return True
    except Exception as exc:
        log(f"[rail_jobs] could not queue {rail}: {exc}")
        return False
