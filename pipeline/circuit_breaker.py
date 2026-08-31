"""
pipeline/circuit_breaker.py

A persisted circuit breaker for the paid external APIs (OpenRouter). Trips
on repeated failures to stop a retry storm from billing a down/degraded API further.

Three states: CLOSED (normal, counts consecutive failures, trips to OPEN at threshold),
OPEN (reject instantly, no call, no bill; moves to HALF_OPEN after cooldown), HALF_OPEN
(one trial call: success → CLOSED, failure → OPEN).

State is persisted per-service in opyt.db, not in-memory, so every session shares one
outage view. Connections are opened per-operation rather than held on the instance, so a
module-level breaker can be shared across threads (SQLite connections are thread-bound).
"""

import sqlite3
import time
from contextlib import closing
from pathlib import Path

from pipeline.sqlite_db import default_db_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS circuit_breaker (
    service      TEXT PRIMARY KEY,
    state        TEXT NOT NULL DEFAULT 'closed',  -- closed | open | half_open
    fail_count   INTEGER NOT NULL DEFAULT 0,
    opened_at    REAL,            -- unix seconds when it tripped (NULL when closed)
    last_failure TEXT             -- last error detail, surfaced by status()
);
"""

DEFAULT_THRESHOLD = 5     # consecutive failures before tripping
DEFAULT_COOLDOWN = 60.0   # seconds OPEN before a HALF_OPEN trial


class CircuitOpenError(Exception):
    """Raised instead of making the call when the breaker is OPEN (fail fast)."""

    def __init__(self, service: str, retry_after: float):
        self.service = service
        self.retry_after = retry_after
        super().__init__(
            f"circuit '{service}' is OPEN — skipping call (retry in ~{retry_after:.0f}s)"
        )


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


class CircuitBreaker:
    """One breaker per external service, keyed by name in the circuit_breaker table.

    Safe to construct once and share (incl. across threads): every method opens and
    closes its own connection, so nothing is bound to the constructing thread.
    """

    def __init__(self, service: str, threshold: int = DEFAULT_THRESHOLD,
                 cooldown: float = DEFAULT_COOLDOWN, db_path: Path | None = None):
        self.service = service
        self.threshold = threshold
        self.cooldown = cooldown
        self._db_path = Path(db_path) if db_path else default_db_path()
        with closing(_connect(self._db_path)) as conn:
            conn.execute("INSERT OR IGNORE INTO circuit_breaker(service) VALUES (?)", (self.service,))
            conn.commit()

    # ── decision ──────────────────────────────────────────────────────────────

    def allow(self) -> bool:
        """May a call go through right now? Advances OPEN → HALF_OPEN once the
        cooldown has elapsed (the returned True is then the single trial)."""
        now = time.time()
        with closing(_connect(self._db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                state, opened_at = conn.execute(
                    "SELECT state, opened_at FROM circuit_breaker WHERE service=?", (self.service,)
                ).fetchone()
                if state == "open":
                    if opened_at is not None and (now - opened_at) >= self.cooldown:
                        conn.execute(
                            "UPDATE circuit_breaker SET state='half_open' WHERE service=?",
                            (self.service,),
                        )
                        conn.commit()
                        return True  # the one trial call
                    conn.commit()
                    return False
                conn.commit()
                return True  # closed or half_open
            except Exception:
                conn.rollback()
                raise

    def retry_after(self) -> float:
        """Seconds until the OPEN breaker will permit a trial (0 if not open)."""
        with closing(_connect(self._db_path)) as conn:
            row = conn.execute(
                "SELECT state, opened_at FROM circuit_breaker WHERE service=?", (self.service,)
            ).fetchone()
        if not row or row[0] != "open" or row[1] is None:
            return 0.0
        return max(0.0, self.cooldown - (time.time() - row[1]))

    # ── outcome recording ─────────────────────────────────────────────────────

    def record_success(self) -> None:
        """A call worked → reset to CLOSED."""
        with closing(_connect(self._db_path)) as conn:
            conn.execute(
                "UPDATE circuit_breaker SET state='closed', fail_count=0, "
                "opened_at=NULL, last_failure=NULL WHERE service=?",
                (self.service,),
            )
            conn.commit()

    def record_failure(self, detail: str | None = None) -> None:
        """A call failed. In HALF_OPEN the trial failing re-opens immediately;
        in CLOSED we trip once consecutive failures reach the threshold."""
        now = time.time()
        with closing(_connect(self._db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                state, fail_count = conn.execute(
                    "SELECT state, fail_count FROM circuit_breaker WHERE service=?", (self.service,)
                ).fetchone()
                if state == "half_open":
                    conn.execute(
                        "UPDATE circuit_breaker SET state='open', opened_at=?, last_failure=? "
                        "WHERE service=?",
                        (now, detail, self.service),
                    )
                else:
                    fail_count += 1
                    if fail_count >= self.threshold:
                        conn.execute(
                            "UPDATE circuit_breaker SET state='open', fail_count=?, opened_at=?, "
                            "last_failure=? WHERE service=?",
                            (fail_count, now, detail, self.service),
                        )
                    else:
                        conn.execute(
                            "UPDATE circuit_breaker SET fail_count=?, last_failure=? WHERE service=?",
                            (fail_count, detail, self.service),
                        )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    # ── wrapper ───────────────────────────────────────────────────────────────

    def call(self, fn, *, ignore: tuple = ()):
        """Run fn() under the breaker.

        OPEN → raise CircuitOpenError without calling. Otherwise call fn; a success
        resets the breaker, an exception records a failure and re-raises. Exceptions
        in ``ignore`` are re-raised WITHOUT counting (model expected non-failures —
        a 404 "item deleted" — as ignored so they don't trip the breaker).
        """
        if not self.allow():
            raise CircuitOpenError(self.service, self.retry_after())
        try:
            result = fn()
        except ignore:
            raise  # expected, not a breaker failure
        except Exception as e:
            self.record_failure(f"{type(e).__name__}: {e}")
            raise
        self.record_success()
        return result


def status(db_path: Path | None = None) -> list[dict]:
    """Snapshot every breaker. One live reader: `pipeline/kb/oracle_refresh.py`, which skips
    sources whose breaker is open rather than paying a call it knows will fail."""
    with closing(_connect(Path(db_path) if db_path else default_db_path())) as conn:
        rows = conn.execute(
            "SELECT service, state, fail_count, opened_at, last_failure FROM circuit_breaker"
        ).fetchall()
    now = time.time()
    return [
        {"service": s, "state": st, "fail_count": fc,
         "open_for_s": (now - oa) if (st == "open" and oa is not None) else None,
         "last_failure": lf}
        for (s, st, fc, oa, lf) in rows
    ]
