"""
pipeline/circuit_breaker.py — a persisted circuit breaker for external APIs.

Trips on repeated failures to stop a retry storm against a down API. CLOSED counts consecutive
failures and trips to OPEN at the threshold; OPEN rejects until the cooldown passes, then allows
one HALF_OPEN trial (success → CLOSED, failure → OPEN).

State is persisted per service so every session shares one outage view, in its own file rather
than `opyt.db`: each decision takes `BEGIN IMMEDIATE`, which must not compete with ingestion's
long write transactions. Connections are opened per operation, so one breaker can be shared
across threads.
"""

import sqlite3
import time
from contextlib import closing
from pathlib import Path

from opyt_core.paths import opyt_path, owner_only

_SCHEMA = """
CREATE TABLE IF NOT EXISTS circuit_breaker (
    service      TEXT PRIMARY KEY,
    state        TEXT NOT NULL DEFAULT 'closed',  -- closed | open | half_open
    fail_count   INTEGER NOT NULL DEFAULT 0,
    opened_at    REAL,            -- unix seconds when it tripped (NULL when closed)
    last_failure TEXT             -- last error detail, surfaced by status()
);
"""


def breaker_db_path() -> Path:
    """`<home>/circuit_breaker.db`, resolved at call time: one hosted process serves many homes."""
    return opyt_path("circuit_breaker.db")


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
    # `sqlite3.connect` raises on a missing parent directory, so create the home first.
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    owner_only(db_path)
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
        self._db_path = Path(db_path) if db_path else breaker_db_path()
        with closing(_connect(self._db_path)) as conn:
            conn.execute("INSERT OR IGNORE INTO circuit_breaker(service) VALUES (?)", (self.service,))
            conn.commit()

    # ── decision ──────────────────────────────────────────────────────────────

    def allow(self) -> bool:
        """Claim the right to call right now. Mutates: advances OPEN → HALF_OPEN once the
        cooldown has elapsed, and the returned True is that single trial.

        Only `call` should invoke this; a pre-check must use `peek`, or the trial is claimed
        twice and no outcome is ever recorded. A HALF_OPEN older than one cooldown was abandoned
        and is re-granted; re-stamping `opened_at` keeps the re-grant single."""
        now = time.time()
        with closing(_connect(self._db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                state, opened_at = conn.execute(
                    "SELECT state, opened_at FROM circuit_breaker WHERE service=?", (self.service,)
                ).fetchone()
                if state in ("open", "half_open"):
                    if opened_at is not None and (now - opened_at) >= self.cooldown:
                        conn.execute(
                            "UPDATE circuit_breaker SET state='half_open', opened_at=? "
                            "WHERE service=?", (now, self.service),
                        )
                        conn.commit()
                        return True  # the one trial call
                    conn.commit()
                    return False
                conn.commit()
                return state == "closed"
            except Exception:
                conn.rollback()
                raise

    def peek(self) -> bool:
        """Would a call be permitted right now? Read-only; what a pre-check must use."""
        now = time.time()
        with closing(_connect(self._db_path)) as conn:
            row = conn.execute(
                "SELECT state, opened_at FROM circuit_breaker WHERE service=?", (self.service,)
            ).fetchone()
        if not row:
            return True
        state, opened_at = row
        if state == "closed":
            return True
        return opened_at is not None and (now - opened_at) >= self.cooldown

    def retry_after(self) -> float:
        """Seconds until a non-CLOSED breaker will permit a trial (0 when one is available now).
        HALF_OPEN counts: a pending trial is as unavailable as an open breaker."""
        with closing(_connect(self._db_path)) as conn:
            row = conn.execute(
                "SELECT state, opened_at FROM circuit_breaker WHERE service=?", (self.service,)
            ).fetchone()
        if not row or row[0] == "closed" or row[1] is None:
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
        in ``ignore`` (expected non-failures, e.g. a 404 "item deleted") re-raise without counting.
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
    """Snapshot every breaker for status surfaces."""
    with closing(_connect(Path(db_path) if db_path else breaker_db_path())) as conn:
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
