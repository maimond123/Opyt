"""
pipeline/circuit_breaker.py

A persisted circuit breaker for external APIs (OpenRouter). Trips
on repeated failures to stop a retry storm against a down/degraded API.

Three states: CLOSED (normal, counts consecutive failures, trips to OPEN at threshold),
OPEN (reject instantly; moves to HALF_OPEN after cooldown), HALF_OPEN
(one trial call: success → CLOSED, failure → OPEN).

State is persisted per-service, not in-memory, so every session shares one outage view.
Connections are opened per-operation rather than held on the instance, so a module-level
breaker can be shared across threads (SQLite connections are thread-bound).

⚠️ ITS OWN FILE, NOT `opyt.db`, AND THAT IS NOT A TIDINESS PREFERENCE. Every decision below
takes `BEGIN IMMEDIATE` — an EXCLUSIVE write lock on the whole database — to answer "is this
service up?". While that lived in `opyt.db` it competed with ingestion for the same lock, and
ingestion wins: it writes atoms, chunks and embeddings in transactions far longer than the 5s
`busy_timeout` here. Measured on the box 2026-09-16 during a connect, with the curation walk and
the `bookmark_catchup` rail both writing: **12 `database is locked` failures**, 6 of them OCR
reads that were then DISCARDED — `ocr_cascade.read_image` catches the OperationalError in its
generic `except Exception`, logs "transcribe failed", and returns None. So a lock this module
took to check on OpenRouter silently cost a blog post its chart transcripts, and the log blamed
the model.

Breaker state has nothing to do with the knowledge base, so it has no business sharing its
write lock. The six services now contend only with each other, over single-row reads that take
microseconds, instead of with a bulk ingest.

The move does NOT migrate the old `circuit_breaker` table out of `opyt.db`. State here is
transient health data with a self-correcting default: a fresh file reads CLOSED, and a service
that really is down re-trips after `threshold` consecutive failures. A migration path would run
forever to buy one deploy a handful of already-doomed calls.
"""

import sqlite3
import time
from contextlib import closing
from pathlib import Path

from opyt_core.paths import opyt_path

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
    """`<home>/circuit_breaker.db` — resolved at CALL time, never bound at import.

    Runtime resolution is the Distributable invariant, and it is what makes a hosted box work at
    all: one process serves many homes through `$OPYT_HOME`, so a path captured at import would
    pin every home's breaker to whichever home happened to load this module first."""
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
    # The HOME first. This used to ride on `opyt.db`, which something else had always created by
    # the time a breaker ran; its own file has no such guarantor, and `sqlite3.connect` raises
    # "unable to open database file" on a missing PARENT, not a missing file. That is the same
    # trap `0e8b5438` fixed in the profile lease three weeks of debugging later — see §2.3 of
    # docs/plans/2026-09-16-hosted-chrome-contention-handoff.md. Paying it once here is cheap.
    db_path.parent.mkdir(parents=True, exist_ok=True)
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
        self._db_path = Path(db_path) if db_path else breaker_db_path()
        with closing(_connect(self._db_path)) as conn:
            conn.execute("INSERT OR IGNORE INTO circuit_breaker(service) VALUES (?)", (self.service,))
            conn.commit()

    # ── decision ──────────────────────────────────────────────────────────────

    def allow(self) -> bool:
        """CLAIM the right to call right now. MUTATES: advances OPEN → HALF_OPEN once the
        cooldown has elapsed, and the returned True IS that single trial.

        ⚠️ Only `call` should invoke this. A caller that asks `allow()` and then calls through
        `call()` claims the trial twice — the second ask sees HALF_OPEN, which is not CLOSED, and
        refuses. Nothing then records an outcome, so the breaker never leaves HALF_OPEN and the
        service is dead forever. That is not hypothetical: `api.openalex.org` sat HALF_OPEN for
        257 hours and `export.arxiv.org` for 30 on the live store (2026-09-08), against a 15-minute
        cooldown, because `_BreakerBacked.available()` asked with this method. Use `peek` to ask.

        An ABANDONED trial is re-granted. A HALF_OPEN older than one cooldown means whoever
        claimed it never recorded an outcome — a crashed process, or the double-claim above — and
        without this the state is absorbing: `call` refuses HALF_OPEN, so no outcome is ever
        recorded, so it stays HALF_OPEN. Re-stamping `opened_at` is what keeps the re-grant single:
        a concurrent claimant finds the clock reset."""
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
        """Would a call be permitted right now? READ-ONLY — never advances the state machine.

        What a pre-check must use. `allow` claims the trial, so asking it in order to decide
        whether to bother asking is how a breaker gets stranded (see `allow`).
        """
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

        HALF_OPEN counts, not just OPEN: a claimed trial whose outcome is still pending is exactly
        as unavailable as an open breaker, and reporting 0 for it told every caller to retry
        immediately into a refusal."""
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
