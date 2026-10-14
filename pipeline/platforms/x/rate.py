"""x.com's per-operation request budget, persisted per home.

Every X request records `x-rate-limit-remaining` / `-reset` off its response here, so detached
rail children start with what the previous pass learned. X only: no other host gives a retry
instant to persist. Its own file, not `opyt.db` (hot-path health data must not wait on a bulk
ingest's write lock), and per home, because the bucket belongs to that home's X session.

Fail-safe throughout: an unreadable or unwritable meter leaves the caller blind for one request,
which the next response fills in.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Literal, TypeAlias

from opyt_core.paths import opyt_path, owner_only

BudgetObservation: TypeAlias = tuple[int, float] | None
BudgetAction: TypeAlias = Literal["allow", "forget", "refuse"]

# x.com meters each GraphQL operation in 15-minute windows, per account; nothing here chooses it.
# `refuse_if_spent` uses the exact reset off each response; this is the ceiling to wait when
# nothing knows better.
X_RATE_WINDOW = 900.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS x_rate_budget (
    op         TEXT PRIMARY KEY,
    remaining  INTEGER NOT NULL,   -- requests left in the current window
    reset      REAL NOT NULL,      -- unix instant the window refills (x-rate-limit-reset)
    updated_at REAL NOT NULL
);
"""


def meter_db_path(home: Path | None = None) -> Path:
    """`<home>/x_rate_meter.db`, resolved at call time: one hosted process serves many homes
    through `$OPYT_HOME`. `home` reads a home this process is not running as."""
    return (Path(home) / "x_rate_meter.db") if home is not None else opyt_path("x_rate_meter.db")


def _connect(db_path: Path) -> sqlite3.Connection:
    # `sqlite3.connect` raises on a missing parent directory, and nothing else creates it first.
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    owner_only(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def record(op: str, remaining: int, reset: float, *, home: Path | None = None) -> None:
    """Store one operation's budget. Never raises.

    The UPSERT keeps the stricter reading (as `stricter_budget` does), not the last write:
    several processes report against the same server-side bucket."""
    try:
        with closing(_connect(meter_db_path(home))) as conn:
            conn.execute(
                "INSERT INTO x_rate_budget(op, remaining, reset, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(op) DO UPDATE SET "
                "  remaining=CASE WHEN excluded.reset > x_rate_budget.reset "
                "                 THEN excluded.remaining "
                "                 WHEN excluded.reset < x_rate_budget.reset "
                "                 THEN x_rate_budget.remaining "
                "                 ELSE MIN(excluded.remaining, x_rate_budget.remaining) END, "
                "  reset=MAX(excluded.reset, x_rate_budget.reset), "
                "  updated_at=excluded.updated_at",
                (op, int(remaining), float(reset), time.time()))
            conn.commit()
    except Exception:
        return            # a meter that cannot be written is the state we had before it existed


def read(op: str, *, home: Path | None = None) -> tuple[int, float] | None:
    """This operation's last recorded `(remaining, reset)`, or None if nothing knows. Never raises."""
    try:
        with closing(_connect(meter_db_path(home))) as conn:
            row = conn.execute("SELECT remaining, reset FROM x_rate_budget WHERE op=?",
                               (op,)).fetchone()
    except Exception:
        return None
    return (int(row[0]), float(row[1])) if row else None


def forget(op: str, *, home: Path | None = None) -> None:
    """Drop a window that has already rolled over. Never raises.

    An expired row would otherwise read as spent to anything that ignores `reset`."""
    try:
        with closing(_connect(meter_db_path(home))) as conn:
            conn.execute("DELETE FROM x_rate_budget WHERE op=?", (op,))
            conn.commit()
    except Exception:
        return


def stricter_budget(local: BudgetObservation,
                    durable: BudgetObservation) -> BudgetObservation:
    """Select the more pessimistic observation of the same server-side bucket.

    A later reset is a newer window and supersedes outright. At the same reset, the smaller
    remaining count is the observation that saw more requests spent.
    """
    if local is None:
        return durable
    if durable is None:
        return local
    local_left, local_reset = local
    durable_left, durable_reset = durable
    if local_reset != durable_reset:
        return local if local_reset > durable_reset else durable
    return min(local_left, durable_left), local_reset


def budget_action(observation: BudgetObservation, now: float) -> BudgetAction:
    """Decide whether to issue a request, clear an expired window, or refuse it locally."""
    if observation is None:
        return "allow"
    remaining, reset = observation
    if remaining > 0:
        return "allow"
    return "forget" if now >= reset else "refuse"


class XRateLimited(RuntimeError):
    """x.com's request budget for this operation's window is spent.

    Distinct from a per-account failure: a multi-account loop must stop entirely here. Carries
    `reset_at`, the unix instant the bucket refills (None when a 429 had no header).
    """

    def __init__(self, message: str, *, op: str | None = None, reset_at: float | None = None):
        super().__init__(message)
        self.op = op
        self.reset_at = reset_at


# ── The per-operation request budget ────────────────────────────────────────
# Each operation holds its own bucket per 15-minute window, read off every response, so nothing
# encodes X's numbers. The dict is a process-local cache in front of the persisted meter; the two
# are merged on read (`rate_budget`), since either may hold a reading the other has not seen.
_RATE_STATE: dict[str, tuple[int, float]] = {}       # op -> (remaining, reset unix-epoch)
_RATE_LOCK = threading.Lock()


def record_rate_values(op: str, remaining: int, reset: float) -> None:
    """Store a response's two rate fields.

    Written to this process's dict and to the home's persisted meter."""
    with _RATE_LOCK:
        _RATE_STATE[op] = (remaining, reset)
    record(op, remaining, reset)


def rate_budget(op: str) -> tuple[int, float] | None:
    """This operation's last-known `(remaining, reset)`, or None if nothing knows.

    Merges this process's dict with the home's persisted meter."""
    with _RATE_LOCK:
        local = _RATE_STATE.get(op)
    return stricter_budget(local, read(op))


def refuse_if_spent(op: str) -> None:
    """Raise before issuing a request the server has already told us it will refuse.

    Refuses rather than sleeps: a rail holds a lease, and a locally refused request never spends
    the bucket on a 429."""
    state = rate_budget(op)
    # Read the clock only when an exhausted observation can need expiry handling.
    now = time.time() if state is not None and state[0] <= 0 else 0.0
    action = budget_action(state, now)
    if action == "allow":
        return
    if action == "forget":                           # the window already rolled over
        with _RATE_LOCK:
            _RATE_STATE.pop(op, None)
        forget(op)                      # …in both places, or the next process
        return                                       # reads a spent-forever bucket
    assert state is not None
    _, reset = state
    raise XRateLimited(
        f"{op}: this 15-minute window's request budget is spent — {round(reset - now)}s until it "
        f"refills. Not issued (refused locally, so it does not count against the bucket).",
        op=op, reset_at=reset)
