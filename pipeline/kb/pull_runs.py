"""
pipeline/kb/pull_runs.py — what a pull is doing, on disk, while it is doing it.

One row per RUN and one per (run × Oracle), recording who was picked, in what order, who has
finished, and what each of them yielded. Nothing on the query path reads this; it exists so that
a call which started a pull can hand back a report MINUTES later, from a different call, or from
a different conversation — which is the whole of `2026-09-14-no-call-waits-for-a-pull.md`.

⚠️ THIS EXISTS BECAUSE THE REPORT USED TO LIVE ONLY IN MEMORY. `_ingest` assembled its
`results` / `presentation` from a Python list at the end of the pull, so a tool call cut off at
the client's 60-second wall lost every word of it and handed the model one fact — `Error:
Request timed out` — which it then used as a universal explanation for three unrelated things,
none of which was a timeout (2026-09-14, six of nine ingest calls). A pull that cannot be
narrated from disk cannot be narrated at all once the call that started it is gone.

NOT DERIVABLE, and nothing here may pretend otherwise. `oracle_refresh_state` draws the same
line and for the same reason: these rows record an ATTEMPT, and an attempt that returned nothing
leaves no trace in the corpus to rebuild from. A dropped table loses the REPORT, never the atoms
— the safe direction. Do not add a "rebuild from `atoms`" path; it would reconstruct successes
and silently forget everyone who was reached and yielded nothing.

UNIX SECONDS (REAL), not ISO text, unlike its neighbour `oracle_sources`. Two readers do numeric
math on these: `progress` reports elapsed seconds, and liveness is a staleness comparison against
a `sync_lock` heartbeat, which is REAL for the reason its own docstring gives — "ISO strings
don't sort correctly when microseconds are sometimes omitted". Matching the thing this is
compared against beats matching the table next door.

THERE IS NO `picks` COLUMN, deliberately, and the handoff plan that specified one was wrong. The
ordered roster IS `pull_run_oracles` read by `position`; a second copy on the parent row is a
second ordering of the same people, which is exactly the hazard `_ordered_picks` refuses when it
declines to add a scoring function beside `Candidate.sort_key`. It also could not survive commit
5's queue-onto, where a second `ingest` appends picks to a run already in flight: two writers,
two representations, one of them stale.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass

from . import schema

_DDL = """
CREATE TABLE IF NOT EXISTS pull_runs (
  run_id      TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,      -- 'ingest' today; room for arm_a / arm_b
  started_at  REAL NOT NULL,      -- unix seconds
  finished_at REAL,               -- NULL = the loop has not closed this run
  heartbeat_at REAL,              -- bumped every HEARTBEAT_SECONDS while the pull is alive
  lookback    TEXT,               -- JSON: the lookback report the final presentation needs
  reported_at REAL                -- when a completion notice was carried to the user
);
CREATE TABLE IF NOT EXISTS pull_run_oracles (
  run_id       TEXT NOT NULL,
  canonical_id TEXT NOT NULL,
  name         TEXT,
  position     INTEGER NOT NULL,  -- `_ordered_picks` order — the roster, and its only copy
  window       TEXT,             -- ISO x_since this pick is to be pulled with; NULL = adapter default
  started_at   REAL,
  finished_at  REAL,
  result       TEXT,              -- JSON: the per-Oracle dict `_ingest` reports in `results`
  PRIMARY KEY (run_id, canonical_id)
);
CREATE INDEX IF NOT EXISTS idx_pull_runs_started ON pull_runs(started_at DESC);
"""


@dataclass(frozen=True)
class PullRun:
    run_id: str
    kind: str
    started_at: float
    finished_at: float | None
    heartbeat_at: float | None
    lookback: dict | None
    reported_at: float | None


@dataclass(frozen=True)
class PullRunOracle:
    run_id: str
    canonical_id: str
    name: str | None
    position: int
    window: str | None
    started_at: float | None
    finished_at: float | None
    result: dict | None

    @property
    def state(self) -> str:
        """`waiting` → `in_flight` → `done`. Derived, never stored: a stored state is a fourth
        thing that can disagree with the two timestamps that already say it."""
        if self.finished_at is not None:
            return "done"
        return "in_flight" if self.started_at is not None else "waiting"


# ── connection + schema ─────────────────────────────────────────────────────────
def init_pull_run_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL, called by every public writer here — a caller may hand us a plain
    `schema.connect()` that has never seen these tables, and `_ingest` does exactly that.

    `CREATE TABLE IF NOT EXISTS` does NOT add a column to a table that already exists, so a new
    column needs an explicit ALTER beside this call, as `oracle_refresh_state` does. There is no
    separate migration hook on purpose: every writer runs this, so there is exactly one place a
    store can be behind."""
    conn.executescript(_DDL)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(pull_runs)")}
    if "heartbeat_at" not in cols:
        conn.execute("ALTER TABLE pull_runs ADD COLUMN heartbeat_at REAL")
    ocols = {r[1] for r in conn.execute("PRAGMA table_info(pull_run_oracles)")}
    if "window" not in ocols:
        conn.execute("ALTER TABLE pull_run_oracles ADD COLUMN window TEXT")
    conn.commit()


def connect(db_path=None, *, read_only: bool = False):
    """The atom-KB store with the run tables guaranteed present. Reuses `schema.connect` (WAL +
    busy_timeout + row_factory + `$OPYT_HOME`); read-only opens skip DDL, matching its contract.

    THE PULL THREAD CALLS THIS FOR ITSELF. SQLite connections are thread-bound and the connection
    on the call that started the run is mid-request and about to be closed — the same rule
    `footprint_enrichment.start_background` records for its own thread."""
    conn = schema.connect(db_path, read_only=read_only)
    if not read_only:
        init_pull_run_schema(conn)
    return conn


def _loads(raw):
    """JSON that fails to parse degrades to None, never raises. A report column is a convenience
    for the narrator; a store that cannot answer "what did this yield" must still be able to
    answer "did this finish", which is the fact a user is actually owed."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _to_run(row: sqlite3.Row) -> PullRun:
    return PullRun(run_id=row["run_id"], kind=row["kind"], started_at=row["started_at"],
                   finished_at=row["finished_at"], heartbeat_at=row["heartbeat_at"],
                   lookback=_loads(row["lookback"]), reported_at=row["reported_at"])


def _to_oracle(row: sqlite3.Row) -> PullRunOracle:
    return PullRunOracle(run_id=row["run_id"], canonical_id=row["canonical_id"],
                         name=row["name"], position=row["position"], window=row["window"],
                         started_at=row["started_at"], finished_at=row["finished_at"],
                         result=_loads(row["result"]))


# ── writers ─────────────────────────────────────────────────────────────────────
def open_run(conn, *, kind: str, picks: list[dict], lookback: dict | None = None) -> str:
    """Open a run over `picks` IN THE ORDER GIVEN, and return its id.

    ⚠️ EVERY PICK GETS A ROW NOW, not when its turn comes. "Who is still waiting" has to be
    answerable one second into a twelve-minute pull, and an Oracle absent from the report reads
    as an Oracle who failed — that is how *"Bryan Johnson's pull also timed out"* got said about
    a pull that was mid-flight and landing 57 files.

    `picks` is `_ordered_picks` output: dicts carrying `canonical_id` and `name`, each optionally
    carrying the `window` (ISO `x_since`) it is to be pulled with.

    ⚠️ THE WINDOW IS STORED PER ROW, not derived by whoever walks it, and that is what makes
    "queued onto the running pull" a true sentence rather than a hopeful one. A second `ingest`
    arriving mid-pull adds its picks here; the loop already in flight re-reads this roster and
    finishes them. It cannot re-derive their window — `x_lookback` was an argument to the CALL
    that queued them, and that call is long gone.
    """
    run_id = uuid.uuid4().hex
    init_pull_run_schema(conn)
    now = time.time()
    conn.execute("INSERT INTO pull_runs (run_id, kind, started_at, heartbeat_at, lookback) "
                 "VALUES (?,?,?,?,?)",
                 (run_id, kind, now, now,
                  json.dumps(lookback) if lookback is not None else None))
    conn.executemany(
        "INSERT INTO pull_run_oracles (run_id, canonical_id, name, position, window) "
        "VALUES (?,?,?,?,?)",
        [(run_id, o["canonical_id"], o.get("name"), i, o.get("window"))
         for i, o in enumerate(picks)])
    conn.commit()
    return run_id


def add_picks(conn, run_id: str, picks: list[dict]) -> int:
    """Append picks to a run already in flight, after whatever is there. Returns how many landed.

    This is what a second `ingest` arriving mid-pull does instead of being refused: the likely
    trigger is "oh, add one more person", not a banned retry, and bouncing it drops writers the
    user just asked for. `INSERT OR IGNORE` because re-asking for somebody already on the roster
    is a no-op, not an error — and it keeps their original position rather than moving them to
    the back for having been named twice."""
    init_pull_run_schema(conn)
    row = conn.execute("SELECT COALESCE(MAX(position), -1) AS p FROM pull_run_oracles "
                       "WHERE run_id = ?", (run_id,)).fetchone()
    nxt = (row["p"] if row else -1) + 1
    before = conn.execute("SELECT COUNT(*) AS c FROM pull_run_oracles WHERE run_id = ?",
                          (run_id,)).fetchone()["c"]
    conn.executemany(
        "INSERT OR IGNORE INTO pull_run_oracles (run_id, canonical_id, name, position, window) "
        "VALUES (?,?,?,?,?)",
        [(run_id, o["canonical_id"], o.get("name"), nxt + i, o.get("window"))
         for i, o in enumerate(picks)])
    conn.commit()
    after = conn.execute("SELECT COUNT(*) AS c FROM pull_run_oracles WHERE run_id = ?",
                         (run_id,)).fetchone()["c"]
    return after - before


def start_oracle(conn, run_id: str, canonical_id: str) -> None:
    """Stamp one Oracle as in flight. Idempotent on re-entry — the FIRST stamp wins, because the
    breadth pass and the depth pass are two visits to one person and the honest answer to "since
    when has this been running" is since breadth started."""
    init_pull_run_schema(conn)
    conn.execute("UPDATE pull_run_oracles SET started_at = ? "
                 "WHERE run_id = ? AND canonical_id = ? AND started_at IS NULL",
                 (time.time(), run_id, canonical_id))
    conn.commit()


def finish_oracle(conn, run_id: str, canonical_id: str, result: dict | None) -> None:
    """Record what one Oracle yielded, and that they are done.

    Written after the DEPTH pass, which is the point both passes over one person have been merged
    — `_merge_passes` is what makes one row out of two, and a reader that saw the breadth row
    land first would count the person twice and split their atoms."""
    init_pull_run_schema(conn)
    conn.execute("UPDATE pull_run_oracles SET finished_at = ?, result = ? "
                 "WHERE run_id = ? AND canonical_id = ?",
                 (time.time(), json.dumps(result) if result is not None else None,
                  run_id, canonical_id))
    conn.commit()


def close_run(conn, run_id: str, *, lookback: dict | None = None) -> None:
    """The loop finished every pick it held. `finished_at` on the PARENT is the only thing that
    means "complete" — a run where every row happens to be done but the parent is open is a run
    whose loop died between the last Oracle and here, and those are different facts."""
    init_pull_run_schema(conn)
    if lookback is not None:
        conn.execute("UPDATE pull_runs SET lookback = ? WHERE run_id = ?",
                     (json.dumps(lookback), run_id))
    conn.execute("UPDATE pull_runs SET finished_at = ? WHERE run_id = ? AND finished_at IS NULL",
                 (time.time(), run_id))
    conn.commit()


def mark_reported(conn, run_id: str) -> None:
    """A completion notice has been carried to the user; do not carry it again.

    Without this the rider becomes permanent furniture on four tools and the reader learns to
    skip the one field that says something happened — the same argument `screen` makes for
    omitting `omitted: 0`."""
    init_pull_run_schema(conn)
    conn.execute("UPDATE pull_runs SET reported_at = ? WHERE run_id = ? AND reported_at IS NULL",
                 (time.time(), run_id))
    conn.commit()


# ── readers ─────────────────────────────────────────────────────────────────────
def get_run(conn, run_id: str) -> PullRun | None:
    init_pull_run_schema(conn)
    row = conn.execute("SELECT * FROM pull_runs WHERE run_id = ?", (run_id,)).fetchone()
    return _to_run(row) if row else None


def latest_run(conn, *, kind: str = "ingest") -> PullRun | None:
    """The most recently STARTED run of a kind, finished or not. `progress` with no `run_id`
    means "the one I just started", and newest-started is that under every ordering — a run that
    finishes fast does not displace one still going."""
    init_pull_run_schema(conn)
    row = conn.execute("SELECT * FROM pull_runs WHERE kind = ? "
                       "ORDER BY started_at DESC LIMIT 1", (kind,)).fetchone()
    return _to_run(row) if row else None


def oracles_for(conn, run_id: str) -> list[PullRunOracle]:
    init_pull_run_schema(conn)
    rows = conn.execute("SELECT * FROM pull_run_oracles WHERE run_id = ? ORDER BY position",
                        (run_id,)).fetchall()
    return [_to_oracle(r) for r in rows]


def unfinished_for(conn, run_id: str) -> list[PullRunOracle]:
    """Roster rows nobody has finished, in position order — what a loop re-reads to discover the
    picks a later call queued onto it. Includes anyone in flight, so the caller compares against
    what it is already walking rather than trusting this to mean "not started"."""
    return [o for o in oracles_for(conn, run_id) if o.finished_at is None]


def unreported_finished(conn, *, kind: str = "ingest") -> PullRun | None:
    """A run that has finished and has never been mentioned — the rider's whole input.

    Only the most recent one. Two unreported runs means the user missed a completion entirely,
    and replaying both would open a conversation with a backlog; the newest is the one they are
    about to ask about."""
    init_pull_run_schema(conn)
    row = conn.execute("SELECT * FROM pull_runs WHERE kind = ? AND finished_at IS NOT NULL "
                       "AND reported_at IS NULL ORDER BY finished_at DESC LIMIT 1",
                       (kind,)).fetchone()
    return _to_run(row) if row else None


# ── liveness ────────────────────────────────────────────────────────────────────
# ⚠️ A HEARTBEAT ON THE RUN ROW, NOT THE SHARED X LEASE — and the handoff plan that said to read
# liveness off `sync_lock` was wrong about which question that lock answers. `CatchupLock`'s name
# for this family is `"oracle-refresh"`, shared DELIBERATELY by the refresh rail and by footprint
# enrichment, because all three walk the same two x.com buckets and a second lease name would let
# two of them split a meter the whole design is organised around. A shared lease says "something
# in this family is working". It cannot say WHICH, so reading it here would report a dead pull as
# running whenever enrichment happened to hold it — precisely the false "it is still going" this
# plan exists to remove.
#
# The MECHANISM is borrowed whole from `pipeline/sync_lock.py`, which is where the reasoning
# lives: a lease, not a mutex, because a SIGKILL'd holder (laptop sleeps, OOM, the user quits the
# client mid-pull) can never release a mutex and every later reader would wait forever. TTL is 3×
# the beat so one missed bump from a GC pause is not a false eviction.
HEARTBEAT_SECONDS = 10.0
LEASE_TTL = 30.0


def beat(conn, run_id: str) -> None:
    """Bump the run's heartbeat. Cheap, idempotent, and never raises — a missed beat costs a
    false `stopped`, and raising out of a timer thread would cost the pull."""
    try:
        conn.execute("UPDATE pull_runs SET heartbeat_at = ? WHERE run_id = ?",
                     (time.time(), run_id))
        conn.commit()
    except Exception:
        pass


def is_alive(run: PullRun, *, ttl: float = LEASE_TTL, now: float | None = None) -> bool:
    """Is something still working on this run right now?

    A finished run is not alive and does not need to be — `run_status` reads `finished_at` first.
    A run with no heartbeat at all predates this column and is treated as dead, which is the safe
    direction: it makes the report say "stopped at 4 of 7", never "still going" about nothing."""
    if run.heartbeat_at is None:
        return False
    return (now or time.time()) - run.heartbeat_at < ttl


class Heartbeat:
    """Keep a run's heartbeat fresh for as long as the pull is running, from its own thread.

    ⚠️ THE BEAT CANNOT RIDE ON THE WORK, and a blog archive is why. Stamping the row as each
    Oracle finishes looks equivalent and is not: one measured archive took ~50 seconds as a single
    uninterruptible unit, so the gap between two stamps can exceed any TTL short enough to notice
    a real death. A pull that is working hard would read as a pull that died.

    Its own connection, opened in its own thread — SQLite connections are thread-bound and the
    caller's belongs to the request that started this."""

    def __init__(self, run_id: str, *, interval: float = HEARTBEAT_SECONDS, db_path=None):
        self.run_id = run_id
        self.interval = interval
        self._db_path = db_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        conn = None
        try:
            conn = connect(self._db_path)
            while not self._stop.wait(self.interval):
                beat(conn, self.run_id)
        except Exception:
            pass
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def __enter__(self) -> "Heartbeat":
        try:
            self._thread = threading.Thread(target=self._run,
                                            name=f"opyt-pull-beat-{self.run_id[:8]}", daemon=True)
            self._thread.start()
        except Exception:
            self._thread = None     # a pull must not fail for want of a narrator
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def in_flight(conn, *, kind: str = "ingest", now: float | None = None) -> PullRun | None:
    """A run that is unfinished AND still beating — what a second `ingest` joins instead of
    starting a rival pull. None when the newest run is finished or its holder is gone.

    Only the newest is considered: an older unfinished run whose process died is not something to
    join, it is something the next call reports as stopped."""
    run = latest_run(conn, kind=kind)
    if run is None or run.finished_at is not None:
        return None
    return run if is_alive(run, now=now) else None


def completion_notice(conn, *, kind: str = "ingest") -> dict | None:
    """A finished pull nobody has mentioned yet, as one short factual notice — or None.

    ⚠️ THIS IS THE ONLY WAY A USER WHO WALKED AWAY EVER FINDS OUT. OPYT cannot push: an MCP
    server speaks when it is called and never otherwise, and no part of this design may pretend
    differently. So a pull that finished while the conversation was closed rides back on whatever
    the user does next — `search` included — and then never again.

    STAMPED AS IT IS TAKEN. Un-stamped it would ride on every call forever and become furniture,
    and the reader would learn to skip the one field that says something happened; that is the
    same argument `screen` makes for omitting `omitted: 0`.

    It carries COUNTS AND NAMES, not a presentation. The full report is one `progress` call away
    with the `run_id` this hands over, and building prose here would put a second opinion about
    an ingest outcome in a module that stores rows.

    Lives here rather than in a tool module because four tools carry it and the tool modules are
    deliberately flat — none imports another; only `server.py` imports them.
    """
    run = unreported_finished(conn, kind=kind)
    if run is None:
        return None
    rows = oracles_for(conn, run.run_id)
    done = [o for o in rows if o.finished_at is not None]
    missed = [o.name or o.canonical_id for o in rows if o.finished_at is None]
    atoms = sum((o.result or {}).get("atoms_added", 0) for o in done)
    mark_reported(conn, run.run_id)

    if missed:
        # STOPPED, and never described as a failure. The likely cause is that the user quit the
        # app, and what landed is durable — so the honest offer is to finish it, not an apology.
        message = (f"A pull that was running earlier reached {len(done)} of {len(rows)} writers "
                   f"({atoms} pieces) before it stopped — the app was probably closed. "
                   f"Still to do: {', '.join(missed)}. Tell the user what landed, offer to "
                   f"finish the rest (a fresh `oracle(action='ingest')` picks up where this "
                   f"stopped), and do NOT call it a failure, a crash or an error.")
    else:
        message = (f"A pull finished while the user was away: {len(done)} writers, {atoms} "
                   f"pieces, all durable. Mention it once, briefly, alongside whatever they just "
                   f"asked for — `oracle(action='progress', run_id=...)` has the full report if "
                   f"they want it. Do not make it the subject of the turn.")

    return {"run_id": run.run_id, "writers": len(done), "atoms": atoms,
            "status": "complete" if not missed else "stopped",
            "unreached": missed, "message": message}


def run_status(run: PullRun, *, alive: bool) -> str:
    """`complete` | `running` | `stopped` — PURE, with liveness injected rather than imported.

    ⚠️ `stopped` IS A READ, NEVER AN INFERENCE FROM ELAPSED TIME. An unfinished run whose worker
    is gone is the wedged-job shape — `started_at` set, `finished_at` NULL, nothing alive — which
    had to be cleared out of the live store by hand on 2026-09-14. The caller answers `alive`
    from the `sync_lock` heartbeat, which is reclaimable precisely because a SIGKILL'd holder can
    never release a mutex. A pull is allowed to take twelve minutes; taking a long time is not
    evidence of anything.

    `alive` is injected so this module never imports the lock, and so a test can state the fact
    rather than fake a lease."""
    if run.finished_at is not None:
        return "complete"
    return "running" if alive else "stopped"
