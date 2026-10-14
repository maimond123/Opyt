"""Curation's local work: the paper-author derivations and identity resolution, run when what they
read has changed, and hourly regardless.

Neither reads a platform, so neither waits for a session: a home that only saves papers derives
its authors like any other. Each is one unit on its own quota, and a pass happens when the store
says it is owed:

  papers    `papers.sync_paper_author_signals`, then `papers.sync_coauthor_signals` (authors
            before coauthors: the order is load-bearing). Owed when the paper atoms or the Oracle
            roster changed since its last pass.
  resolve   `resolve.resolve_entities`. Owed when the entities changed since its last pass, which
            a papers pass that minted an author is one way to cause.

An edit inside an existing row (a paper's backfilled author list, an entity's new identity link)
does not change what is fingerprinted; the hourly pass covers it. Each pass records the
fingerprint it ran against, so its own writes do not make it owed again: papers records what it
read before running (its writes are signals and entities, not its inputs), resolve what it left
(its backfill may add the entities it then resolves).

A failure raises to the quota loop, whose backstop backs the unit off; nothing is half-written,
since both jobs are idempotent re-reads of the whole store.
"""

import sqlite3
from datetime import datetime, timedelta

from pipeline.curation import papers
from pipeline.ingestion.utils import log
from pipeline.kb import resolve
from pipeline.kb.writes import write_tx
from pipeline.timeparse import parse_ts, utc_iso
from pipeline.worker.quota_loop.source import YOURS, Outcome, Unit

QUOTA = "curation:maintenance"
SAFETY = timedelta(hours=1)

_DDL = """
CREATE TABLE IF NOT EXISTS curation_maintenance (
    job TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, ran_at TEXT NOT NULL
);
"""

# What each job reads, as a cheap fingerprint of whole-row arrivals.
_INPUTS = {
    "papers": "SELECT (SELECT COUNT(*) || ':' || COALESCE(MAX(rowid), 0) FROM atoms "
              "        WHERE source_type='paper') || '|' || (SELECT COUNT(*) FROM oracles)",
    "resolve": "SELECT COUNT(*) || ':' || COALESCE(MAX(rowid), 0) FROM entities",
}


def init_schema(conn: sqlite3.Connection) -> None:
    # executescript commits a caller's transaction, so initialization is entry-only.
    if conn.in_transaction:
        raise RuntimeError("initialize the maintenance table before starting a transaction")
    conn.executescript(_DDL)


def fingerprint(conn: sqlite3.Connection, job: str) -> str:
    return conn.execute(_INPUTS[job]).fetchone()[0]


def _last(conn: sqlite3.Connection, job: str) -> tuple[str, str] | None:
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='curation_maintenance'").fetchone():
        return None
    row = conn.execute("SELECT fingerprint, ran_at FROM curation_maintenance WHERE job=?",
                       (job,)).fetchone()
    return (row[0], row[1]) if row else None


def due_at(last: tuple[str, str] | None, current: str, now: datetime) -> datetime:
    """When a job is next owed: now if it never ran or its inputs changed, else an hour after
    its last pass."""
    if last is None or last[0] != current:
        return now
    ran = parse_ts(last[1])
    return ran + SAFETY if ran else now


def derive_papers(conn: sqlite3.Connection) -> dict:
    """Saved papers' authors, then Oracles' coauthors."""
    return {"paper_authors": papers.sync_paper_author_signals(conn),
            "paper_coauthors": papers.sync_coauthor_signals(conn)}


class Maintenance:
    """One job per unit, whichever is owed first: papers before resolve, so a minted author is
    merged in the same round."""
    name, klass, floor = "curation_maintenance", YOURS, 0
    jobs = ("papers", "resolve")

    def __init__(self, quota: str = QUOTA) -> None:
        self.quota = quota

    def candidates(self, conn, now) -> list[Unit]:
        return [Unit(f"maintenance:{job}", (job, now)) for job in self.jobs
                if due_at(_last(conn, job), fingerprint(conn, job), now) <= now]

    def next_unit(self, conn, now):
        return next(iter(self.candidates(conn, now)), None)

    def fetch(self, conn, unit, ctx):
        job, _ = unit.payload
        if job == "papers":
            read = fingerprint(conn, job)
            return read, derive_papers(conn)
        out = resolve.resolve_entities(conn).as_dict()
        return fingerprint(conn, job), out

    def record(self, conn, unit, fetched):
        job, now = unit.payload
        read, out = fetched
        if job == "papers" and (due := out["paper_authors"].get("needs_backfill")):
            # The backfill's detector: the one place a store's pre-author-list papers are named.
            log(f"[curation] {due} paper atom(s) carry no author list: "
                f"{out['paper_authors']['remedy']}")
        with write_tx(conn, site="curation.maintenance"):
            conn.execute("INSERT INTO curation_maintenance(job,fingerprint,ran_at) VALUES(?,?,?) "
                         "ON CONFLICT(job) DO UPDATE SET fingerprint=excluded.fingerprint, "
                         "ran_at=excluded.ran_at", (job, read, utc_iso(now)))
        return Outcome(job)

    def next_due(self, conn, now):
        return min(due_at(_last(conn, job), fingerprint(conn, job), now) for job in self.jobs)
