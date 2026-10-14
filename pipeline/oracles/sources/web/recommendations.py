"""Recommendation reads: one publication per turn, with a durable weekly roster pass, and the pass's
tables (`web_recommendation_pass`, `web_recommendation_reads`).

The clock records attempts, including refusals; a database lock leaves the read owed. Signals
are aggregated after the pass so corroboration counts include every publication it read; what
they mean is curation's (`pipeline/curation/recommended.py`).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from urllib.parse import urlsplit

from pipeline.curation import clock, recommended
from pipeline.kb.writes import is_lock_error, write_tx
from pipeline.oracles import roster
from pipeline.oracles.sources.web.platform import QUOTA
from pipeline.platforms.substack import publication as substack_publication
from pipeline.timeparse import parse_ts, utc_now, utc_iso
from pipeline.worker.quota_loop.source import SCOUTING, Outcome, Unit
from pipeline.worker.quota_loop.source import PoolSource
from pipeline.oracles.sources import pair_rows

# A pass re-reads every Substack Oracle's list, and lists are edited by hand on the order of
# months, so a pass repeats weekly; a Substack Oracle added since the last pass makes one due
# sooner (`due`). The gate also spares Substack's Cloudflare 403 window.
FLOOR_HOURS = 7 * 24.0

# The durable per-publication observations of a recommendation pass.
DDL = """
CREATE TABLE IF NOT EXISTS web_recommendation_pass (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1), started_at TEXT NOT NULL, done INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS web_recommendation_reads (
    canonical_id TEXT NOT NULL, url TEXT NOT NULL, status TEXT, payload TEXT, detail TEXT,
    PRIMARY KEY(canonical_id,url)
);
"""


def init_schema(conn):
    conn.executescript(DDL)


def active(conn):
    return conn.execute("SELECT started_at FROM web_recommendation_pass WHERE done=0").fetchone()


def read_publication(url):
    """One publication's published recommendations; errors reach the owning loop."""
    return substack_publication.fetch_recommendations(substack_publication.fetch_publication_id(url))


def due(conn: sqlite3.Connection, now: datetime | None = None) -> bool:
    """A week since the last attempt, or a Substack Oracle added after it (`pass_due`)."""
    return pass_due(clock.get_run(conn, recommended.COLLECTOR), pair_rows.walked(conn, ("substack",)),
                    now or utc_now())


def pass_due(last_run: clock.CollectorRun | None, substack_rows: list, now: datetime) -> bool:
    """Is a pass due? `last_run` is the collector's clock row, `substack_rows` the Substack
    pairs the scrapers walk.

    | last attempt                                       | due |
    |----------------------------------------------------|-----|
    | none, or a week or more ago                        | yes |
    | within the week, a Substack Oracle added after it  | yes |
    | within the week, otherwise                         | no  |
    """
    at = clock.due_at(last_run, floor_hours=FLOOR_HOURS)
    if at is None or now >= at:
        return True
    last = parse_ts(last_run.last_attempt_at)
    return any((added := parse_ts(r.added_at)) is not None and added > last
               for r in substack_rows)


class Recommendations(PoolSource):
    """One publication per turn, with a durable pass preserving full-roster corroboration."""
    name, quota, klass, floor = "recommendations", QUOTA, SCOUTING, 0

    def candidates(self, conn, now):
        if active(conn) is None:
            return [Unit("recommendations:start", now)] if pair_rows.walked(conn, ("substack",)) and due(conn, now) else []
        pending = conn.execute("SELECT * FROM web_recommendation_reads WHERE status IS NULL ORDER BY canonical_id,url").fetchall()
        if not pending:
            return [Unit("recommendations:finish")]
        return [Unit(f"recommendations:{r['canonical_id']}:{r['url']}", dict(r), host=urlsplit(r["url"]).hostname)
                for r in pending]

    def next_unit(self, conn, now):
        return next(iter(self.candidates(conn, now)), None)

    def fetch(self, conn, unit, ctx):
        if unit.key in ("recommendations:start", "recommendations:finish"):
            return None
        if not roster.on_roster(conn, unit.payload["canonical_id"]):
            return {"status": "forgotten", "recommendations": []}
        try:
            return {"status": "ok", "recommendations": read_publication(unit.payload["url"])}
        except Exception as exc:
            if is_lock_error(exc):
                raise
            return {"status": "error", "recommendations": [], "detail": f"{type(exc).__name__}: {exc}"}

    def record(self, conn, unit, fetched):
        if unit.key == "recommendations:start":
            with write_tx(conn, site="recommendations.start"):
                if active(conn) is None:
                    conn.execute("DELETE FROM web_recommendation_reads")
                    conn.execute("INSERT OR REPLACE INTO web_recommendation_pass VALUES(1,?,0)",
                                 (utc_iso(unit.payload),))
                    conn.executemany("INSERT INTO web_recommendation_reads(canonical_id,url) VALUES(?,?)",
                                     [(r.canonical_id, r.source_key) for r in pair_rows.walked(conn, ("substack",))])
            return Outcome("started")
        if unit.key == "recommendations:finish":
            started = active(conn)
            if started is None:
                return Outcome("changed")
            observations = [dict(r) for r in conn.execute("SELECT * FROM web_recommendation_reads")]
            if any(r["status"] is None for r in observations):
                return Outcome("changed")
            results = [{**r, "recommendations": json.loads(r["payload"] or "[]")} for r in observations
                       if roster.on_roster(conn, r["canonical_id"])]
            # Existing entity/signal writers commit individually. This idempotent aggregate
            # finishes before the attempt clock, so a failure retries the same saved observations.
            out = recommended.store_recommendations(conn, results)
            status = "partial" if out["skipped_oracles"] else "ok"
            clock.record_run(conn, recommended.COLLECTOR, status=status,
                                      found=out["publications"], stored_after=out["signalled"],
                                      now=started[0])
            with write_tx(conn, site="recommendations.complete"):
                conn.execute("UPDATE web_recommendation_pass SET done=1")
            return Outcome(status)
        with write_tx(conn, site="recommendations.read"):
            conn.execute("UPDATE web_recommendation_reads SET status=?,payload=?,detail=? "
                         "WHERE canonical_id=? AND url=?",
                         (fetched["status"], json.dumps(fetched["recommendations"]), fetched.get("detail"),
                          unit.payload["canonical_id"], unit.payload["url"]))
        return Outcome(fetched["status"])

    def next_due(self, conn, now):
        at = clock.due_at(clock.get_run(conn, recommended.COLLECTOR), floor_hours=FLOOR_HOURS)
        if at is None or not pair_rows.walked(conn, ("substack",)):
            return None
        return at
