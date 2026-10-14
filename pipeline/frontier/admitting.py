"""
pipeline/frontier/admitting.py — Frontier stage 3 on the scrapers process: one candidate the
host kept admitted per unit, oldest first (`admit.admit_one`), and its status settled by the
give-up rule. The only frontier work that writes `atoms`. A find nobody has checked yet is not
admitted: the host checks finds before they reach the library.

The unit runs in the web pool, keyed on its first request's host (Semantic Scholar for a paper,
GitHub's API for a repo, x.com for an X post, Substack for a Substack post), so admissions run one
at a time per service. What it may do now is
decided before it is offered:

  | rule                           | until                                                   |
  |--------------------------------|---------------------------------------------------------|
  | the hour's admission allowance | the oldest attempt leaves the hour (`allowance`)        |
  | a failed attempt's wait        | an hour after it                                        |
  | the platform's outage pause    | the pause ends                                          |

Run as a module, `--requeue-rejected [SLUG]` is the operator's reset of rejected candidates.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Callable, NamedTuple

from pipeline.frontier import admit, allowance, candidates
from pipeline.ingestion.utils import log
from pipeline.kb import schema
from pipeline.kb.writes import write_tx
from pipeline.timeparse import parse_ts, utc_iso, utc_now
from pipeline.worker.quota_loop.retries import Outages
from pipeline.worker.quota_loop.source import NEW_POSTS, Changed, Outcome, Unit

QUOTA = "web"
# Forces `rejected` after this many failed attempts, regardless of classification.
ADMIT_MAX_ATTEMPTS = int(os.environ.get("OPYT_ADMIT_MAX_ATTEMPTS", 5))
# The service an admission's first request goes to, by the candidate's kind. A Substack post's
# goes to its publication's own host; one key keeps Substack's admissions one at a time.
HOSTS = {"paper": "api.semanticscholar.org", "repo": "api.github.com", "x": "x.com",
         "substack": "substack.com"}


class Settled(NamedTuple):
    """The admission outcome to write, including whether the retry cap forced rejection."""
    status: str
    attempts: int
    capped: bool


def settle(status: str, attempts: int | None) -> Settled:
    """The give-up rule: an attempt's status and the row's attempt count → what is written.

    A failure (`kept`, still owed) or rejection counts one attempt; reaching `ADMIT_MAX_ATTEMPTS`
    forces `rejected`. A success never counts, and a busy store (`locked`) is not the candidate's
    failure: it stays `kept` with its count unchanged.
    """
    n = attempts or 0
    if status == "locked":
        return Settled(candidates.KEPT, n, False)
    if status == candidates.KEPT:
        n += 1
        return Settled("rejected" if n >= ADMIT_MAX_ATTEMPTS else candidates.KEPT, n,
                       n >= ADMIT_MAX_ATTEMPTS)
    if status == "rejected":
        return Settled("rejected", n + 1, False)
    return Settled(status, n, False)


def _attempts(conn, now: datetime) -> list[datetime]:
    return [at for raw in candidates.attempts_since(conn, utc_iso(now - allowance.HOUR))
            if (at := parse_ts(raw)) is not None]


class Admissions:
    """Frontier stage 3: the `kept` candidates, oldest first."""
    name, quota, klass, floor = "frontier_admissions", QUOTA, NEW_POSTS, 0

    def __init__(self, outages: Outages, clock: Callable[[], datetime] = utc_now):
        self.outages = outages
        self.clock = clock

    def candidates(self, conn, now: datetime) -> list[Unit]:
        spent = _attempts(conn, now)
        if allowance.frees_at(spent, allowance.ADMISSIONS_PER_HOUR, now) is not None:
            return []
        rows = candidates.due_for_admission(
            conn, tried_before=utc_iso(now - allowance.HOUR),
            limit=allowance.ADMISSIONS_PER_HOUR - len(spent))
        return [Unit(f"frontier-admit:{row['candidate_id']}", row,
                     host=HOSTS.get(candidates.kind_of(row["kind"])))
                for row in rows if not self.outages.paused(row["source"], now)]

    def next_unit(self, conn, now: datetime) -> Unit | None:
        return next(iter(self.candidates(conn, now)), None)

    def fetch(self, conn, unit: Unit, ctx) -> Settled:
        row = unit.payload

        attempt = admit.admit_one(conn, row)
        settled = settle(attempt.status, row["attempts"])
        if settled.capped:
            log(f"[frontier-admit] {row['candidate_id']} rejected at the attempt cap "
                f"({settled.attempts}/{ADMIT_MAX_ATTEMPTS}), last_error={attempt.slug}")
        with write_tx(conn, site="frontier.admit"):
            if ctx.lease_lost():
                raise Changed("web quota lease lost")
            candidates.set_status(conn, row["candidate_id"], status=settled.status,
                                  attempts=settled.attempts, last_error=attempt.slug,
                                  atom_id=attempt.atom_id, at=utc_iso(self.clock()))
        return settled

    def record(self, conn, unit: Unit, fetched: Settled) -> Outcome:
        return Outcome(fetched.status)

    def next_due(self, conn, now: datetime) -> datetime | None:
        """When a spent hour frees, or a failed candidate's wait ends."""
        spent = allowance.frees_at(_attempts(conn, now), allowance.ADMISSIONS_PER_HOUR, now)
        if spent is not None:
            return spent if candidates.backlog(conn) else None
        oldest = parse_ts(candidates.next_retry(conn, utc_iso(now - allowance.HOUR)))
        retry = oldest + allowance.HOUR if oldest is not None else None
        return retry if retry is not None and retry > now else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Frontier stage 3 — reset rejected candidates")
    ap.add_argument("--requeue-rejected", nargs="?", const="", default=None, metavar="SLUG",
                    help="reset rejected rows to kept (optionally only those with this last_error)")
    args = ap.parse_args(argv)
    if args.requeue_rejected is None:
        ap.print_help()
        return 2
    conn = schema.connect()
    try:
        n = candidates.requeue_rejected(conn, last_error=args.requeue_rejected or None)
        print(f"requeued {n} rejected candidate(s)")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
