"""
pipeline/oracles/sources/timeline/scouting.py — sampling candidates' own posts, the lowest class
on X's `UserOriginalsTimeline` quota.

One unit is one page of one candidate's Originals timeline, stored in the probe store (never in
`atoms`) so `oracle(action='candidates')` can answer from what people write. A candidate is
sampled once: up to three pages, stopping at the first page older than 90 days, strongest
candidate first (`screen`'s rank). `ok`, `empty` and `unavailable` are final; a candidate
is never re-sampled. Scouting is X only.

A candidate known only by the X handle they declared on Substack (`samples.candidate_queue`) first
gets one unit that resolves the handle (`UserByScreenName`) and mints `x:user:<id>` attesting their
publication, so `resolve` merges it into their cluster and the next ranking queues them by id. A
handle X cannot read is final as `unavailable`, under the `x:@<handle>` key.

Scouting runs only when nothing of yours and no new posts are owed on this quota (the scraper's
class order), only once there is an Oracle, and only while the meter shows more than 50 left.
That floor is the only reserve on the quota.

A page and the candidate's standing (`probe_pulls`) commit together: the fetch reads and renders
the page, then writes both in one transaction.

A failed page is charged by the quota loop, under the one retry policy
(`pipeline/worker/quota_loop/retries.py`), on the candidate's own `probe_pulls` row through
`PROBE_ROWS`. A candidate's failures all wait as a blip. An unreadable account is final here
rather than retried: it is a sample. So is a candidate that failed ten times in a row: it is
skipped for good, and nothing re-arms it.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from pipeline.ingestion.utils import SyncAuthError
from pipeline.kb import resolve
from pipeline.kb import schema
from pipeline.kb import writes
from pipeline.oracles import probe_store
from pipeline.oracles import roster
from pipeline.oracles.sources.timeline import samples
from pipeline.oracles.sources.timeline.walks import PLATFORM
from pipeline.oracles.sources.timeline.walks import QUOTA
from pipeline.oracles.sources.timeline.walks import page_kit
from pipeline.platforms.x import profile as x_profile
from pipeline.platforms.x.originals import XUserUnavailable
from pipeline.timeparse import parse_ts, utc_now
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import (SCOUTING, Changed, Context, Item, Outcome, Retry,
                                               Unit)
from pipeline.worker.quota_loop.source import RetryRows
from pipeline.worker.quota_loop.source import Source


PAGES = 3                                  # pages per candidate's sample
SPAN = timedelta(days=90)                  # a sample stops at the first page older than this
FLOOR = 50                                 # scouting never takes a window below this
_RANK_TTL_S = 300.0


class _ProbeRows(RetryRows):
    """`probe_pulls`' retry state, keyed by `(who_id,)`. A candidate's failures all wait as a
    blip, so the row keeps no kind: status `failed` is the blip, and the count restarts after any
    other status."""

    def read(self, row: dict | None) -> Retry:
        if row is None:
            return Retry()
        status = retries.BLIP if row["status"] == probe_store.STATUS_FAILED else row["status"]
        return Retry(status, row.get("failures"), row.get("retry_at"), row.get("detail"))

    def write(self, conn: sqlite3.Connection, key: tuple, retry: Retry) -> None:
        """A charge (status `failed`) or a charge given back (status `sampling`). A candidate
        never sampled gets its row here, with nothing read yet."""
        status = (probe_store.STATUS_FAILED if retry.status == retries.BLIP
                  else probe_store.STATUS_SAMPLING)
        with writes.write_tx(conn, site="scouting"):
            probe_store.record_retry(conn, key[0], status, retry_at=retry.retry_at,
                                     failures=retry.failures, detail=retry.last_error)

    def clear(self, conn: sqlite3.Connection, key: tuple) -> None:
        with writes.write_tx(conn, site="scouting"):
            probe_store.clear_retry(conn, key[0])


PROBE_ROWS = _ProbeRows()


@dataclass(frozen=True)
class Sampled:
    """One page of a candidate's sample, fetched and stored in the probe store, or the account
    unreadable; with where the sample stood before it."""
    page: dict | None
    unavailable: str | None
    atoms: int
    pages: int
    below: str | None
    now: datetime


@dataclass(frozen=True)
class Standing:
    """Where a candidate's sample stands after a page: its `probe_pulls` row."""
    status: str
    atoms: int
    pages: int
    tail: str | None
    detail: str | None = None


def standing(fetched: Sampled) -> Standing:
    """A sample after one page: unavailable, sampling while pages are owed, then `ok`, or `empty`
    for an account with nothing posted at all."""
    page, atoms, pages, below = fetched.page, fetched.atoms, fetched.pages, fetched.below
    if fetched.unavailable is not None:
        return Standing(probe_store.STATUS_UNAVAILABLE, atoms, pages, below, fetched.unavailable)
    pages += 1
    atoms += page["written"]
    finished = (page["ended"] or pages >= PAGES
                or (page["oldest_at"] is not None and page["oldest_at"] < fetched.now - SPAN))
    if not finished:
        status = probe_store.STATUS_SAMPLING
    elif pages == 1 and not page["fetched"]:
        status = probe_store.STATUS_EMPTY       # an account with nothing posted at all
    else:
        status = probe_store.STATUS_OK
    return Standing(status, atoms, pages, page["oldest_id"] or below)


@dataclass(frozen=True)
class Resolved:
    """A declared handle looked up: the X id it names, or None when X cannot read it."""
    who_id: str | None


class Scouting(Source):
    name, quota, klass, floor = "scouting", QUOTA, SCOUTING, FLOOR

    def __init__(self) -> None:
        self._ranked: tuple[float, list[dict]] | None = None

    def _candidates(self, conn) -> list[dict]:
        # The ranking is a prior over who to sample first, not owed state, and it is the
        # expensive part (every curation signal). Recomputed every few minutes; which candidates
        # are owed is read from `probe_pulls` on every call.
        if self._ranked is None or time.monotonic() - self._ranked[0] > _RANK_TTL_S:
            self._ranked = (time.monotonic(), samples.candidate_queue(conn))
        return self._ranked[1]

    def next_unit(self, conn, now: datetime) -> Unit | None:
        if not roster.list_oracles(conn):
            return None
        pulls = probe_store.pull_states(conn)
        for cand in self._candidates(conn):
            row = pulls.get(cand["who_id"])
            if row is None or row["status"] == probe_store.STATUS_SAMPLING or (
                    row["status"] == probe_store.STATUS_FAILED
                    and retries.may_run(row.get("retry_at"), row.get("failures"), now)):
                item = Item(PROBE_ROWS, (cand["who_id"],), PROBE_ROWS.read(row), PLATFORM,
                            kind=retries.BLIP)
                return Unit(cand["who_id"], cand, item=item)
        return None

    def next_due(self, conn, now: datetime) -> datetime | None:
        retry = parse_ts(probe_store.next_retry_at(conn))
        return retry if retry is not None and retry > now else None

    def fetch(self, conn, unit: Unit, ctx: Context) -> Sampled | Resolved:

        cand = unit.payload
        probe_store.init_probe_schema(conn)
        if cand["user_id"] is None:
            return self._resolve(conn, cand, ctx)
        prior = probe_store.pull_states(conn).get(cand["who_id"])
        resuming = prior is not None and prior["status"] in (
            probe_store.STATUS_SAMPLING, probe_store.STATUS_FAILED) and prior.get("pages_done")
        pages = int(prior["pages_done"]) if resuming else 0
        below = prior.get("tail_id") if resuming else None
        atoms = int(prior.get("atoms") or 0) if resuming else 0
        now = utc_now()
        kit = page_kit(conn, ctx)
        try:
            page = samples.sample_page(conn, kit.session(), cand, below_id=below)
            sampled = Sampled(page, None, atoms, pages, below, now)
        except XUserUnavailable as exc:
            # Final, not a failure: an unreadable account is a sample.
            sampled = Sampled(None, str(exc), atoms, pages, below, now)
        except SyncAuthError:
            kit.forget_session()
            raise
        stands = standing(sampled)
        with writes.write_tx(conn, site="scouting"):
            if ctx.lease_lost():
                raise Changed("timeline quota lease lost")
            if page := sampled.page:
                samples.write_sample(conn, page)
            probe_store.record_sample(conn, cand["who_id"], stands.status, atoms=stands.atoms,
                                      pages_done=stands.pages, tail_id=stands.tail,
                                      detail=stands.detail)
        return sampled

    def _resolve(self, conn, cand: dict, ctx: Context) -> Resolved:
        """Look up a declared handle and mint its X account into the candidate's cluster. The
        handle's own `probe_pulls` row records the lookup, so it is made once."""
        kit = page_kit(conn, ctx)
        try:
            prof = x_profile.fetch_user_profile(kit.session(), cand["handle"])
        except SyncAuthError:
            kit.forget_session()
            raise
        who_id = f"x:user:{prof['user_id']}" if prof else None
        with writes.write_tx(conn, site="scouting.resolve"):
            if ctx.lease_lost():
                raise Changed("timeline quota lease lost")
            if who_id:
                schema.upsert_entity(conn, who_id, name=prof.get("display_name") or None,
                                     identity_links=list(cand.get("links") or []),
                                     profile={"handle": prof.get("handle") or cand["handle"],
                                              "bio": prof.get("bio"),
                                              "verified": prof.get("verified"),
                                              "followers": prof.get("followers")})
            probe_store.record_sample(
                conn, cand["who_id"],
                probe_store.STATUS_OK if who_id else probe_store.STATUS_UNAVAILABLE,
                atoms=0, pages_done=0, tail_id=None,
                detail=f"resolved to {who_id}" if who_id else "X cannot read this handle")
        if who_id:
            resolve.resolve_entities(conn)
            self._ranked = None             # queue them by id on the next unit
        return Resolved(who_id)

    def record(self, conn, unit: Unit, fetched: Sampled | Resolved) -> Outcome:
        if isinstance(fetched, Resolved):
            if fetched.who_id is None:
                return Outcome(probe_store.STATUS_UNAVAILABLE, answered=False)
            return Outcome("resolved")
        stands = standing(fetched)
        if stands.status == probe_store.STATUS_UNAVAILABLE:
            return Outcome(probe_store.STATUS_UNAVAILABLE, answered=False)
        return Outcome(stands.status)
