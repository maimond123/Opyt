"""Oracle archive and refresh priorities; archive.run resumes their listing and document units."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from functools import partial

from pipeline.kb import schema
from pipeline.kb import resolve
from pipeline.kb import writes
from pipeline.oracles import coverage
from pipeline.oracles import found
from pipeline.oracles import registry
from pipeline.oracles import roster
from pipeline.oracles import windows
from pipeline.oracles.asks import ask_for_head, asks_by_head
from pipeline.oracles.pairs import pairs_by_entity
from pipeline.oracles.sources.archive import run as archive_run
from pipeline.oracles.sources.web import platform
from pipeline.oracles.sources.web.platform import QUOTA
from pipeline.timeparse import parse_ts, utc_iso, utc_now
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.retries import Outages
from pipeline.worker.quota_loop.source import (NEW_POSTS, YOURS, Footprint, Item, Outcome,
                                              Retry, Unit)
from pipeline.oracles.sources import first_pull
from pipeline.worker.quota_loop.source import PoolSource
from pipeline.worker.quota_loop.source import RetryRows
from pipeline.oracles.sources.pair_rows import pair_key
from pipeline.oracles.sources import pair_rows

WEB_TYPES = ("substack", "blog", "github", "openalex")


# ── a pair ────────────────────────────────────────────────────────────────────────
def pair_unit(row: registry.SourceRow, payload) -> Unit:
    """A unit on a registered pair: charged to the pair, counted toward its platform."""
    return Unit(pair_rows.unit_key(row), payload, item=pair_rows.item(row, platform.platform_key(row.source_type, row.source_key)),
                footprint=Footprint(row.canonical_id))


# ── a found source ────────────────────────────────────────────────────────────────
def _set_found(conn, f: found.FoundSource, **cols) -> None:
    sets = ", ".join(f"{k}=?" for k in cols)
    with writes.write_tx(conn, site="archives.found"):
        conn.execute(f"UPDATE found_sources SET {sets} "
                     "WHERE canonical_id=? AND source_type=? AND url=?",
                     (*cols.values(), f.canonical_id, f.source_type, f.url))


class _FoundRows(RetryRows):
    """`found_sources`' retry columns, keyed by `(canonical_id, source_type, url)`. `last_status`
    names the kind while the first pull is failing, and its action once it is done."""

    def read(self, f: found.FoundSource) -> Retry:
        return Retry(f.last_status, f.failures, f.retry_at, f.last_error)

    def write(self, conn: sqlite3.Connection, key: tuple, retry: Retry) -> None:
        with writes.write_tx(conn, site="archives.found"):
            conn.execute("UPDATE found_sources SET failures=?, last_status=?, retry_at=?, "
                         "last_error=? WHERE canonical_id=? AND source_type=? AND url=?",
                         (retry.failures, retry.status, retry.retry_at, retry.last_error, *key))

    def clear(self, conn: sqlite3.Connection, key: tuple) -> None:
        with writes.write_tx(conn, site="archives.found"):
            conn.execute("UPDATE found_sources SET failures=0, retry_at=NULL, last_error=NULL "
                         "WHERE canonical_id=? AND source_type=? AND url=?", key)


FOUND_ROWS = _FoundRows()


def found_unit(f: found.FoundSource) -> Unit:
    """A unit on a found source's first pull: charged to its row, counted toward its platform."""
    item = Item(FOUND_ROWS, (f.canonical_id, f.source_type, f.url), FOUND_ROWS.read(f),
                platform.platform_key(f.source_type, f.url))
    return Unit(f"found|{f.canonical_id}|{f.source_type}|{f.url}", ("found", f), item=item,
                footprint=Footprint(f.canonical_id))


def record_found(conn, f: found.FoundSource, pulled) -> Outcome:
    """Record what a found source's first pull came to."""
    rec = (pulled.get("results") or [{"action": "skipped", "detail": "no url"}])[0]
    action = rec.get("action")
    if (rec.get("stats") or {}).get("archive_started"):
        return Outcome("started")
    # Done, whatever it found: an affiliation, a source left for review and a refusal by the gate
    # are first pulls too, and nothing about them is retried.
    _set_found(conn, f, pulled_at=utc_iso(utc_now()), last_status=action)
    return Outcome(action)


# ── the sources ───────────────────────────────────────────────────────────────────
class Archives(PoolSource):
    """YOURS: every first pull of a found source, and every pair not pulled since its Oracle's
    last ask, oldest ask first, found sources before pairs."""
    name, quota, klass, floor = "archives", QUOTA, YOURS, 0

    def __init__(self, health: Outages | None = None) -> None:
        self.health = health or Outages(retries.PLATFORM_PAUSES_S)

    def _owed(self, conn, now: datetime):
        """`[(order, unit)]` that may run now, and the retry times of those that may not."""
        asks = asks_by_head(conn)
        units, later, first_pulls = [], [], set()
        for f in found.list_found(conn):
            if f.pulled_at is not None or not roster.on_roster(conn, f.canonical_id):
                continue
            head = schema.current_canonical(conn, f.canonical_id)
            first_pulls.add((head, f.source_type))
            if not retries.may_run(f.retry_at, f.failures, now) or self.health.paused(
                    platform.platform_key(f.source_type, f.url), now):
                if (retry := parse_ts(f.retry_at)) is not None:
                    later.append(retry)
                continue
            ask = asks.get(head)
            units.append(((ask.position if ask else float("inf"), 0, f.source_type, f.url),
                          found_unit(f)))
        rows = [r for r in pair_rows.walked(conn, WEB_TYPES)
                if windows.owed_on_ask(r, asks.get(r.canonical_id))
                and (r.canonical_id, r.source_type) not in first_pulls]
        later += [t for t in (pair_rows.next_free(rows, now),) if t is not None]
        for r in pair_rows.free(rows, now):
            if self.health.paused(platform.platform_key(r.source_type, r.source_key), now):
                continue
            ask = asks.get(r.canonical_id)
            units.append(((ask.position if ask else float("inf"), 1, r.source_type,
                           r.source_key),
                          pair_unit(r, ("pair", r))))
        return units, later

    def candidates(self, conn, now):
        units, _later = self._owed(conn, now)
        for _priority, parent in sorted(units, key=lambda u: u[0]):
            yield from archive_run.offer(conn, parent, now)

    def next_unit(self, conn, now):
        return next(self.candidates(conn, now), None)

    def fetch(self, conn, unit, ctx):
        if isinstance(unit.payload, archive_run.Work):
            return archive_run.fetch(conn, unit.payload, ctx)
        what, row = unit.payload
        if what == "found":
            ask = ask_for_head(conn, row.canonical_id)
            since = windows.window(ask.web_from) if ask else None
            oracle = roster.oracle_row(conn, row.canonical_id)
            return first_pull.onboard_footprint(
                conn, row.canonical_id, [row.source()],
                author_name=oracle["name"] if oracle else None, since=since, isolate=False,
                begin=partial(begin_walk, parent=unit, source_type=row.source_type))
        return _begin_pair(conn, unit, row,
                           windows.ask_since(row, ask_for_head(conn, row.canonical_id)))

    def record(self, conn, unit, fetched):
        if isinstance(unit.payload, archive_run.Work):
            return archive_run.record(conn, unit.payload, fetched, partial(_finish, conn))
        what, row = unit.payload
        if what == "found":
            rec = (fetched.get("results") or [{}])[0]
            if not (rec.get("stats") or {}).get("archive_started") and roster.on_roster(conn, row.canonical_id):
                ask = ask_for_head(conn, row.canonical_id)
                coverage.stamp_source(conn, row.canonical_id, rec, x_since=None,
                                      web_since=windows.window(ask.web_from) if ask else None,
                                      scholar_since=None, include_x=False)
            return record_found(conn, row, fetched)
        return fetched

    def next_due(self, conn, now: datetime) -> datetime | None:
        _units, later = self._owed(conn, now)
        times = [t for t in later if t > now] + _retry_times(conn, now)
        if (resume := self.health.next_resume(now)) is not None:
            times.append(resume)
        return min(times) if times else None


class WebNewPosts(PoolSource):
    """`NEW_POSTS`: a pair past its refresh time, worst lag first, over the refresh window
    (`windows.since_for`). What an ask owes is `archives`', and a pair whose kind its
    Oracle's ask left out waits for one (`waits_for_an_ask`)."""
    name, quota, klass, floor = "web_new_posts", QUOTA, NEW_POSTS, 0

    def __init__(self, health: Outages | None = None) -> None:
        self.health = health or Outages(retries.PLATFORM_PAUSES_S)

    def _rows(self, conn, now: datetime) -> list[registry.SourceRow]:
        asks = asks_by_head(conn)
        return [r for r in pair_rows.walked(conn, WEB_TYPES)
                if (windows.walking(r) or windows.is_stale(r, now))
                and not windows.owed_on_ask(r, asks.get(r.canonical_id))
                and not windows.waits_for_an_ask(r, asks.get(r.canonical_id))]

    def candidates(self, conn, now):
        rows = [r for r in pair_rows.free(self._rows(conn, now), now)
                if not self.health.paused(platform.platform_key(r.source_type, r.source_key), now)]
        for row in sorted(rows, key=lambda r: (windows.walking(r), windows.staleness_hours(r, now)), reverse=True):
            parent = pair_unit(row, ("pair", row))
            yield from archive_run.offer(conn, parent, now)

    def next_unit(self, conn, now):
        return next(self.candidates(conn, now), None)

    def fetch(self, conn, unit, ctx):
        if isinstance(unit.payload, archive_run.Work):
            return archive_run.fetch(conn, unit.payload, ctx)
        _, row = unit.payload
        return _begin_pair(conn, unit, row, windows.since_for(row))

    def record(self, conn, unit, fetched):
        if isinstance(unit.payload, archive_run.Work):
            return archive_run.record(conn, unit.payload, fetched, partial(_finish, conn))
        return fetched

    def next_due(self, conn, now: datetime) -> datetime | None:
        times = _retry_times(conn, now)
        for r in pair_rows.walked(conn, WEB_TYPES):
            last = parse_ts(r.last_pulled_at)
            if last is not None:
                times.append(last + timedelta(hours=windows.pair_ttl_hours(r)))
        waiting = pair_rows.next_free(self._rows(conn, now), now)
        if waiting is not None:
            times.append(waiting)
        if (resume := self.health.next_resume(now)) is not None:
            times.append(resume)
        later = [t for t in times if t > now]
        return min(later) if later else None


def _retry_times(conn, now):
    return [at for row in conn.execute("SELECT retry_at FROM archive_units WHERE done=0 AND retry_at IS NOT NULL")
            if (at := parse_ts(row[0])) and at > now]


def begin_walk(conn, url, *, parent, source_type, **kwargs):
    """The `begin` a first pull is handed: begin the source's archive walk under `parent`, so
    this source's units carry the walk."""
    return archive_run.begin(conn, parent, source_type, url, **kwargs)


def _begin_pair(conn, unit, row, since):
    url = f"https://github.com/{row.source_key}" if row.source_type == "github" else row.source_key
    if row.source_type == "github" and (windows.walking(row) or row.last_pulled_at is None):
        since = None
    summary = archive_run.begin(conn, unit, row.source_type, url, author_name=row.name,
                                since=since, topics=row.topic_filter)
    if summary is None:
        result = coverage.save_pull(conn, row, {}, "skipped", "invalid source URL", since=since)
        return Outcome(result["status"])
    return Outcome("started")


def _finish(conn, work, summary):
    what, row = work.parent.payload
    since = parse_ts(work.walk.spec.get("since"))
    if what == "found":
        resolve.resolve_entities(conn)
        rec = {"type": row.source_type, "url": row.url, "action": "ingested", "stats": summary}
        # The foreground stamping flow also records review/identity information.
        coverage.stamp_source(conn, row.canonical_id, rec, x_since=None, web_since=since,
                              scholar_since=None, include_x=False)
        head = schema.current_canonical(conn, row.canonical_id)
        # A long archive stamps its observed top, so new posts published during it stay owed.
        entity = coverage.result_entity_id(rec)
        pair = pairs_by_entity(conn, head).get(entity)
        if pair:
            with writes.write_tx(conn, site="archive.first_pull"):
                registry.cap_last_pulled(conn, (head, *pair), work.walk.generation)
        result = record_found(conn, row, {"results": [rec]})
    else:
        result = Outcome(coverage.save_pull(
            conn, row, summary, "ingested", None, since=since,
            stamp_at=parse_ts(work.walk.generation))["status"])
        with writes.write_tx(conn, site="archive.frontier"):
            registry.end_forward(conn, pair_key(row))
    return result
