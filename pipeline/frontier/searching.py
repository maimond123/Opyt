"""
pipeline/frontier/searching.py — Frontier stage 2 on the scrapers process: one standing query
searched on one source per unit, its page staged and its pair stamped in one transaction.

Runs standing queries against artifact sources on a cadence, remembers how far each (query,
source) pair got, and writes candidates to a staging table. Nothing here writes to `atoms`
(stage 3, `admitting.py`, owns that), so a bad query costs inbox noise, never KB pollution.

The unit runs in the web pool, on its source's API host. What it may do now is decided before it
is offered, so no slot is held while it waits:

  | rule                         | where                     | until                               |
  |------------------------------|---------------------------|-------------------------------------|
  | the pair is due              | `cadence.due_at`          | its TTL, or an hour after a failure |
  | the hour's request allowance | `allowance.frees_at`      | the oldest request leaves the hour  |
  | the source's courtesy gap    | `min_interval_s`, memory  | the gap after its last request      |
  | the source's breaker         | `available()`             | the breaker's own cooldown          |
  | the platform's outage pause  | the web pool's `Outages`  | the pause ends                      |

The allowance counts finished requests, so up to one more per source can be in flight when it
fills. A failed pull never stamps; a pair whose source has no adapter is recorded once as
`no_adapter` and counted, never pulled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from pipeline.frontier import adapters, allowance, cadence, candidates, intake, pulls
from pipeline.frontier import queries as fq
from pipeline.frontier.source_search import Adapter, available
from pipeline.platforms.errors import SourceError
from pipeline.ingestion.utils import log
from pipeline.kb.writes import write_tx
from pipeline.timeparse import utc_iso, utc_now
from pipeline.worker.quota_loop.retries import Outages
from pipeline.worker.quota_loop.source import NEW_POSTS, Changed, Outcome, Unit

QUOTA = "web"
# A watch names one thing, so a month of it fits one page; the OpenAlex window re-reads 30 days.
PER_QUERY_LIMIT = 50
NO_ADAPTER = "no_adapter"


@dataclass(frozen=True)
class Pulled:
    """One pair's search: `no_adapter`, `window_refused`, `error`, `ok` or `empty`, and the page
    it kept."""
    status: str
    found: list = field(default_factory=list)


def pull(pair: cadence.DuePair, adapter: Adapter, now: datetime) -> Pulled:
    """The request's own rules, in order: the window, the request, the name check."""
    q, source = pair.query, pair.source
    since = cadence.lookback_floor(adapter, cadence.since_for(pair.mark, now=now), now)
    if not cadence.window_ok(since, now):
        log(f"[frontier] window refused for {q['text']!r}/{source}: since={since}")
        return Pulled("window_refused")
    try:
        found = adapter.search(q["text"], since=since, limit=PER_QUERY_LIMIT, now=now,
                               context=fq.context(q))
    except SourceError as e:
        log(f"[frontier] {source} failed for {q['text']!r}: {e}")
        return Pulled("error")
    except Exception as e:                                   # an adapter bug is not fatal
        log(f"[frontier] {source} raised for {q['text']!r}: {type(e).__name__}: {e}")
        return Pulled("error")
    found, cut = intake.named(found, q["text"])
    if cut:
        # Never a silent cap: what the check dropped is said.
        log(f"[frontier] {source} {q['text']!r}: dropped {cut} of {cut + len(found)} "
            f"candidates that do not name it")
    return Pulled("ok" if found else "empty", found)


def store(conn, pair: cadence.DuePair, pulled: Pulled, now: datetime) -> int:
    """One pair's only writes: its page and its stamp together, or its status alone. A failure
    partway through stages none of the page and leaves the pair unstamped, so the next try pulls
    it again from the same mark. Returns how many candidates were new."""
    qid, source = pair.query["query_id"], pair.source
    pulled_page = pulled.status in ("ok", "empty")
    with write_tx(conn, site="frontier.search"):
        new = candidates.stage(conn, pulled.found, qid, now=utc_iso(now))[0] if pulled_page else 0
        pulls.record_pull(conn, qid, source, last_status=pulled.status, stamp=pulled_page,
                          now=now)
    return new


def _key(pair: cadence.DuePair) -> str:
    return f"frontier-search:{pair.query['query_id']}|{pair.source}"


class Searches:
    """Frontier stage 2: the due (query, source) pairs, stalest first."""
    name, quota, klass, floor = "frontier_searches", QUOTA, NEW_POSTS, 0

    def __init__(self, outages: Outages, registry: dict[str, Adapter] | None = None,
                 clock: Callable[[], datetime] = utc_now):
        self.outages = outages
        self._registry = registry
        self.clock = clock
        self._asked: dict[str, datetime] = {}     # source → when its last request went out

    def registry(self) -> dict[str, Adapter]:
        if self._registry is None:
            self._registry = adapters.adapters()
        return self._registry

    def _held(self, source: str, now: datetime) -> datetime | None:
        """When `source` may be asked again: after its courtesy gap, and its platform's pause."""
        gap = float(getattr(self.registry()[source], "min_interval_s", 0.0) or 0.0)
        waits = [asked + timedelta(seconds=gap)] if (asked := self._asked.get(source)) else []
        if self.outages.paused(source, now):
            waits.append(self.outages.until[source])
        later = [at for at in waits if at > now]
        return max(later) if later else None

    def _spent_until(self, conn, now: datetime) -> datetime | None:
        return allowance.frees_at(pulls.requests_since(conn, now - allowance.HOUR),
                                  allowance.SEARCHES_PER_HOUR, now)

    def candidates(self, conn, now: datetime) -> list[Unit]:
        registry = self.registry()
        plan = cadence.plan(fq.active_queries(conn), pulls.watermarks(conn), set(registry),
                            now=now)
        units = [Unit(_key(pair), pair) for pair in plan.no_adapter
                 if pair.mark is None or pair.mark["last_status"] != NO_ADAPTER]
        if self._spent_until(conn, now) is not None:
            return units
        for pair in plan.due:
            adapter = registry[pair.source]
            if self._held(pair.source, now) is None and available(adapter):
                units.append(Unit(_key(pair), pair, host=adapter.host))
        return units

    def next_unit(self, conn, now: datetime) -> Unit | None:
        return next(iter(self.candidates(conn, now)), None)

    def fetch(self, conn, unit: Unit, ctx) -> Pulled:
        pair = unit.payload
        now = self.clock()
        adapter = self.registry().get(pair.source)
        if adapter is None:
            pulled = Pulled(NO_ADAPTER)
        else:
            self._asked[pair.source] = now
            pulled = pull(pair, adapter, now)
        with write_tx(conn, site="frontier.search"):
            if ctx.lease_lost():
                raise Changed("web quota lease lost")
            store(conn, pair, pulled, now)
        return pulled

    def record(self, conn, unit: Unit, fetched: Pulled) -> Outcome:
        return Outcome(fetched.status)

    def next_due(self, conn, now: datetime) -> datetime | None:
        """When a pair falls due, or a held source or a spent hour frees for one already due."""
        registry, marks = self.registry(), pulls.watermarks(conn)
        times = []
        for q in fq.active_queries(conn):
            for source in json.loads(q["target_sources"] or "[]"):
                if source not in registry:
                    continue
                at = cadence.due_at(marks.get((q["query_id"], source)), source,
                                    query_id=q["query_id"])
                if at is None or at <= now:
                    at = self._held(source, now) or now
                times.append(at)
        if (spent := self._spent_until(conn, now)) is not None:
            times = [max(at, spent) for at in times]
        later = [at for at in times if at > now]
        return min(later) if later else None
