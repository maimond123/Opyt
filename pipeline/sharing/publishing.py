"""
pipeline/sharing/publishing.py — keeps the served copy of this knowledge base current, by itself:
one unit is one gate-and-publish pass, on the scrapers process.

A pass runs every `EVERY` while the Mac is awake, and at once when a push is owed (`share`,
the Google connection, `share(refresh=True)`). For `FAST_FOR` after a share or the Google
connection it runs every `FAST` instead: that is when a new user opens their copy, and when it
moves most. On the onboarding runs of 2026-10-13/14, about 90% of the library arrived in the
first 10 minutes, most of the rest within the hour, and 0-2% in the second hour. The gate is `push_gate.py` and reads only this
install, so a pass on an unchanged store asks the service nothing. An install with no
`OPYT_SERVICE_TOKEN` is offered no pass: one that never shared, and one whose token is gone, by
`unshare` or by the service refusing it (`client`).

Each push sends only the rows that changed since the last push that landed (`publish.py`):
about 1 MB a day for a settled library, measured. The whole export (64 MB free, 474 MB
own key) goes on the first share, after a refusal, or when a change is most of
the library. `EVERY` was set when every push was whole.

On its own serial quota, so a long upload holds no web-fetch slot. The watermark is written in
`record`, after the upload returned, so a pass killed mid-upload leaves none and the next one
uploads again. A push lands one pass behind the ingest that caused it; do not try to order them.

An install that shares also asks the service for the tool calls the web and agents make
(`mcp_server/line.py`, the worker's `line` job). Every such install passes here, one that shared
before that job existed included, so a pass is what gives its home the job (`job_queue.enroll`).
The job keeps its own cadence from then on.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import NamedTuple

from pipeline.ingestion.utils import log
from pipeline.sharing import account, publish, push_gate
from pipeline.worker import job_queue
from pipeline.worker.quota_loop.source import YOURS, Outcome, Unit
from pipeline.worker.registry import LINE

QUOTA = "publishing"
EVERY = timedelta(hours=1)
FAST, FAST_FOR = timedelta(minutes=5), timedelta(hours=1)


def due_at(last_pass: datetime | None, owed: datetime | None, now: datetime,
           fast_since: datetime | None = None) -> datetime:
    """Now if no pass was ever recorded, or a push was owed since the last (in the same second
    counts as since: one pass too many, never one missed); else `FAST` after the last pass while
    within `FAST_FOR` of `fast_since` (the last share or Google connection), `EVERY` after it
    otherwise. An owed push whose pass failed waits like any other, rather than retrying at
    once."""
    if last_pass is None or (owed is not None and owed >= last_pass):
        return now
    fast = fast_since is not None and now < fast_since + FAST_FOR
    return last_pass + (FAST if fast else EVERY)


class Passed(NamedTuple):
    """What a pass came to, and the store position it judged."""
    result: dict
    position: dict | None


def run_pass(acct: account.Account) -> Passed:
    """Read the gate, and push if the store moved. Every outcome is a status: a failure of the
    local build itself (see `publish.publish`) is logged as `failed`, so the pass waits for the
    next one as any other failure does."""
    position = push_gate.store_position()
    skip = push_gate.decide(position, push_gate.read_watermark(),
                            owed=push_gate.owed_at() is not None)
    if skip is not None:
        return Passed(skip, position)
    try:
        res = publish.publish(acct)
    except Exception as e:
        res = {"status": "failed", "message": f"{type(e).__name__}: {e}"}
    if res["status"] != "ok":
        log(f"[publishing] {res['status']}: {res.get('message')}")
    return Passed(res, position)


class Publications:
    """The served copy's catch-up: one pass per unit."""
    name, quota, klass, floor = "publications", QUOTA, YOURS, 0

    def _due(self, now: datetime) -> datetime | None:
        if not account.load().token:
            return None
        return due_at(push_gate.passed_at(), push_gate.owed_at(), now, push_gate.fast_since())

    def next_unit(self, conn, now: datetime) -> Unit | None:
        due = self._due(now)
        return Unit("publishing:pass", now) if due is not None and due <= now else None

    def fetch(self, conn, unit: Unit, ctx) -> Passed:
        return run_pass(account.load())

    def record(self, conn, unit: Unit, fetched: Passed) -> Outcome:
        push_gate.record(fetched.result["status"], fetched.position, unit.payload)
        job_queue.enroll(LINE)
        return Outcome(fetched.result["status"])

    def next_due(self, conn, now: datetime) -> datetime | None:
        return self._due(now)
