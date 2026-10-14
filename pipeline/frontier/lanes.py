"""
pipeline/frontier/lanes.py — search a few names on the watch sources now, side by side, under a
time budget, and show what each found: what a topic's preview (`watch(action="preview")`) and
`look` both ask, so their counts and results never disagree. Writes nothing.

Each source answers one lane: its spellings in turn, paced by its courtesy gap, through the
adapters and the name check the daily pulls use. A source that cannot usefully be asked as far
back as the caller wants is searched over its own `max_window_days` instead (`window_days`), so a
long ask does not come back empty from the sources that cannot serve it. What a lane could not do
says so:

  | status        | means                                                         |
  |---------------|---------------------------------------------------------------|
  | `unavailable` | the source cannot be asked now (X: not connected)             |
  | `unchecked`   | the lane ran out of time before this spelling                 |
  | `failed`      | the request failed                                            |
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from pipeline.frontier import intake
from pipeline.frontier.candidates import Candidate
from pipeline.frontier.source_search import available
from pipeline.ingestion.utils import log
from pipeline.retrieval.cards import cite

UNAVAILABLE, UNCHECKED, FAILED = "unavailable", "unchecked", "failed"
# As much of a find's text as `latest` hands over: enough to judge it by.
SNIPPET_CHARS = 400


def search(rows: list[dict], registry: dict, *, since: datetime, now: datetime, limit: int,
           budget_s: float) -> dict:
    """`{(text, source): [candidates] | status}`: every row's `text` on each of its `sources`,
    narrowed by its `context`, the finds kept by the name check. Never raises: a source's failure
    is its own status."""
    lanes: dict[str, list[dict]] = {}
    for row in rows:
        for source in row["sources"]:
            lanes.setdefault(source, []).append(row)
    deadline = time.monotonic() + budget_s
    asked_days = (now - since).total_seconds() / 86400.0

    def lane(source: str) -> dict:
        adapter, out = registry[source], {}
        if not available(adapter):
            return {(row["text"], source): UNAVAILABLE for row in lanes[source]}
        gap = float(getattr(adapter, "min_interval_s", 0.0) or 0.0)
        since_here = now - timedelta(days=window_days(adapter, asked_days))
        for i, row in enumerate(lanes[source]):
            text = row["text"]
            if i and gap:
                time.sleep(gap)
            if time.monotonic() > deadline:
                out[(text, source)] = UNCHECKED
                continue
            try:
                page = adapter.search(text, since=since_here, limit=limit, now=now,
                                      context=row["context"])
                out[(text, source)] = intake.named(page, text)[0]
            except Exception as e:
                log(f"[lanes] {source} failed for {text!r}: {type(e).__name__}: {e}")
                out[(text, source)] = FAILED
        return out

    if not lanes:
        return {}
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        return {k: v for part in pool.map(lane, lanes) for k, v in part.items()}


def window_days(adapter, days: float) -> float:
    """How far back a source is actually searched: `days`, narrowed to the source's own
    `max_window_days` when it declares one.

    Declared on the adapter and applied here rather than in the adapter, the shape
    `cadence.lookback_floor` already uses for the opposite bound, so one decision serves both the
    search and the answer that reports the window it used.
    """
    cap = float(getattr(adapter, "max_window_days", 0.0) or 0.0)
    return days if cap <= 0 else min(days, cap)


def card(c: Candidate) -> dict:
    """A find as the host shows it: who or what, when, its link, and the opening of its text."""
    return {"title": c.title, "when": (c.published or "")[:10] or None,
            "cite": cite(c.url, c.title, c.kind, c.published),
            "snippet": (c.summary or "")[:SNIPPET_CHARS]}
