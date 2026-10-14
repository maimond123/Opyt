"""
pipeline/oracles/checks.py — check-on-ask: an Oracle's newest posts, asked for now.

  `request_check(conn, canonical_id)` stamps the Oracle's X pairs and wakes the scrapers.

A check is not a walk of its own. It stamps `check_requested_at` on the Oracle's X pairs and
wakes the scrapers, and the `check` source (`oracles/sources/timeline/sources.py`) moves those
pairs' ordinary forward walk to the front. It stays owed through every block until a walk that
started after it finishes. Where it stands is `progress`'s to report (`milestones`); nothing
polls a check.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from pipeline.kb import schema
from pipeline.kb import writes
from pipeline.oracles import asks
from pipeline.oracles import registry
from pipeline.oracles.sources.timeline.walks import x_pairs
from pipeline.timeparse import utc_now
from pipeline.oracles.sources.pair_rows import pair_key


def request_check(conn: sqlite3.Connection, canonical_id: str, *,
                  now: datetime | None = None) -> bool:
    """Ask for this Oracle's newest posts now: mark every X pair the scraper walks and wake it.
    An Oracle with no such pair has nothing to check, and nothing is woken. True when a check was
    asked."""
    cid = schema.current_canonical(conn, canonical_id)
    requested_at = (now or utc_now()).isoformat()
    pairs = x_pairs(conn, cid)
    if pairs:
        with writes.write_tx(conn, site="request_check"):
            registry.mark_check(conn, [pair_key(r) for r in pairs], requested_at)
        asks.wake()
    return bool(pairs)
