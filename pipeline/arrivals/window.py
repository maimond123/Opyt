"""
pipeline/arrivals/window.py — which atoms arrived since the user last looked, or in the last week
when they have never looked. Which of those are new in time is `fresh.py`'s decision.

An atom enters when it is first stored, or when a human ingest promotes a crawled one, so its
arrival is `COALESCE(promoted_at, first_seen)`. A watch's find is not an arrival: the host showed it
when it checked it, before it was stored. The last look is one row, `arrivals_seen`, moved only by
an unscoped `latest`.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from pipeline.kb.writes import write_tx
from pipeline.timeparse import utc_iso

# How far back "new" reaches for a user who has never looked.
FIRST_LOOK = timedelta(days=7)

# Both sides go through `datetime()`: SQLite stamps `first_seen` "YYYY-MM-DD HH:MM:SS" and the last
# look is an ISO stamp, which compare wrongly as text.
_ARRIVED = "COALESCE(promoted_at, first_seen)"
_AFTER = f"datetime({_ARRIVED}) > datetime(?) AND entry_mode != 'frontier'"


def start(conn, now: datetime) -> str:
    """Where "new" starts: the user's last look, or a week before `now`."""
    row = conn.execute("SELECT seen_at FROM arrivals_seen WHERE id = 1").fetchone()
    return row[0] if row else utc_iso(now - FIRST_LOOK)


def arrivals(conn, since: str) -> list[dict]:
    """Every atom that arrived after `since`, newest arrival first: how it entered, its author,
    when it arrived and was published, and, for one with no date of its own, the dates of the
    posts that linked it (`fresh.published`)."""
    rows = conn.execute(
        f"SELECT atom_id, entry_mode, who_id, when_ts, {_ARRIVED} AS arrived, "
        f"CASE WHEN COALESCE(when_ts, '') = '' THEN payload END AS payload "
        f"FROM atoms WHERE {_AFTER} ORDER BY datetime({_ARRIVED}) DESC, atom_id", (since,))
    return [{"atom_id": r["atom_id"], "entry_mode": r["entry_mode"], "who_id": r["who_id"],
             "when_ts": r["when_ts"], "arrived": r["arrived"], "linked_on": _linked_on(r["payload"])}
            for r in rows]


def _linked_on(payload: str | None) -> list[str]:
    if not payload:
        return []
    try:
        refs = json.loads(payload).get("referenced_by") or []
    except (ValueError, AttributeError):
        return []
    return [ref["date"] for ref in refs if isinstance(ref, dict) and ref.get("date")]


def mark_seen(conn, now: datetime) -> None:
    """The user has looked: what arrived before `now` is no longer new."""
    with write_tx(conn, site="arrivals.seen"):
        conn.execute("INSERT INTO arrivals_seen (id, seen_at) VALUES (1, ?) "
                     "ON CONFLICT(id) DO UPDATE SET seen_at = excluded.seen_at", (utc_iso(now),))
