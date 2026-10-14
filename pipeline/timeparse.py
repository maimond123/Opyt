"""
pipeline/timeparse.py — the one timestamp parse, and the one timestamp format.

`parse_ts` turns a stored stamp into an aware UTC datetime, so a staleness subtraction never raises
on a mixed aware/naive pair. `utc_now` (a datetime) and `utc_iso` (a string) are the write side,
kept here so the parse and the format agree on what a stored stamp looks like.
"""

from __future__ import annotations

from datetime import datetime, timezone


def parse_ts(ts) -> datetime | None:
    """An atom/state timestamp → an aware UTC datetime, or None when it carries no usable
    time. Accepts full ISO (tz-aware or not) or a date-only prefix; naive values are assumed
    UTC. Total: unparseable or non-string input returns None, which callers treat as unknown."""
    if not ts:
        return None
    for candidate in (str(ts).strip(), str(ts).strip()[:10]):
        try:
            d = datetime.fromisoformat(candidate)
        except (ValueError, TypeError):
            continue
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    return None


def utc_now() -> datetime:
    """Now, as an aware UTC datetime (a naive one would raise against a `parse_ts` result)."""
    return datetime.now(timezone.utc)


def utc_iso(d: datetime | None = None) -> str:
    """`d` (default: now) → a seconds-precision UTC ISO string, the stored stamp format.

    Seconds precision is the contract: stamps are ordered as text in SQL, which only works when
    every writer produces the same width. Naive input is assumed UTC, matching `parse_ts`."""
    d = d if d is not None else datetime.now(timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).isoformat(timespec="seconds")
