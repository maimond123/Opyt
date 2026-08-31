"""
pipeline/timeparse.py — the one timestamp parse, and the one timestamp FORMAT.

`parse_ts` turns a stored stamp (an atom's `when_ts`, an Oracle refresh watermark, a sitting's
read stamp) into an aware UTC datetime, so a staleness subtraction never raises on a mixed
aware/naive pair. Every rail that has a freshness clock needs it.

`utc_now` and `utc_iso` are the WRITE side of the same job, and they live here because the
parse and the format must agree about what a stored stamp looks like.

One name per meaning, and that rule was bought. Nine call sites across five modules had
private `_now()` helpers under ONE name with THREE different return types — a `datetime` in
`curation_state` and `oracle_refresh_state`, a seconds-truncated ISO string in
`frontier_queries` and `frontier_surface`, and a FULL-PRECISION ISO string in `dedup_store`.
A reader who learned what `_now()` meant in one module learned something false about the
other two, and stamps written by two of them do not sort against each other as strings. So
the type is now in the name: `utc_now` returns a datetime, `utc_iso` returns a string.

Why it sits at the top of `pipeline` and not inside a rail: A LIVE RAIL MUST NEVER TAKE A
DEPENDENCY ON A DEAD ONE'S LIFETIME. That is the rule this module's location enforces, and it
is the same one `pipeline/rank.py` exists to undo (the atom rail importing rank fusion from the
vault rail) and the same one the `atom-rail-not-welded-to-catchup` guard names. A shared
primitive belongs in a neutral single-purpose module at the top of `pipeline/`, never borrowed
across a rail boundary — because the borrowed-from rail is then undeletable.

It was bought by a real weld, since resolved: this body used to live in a pre-atom-KB rail that
a live frontier module imported across the boundary, so deleting the dead rail would have
raised in live code. Both of those modules have since been deleted; naming them here would only
send a reader grepping for files that are gone.

The parse had ALSO forked before it was consolidated — three implementations, and the
`pipeline/kb` copy had silently added the `str()` coercion below that the others lacked. Nobody
noticed, because nothing compared them. That is chapter two of the `forked-slugify` story
(seven copies, drifted on hyphen runs and truncate order), which is why the fix was one shared
body rather than a fourth private one. The body kept is the more defensive one: `str()` means a
non-string input returns None instead of raising.
"""

from __future__ import annotations

from datetime import datetime, timezone


def parse_ts(ts) -> datetime | None:
    """An atom/state timestamp → an aware UTC datetime, or None when it carries no usable
    time. Accepts full ISO (tz-aware or not) or a date-only prefix; naive values are assumed
    UTC so a comparison never raises on a mixed pair. Total: any unparseable or non-string
    input returns None rather than raising, because every caller is a staleness check whose
    fail-safe answer is 'treat it as never/unknown', not a crash."""
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
    """Now, as an aware UTC datetime. The only clock read in the tree that returns a datetime.

    Aware, never naive: a naive `now()` compared against a stored stamp that `parse_ts` made
    aware raises TypeError, and that comparison is exactly what every freshness check does."""
    return datetime.now(timezone.utc)


def utc_iso(d: datetime | None = None) -> str:
    """`d` (default: now) → a seconds-precision UTC ISO string — the STORED stamp format.

    Seconds precision is the contract, not a rounding convenience. These strings are compared
    and ORDERED as text in SQL (`WHERE ran_at > ?`, `ORDER BY read_at`), and text ordering is
    only meaningful when every writer produces the same width. A microsecond-bearing stamp and
    a seconds-truncated one sort against each other by their shared prefix and then diverge on
    a character that means nothing.

    Any aware or naive input works: naive is assumed UTC, matching `parse_ts`."""
    d = d if d is not None else datetime.now(timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc).isoformat(timespec="seconds")
