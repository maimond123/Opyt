"""
pipeline/oracles/sources/timeline/walks.py — walking Oracles' timelines on X's
`UserOriginalsTimeline` quota, one page per unit.

Three sources drive the walks; scouting (`scouting.py`) spends the same quota on candidates:

  | source      | class     | one unit                                                         |
  |-------------|-----------|------------------------------------------------------------------|
  | `picks`     | yours     | one page further back on a pair that owes history                |
  | `check`     | yours     | one page of the newest posts on a pair someone asked about       |
  | `new_posts` | new posts | one page of the newest posts on a pair past its refresh time     |

Every unit is computed from the store when the scraper asks, so a scraper killed between pages
loses nothing: the next one reads the same rows and carries on.

Backward (`picks`). A pair owes history while `covered_from` is unset or later than its target:
the oldest of what was asked for it (`requested_from`), what was asked for its Oracle
(`oracle_asks.x_from`, so a timeline found later inherits the ask), and six months, clamped at
X's two-year ceiling. The first page reads the top of the timeline and stamps `last_pulled_at`
with the moment it did, which gives the forward walk its window. Every later page resumes below
the oldest post read (`back_tail_id`), so twenty posts inside one minute cannot stall it. The
frontier moves only after the page is written. A pair's first page also folds its account into
the Oracle's cluster when it is not there yet (`join_cluster`): the X handle discovery found for a
Substack- or blog-rooted Oracle.

Forward (`check`, `new_posts`). A forward walk keeps its own frontier (`fwd_*`) and walks down
from the top one page at a time until it reaches the last pull, minus the six-hour overlap. When
it does, `last_pulled_at` becomes the moment it read the top, not the moment it finished, so a
post that appeared during a long walk is caught by the next one. Both sources drive the same
walk: a check is the ordinary refresh moved to the front.

Nothing here is gated on consent: every confirmed Oracle is kept current; to stop, the user
removes the Oracle.

Each unit is a fetch and a record. The fetch reads one page and renders it, reading no photo, then
writes the page, the photos its posts still owe (`footprint_photos`, read on their own loop by
`x_photos.py`) and the pair's frontier in one transaction: a frontier never claims a page that is
not stored, and a page with a post that failed to render writes what did and leaves the frontier
where it was.
The record reports what the page came to. What a failure costs is the quota loop's: it charges
the pair's own `oracle_sources` row through `pair_rows.PAIR_ROWS`, keeps X's outage streak (every
pair is one platform, `x`), and removes a page saved for an Oracle forgotten meanwhile, before
the record step runs.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from pipeline.ingestion.utils import SyncAuthError, log
from pipeline.kb import resolve, schema
from pipeline.kb import writes
from pipeline.oracles import registry
from pipeline.oracles import roster
from pipeline.oracles import windows
from pipeline.oracles.asks import Ask
from pipeline.oracles.sources.timeline import pages
from pipeline.platforms.x.originals import XUserUnavailable, id_at
from pipeline.timeparse import parse_ts, utc_iso, utc_now
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import (Changed, Context, Footprint,
                                               Outcome, Unit)
from pipeline.oracles.sources.pair_rows import pair_key
from pipeline.oracles.sources import pair_rows

QUOTA = "x:UserOriginalsTimeline"
PLATFORM = "x"                             # every unit on this quota: one outage streak


# ── what the sources on this quota share ──────────────────────────────────────────
def page_kit(conn: sqlite3.Connection, ctx: Context):
    """The page kit (`pages.PageKit`), loaded on the first unit and kept for the
    scraper's lifetime. Scouting uses it too."""
    kit = ctx.cache.get("page_kit")
    if kit is None:
        kit = ctx.cache["page_kit"] = pages.PageKit()
    return kit


def x_pairs(conn: sqlite3.Connection, canonical_id: str | None = None) -> list[registry.SourceRow]:
    """Every X pair the scraper walks (`pair_rows.walked`), scoped to one Oracle when `canonical_id`
    is given. The one definition: a check that counted a pair this leaves out would wait on it
    forever."""
    return pair_rows.walked(conn, ("x",), canonical_id)


@dataclass(frozen=True)
class Fetched:
    """One page of a pair's timeline, fetched and saved, with what its record step needs."""
    page: Any                              # `pages.Page`
    uid: str
    below: str | None
    now: datetime


def fetch_page(conn, row: registry.SourceRow, ctx: Context, unit: Unit, *, below_id: str | None,
               save: Callable[[sqlite3.Connection, Fetched], None]) -> Fetched:
    """One page of `row`'s timeline, read, then written with the pair's frontier (`save`) in one
    transaction. Raises what failed; a dead session also drops the kit's session, so the next
    page reads the cookies again. `Changed` when the lease went, or another lane rewrote one of
    the page's posts while it was read: the page is written by a later unit, from what is there.
    `Changed` too when the Oracle was forgotten meanwhile, so nothing lands after a forget."""
    now = utc_now()
    kit = page_kit(conn, ctx)
    try:
        uid = kit.user_id(conn, row.canonical_id, row.source_key)
        if uid is None:
            raise XUserUnavailable(f"@{row.source_key}: X profile unreadable "
                                   f"(suspended, protected or renamed)")
        page = pages.walk_page(conn, kit, user_id=uid, handle=row.source_key, below_id=below_id,
                         author_name=row.name)
    except SyncAuthError:
        kit.forget_session()               # read the cookies again: the user may reconnect
        raise
    fetched = Fetched(page, uid, below_id, now)
    with writes.write_tx(conn, site="timeline.page"):
        if ctx.lease_lost():
            raise Changed("timeline quota lease lost")
        if not roster.on_roster(conn, row.canonical_id):
            raise Changed(f"x:{row.source_key}: its Oracle was forgotten while the page was read")
        if not pages.page_current(conn, page):
            raise Changed(f"x:{row.source_key}: a post on the page changed while it was read")
        pages.write_page(page)
        if page.complete:
            save(conn, fetched)
    return fetched


def unsaved(fetched: Fetched) -> Outcome | None:
    """A page that came back but would not save, a post on it having failed to render or write:
    X answered, and the pair is charged."""
    page = fetched.page
    if page.complete:
        return None
    return Outcome(retries.UNSAVED, failure=retries.UNSAVED,
                   detail=f"page fetched but not saved ({page.added} atoms written, "
                          f"a post failed)")


def _status(fetched: Fetched) -> str:
    return "ingested" if fetched.page.added else "empty"


def unit_for(row: registry.SourceRow, key: str, payload) -> Unit:
    """A unit on `row`: its failures are charged to the pair, and its pages saved for its Oracle."""
    return Unit(key, payload, item=pair_rows.item(row, PLATFORM),
                footprint=Footprint(row.canonical_id))


def _cursor(conn, uid: str) -> str | None:
    return registry.latest_atom_ts(conn, "x", [f"x:user:{uid}"])


# ── the walks ─────────────────────────────────────────────────────────────────────
def below_back(row: registry.SourceRow) -> str | None:
    """Where a backward page starts: below the oldest post read, or from the top for a first
    page."""
    below = row.back_tail_id
    if below is None and row.covered_from:
        # Coverage with no post id: resume once at the claimed date plus the overlap; every page
        # after keeps the id.
        resume_at = parse_ts(row.covered_from) + windows.X_RESUME_OVERLAP
        below = str(id_at(resume_at) + 1)
    return below


def save_back(conn, row: registry.SourceRow, ask: Ask | None, fetched: Fetched) -> None:
    """The frontier after one page further back on `row`, or its first page from the top."""
    page, uid, now = fetched.page, fetched.uid, fetched.now
    from_top = fetched.below is None
    # A page with nothing below it has read the whole timeline: the pair is covered as far as
    # it was asked, or as far as the page went, whichever is older.
    claim = page.oldest_at
    if page.ended:
        claim = min(t for t in (windows.target(row, ask, now), page.oldest_at) if t is not None)
    covered = utc_iso(claim) if claim is not None else None
    cursor = _cursor(conn, uid)
    with writes.write_tx(conn, site="timeline.back"):
        registry.save_back(conn, pair_key(row), status=_status(fetched), cursor=cursor,
                           covered=covered, tail_id=page.oldest_id,
                           read_at=page.read_at.isoformat(), from_top=from_top)


def record_back(conn, row: registry.SourceRow, fetched: Fetched) -> Outcome:
    """What one page further back on `row` came to. A pair's first page folds its account into
    the Oracle's cluster."""
    if (failed := unsaved(fetched)) is not None:
        return failed
    if row.last_pulled_at is None:
        join_cluster(conn, row, fetched.uid)
    return Outcome(_status(fetched))


def join_cluster(conn, row: registry.SourceRow, uid: str) -> None:
    """After a pair's first page: fold its X account into the Oracle's cluster when it is not
    there yet (the pair discovery registered for a Substack- or blog-rooted Oracle); nothing else
    on this path merges entities. Never raises: the page is already written.

    The page wrote the account's entity with its handle as X spells it. The pair's key is aligned
    to that spelling first, because `seed_from_entities` derives an X pair from the entity, and a
    key differing only in case would be a second row walking the same timeline."""
    try:
        if pages.stored_x_user_id(conn, row.canonical_id, row.source_key) is not None:
            return
        entity = schema.get_entity(conn, f"x:user:{uid}")
        profile = entity["profile"] if entity is not None else None
        spelled = ((json.loads(profile) if isinstance(profile, str) else profile or {})
                   .get("handle") or "").lstrip("@")
        if spelled and spelled != row.source_key and spelled.lower() == row.source_key.lower():
            with writes.write_tx(conn, site="timeline.join"):
                registry.respell_x(conn, pair_key(row), spelled)
        resolve.resolve_entities(conn)
    except Exception as exc:
        log(f"[scrapers] x:{row.source_key}: could not fold the account into its Oracle: "
            f"{type(exc).__name__}: {exc}")


def below_forward(row: registry.SourceRow) -> str | None:
    """Where a forward page starts: from the top, or below where the walk in progress got."""
    return row.fwd_tail_id if row.fwd_started_at else None


def forward_done(row: registry.SourceRow, page) -> bool:
    """Has the forward walk reached the last pull, minus the overlap, or the timeline's end?"""
    boundary = parse_ts(row.last_pulled_at) - windows.OVERLAP
    return page.ended or page.oldest_at is None or page.oldest_at <= boundary


def save_forward(conn, row: registry.SourceRow, fetched: Fetched) -> None:
    """The frontier after one page of `row`'s forward walk."""
    page = fetched.page
    started = parse_ts(row.fwd_started_at) if row.fwd_started_at else page.read_at
    cursor = _cursor(conn, fetched.uid)
    status = _status(fetched)
    with writes.write_tx(conn, site="timeline.forward"):
        registry.save_forward(conn, pair_key(row), status=status, cursor=cursor,
                              started=started.isoformat(), reached=utc_iso(page.oldest_at),
                              tail_id=page.oldest_id, done=forward_done(row, page))


def record_forward(row: registry.SourceRow, fetched: Fetched) -> Outcome:
    """What one page of `row`'s forward walk came to."""
    if (failed := unsaved(fetched)) is not None:
        return failed
    return Outcome(_status(fetched) if forward_done(row, fetched.page) else "walking")


# ── the sources ───────────────────────────────────────────────────────────────────


