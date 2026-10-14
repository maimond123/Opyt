"""The saved-posts imports, read from their durable work: which have not landed, and which were
never begun.

An import is the walk of a platform's saved list (`bookmark_listing`, `substack_saved_listing`)
and the saved items it offered. The background process begins one on its own once the platform is
connected, so "never begun" is what `onboard` wakes the process for, and "not landed" is what the
user was promised and has not got yet:
  - its list has never been walked to the end, or
  - a saved post is still owed and still being tried (one the loop gave up on is not waited for).

A platform with no walk at all has promised nothing, so these reads never ask a platform whether
it is connected: that can be a live request, and a status read must not make one.
"""

import sqlite3

from pipeline.curation import (bookmark_listing, listings, saved_items, substack_saved_listing,
                               substack_walks, x_walks)
from pipeline.kb import schema

# {platform: (its import's short code, the walk of its saved list)}
IMPORTS = {x_walks.PLATFORM: ("bookmarks_import", bookmark_listing.COLLECTOR),
           substack_walks.PLATFORM: ("substack_saved_import", substack_saved_listing.COLLECTOR)}


def _landed(conn: sqlite3.Connection, platform: str, collector: str) -> bool | None:
    """None: never begun. False: begun and not landed. True: landed."""
    if listings.current(conn, collector) is None:
        return None
    if listings.accepted(conn, collector) is None:
        return False
    return not any(s.retry.failures is None or s.retry.retry_at is not None
                   for s in saved_items.owed(conn, platform))


def _states() -> dict:
    conn = schema.connect()
    try:
        return {platform: _landed(conn, platform, collector)
                for platform, (_, collector) in IMPORTS.items()}
    finally:
        conn.close()


def imports_outstanding() -> list[str]:
    """The imports begun and not landed, as short codes (`bookmarks_import`,
    `substack_saved_import`). `[]` when none, and on any error."""
    try:
        return [IMPORTS[platform][0] for platform, landed in _states().items()
                if landed is False]
    except Exception:
        return []


def in_progress(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    """Each platform whose import is begun and not landed: (posts saved, posts offered so far).
    The offered count can still grow while the list is being walked. `{}` on any error: this
    only qualifies what the user is told about their library."""
    try:
        return {platform: tuple(conn.execute(
                    "SELECT COALESCE(SUM(state='done'), 0), COUNT(*) FROM curation_saved_items "
                    "WHERE platform=? AND state!='superseded'", (platform,)).fetchone())
                for platform, (_, collector) in IMPORTS.items()
                if _landed(conn, platform, collector) is False}
    except Exception:
        return {}


def imports_never_queued(platforms) -> list[str]:
    """Of these platforms, the ones whose import was never begun, sorted: what `onboard` wakes
    the background process for. `[]` on any error: a store that cannot be read is asked again on
    the next `onboard` call."""
    try:
        states = _states()
        return sorted(p for p in platforms if p in states and states[p] is None)
    except Exception:
        return []
