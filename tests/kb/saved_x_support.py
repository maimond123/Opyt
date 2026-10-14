"""Land saved X posts the way the scrapers process does: the Bookmarks listing offers them as
saved-item work, `SavedXPosts` saves them a batch per unit, and `SavedXImages` reads their photos,
the loop charging a failure to each unit's head."""
from datetime import datetime, timezone

from pipeline.curation import bookmarks, saved_images, saved_items, saved_x_images
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.retries import PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.source import Context

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


def save(conn, *posts, account="me", now=NOW) -> list[str]:
    """Offer `posts`, then run every owed post's unit that may run now; what each came to."""
    saved_items.init_schema(conn)
    saved_images.init_schema(conn)
    saved_items.offer(conn, "x", account,
                      [(str(p["id"]), saved_items.x_revision(p), p) for p in posts], now)
    source, outages = bookmarks.SavedXPosts(), Outages(PLATFORM_PAUSES_S)
    ctx = Context(bookmarks.QUOTA)
    out = []
    while (unit := source.next_unit(conn, now)) is not None:
        outcome = source.record(conn, unit, source.fetch(conn, unit, ctx))
        if outcome.failure:
            retries.charge(conn, unit.item, outcome, outages, now)
        out.append(outcome.status)
    return out


def read_images(conn, now=NOW) -> list[str]:
    """Run every owed saved-images unit that may run now; what each came to."""
    source, outages = saved_x_images.SavedXImages(), Outages(PLATFORM_PAUSES_S)
    ctx = Context(saved_x_images.QUOTA)
    out = []
    while (unit := source.next_unit(conn, now)) is not None:
        outcome = source.record(conn, unit, source.fetch(conn, unit, ctx))
        if outcome.failure:
            retries.charge(conn, unit.item, outcome, outages, now)
        out.append(outcome.status)
    return out
