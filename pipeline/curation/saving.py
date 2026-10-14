"""The run every saved-post source shares: the units it owes and when it is next due, its image
cache, and the lease check before its write.

Each source keeps only what differs: X's version check and conversation mark
(`bookmarks.SavedXPosts`), Substack's body fetch, publication host and outage
(`substack_saved.SavedSubstackPosts`).
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Iterator

from opyt_core.paths import opyt_home
from pipeline.curation import saved_items
from pipeline.image_cache import load_image_cache, save_image_cache
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import Changed, Item, Unit


def owed_units(conn, platform: str, prefix: str, now: datetime,
               host: Callable[[saved_items.Saved], str | None] = lambda saved: None) -> list[Unit]:
    """One unit per owed saved post that may run now, charged to its saved-item row."""
    return [Unit(f"{prefix}:{s.item}", s,
                 item=Item(saved_items.ROWS, (s.platform, s.item, s.revision), s.retry,
                           platform=platform),
                 host=host(s))
            for s in saved_items.owed(conn, platform)
            if retries.may_run(s.retry.retry_at, s.retry.failures, now)]


def next_due(conn, platform: str) -> datetime | None:
    """The earliest retry among the owed saved posts; None when none waits."""
    return retries.earliest_retry(s.retry for s in saved_items.owed(conn, platform))


@contextmanager
def image_cache() -> Iterator[dict]:
    """The image cache for one unit, merged back into what is on disk afterwards, so two sources
    describing images at once keep each other's entries."""
    home = opyt_home()
    cache = load_image_cache(home)
    try:
        yield cache
    finally:
        save_image_cache(home, {**load_image_cache(home), **cache})


def require_lease(lease_lost: Callable[[], bool]) -> None:
    """The first refusal before a saved post's write, inside its transaction."""
    if lease_lost():
        raise Changed("saved-posts lease lost")
