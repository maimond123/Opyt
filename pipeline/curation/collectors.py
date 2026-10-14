"""Curation's work for the background process, and the public reads of its collectors' state.

Curation supplies; the composition entry (`pipeline/background.py`) assembles. Curation's sources
join the process in two ways:
  - Substack's walks and saved content run in the one web pool the process already has, beside
    oracles' web sources and under the same outage state (`web_sources`).
  - Everything else runs on curation's own quotas (`loops`): one loop per X operation (Bookmarks,
    Following, Likes, owned Lists, List members), saved X content, its photos, and the local
    maintenance.

Each collector's success is stamped on its clock (`clock`, `collector_runs`) under the
durable keys below, which `screen`, `onboard` and `oracle` read.
"""

import sqlite3
from typing import Callable

from pipeline.curation import (bookmark_listing, bookmarks, clock, listings, maintenance,
                               saved_images, saved_items, saved_x_images, substack_people,
                               substack_saved, substack_saved_listing, substack_walks, walks,
                               x_collectors, x_walks)
from pipeline.kb import schema
from pipeline.worker.quota_loop.loop import QuotaLoop
from pipeline.worker.quota_loop.retries import OUTAGE_PAUSES_S, PLATFORM_PAUSES_S, Outages
from pipeline.worker.quota_loop.serial import Serial

# Every collector clock key, with the platform whose session it reads.
CLOCKS: dict[str, str] = {
    x_collectors.Lists.collector: x_walks.PLATFORM,
    substack_people.Follows.collector: substack_walks.PLATFORM,
    substack_people.Subscriptions.collector: substack_walks.PLATFORM,
    substack_saved_listing.SIGNAL_COLLECTOR: substack_walks.PLATFORM,
    x_collectors.Following.collector: x_walks.PLATFORM,
    x_collectors.Likes.collector: x_walks.PLATFORM,
    bookmark_listing.SIGNAL_COLLECTOR: x_walks.PLATFORM,
}
COLLECTORS: tuple[str, ...] = tuple(CLOCKS)


def init_schema(conn: sqlite3.Connection) -> None:
    """The tables curation's sources write, on a loop's connection, before any transaction."""
    listings.init_schema(conn)
    saved_items.init_schema(conn)
    saved_images.init_schema(conn)
    maintenance.init_schema(conn)
    clock.init_state_schema(conn)


def web_sources(quota: str, outages: Outages) -> list:
    """Substack's account walks and saved posts, for the web pool spending `quota`; `outages` is
    that pool's, so a Substack pause holds every Substack read at once."""
    via = substack_walks.SubstackAccount(quota, outages)
    return [walks.WalkSource(substack_people.Follows(), via),
            walks.WalkSource(substack_people.Subscriptions(), via),
            walks.WalkSource(substack_saved_listing.SavedPosts(), via),
            substack_saved.SavedSubstackPosts(quota, outages)]


def loops(connect: Callable[[], sqlite3.Connection]) -> list[QuotaLoop]:
    """Curation's own quotas. An X loop is one platform, so an outage stops it whole, as on
    oracles' X loops."""
    def x_loop(source) -> QuotaLoop:
        return QuotaLoop(source.quota, [source], executor=Serial(),
                         outages=Outages(OUTAGE_PAUSES_S, whole_quota=True), connect=connect)

    x, lists = x_walks.XAccount(), x_collectors.Lists()
    return [x_loop(walks.WalkSource(bookmark_listing.Bookmarks(), x)),
            x_loop(walks.WalkSource(x_collectors.Following(), x)),
            x_loop(walks.WalkSource(x_collectors.Likes(), x)),
            # Lists' enumeration starts and finishes the walk; its members read on their own quota.
            x_loop(walks.WalkSource(lists, x)),
            x_loop(walks.WalkSource(lists, x, lists.members, ends=False)),
            QuotaLoop(bookmarks.QUOTA, [bookmarks.SavedXPosts()], executor=Serial(),
                      outages=Outages(PLATFORM_PAUSES_S), connect=connect),
            # Its own loop, so reading photos never holds up saving the posts.
            QuotaLoop(saved_x_images.QUOTA, [saved_x_images.SavedXImages()], executor=Serial(),
                      outages=Outages(PLATFORM_PAUSES_S), connect=connect),
            QuotaLoop(maintenance.QUOTA, [maintenance.Maintenance()], executor=Serial(),
                      outages=Outages(PLATFORM_PAUSES_S), connect=connect)]


def curation_pending(connected: set[str]) -> list[str]:
    """Connected platforms none of whose own collectors has ever succeeded, sorted. Judged per
    platform, because platforms are connected one at a time."""
    conn = schema.connect()
    try:
        ok = {r.collector for r in clock.list_runs(conn) if r.ok}
    finally:
        conn.close()
    read = {platform for collector, platform in CLOCKS.items() if collector in ok}
    return sorted(connected - read)
