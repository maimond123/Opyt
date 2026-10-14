"""The photos of saved X posts, read after the posts are saved.

`SavedXImages` takes a batch of owed `saved_images` rows per unit, reads their photos on a few
threads into the image cache, and decides each post:

| All photos read now | Thread lane owes the post | Result                                       |
|---------------------|---------------------------|----------------------------------------------|
| no                  | —                         | stays owed (charged only when it is the head)|
| yes                 | yes                       | settled: the thread lane's re-save renders it|
| yes                 | no                        | re-rendered; rewritten only if it changed    |

So a post with both photos and a thread is written at most twice: its first save, and the thread
lane's, which waits for this row. Like `SavedXPosts`, the unit charges its head and the riders
that fail are left owed and uncharged. While no photo can be read here (`chart_read.refused`: a
free install has no model), it offers nothing: the rows wait, uncharged, for a key.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

from pipeline import chart_read
from pipeline.curation import bookmarks, conversations, saved_images, saved_items, saving
from pipeline.kb import link_cards, vision
from pipeline.kb.ingest_common import store_atom
from pipeline.kb.writes import write_tx
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import CHANGED, YOURS, Changed, Item, Outcome, Unit

PLATFORM = "x"
QUOTA = "images:x"      # no X request: photos come from X's media host, reads from the model
BATCH = 8               # posts per unit
READERS = 4             # photos read at once within a unit

PENDING, SETTLED, RESAVED = "images_pending", "settled", "resaved"


def outcome(pending: bool, thread_owed: bool) -> str:
    """What one post comes to once its photos were tried (deciding): the table above, before
    the re-render, which may still find the render unchanged."""
    if pending:
        return PENDING
    return SETTLED if thread_owed else RESAVED


def _read(norms: list[dict], cache: dict) -> None:
    """Read every unread photo of these posts into `cache` (in place). A failed read leaves its
    photo pending; the cache's writes are locked, so readers may share it."""
    with ThreadPoolExecutor(max_workers=READERS) as pool:
        list(pool.map(lambda n: vision.enrich_tweet_media(n, cache), norms))


def _settle(conn, row, lease_lost) -> None:
    with write_tx(conn, site="saved_images.done"):
        saving.require_lease(lease_lost)
        saved_images.settle(conn, row)


def read_batch(conn, rows: list[saved_images.Owed], cache: dict, lease_lost) -> dict[str, str]:
    """Read a batch of saved posts' photos, then settle or re-save each post. What each came to,
    by item. A post refused for its own reason is `changed`; a lost lease stops the batch."""
    norms = [copy.deepcopy(r.saved.data) for r in rows]
    _read(norms, cache)
    owed = set(conversations.owed_conversations(conn, [r.saved.item for r in rows]))
    cards = link_cards.load(conn)
    came, renders = {}, []
    for row, norm in zip(rows, norms):
        aid = f"x:{row.saved.item}"
        step = outcome(vision._photos_pending(norm, cache), row.saved.item in owed)
        previous = bookmarks.stored(conn, aid)
        result = None
        if step == RESAVED and previous:
            kept = saved_items.rendering(conn, PLATFORM, row.saved.item)
            result = bookmarks.render(norm, previous, kept.chain, cards)
        if step == PENDING:
            came[row.saved.item] = PENDING
        elif result is not None and "atom" in result:
            renders.append((row, previous, result))
        else:
            # Thread-owed, unchanged, or no atom left to re-save (forgotten): nothing to write.
            try:
                _settle(conn, row, lease_lost)
                came[row.saved.item] = SETTLED
            except Changed:
                if lease_lost():
                    raise
                came[row.saved.item] = CHANGED
    for row, previous, result in renders:
        try:
            with write_tx(conn, site="saved_images.atom"):
                saving.require_lease(lease_lost)
                bookmarks.check_version(conn, result["atom"]["atom_id"], previous)
                saved_images.require_current(conn, row)
                store_atom(conn, result["atom"], result["md"])
                saved_items.record_rendered(conn, PLATFORM, row.saved.item,
                                            teaser=bookmarks.is_teaser(row.saved.data))
                saved_images.settle(conn, row)
            came[row.saved.item] = RESAVED
        except Changed:
            if lease_lost():
                raise
            came[row.saved.item] = CHANGED
    return came


class SavedXImages:
    """A batch of saved X posts' unread photos per unit."""
    name, quota, klass, floor = "saved_x_images", QUOTA, YOURS, 0

    def next_unit(self, conn, now):
        if chart_read.refused() is not None:
            return None
        rows = [r for r in saved_images.owed(conn, PLATFORM)
                if retries.may_run(r.retry.retry_at, r.retry.failures, now)]
        if not rows:
            return None
        head = rows[0]
        key = (head.saved.platform, head.saved.item, head.saved.revision)
        return Unit(f"saved-x-images:{head.saved.item}", rows[:BATCH],
                    item=Item(saved_images.ROWS, key, head.retry, platform=PLATFORM))

    def fetch(self, conn, unit, ctx):
        with saving.image_cache() as cache:
            came = read_batch(conn, unit.payload, cache, ctx.lease_lost)
        return came[unit.payload[0].saved.item]

    def record(self, conn, unit, fetched):
        if fetched == PENDING:
            return Outcome(PENDING, failure=retries.BLIP,
                           detail="a photo could not be read yet; the post is saved without it")
        return Outcome(fetched)

    def next_due(self, conn, now):
        if chart_read.refused() is not None:
            return None
        return retries.earliest_retry(r.retry for r in saved_images.owed(conn, PLATFORM))
