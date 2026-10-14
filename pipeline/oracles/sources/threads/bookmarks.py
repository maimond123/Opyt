"""One saved bookmark's conversation, read from its atom rather than the Bookmarks feed. The post
it renders and the chain it keeps live on the post's saved row (`curation/saved_items.py`)."""

import copy
from dataclasses import dataclass

from opyt_core.paths import opyt_home
from pipeline.curation import saved_items
from pipeline.curation.bookmarks import is_teaser
from pipeline.image_cache import load_image_cache
from pipeline.ingestion.x_render import _article_shape, tweet_to_markdown
from pipeline.kb import derive, ingest_x, link_cards, vision
from pipeline.kb.ingest_common import changed_hash
from pipeline.platforms.x import conversation as x_conversation
from pipeline.worker.quota_loop.source import YOURS, Outcome

from pipeline.oracles.sources.threads import reading
from pipeline.oracles.sources.threads import saving
from pipeline.oracles.sources.threads import state
from pipeline.oracles.sources.threads.walk import Walk
from pipeline.worker.quota_loop.source import Source


@dataclass(frozen=True)
class Fetched:
    """A bookmark's conversation, written and completed, or found gone."""
    gone: bool = False


class Bookmarks(Source):
    name, quota, klass, floor = "bookmark_threads", state.QUOTA, YOURS, 0

    def next_unit(self, conn, now):
        return state.next_unit(conn, "bookmarks", now)

    def next_due(self, conn, now):
        return state.next_due(conn, "bookmarks", now)

    def fetch(self, conn, unit, ctx):
        work = unit.payload
        page = list(work.walk.tweets)
        if not work.walk.focused:
            page = reading.page(ctx, work.root)
            work = state.save_walk(conn, work, Walk(tuple(page), (work.root,)))
        focal = next((t for t in page if str(t.get("id")) == work.root), None)
        if focal is None:
            def finish_gone(written):
                state.finish(conn, written, gone=True)
                saved_items.record_chain(conn, "x", work.root, [])

            saving.complete(conn, ctx, work, finish_gone)
            return Fetched(gone=True)
        # A post saved by hand has no saved row: the page's copy is its input.
        kept = saved_items.rendering(conn, "x", work.root)
        norm = copy.deepcopy(kept.data if kept else focal)
        if focal.get("article") and _article_shape(focal["article"])[1]:
            norm = {**norm, "article": focal["article"]}
        # The saved-images source read the photos into the cache and left this re-save to render
        # them (it waits for that), so the post is rewritten once for both.
        vision.attach_cached(norm, load_image_cache(opyt_home()))
        chain = [norm if str(t.get("id")) == work.root else t
                 for t in x_conversation.reconstruct_chain(page, work.root)]
        cards = link_cards.load(conn)
        for t in [norm, *chain]:
            link_cards.attach(t, cards)

        def finish(written):
            state.finish(conn, written)
            saved_items.record_chain(conn, "x", work.root, chain)

        def finish_published(written):
            finish(written)
            saved_items.record_rendered(conn, "x", work.root, teaser=is_teaser(norm))

        md = tweet_to_markdown(norm, article=norm.get("article"), thread_tweets=chain or None,
                               source="x-bookmark", footer_label="Bookmarked")
        decided = changed_hash(work.atom_id, md, {work.atom_id: work.raw_hash})
        if decided is None:
            saving.complete(conn, ctx, work, finish)
            return Fetched()
        atom = ingest_x.build_x_atom(norm, work.atom_id, raw_hash=decided,
                                    meta=derive.derive_x(norm), thread_tweets=chain or None)
        saving.publish(conn, ctx, work, atom, md, finish_published)
        return Fetched()

    def record(self, conn, unit, fetched):
        return Outcome("gone" if fetched.gone else "fetched")
