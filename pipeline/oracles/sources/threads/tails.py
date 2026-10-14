"""An Oracle's thread tail, one request per turn with progress kept in `thread_walks`."""

from dataclasses import dataclass

from pipeline.kb import footprint_photos, ingest_x_footprint as footprint
from pipeline.kb.ingest_common import StageTimer
from pipeline.worker.quota_loop.source import YOURS, Outcome
from pipeline.oracles.sources.threads import tail_state

from pipeline.oracles.sources.threads import reading
from pipeline.oracles.sources.threads import saving
from pipeline.oracles.sources.threads import state
from pipeline.oracles.sources.threads.walk import advance
from pipeline.worker.quota_loop.source import Source


@dataclass(frozen=True)
class Fetched:
    """One turn of a thread's tail: a request made and its walk saved, and when the walk is done,
    the thread written and completed, or found gone."""
    done: bool
    gone: bool = False


class Tails(Source):
    name, quota, klass, floor = "thread_tails", state.QUOTA, YOURS, 0

    def next_unit(self, conn, now):
        return state.next_unit(conn, "tails", now)

    def next_due(self, conn, now):
        return state.next_due(conn, "tails", now)

    def fetch(self, conn, unit, ctx):
        work = unit.payload
        walk = work.walk
        if focal := walk.next_focal(work.root):
            author = str(work.payload.get("thread_author") or "")
            if not author:
                raise ValueError("owed thread has no author")
            page = reading.page(ctx, focal)
            walk = advance(walk, work.root, author, focal, page)
            work = state.save_walk(conn, work, walk)
        if walk.next_focal(work.root) is not None:
            return Fetched(done=False)
        if walk.gone:
            saving.complete(conn, ctx, work, lambda w: state.finish(conn, w, gone=True))
            return Fetched(done=True, gone=True)
        if not walk.tweets:
            raise ValueError("thread response contained none of the author's posts")
        group = [t for g in footprint._filter_and_stitch(list(walk.tweets)) for t in g]
        if not group:
            raise ValueError("thread response has no usable author chain")
        writer = footprint._GroupWriter(conn, img_cache=reading.images(ctx),
                                         seen={work.atom_id: work.raw_hash}, timer=StageTimer(),
                                         tail_rules=tail_state)
        prepared = writer.produce(group, atom_id=work.atom_id, decision=tail_state.FETCHED)
        complete = prepared is not None and not prepared["article_incomplete"]
        # The thread renders the photo reads the cache holds; one still unread is owed with the
        # write, to the photo source.
        owes_photos = (prepared is not None and prepared["decided"] is not None
                       and prepared["photos_owed"])

        def finish(written):
            state.finish(conn, written, complete=complete)
            if owes_photos:
                footprint_photos.owe(conn, work.atom_id, prepared["who_id"], prepared["group"])

        if prepared is not None and prepared["decided"] is not None:
            owed = writer._owed_links(group, work.atom_id)
            atom = footprint.group_atom(prepared, owed)
            saving.publish(conn, ctx, work, atom, prepared["md"], finish)
        else:
            saving.complete(conn, ctx, work, finish)
        return Fetched(done=True)

    def record(self, conn, unit, fetched):
        if not fetched.done:
            return Outcome("walking")
        return Outcome("gone" if fetched.gone else "fetched")
