"""The photos of Oracles' X posts, read after the posts are written.

A timeline page and a thread tail write their posts with the photo reads the image cache already
holds, and owe a `footprint_photos` row for each post with a photo still unread. `OraclePhotos`
takes the oldest owed rows of one Oracle per unit, up to `PHOTOS` photos, reads them on `READERS`
threads into the image cache, then decides each post (`step`):

| All photos read now | Its thread tail is owed | Post stored | Result                         |
|---------------------|-------------------------|-------------|--------------------------------|
| no                  | —                       | —           | stays owed (charged if head)   |
| yes                 | yes                     | —           | settled: the tail lane renders |
| yes                 | no                      | no          | settled: nothing to render on  |
| yes                 | no                      | yes         | rendered again, then settled   |

A render goes through the same `_GroupWriter` the page used, forced to the stored tail state, so
the post renders as it would have with its photos read inline: rewritten only if the render
changed. The tail lane waits on an owed row, so a thread with photos is rewritten once for both. A
post not stored is an atom forgotten since it was owed, or one whose write failed on its page,
which the page writes again.

The posts are rendered first and then land one transaction each, a rewrite with its settle, so a
unit holds the write lock for one post at a time and a post that changed meanwhile skips only
itself. Like `SavedXImages`, the unit charges its head; riders that fail stay owed and uncharged.
It runs on its own loop (`loops.x_photos_loop`), beside the timeline's, so a page never waits on
it. While no photo can be read here (`chart_read.refused`: a free install has no model), it offers
nothing: the rows wait, uncharged, for a key, instead of failing until they are given up.
"""

from __future__ import annotations

import copy

from opyt_core.paths import opyt_home
from pipeline import chart_read
from pipeline.image_cache import load_image_cache, save_image_cache
from pipeline.kb import footprint_photos, schema, vision
from pipeline.kb import ingest_x_footprint as footprint
from pipeline.kb.ingest_common import StageTimer
from pipeline.kb.writes import write_tx
from pipeline.oracles import roster
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import (CHANGED, YOURS, Changed, Footprint, Item, Outcome,
                                              Unit)

from pipeline.oracles.sources.threads import tail_state
from pipeline.oracles.sources.timeline import pages

PLATFORM = "x"
QUOTA = "images:x-oracles"    # no X request: photos come from X's media host, reads from the model
PHOTOS = 30                   # photos per unit, at least one post's
READERS = 12                  # photos read at once: the vision gate's own ceiling

PENDING, SETTLED, RENDER = "images_pending", "settled", "render"
WRITTEN = "rewritten"


def step(pending: bool, tail_owed: bool, stored: bool) -> str:
    """What one post comes to once its photos were tried (deciding): the table above, before the
    render, which may still find nothing changed."""
    if pending:
        return PENDING
    if tail_owed or not stored:
        return SETTLED
    return RENDER


def _photos(row: footprint_photos.Owed) -> int:
    return sum(1 for t in row.group for _ in vision._iter_node_photos(t))


def pick(rows: list[tuple[footprint_photos.Owed, str]]) -> list[footprint_photos.Owed]:
    """The head and the next rows of its Oracle, up to `PHOTOS` photos (deciding)."""
    head, oracle = rows[0]
    picked, n = [head], _photos(head)
    for row, of in rows[1:]:
        if of != oracle:
            continue
        n += _photos(row)
        if n > PHOTOS:
            break
        picked.append(row)
    return picked


def render(conn, rows: list[footprint_photos.Owed], cache: dict,
           lease_lost) -> dict[str, str]:
    """Render each read post again, then land each post in its
    own transaction: its rewrite, if any, with its settle. What each came to, by atom id. A post
    that changed while its photos were read, or whose row did, comes to `CHANGED` and the rest go
    on; the next unit renders it from what is there. A lost lease raises `Changed`, stopping the
    unit; the posts already landed stay."""
    ids = [r.atom_id for r in rows]
    seen, tails, links = pages.page_ledgers(conn, ids)
    writer = footprint._GroupWriter(conn, img_cache=cache, seen=dict(seen),
                                    timer=StageTimer(), tails=tails, links_known=links,
                                    tail_rules=tail_state)
    came, landing = {}, []
    for row in rows:
        group = copy.deepcopy(row.group)
        tail = (tails.get(row.atom_id) or {}).get("thread_tail")
        decided = step(any(vision._photos_pending(t, cache) for t in group),
                       tail == tail_state.TAIL_OWED, row.atom_id in seen)
        if decided == PENDING:
            came[row.atom_id] = PENDING
            continue
        # Forced to the stored tail state; a post with none is decided as a page would.
        forced = tail_state.TailDecision(tail=tail) if tail is not None else None
        res = (writer.produce(group, atom_id=row.atom_id, decision=forced)
               if decided == RENDER else None)
        if res is None or res.get("decided") is None:
            landing.append((row, False))  # settled, skipped, or a render that changed nothing
            continue
        writer.consume(res)
        landing.append((row, True))
    for row, rewrite in landing:
        try:
            with write_tx(conn, site="oracle_photos.write"):
                _refuse(conn, row, seen.get(row.atom_id), lease_lost)
                if rewrite:
                    writer.write_one(row.atom_id)
                footprint_photos.settle(conn, row)
        except Changed:
            if lease_lost():
                raise
            came[row.atom_id] = CHANGED
            continue
        came[row.atom_id] = WRITTEN if rewrite else SETTLED
    return came


def _refuse(conn, row: footprint_photos.Owed, seen_hash: str | None, lease_lost) -> None:
    """Every refusal before one post lands, inside its transaction."""
    if lease_lost():
        raise Changed("photo quota lease lost")
    if not roster.on_roster(conn, row.who_id):
        raise Changed("the Oracle was forgotten while its photos were read")
    if pages.page_ledgers(conn, [row.atom_id])[0].get(row.atom_id) != seen_hash:
        raise Changed(f"{row.atom_id} changed while its photos were read")
    footprint_photos.require_current(conn, row)


class OraclePhotos:
    """A batch of one Oracle's posts' unread photos per unit."""
    name, quota, klass, floor = "oracle_photos", QUOTA, YOURS, 0

    def _owed(self, conn, now) -> list[tuple[footprint_photos.Owed, str]]:
        """Rows that may run now, with the Oracle each belongs to, for Oracles on the roster: a
        forgotten Oracle's rows wait, inert, and a re-add picks them up."""
        heads = roster.heads(conn)
        out = []
        for row in footprint_photos.owed(conn):
            if not retries.may_run(row.retry.retry_at, row.retry.failures, now):
                continue
            oracle = schema.current_canonical(conn, row.who_id)
            if oracle in heads:
                out.append((row, oracle))
        return out

    def next_unit(self, conn, now):
        rows = self._owed(conn, now) if chart_read.refused() is None else []
        if not rows:
            return None
        head, oracle = rows[0]
        return Unit(f"oracle-photos:{head.atom_id}", pick(rows),
                    item=Item(footprint_photos.ROWS, (head.atom_id,), head.retry,
                              platform=PLATFORM),
                    footprint=Footprint(oracle))

    def fetch(self, conn, unit, ctx):
        rows = unit.payload
        home = opyt_home()
        cache = load_image_cache(home)
        vision.read_group_photos([r.group for r in rows], cache, workers=READERS)
        save_image_cache(home, cache)    # merged with what is on disk, before anything else
        came = render(conn, rows, cache, ctx.lease_lost)
        return came[rows[0].atom_id]

    def record(self, conn, unit, fetched):
        if fetched == PENDING:
            return Outcome(PENDING, failure=retries.BLIP,
                           detail="a photo could not be read yet; the post is kept without it")
        return Outcome(fetched)

    def next_due(self, conn, now):
        if chart_read.refused() is not None:
            return None
        return retries.earliest_retry(r.retry for r in footprint_photos.owed(conn))
