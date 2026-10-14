"""Saved X posts; their unread photos are owed to the saved-images source, and their
conversations to the TweetDetail lane.

`SavedXPosts` saves a batch of posts per unit from the work the Bookmarks listing offered
(`saved_items`), preparing each through `_prepare` and writing each in its own transaction. It reads
no image and spends no TweetDetail: a post renders with the image reads the cache already holds,
owes a `saved_images` row for any photo still unread, and a post with replies is owed to the
thread lane, which reads owed conversations from the atoms. What a re-render
needs beyond the post (its chain, and whether its atom came from an article teaser) is kept on the
post's `saved_items` row, never on the atom.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from opyt_core.paths import opyt_home
from pipeline.curation import (conversations, saved_images, saved_items, saving, signals,
                               x_collectors)
from pipeline.ingestion.x_render import _article_shape, tweet_to_markdown
from pipeline.kb import derive, link_cards, vision
from pipeline.kb.ingest_common import changed_hash, promote_atom, store_atom
from pipeline.kb.ingest_x import build_x_atom
from pipeline.kb.writes import write_tx
from pipeline.worker.quota_loop.source import CHANGED, YOURS, Changed, Outcome

PLATFORM = "x"
QUOTA = "saved:x"       # no X request: the posts come from the Bookmarks listing's work
BATCH = 25              # posts per unit


@dataclass(frozen=True)
class Stored:
    version: int
    raw_hash: str
    payload: dict


PRESENT, RECOVER, RENDER = "present", "recover", "render"


def what_to_save(*, stored: bool, checked: bool, rerender: bool, chain_missing: bool) -> str:
    """What a saved X post comes to before anything is read for it (deciding). `rerender`: an
    article teaser now or before, or a photo not yet described. `chain_missing`: the stored
    atom is a thread whose saved row keeps no chain (it was saved by hand before the listing
    offered it); its body is kept until the thread lane recovers the chain, rather than
    overwritten by a solo update.

    | Stored | Conversation checked | `rerender` | `chain_missing` | Result    |
    |--------|----------------------|------------|-----------------|-----------|
    | yes    | yes                  | no         | —               | `PRESENT` |
    | yes    | otherwise            |            | yes             | `RECOVER` |
    | yes    | otherwise            |            | no              | `RENDER`  |
    | no     | —                    | —          | —               | `RENDER`  |
    """
    if stored and checked and not rerender:
        return PRESENT
    if stored and chain_missing:
        return RECOVER
    return RENDER


def is_teaser(norm: dict) -> bool:
    """The post carries an X Article that arrived as its teaser, without the body."""
    return bool(norm.get("article")) and not _article_shape(norm["article"])[1]


def _prepare(norm, previous: Stored | None, kept: saved_items.Rendering, checked: bool,
             cache: dict, cards: dict):
    norm = copy.deepcopy(norm)
    tid = str(norm.get("id") or "")
    if not tid:
        return None
    payload = previous.payload if previous else {}
    rerender = is_teaser(norm) or kept.rendered_from_teaser or vision._photos_pending(norm, cache)
    step = what_to_save(stored=previous is not None, checked=checked, rerender=rerender,
                        chain_missing=bool(payload.get("is_thread") and not kept.chain))
    if step == PRESENT:
        return {"present": f"x:{tid}"}
    if step == RECOVER:
        return {"recover": f"x:{tid}"}
    # Only what the cache already holds: the photos still unread are the saved-images source's.
    vision.attach_cached(norm, cache)
    return render(norm, previous, kept.chain, cards)


def render(norm: dict, previous: Stored | None, chain: list | None, cards: dict) -> dict:
    """A saved post rendered from `norm` as it stands, its photos' reads already attached, with
    the thread chain its saved row keeps and the stored link cards (`link_cards.load`): the new
    atom, or `unchanged` when the render hashes the same as the stored one."""
    tid = str(norm["id"])
    chain = [norm if str(t.get("id")) == tid else t for t in chain or []]
    for t in [norm, *chain]:
        link_cards.attach(t, cards)
    meta = derive.derive_x(norm)
    md = tweet_to_markdown(norm, article=norm.get("article"), thread_tweets=chain or None,
                           source="x-bookmark", footer_label="Bookmarked")
    atom_id = f"x:{tid}"
    seen = {atom_id: previous.raw_hash} if previous else {}
    decided = changed_hash(atom_id, md, seen)
    complete = (not is_teaser(norm) and not norm.get("isReply")
                and not (norm.get("replyCount") or 0))
    if decided is None:
        return {"unchanged": atom_id, "complete": complete}
    atom = build_x_atom(norm, atom_id, raw_hash=decided, meta=meta,
                        thread_tweets=chain or None)
    return {"atom": atom, "md": md, "previous": previous, "complete": complete}


def check_version(conn, atom_id, previous):
    row = conn.execute("SELECT version FROM atoms WHERE atom_id=?", (atom_id,)).fetchone()
    if (row is None) != (previous is None) or (row is not None and row[0] != previous.version):
        raise Changed(f"{atom_id} changed during bookmark preparation")


def stored(conn, atom_id: str) -> Stored | None:
    row = conn.execute("SELECT version, raw_hash, payload FROM atoms WHERE atom_id=?",
                       (atom_id,)).fetchone()
    return Stored(row[0], row[1], json.loads(row[2] or "{}")) if row else None


def _refuse(conn, saved: saved_items.Saved, previous: Stored | None, lease_lost) -> None:
    """Every refusal before a saved post's write, inside its transaction."""
    saving.require_lease(lease_lost)
    check_version(conn, f"x:{saved.item}", previous)
    saved_items.require_current(conn, saved)


def _done(conn, saved: saved_items.Saved, cache: dict, now: datetime) -> None:
    """A saved post's done-state, with its unread photos owed (inside its transaction)."""
    saved_items.settle(conn, saved)
    if vision._photos_pending(saved.data, cache):
        saved_images.owe(conn, saved, now)


def _settle_unwritten(conn, saved, previous, result, cache, lease_lost, now) -> str:
    """A post whose atom is not rewritten: already stored and current, or only its input
    recorded."""
    aid = f"x:{saved.item}"
    if "present" in result:
        _refuse(conn, saved, previous, lease_lost)     # before the promotion, which commits itself
        promote_atom(conn, aid, "user-saved")
        with write_tx(conn, site="saved_x.done"):
            _refuse(conn, saved, previous, lease_lost)
            saved_items.settle(conn, saved)
        return "present"
    with write_tx(conn, site="saved_x.inputs"):
        _refuse(conn, saved, previous, lease_lost)
        saved_items.record_rendered(conn, PLATFORM, saved.item, teaser=is_teaser(saved.data))
        if "recover" in result:
            conn.execute("UPDATE atoms SET payload=json_set(payload, '$.bookmark_refresh', "
                         "json('true')) WHERE atom_id=?", (aid,))
        elif result["complete"]:
            conversations.mark_checked(conn, saved.item)
    promote_atom(conn, aid, "user-saved")
    with write_tx(conn, site="saved_x.done"):
        _refuse(conn, saved, previous, lease_lost)
        _done(conn, saved, cache, now)
    return "recover" if "recover" in result else "unchanged"


def save_batch(conn, batch: list[saved_items.Saved], cache: dict, lease_lost,
               now: datetime) -> dict[str, str]:
    """Prepare and save a batch of saved X posts, written one post per transaction: each post's
    atom or its recorded input, its conversation checked when it has none, its done-state, and its
    unread photos owed. What each post came to, by item.

    A post refused for its own reason (`changed`: re-offered meanwhile) is left as it was, and the
    rest go on. A lost lease stops the batch."""
    ledger = conversations.checked(opyt_home())
    try:
        checked = {s.item for s in batch if s.item in ledger}
    finally:
        ledger.close()
    came, renders = {}, []
    cards = link_cards.load(conn)
    for saved in batch:
        previous = stored(conn, f"x:{saved.item}")
        kept = saved_items.rendering(conn, PLATFORM, saved.item)
        result = _prepare(saved.data, previous, kept, saved.item in checked, cache, cards)
        if "atom" in result:
            renders.append((saved, previous, result))
            continue
        try:
            came[saved.item] = _settle_unwritten(conn, saved, previous, result, cache,
                                                 lease_lost, now)
        except Changed:
            if lease_lost():
                raise
            came[saved.item] = CHANGED
    for saved, previous, result in renders:
        try:
            with write_tx(conn, site="saved_x.atom"):
                _refuse(conn, saved, previous, lease_lost)
                store_atom(conn, result["atom"], result["md"])
                saved_items.record_rendered(conn, PLATFORM, saved.item,
                                            teaser=is_teaser(saved.data))
                if result["complete"]:
                    conversations.mark_checked(conn, saved.item)
                _done(conn, saved, cache, now)
        except Changed:
            if lease_lost():
                raise
            came[saved.item] = CHANGED
            continue
        person = x_collectors.bookmark_author(saved.data, saved.account)
        if person is not None:
            signals.assert_saved_author(conn, person, PLATFORM)
        came[saved.item] = "saved"
    return came


class SavedXPosts:
    """A batch of saved X posts per unit, from the work the Bookmarks listing offered. It never
    reads the feed, so a failed post never costs another Bookmarks request.

    The unit charges its head, the oldest owed post that may run. The rest of the batch ride
    along: a rider that saves is done like the head, and a rider that fails is left owed and
    uncharged, to be charged once it is a head itself."""
    name, quota, klass, floor = "saved_x_posts", QUOTA, YOURS, 0

    def next_unit(self, conn, now):
        units = saving.owed_units(conn, PLATFORM, "saved-x", now)
        if not units:
            return None
        return replace(units[0], payload=[u.payload for u in units[:BATCH]])

    def fetch(self, conn, unit, ctx):
        with saving.image_cache() as cache:
            came = save_batch(conn, unit.payload, cache, ctx.lease_lost,
                              datetime.now(timezone.utc))
        return came[unit.payload[0].item]

    def record(self, conn, unit, fetched):
        return Outcome(fetched)

    def next_due(self, conn, now):
        return saving.next_due(conn, PLATFORM)
