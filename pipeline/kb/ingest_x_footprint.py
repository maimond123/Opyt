"""
pipeline/kb/ingest_x_footprint.py — a confirmed Oracle's own X posts → opinion atoms, one
timeline page at a time.

The X timeline scraper (`oracles/sources/timeline/walks.py`) reads the Oracle's
`UserOriginalsTimeline` one request per unit. The page is oracles' (`oracles/sources/timeline/
pages.py`); this module is the library half it renders and writes through (`_GroupWriter`,
`group_atom`): each self-thread or original post becomes one full-body opinion atom. A
thread's tail is fetched later by the tail queue (`threads.tails`), which renders through the
same `_GroupWriter`.

Atom ids are `xprofile:{root_id}`, apart from bookmarks' `x:{tweet_id}`, so the two never collide;
`source_type` stays "x".

Filter: drop RTs and replies-to-others, keep originals + self-threads, no length gate. Filtering
runs before stitching by conversationId, since a conversation includes the author's replies to
commenters.

Photos are never read here: a post renders the reads the image cache holds, and one with a photo
still unread owes a `footprint_photos` row with its write.

Dedup is by content hash: a group is rewritten only when its hash changed, and a page another
lane rewrote meanwhile is not written over (`pages.page_current`). No eligibility gate: one account's own
timeline is single-author by construction.

Fail-safe: a group that fails to render or write skips (no atom, no `seen` mark).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from functools import partial
from typing import Callable

from pipeline.platforms.url_canon import host_of

from . import (derive, footprint_photos, link_cards, post_people, schema, vision, writes)
from .ingest_common import (BASIS_OBSERVED, BODY_COMPLETE, BODY_PARTIAL, StageTimer,
                            body_fields, changed_hash, store_atom)
from .owed_links import owed_links

# Producer-thread ceiling; per-group producers feed one serial consumer that owns conn.
_INGEST_WORKERS = int(os.environ.get("OPYT_INGEST_WORKERS", "20"))

def _has_substantive_media(group: list[dict]) -> bool:
    """Any of the author's own photos read as a document/chart (substance=True). A quoted image
    does not count."""
    return any((m.get("media_read") or {}).get("substance")
               for t in group for m in vision._photos(t))


def _dedup_by_id(raw: list[dict]) -> list[dict]:
    """Order-preserving dedup by tweet id (the API repeats tweets across pages)."""
    seen_ids: set[str] = set()
    unique: list[dict] = []
    for t in raw:
        tid = str(t.get("id", ""))
        if tid and tid not in seen_ids:
            seen_ids.add(tid)
            unique.append(t)
    return unique


def _majority_author(unique: list[dict]) -> tuple[str, str]:
    """(numeric id, handle) of the pull's author by majority vote. ("", "") when no tweet carries
    it, and then the filter drops every reply with a known target."""
    from collections import Counter

    ids = [str((t.get("author") or {}).get("id")) for t in unique if (t.get("author") or {}).get("id")]
    handles = [(t.get("author") or {}).get("userName") for t in unique
               if (t.get("author") or {}).get("userName")]
    return (Counter(ids).most_common(1)[0][0] if ids else "",
            Counter(handles).most_common(1)[0][0] if handles else "")


def _filter_and_stitch(raw: list[dict]) -> list[list[dict]]:
    """Dedup → filter (drop RTs + replies to others, keep originals + self-replies) → stitch by
    conversationId. Filter before stitch: a conversation includes the author's replies to
    commenters, which grouping first would weld into the thread. Returns chrono-sorted groups
    (solo posts = 1-element lists). Pure + offline.

    A reply is kept only when `inReplyToUserId` equals the author's numeric id; the handle field
    is not emitted, and an unknown target drops the reply."""
    from pipeline.ingestion.x_render import _stitch_threads

    unique = _dedup_by_id(raw)
    author_id, _ = _majority_author(unique)

    kept: list[dict] = []
    for t in unique:
        if t.get("isRetweet"):          # from: excludes these; belt-and-suspenders (a re-share isn't theirs)
            continue
        # Keep originals + self-replies; drop replies to anyone else (or an unknown target).
        if t.get("isReply") and str(t.get("inReplyToUserId") or "") != author_id:
            continue
        kept.append(t)                   # no length / engagement gate — short aphorisms stay

    groups = _stitch_threads(kept)       # {conv_id: [chrono tweets]} — group after the filter
    return [g for g in groups.values() if g]


def _own_site_hosts(conn, handle: str) -> frozenset[str]:
    """The hosts of the Oracle's confirmed blog sources, found through its X handle. Their posts
    belong to the blog footprint; minted here they would be stuck as `author_referenced`. No table
    or row → no hosts."""
    try:
        rows = conn.execute(
            "SELECT b.source_key FROM oracle_sources x JOIN oracle_sources b "
            "ON b.canonical_id = x.canonical_id WHERE x.source_type = 'x' "
            "AND lower(x.source_key) = lower(?) AND b.source_type = 'blog'", (handle,)).fetchall()
    except sqlite3.Error:
        return frozenset()
    return frozenset(host_of(r[0]) for r in rows if r[0])


# ── stage nesting: which timed stages sit inside which, so the untimed remainder is reportable ──
# Keep in step with the stage names below; a stale child name would read as "measured, negligible".
_STAGE_NESTING = {
    "produce": ("render", "snapshot"),
    "consume": ("link_queue", "write"),
}


def _stage_residual(totals: dict[str, float]) -> dict[str, float]:
    """Seconds inside a parent stage that no child stage measured; a nonzero residual names the
    parent hiding untimed work."""
    return {p: round(totals[p] - sum(totals.get(c, 0.0) for c in kids), 3)
            for p, kids in _STAGE_NESTING.items() if p in totals}


def group_atom(res: dict, owed: list[dict]) -> dict:
    """A prepared thread and its links become one atom, regardless of which lane fetched it."""
    raw_hash = res["decided"]
    meta, root, group = res["meta"], res["root"], res["group"]
    who_id = res["who_id"]
    return {
        "atom_id": res["atom_id"],
        "source_type": "x",           # X content — one family with bookmarks for retrieval
        "what_kind": "opinion",
        "who_id": who_id,
        "when_ts": meta["when_ts"],
        "when_precision": meta["when_precision"],
        "about_entities": meta["about_entities"],  # @-mention slugs
        "source_url": root.get("url") or f"https://x.com/{res['handle']}/status/{res['root_id']}",
        "raw_hash": raw_hash,
        "description": meta["description"],
        # Structural signals. The author's footprint, so no user curation act is recorded.
        "payload": {
            "like_count": root.get("likeCount", 0),
            "reply_count": root.get("replyCount", 0),
            "is_thread": res["is_thread"],
            "thread_len": len(group),
            "is_article": bool(res["article"]),
            # Also read the quoted object: the fetch can null isQuote on real quote tweets.
            "is_quote": bool(root.get("isQuote") or root.get("quoted_tweet")),
            "has_media": any((t.get("extendedEntities") or {}).get("media") for t in group),
            "media_substance": _has_substantive_media(group),   # a photo read as a chart
            "source_tags": meta["source_tags"],                 # hashtags — author-declared (§6)
            # Partial if the author's own article body is missing or the thread's tail is owed.
            **body_fields(BODY_PARTIAL if res["article_incomplete"] or res["owes_tail"]
                          else BODY_COMPLETE, BASIS_OBSERVED),
            # The tail queue's key: the conversation, its author, and the tweets this body holds.
            **({"thread_tail": res["tail"], "thread_root": res["thread_root"],
                "thread_author": res["thread_author"], "thread_ids": res["thread_ids"]}
               if res["tail"] else {}),
            # The referenced-links queue; recomputed whole on a rewrite.
            "links_owed": owed,
            # Who the post quotes, mentions or links, with the Oracle's own words (the mention arm).
            "engages": post_people.post_people(group),
        },
        "entry_mode": "oracle-footprint",   # not user-saved (curation) / crawled (radar)
    }


class _GroupWriter:
    """One stitched group's path to an atom — produce (pool thread: attach cached photo reads,
    render, hash) → consume (serial: queue links, owe unread photos, hold the atom's write).

    The seam every Oracle X render passes through (timeline pages, the tail queue, the photo
    source), so a thread renders, and hashes, the same whichever writes it. Photos are never read
    here; an unread one owes a `footprint_photos` row in the write's transaction.

    Producers touch only per-group locals and the tallies under `lock`; the consumer alone owns
    `conn` and the ledgers. `owed_convs` / `tails` are the walk's inputs; the tail queue
    and the photo source force the decision instead. `tail_rules` is the thread lane's tail state
    (`oracles.sources.threads.tail_state`), handed in so this module never imports oracles."""

    def __init__(self, conn, *, img_cache: dict, seen: dict, timer: "StageTimer",
                 author_name: str | None = None, owed_convs: set[str] | frozenset = frozenset(),
                 tails: dict | None = None, links_known: set[str] | None = None,
                 tail_rules=None):
        self.conn, self.author_name = conn, author_name
        self.img_cache, self.seen, self.timer = img_cache, seen, timer
        self.owed_convs, self.tails = owed_convs, tails or {}
        self.tail_rules = tail_rules
        # The writes the consumer decided (tail state, an unchanged post's links, owed photos,
        # the atom itself), by the atom each belongs to, made by `write` (all) or `write_one`
        # (one atom's).
        self.local: dict[str, list[Callable[[], None]]] = {}
        self.lock = threading.Lock()
        self.added = self.skipped = self.failed = self.threads = 0   # consumer
        self.submitted = 0                      # atoms held for a write; `added` is what landed
        self.links_queued = 0                   # `links_owed` entries this run recorded
        self._own_hosts: dict[str, frozenset[str]] = {}
        # Posts whose links the queue already knows (`links_owed` present, even empty); an
        # unchanged post not in here gets its links queued without a rewrite. Consumer-only.
        self.links_known = _links_known(conn) if links_known is None else links_known
        self.owed_held = 0                     # producer-phase tally (many threads) → `lock`
        self.photos_owed = 0                   # posts written without a photo's read
        self.cards = link_cards.load(conn)     # stored link cards, read-only on the producers

    def _owed_links(self, group: list[dict], post_atom_id: str) -> list[dict]:
        handle = (group[0].get("author") or {}).get("userName") or ""
        if handle not in self._own_hosts:
            self._own_hosts[handle] = _own_site_hosts(self.conn, handle)
        return owed_links(group, post_atom_id, own_hosts=self._own_hosts[handle])

    def produce(self, group: list[dict], *, atom_id: str | None = None, decision=None):
        """Producer (pool thread): attach cached photo reads, read any X-Article body, render,
        hash. No conn writes. Returns a write-ready result (even for an unchanged group, whose links
        the consumer still queues), or None to skip.

        `atom_id` pins the atom this group rewrites (the tail queue's owed atom); `decision`
        forces the tail decision instead of asking `tail_rules.decide`."""
        from pipeline.ingestion import x_render as xt
        from pipeline.ingestion.utils import log

        timer = self.timer
        root = group[0]                    # chronologically first tweet = the atom's canonical
        root_id = str(root.get("id", ""))
        if not root_id:
            return None
        # A group's identity is its conversation, so later self-replies on a newer page than
        # their root key to the thread's atom rather than a second one.
        conv = str(root.get("conversationId") or root_id)
        has_root = conv == root_id
        atom_id = atom_id or f"xprofile:{conv}"
        is_thread = len(group) > 1
        h = (root.get("author") or {}).get("userName") or ""
        ids = [str(t.get("id")) for t in group]
        author = str((root.get("author") or {}).get("id") or "")

        if decision is None:
            decision = self.tail_rules.decide(self.tails.get(atom_id), conv=conv, ids=ids,
                                              has_root=has_root, stored=atom_id in self.seen,
                                              owed_convs=self.owed_convs)
        if decision.skip:
            return None
        if decision.mark_owed:
            with self.lock:
                self.owed_held += 1
            return {"mark_owed": True, "atom_id": atom_id, "tail": decision.tail,
                    "thread_root": conv, "thread_author": author, "group": group}

        # Cached photo reads and stored link cards, put on each node in place (`description` for
        # the render, `media_read` for `media_substance`, `link_cards` for the `## Links` entry).
        # An unread photo is owed to the photo source.
        for t in group:
            vision.attach_cached(t, self.img_cache)
            link_cards.attach(t, self.cards)
        photos_owed = any(vision._photos_pending(t, self.img_cache) for t in group)

        # X-Article body is a field read: the walk asks for `withArticleRichContentState`. A
        # teaser node (no blocks) is truthy, so test for a body, and mark the atom partial.
        article = None
        article_incomplete = False
        for t in group:
            if not xt._article_tweet_id(t):
                continue
            article = t.get("article")
            _, blocks = xt._article_shape(article or {})
            article_incomplete = not blocks
            if article_incomplete:
                log(f"[footprint] x @{h}: article {t.get('id')} arrived without a body "
                    f"(rendering the teaser) — check `withArticleRichContentState`.")
            break

        if decision.owes:
            with self.lock:
                self.owed_held += 1

        meta = derive.derive_x(root)
        with timer.stage("render"):
            md = xt.tweet_to_markdown(root, article=article,
                                      thread_tweets=group if is_thread else None,
                                      source="x-profile", footer_label="Profile extract")
        # Content-hash dedup: rewrite only on a new/changed body.
        with timer.stage("snapshot"):      # hashes the snapshot against the stored one
            decided = (None if decision.keep_stored
                       else changed_hash(atom_id, md, self.seen))
        return {"group": group, "root": root, "root_id": root_id, "atom_id": atom_id,
                "who_id": meta["who_id"], "meta": meta, "md": md, "is_thread": is_thread,
                "article": article, "decided": decided,
                "photos_owed": photos_owed,
                "article_incomplete": article_incomplete, "tail": decision.tail,
                "owes_tail": decision.owes, "thread_root": conv,
                "thread_author": author, "thread_ids": ids, "handle": h}

    def work(self, group: list[dict]):
        """`produce` under a parent timer, so its children sum against a known total."""
        with self.timer.stage("produce"):
            return self.produce(group)

    def _mark_written(self, atom_id: str, raw_hash: str, is_thread: bool, links: int) -> None:
        """Fires after an atom is durably written, so the summary counts durable atoms."""
        self.seen[atom_id] = raw_hash
        self.added += 1
        self.threads += int(is_thread)
        self.links_queued += links
        self.links_known.add(atom_id)

    def _consume_body(self, res: dict) -> None:
        """Consumer (caller's thread, serial), the sole owner of the write path: works out the
        group's owed links, holds the changed atom's write, and owes its unread photos."""

        timer = self.timer
        if res.get("mark_owed"):
            self._side(res["atom_id"], partial(self.tail_rules.set_tail_state, self.conn,
                                               res["atom_id"], res["tail"], body=BODY_PARTIAL,
                                               root=res["thread_root"],
                                               author=res["thread_author"]))
            return
        group = res["group"]
        who_id = res["who_id"]
        with timer.stage("link_queue"):
            owed = self._owed_links(group, res["atom_id"])

        decided = res["decided"]
        if decided is None:                         # snapshot unchanged → no rewrite
            self.skipped += 1
            if res["atom_id"] not in self.links_known:
                # Its links are not queued yet: queue them in place, with no rewrite.
                self._side(res["atom_id"], partial(self._queue_links, res["atom_id"], owed))
                self.links_known.add(res["atom_id"])
            # A post stored before `engages` existed gains it the same way; a no-op once it has it.
            self._side(res["atom_id"], partial(_fill_engages, self.conn, res["atom_id"],
                                               post_people.post_people(group)))
            return
        meta = res["meta"]
        root = res["root"]
        # Persist the author as a person with the @handle (same convention as ingest_x); idempotent.
        site = meta.get("who_site")
        schema.upsert_entity(
            self.conn, who_id, name=meta.get("who_name") or self.author_name,
            identity_links=[site] if site else None,
            profile={"handle": meta["who_handle"]} if meta.get("who_handle") else None)

        atom = group_atom(res, owed)
        aid, it, n = res["atom_id"], res["is_thread"], len(owed)
        self.submitted += 1
        if res["photos_owed"]:
            self._side(aid, partial(footprint_photos.owe, self.conn, aid, who_id, group))
            self.photos_owed += 1
        # Bookkeeping rides on_written so the counters count durable atoms, not submitted ones.
        self._side(aid, partial(store_atom, self.conn, atom, res["md"],
                                on_written=partial(self._mark_written, aid, atom["raw_hash"],
                                                   it, n)))

    def _side(self, atom_id: str, write: Callable[[], None]) -> None:
        self.local.setdefault(atom_id, []).append(write)

    def consume(self, res: dict) -> None:
        """`_consume_body` under a parent timer. The consumer is serial, so its total is wall
        clock, comparable to `process` (`produce` is thread-seconds)."""
        with self.timer.stage("consume"):
            self._consume_body(res)

    def _queue_links(self, atom_id: str, owed: list[dict]) -> None:
        if _queue_links(self.conn, atom_id, owed):
            writes.after_commit(self.conn, partial(self._count_links, len(owed)))

    def _count_links(self, n: int) -> None:
        self.links_queued += n

    def write(self) -> None:
        """Every write the consumer decided, atoms included, in one transaction, or in the
        caller's when one is open."""
        with self.timer.stage("consume"), self.timer.stage("write"), \
                writes.write_tx(self.conn, site="x.footprint"):
            for side in self.local.values():
                for write in side:
                    write()
        self.local = {}

    def write_one(self, atom_id: str) -> None:
        """One atom's share of `write`, in one transaction or in the caller's when one is open.
        The photo source lands each post with its settle this way, so one post never waits on
        another's."""
        with self.timer.stage("consume"), self.timer.stage("write"), \
                writes.write_tx(self.conn, site="x.footprint"):
            for write in self.local.pop(atom_id, []):
                write()

    def close(self) -> None:
        """Write whatever remains."""
        self.write()

    def summary(self) -> dict:
        """The counters and diagnostics every caller reports, in the walk's summary shape."""
        return {
            "added": self.added, "skipped": self.skipped,
            "failed": self.failed, "threads": self.threads,
            # Posts written without a photo's read, which went to the photo source.
            "photos_owed": self.photos_owed,
            # Link entries this run queued for the reference sources — what tells a caller the
            # queue has work. Captured links are counted by the queue, not here.
            "links_owed": self.links_queued,
            # Parent−Σ(children) per nested stage.
            "stage_residual": _stage_residual(self.timer.totals),
            "stage_seconds": self.timer.totals, "stage_latency": self.timer.distribution(),
        }


# ── The referenced-links queue: what the walk queued, captured after it ──

def _links_known(conn) -> set[str]:
    """X atoms that carry `links_owed` at all, even `[]`."""
    return {r[0] for r in conn.execute(
        "SELECT atom_id FROM atoms WHERE source_type = 'x' "
        "AND json_type(payload, '$.links_owed') IS NOT NULL")}


def _queue_links(conn, atom_id: str, owed: list[dict]) -> bool:
    """Queue an unchanged post's links in place (no rewrite), only where nothing is queued yet.
    The caller commits."""
    cur = conn.execute(
        "UPDATE atoms SET payload = json_set(COALESCE(payload, '{}'), '$.links_owed', json(?)) "
        "WHERE atom_id = ? AND json_type(payload, '$.links_owed') IS NULL",
        (json.dumps(owed), atom_id))
    return cur.rowcount > 0


def _fill_engages(conn, atom_id: str, engages: list[dict]) -> None:
    """Set an unchanged post's `engages` in place (no rewrite), only where it has none yet. The
    caller commits."""
    conn.execute(
        "UPDATE atoms SET payload = json_set(COALESCE(payload, '{}'), '$.engages', json(?)) "
        "WHERE atom_id = ? AND json_type(payload, '$.engages') IS NULL",
        (json.dumps(engages), atom_id))
