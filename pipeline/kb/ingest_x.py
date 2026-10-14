"""Foreground X saves, referenced posts, and the shared X atom builder.

Bulk saved-feed import lives in curation/bookmarks. Its background conversations and Oracle
thread tails are sources of the TweetDetail quota loop.
"""

from __future__ import annotations

import sqlite3
import threading
from functools import partial

from pipeline.platforms.x import conversation as x_conversation
import pipeline.platforms.x.rate as x_rate
from pipeline.platforms import sessions

from . import derive, link_cards, schema
from .ingest_common import (BASIS_OBSERVED, BODY_COMPLETE, body_fields, promote_atom,
                            changed_hash, store_atom)
from .owed_links import referenced_post_links, saved_post_links



class _ConvoFetcher:
    """Read conversation context serially and record only successful reads.

    An empty list is a successful standalone result; None is a failed or disabled read.
    Session and rate refusals disable further conversation requests for this run.
    """

    def __init__(self, checked: set):
        self.checked = checked
        self.enabled = True
        # Admission and counters stay together when a caller shares this fetcher.
        self._lock = threading.Lock()
        self.n_calls = self.n_failed = self.n_chains = 0
        self.session = sessions.x_session("https://x.com/home")

    def chain(self, tid: str) -> list[dict] | None:
        from pipeline.ingestion.utils import log, SyncAuthError

        with self._lock:
            if not self.enabled:
                return None
            self.n_calls += 1
            try:
                chain = x_conversation.fetch_conversation(tid, self.session)
            except (SyncAuthError, x_rate.XRateLimited) as e:
                self.enabled = False
                self.n_failed += 1
                log(f"[kb] conversation reads stopped for this run; the rest of the walk records "
                    f"its saves and re-reads next run: {e}")
                return None
            except Exception as e:
                self.n_failed += 1
                log(f"[kb] conversation fetch failed for {tid}; save skipped: {e}")
                return None
            self.checked.add(str(tid))
            if chain:
                self.n_chains += 1
            return chain


def build_x_atom(norm: dict, atom_id: str, *, raw_hash: str, meta: dict,
                 thread_tweets: list | None = None) -> dict:
    """A rendered tweet → the atom row, shared by every X producer so the payload keeps one shape.

    A bookmark and a hand-dumped post are both personal saves, so this owns their `user-saved`
    entry mode, and the links they owe the referenced-links queue."""
    return {
        "atom_id": atom_id,
        "source_type": "x",
        "what_kind": "opinion",
        "who_id": meta["who_id"],
        "when_ts": meta["when_ts"],
        "when_precision": meta["when_precision"],
        "about_entities": meta["about_entities"],
        "source_url": norm.get("url"),
        "raw_hash": raw_hash,
        "description": meta["description"],
        # Structural fields, incl. the free substantiveness signals (length, thread, article,
        # link/media), kept so a later substantiveness view needs no re-ingest.
        "payload": {
            "like_count": norm.get("likeCount", 0),
            "reply_count": norm.get("replyCount", 0),
            "is_quote": bool(norm.get("isQuote")),
            "is_reply": bool(norm.get("isReply")),
            "is_thread": bool(thread_tweets),
            "is_article": bool(norm.get("article")),
            "has_link": bool((norm.get("entities") or {}).get("urls")),
            "has_media": bool((norm.get("extendedEntities") or {}).get("media")),
            "text_len": len(norm.get("text") or ""),
            "source_tags": meta["source_tags"],   # hashtags — author-declared (§6)
            # Preparation admits only complete reads; failures never reach the atom writer.
            **body_fields(BODY_COMPLETE, BASIS_OBSERVED),
            # The referenced-links queue; recomputed whole on a rewrite.
            "links_owed": saved_post_links(norm, thread_tweets, atom_id),
        },
        "entry_mode": "user-saved",
    }


def peek_tweet(tid: str) -> dict | None:
    """The tweet itself, with no conversation walk, so a caller can show the user a post before
    ingesting it (x.com serves unauthenticated fetchers only a JS shell).

    Uses this machine's X session. `None` on a fetch failure or a deleted / protected / suspended
    post."""
    from pipeline.ingestion.utils import log

    try:
        session = sessions.x_session("https://x.com/home")
        got = x_conversation.fetch_tweets_by_ids(session, [str(tid)]) or []
    except Exception as e:
        log(f"[hopper] tweet peek failed for {tid}: {e}")
        return None
    return next((t for t in got if str(t.get("id")) == str(tid)), None)


def _fetch_one_tweet(tid: str) -> tuple[dict | None, list | None]:
    """Read a focal tweet and its context. Any incomplete read returns `(None, None)`.

    The focal read carries the article body; the conversation read supplies ancestors and
    self-continuations. A successful empty conversation remains a valid standalone save.
    """
    from pipeline.ingestion.x_render import _article_shape
    from pipeline.ingestion.utils import log

    tweet = peek_tweet(tid)
    if not tweet:                           # deleted / protected / suspended, or the fetch failed
        return None, None
    article = tweet.get("article")
    if article and not _article_shape(article)[1]:
        log(f"[hopper] X article {tid} arrived without its body; save skipped.")
        return None, None
    chain = _ConvoFetcher(set()).chain(str(tid))
    if chain is None:
        return None, None
    if chain:                               # splice our copy of the focal in — same as the walk
        chain = [tweet if str(t.get("id")) == str(tid) else t for t in chain]
    return tweet, chain or None


def x_atom_from_url(conn: sqlite3.Connection, url: str, *,
                    entry_mode: str = "user-saved", seen: dict | None = None,
                    img_cache: dict | None = None,
                    on_written=None) -> tuple[str, str | None]:
    """Fetch one X post by URL → an opinion atom; the adapter Hopper routes a status link to, and
    what admits a post a topic's watch found (`entry_mode="frontier"`).
    Returns `(status, atom_id)`: "present" | "saved" | "failed"; `atom_id` is set for the first two.

    Keys on `x:{tweet_id}`, the bookmark walk's id, so a post saved both ways is one atom. Writes
    the author entity; a post the user saved also writes a `save` signal like a bookmark and owes
    its links like one, while a post a watch found is a stranger's, so it signals nothing and owes
    only its papers, like a post an Oracle linked. It never creates an Oracle.

    A failed fetch or a missing tweet skips with no writes. Never raises."""
    from pipeline.ingestion.utils import log
    from pipeline.ingestion.x_render import tweet_to_markdown
    from pipeline.image_cache import load_image_cache, save_image_cache
    from opyt_core.paths import opyt_home

    from . import link_router
    from .vision import enrich_tweet_media

    tid = link_router.parse_tweet_id(url or "")
    if not tid:
        return "failed", None
    atom_id = f"x:{tid}"
    if seen is None:
        seen = schema.load_hashes(conn, "x")
    if atom_id in seen:      # already have it → no fetch, no thread call, no VLM
        # A hand deposit of a post the frontier already crawled is attestation: promote it.
        promote_atom(conn, atom_id, entry_mode)
        return "present", atom_id
    saved = entry_mode == "user-saved"

    try:
        home = opyt_home()
        own_cache = img_cache is None
        cache = load_image_cache(home) if own_cache else img_cache
        norm, chain = _fetch_one_tweet(tid)
        if not norm:
            return "failed", None

        # Read every image, matching the bookmark path — a chart in a post you
        # saved by hand is exactly the image worth reading.
        made = enrich_tweet_media(norm, cache)
        cards = link_cards.load(conn)
        for t in [norm, *(chain or [])]:
            link_cards.attach(t, cards)
        md = tweet_to_markdown(norm, article=norm.get("article"), thread_tweets=chain,
                               source="x-bookmark" if saved else "x-watch",
                               footer_label="Bookmarked" if saved else "Found by a watch")
        raw_hash = changed_hash(atom_id, md, seen)
        if raw_hash is None:                       # unchanged snapshot → nothing to rewrite
            promote_atom(conn, atom_id, entry_mode)
            return "present", atom_id

        meta = derive.derive_x(norm)
        atom = {**build_x_atom(norm, atom_id, raw_hash=raw_hash, meta=meta,
                               thread_tweets=chain), "entry_mode": entry_mode}
        if not saved:                             # nobody chose it: only its papers are owed
            atom["payload"]["links_owed"] = referenced_post_links(norm, atom_id)
        schema.upsert_entity(conn, meta["who_id"], name=meta.get("who_name"),
                             identity_links=[meta["who_site"]] if meta.get("who_site") else None,
                             profile={"handle": meta["who_handle"]} if meta.get("who_handle") else None)
        if saved:                                 # a stranger a watch found is no curation signal
            schema.ensure_signal(conn, meta["who_id"], "save", "x")
        store_atom(conn, atom, md, on_written=on_written and partial(on_written, atom_id))
        seen[atom_id] = raw_hash
        if made and own_cache:
            save_image_cache(home, cache)        # persist VLM descriptions for the next run
        return "saved", atom_id
    except Exception as e:                       # fail-safe: a bad post never raises out
        log(f"[hopper] x post ingest failed for {url}: {e}")
        return "failed", None


def render_referenced_post(norm: dict, img_cache: dict) -> str | None:
    """A post an Oracle linked to → its markdown, or None when it cannot be rendered whole. Touches
    no `conn`, so the X footprint runs it on a pool thread beside the walk.

    Images go through the Oracle-post cascade, not the bookmark read-everything. No thread
    context: the body is written from the post alone. An X Article teaser (no body) renders None."""
    from pipeline.ingestion.x_render import _article_shape, tweet_to_markdown

    from .vision import enrich_tweet_media

    article = norm.get("article")
    if article and not _article_shape(article)[1]:
        return None
    enrich_tweet_media(norm, img_cache)
    return tweet_to_markdown(norm, article=article, thread_tweets=None,
                             source="x-referenced", footer_label="Referenced")


def referenced_x_atom(conn: sqlite3.Connection, norm: dict, md: str, *, seen: dict,
                      on_written=None, isolate=True, store=store_atom) -> tuple[str, str | None]:
    """A post an Oracle linked to, already rendered → an `author_referenced` atom. Returns
    `(status, atom_id)`: "present" | "saved" | "failed".

    Keys on `x:{tweet_id}`, the bookmark namespace, so a later bookmark collapses to one atom and
    becomes `user-saved` while the referrer survives the rewrite (`schema._PAYLOAD_SET`).

    Unlike `x_atom_from_url`: no `save` signal and no promotion to `user-saved`, since nobody saved
    it. Of its own links it owes only its papers (`owed_links.referenced_post_links`), which get
    lines. The author still joins the entity graph. `store` is the write (`store_atom`); a caller
    that lands the atom with its own state later passes one that holds it."""
    from pipeline.ingestion.utils import log

    tid = str(norm.get("id") or "")
    if not tid or not md:
        return "failed", None
    atom_id = f"x:{tid}"
    if atom_id in seen:
        return "present", atom_id
    try:
        raw_hash = changed_hash(atom_id, md, seen)
        if raw_hash is None:
            return "present", atom_id
        meta = derive.derive_x(norm)
        # The saved shape, re-stamped: `build_x_atom` takes no entry mode, and every X atom must
        # carry the same payload shape. Nobody saved it, so of its own links it owes only its papers.
        atom = {**build_x_atom(norm, atom_id, raw_hash=raw_hash, meta=meta),
                "entry_mode": "author_referenced"}
        atom["payload"]["links_owed"] = referenced_post_links(norm, atom_id)
        schema.upsert_entity(conn, meta["who_id"], name=meta.get("who_name"),
                             identity_links=[meta["who_site"]] if meta.get("who_site") else None,
                             profile={"handle": meta["who_handle"]} if meta.get("who_handle") else None)
        store(conn, atom, md, on_written=on_written and partial(on_written, atom_id))
    except Exception as e:                       # foreground callers isolate individual posts
        if not isolate:
            raise
        log(f"[footprint] referenced x post {tid} not saved: {e}")
        return "failed", None
    seen[atom_id] = raw_hash
    return "saved", atom_id
