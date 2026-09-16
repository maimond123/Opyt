"""
pipeline/kb/ingest_x.py — X bookmarks → OPINION atoms (direct-to-atom, free source).

A bookmark is content David deliberately SAVED, so `entry_mode="user-saved"` (saved, not
authored) and `what_kind="opinion"`. This reuses the existing cookie-scrape iterator and
markdown renderer wholesale — the atom-KB adds the routing card + chunk embeddings + the
factual edge graph on top of the SAME snapshot the vault path already knew how to make.

Uses `x_graphql.iterate_bookmarks` for normalized bookmark input and
`x_render.tweet_to_markdown` for snapshot text.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime
from typing import Callable

from . import derive, schema
from .embed import assert_model
from .ingest_common import (BASIS_OBSERVED, BODY_COMPLETE, AtomSink, StageTimer,
                            body_fields, promote_atom, run_concurrent, snapshot_and_hash,
                            submit_atom)

# The pool prepares bookmarks concurrently; the shared fetcher serializes conversation reads.
# Model seams retain their own concurrency controls, and the write path stays serial.
_INGEST_WORKERS = int(os.environ.get("OPYT_INGEST_WORKERS", "20"))
# How many new image descriptions may accrue before the consumer persists the cache — bounds the
# repeated image descriptions after a mid-run crash without a whole-dict JSON write per bookmark.
_CACHE_FLUSH_EVERY = 16
# Checkpoint the resolved-conversation ledger every N durable atoms, not only at end-of-run, so a
# crash mid-backfill re-fetches ≤N threads next run. Safe to checkpoint early: a ledger entry ahead
# of its atom just costs a re-fetch, never a wrong skip.
_LEDGER_FLUSH_EVERY = 64


class _ConvoFetcher:
    """Read conversation context serially and record only successful reads.

    An empty list is a successful standalone result; None is a failed or disabled read.
    Session and rate refusals disable further conversation requests for this run.
    """

    def __init__(self, checked: set):
        self.checked = checked
        self.enabled = True
        # The bookmark pool shares this fetcher. One lock owns request admission and counters.
        self._lock = threading.Lock()
        self.n_calls = self.n_failed = self.n_chains = 0
        from pipeline.ingestion import x_graphql_core as core
        self.session = core.x_session("https://x.com/home")

    def chain(self, tid: str) -> list[dict] | None:
        from pipeline.ingestion import x_graphql_core as core
        from pipeline.ingestion.utils import log, SyncAuthError

        with self._lock:
            if not self.enabled:
                return None
            self.n_calls += 1
            try:
                chain = core.fetch_conversation(tid, self.session, self.session)
            except (SyncAuthError, core.XRateLimited) as e:
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


def build_x_atom(norm: dict, atom_id: str, *, raw_ref: str, raw_hash: str, meta: dict,
                 thread_tweets: list | None = None) -> dict:
    """A rendered tweet → the atom row. Extracted 2026-08-13 from `sync_bookmarks._work` so the
    walk-my-bookmarks path and the hand-dump-one-post path build the SAME shape rather than two
    that drift — the payload here is the substantiveness signal set, and a field missing from one
    producer would look like a real absence to every later reader.

    A bookmark and a hand-dumped post are both personal saves, so this adapter owns
    their `user-saved` entry mode."""
    return {
        "atom_id": atom_id,
        "source_type": "x",
        "what_kind": "opinion",
        "who_id": meta["who_id"],
        "when_ts": meta["when_ts"],
        "when_precision": meta["when_precision"],
        "about_entities": meta["about_entities"],
        "source_url": norm.get("url"),
        "raw_ref": raw_ref,
        "raw_hash": raw_hash,
        "description": meta["description"],
        # Structural fields, incl. the FREE substantiveness signals (length, is_thread,
        # is_article, has_link/media). Recomputable insurance for the deferred Oracle-ingest
        # substantiveness VIEW (docs/Old-Investigations/2026-07-16-oracle-post-substantiveness-
        # signal.md) — bookmarks don't need it (curation is the filter), but landing it now
        # means that view is computable later without re-ingesting.
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
        },
        "entry_mode": "user-saved",
    }


def peek_tweet(tid: str) -> dict | None:
    """The tweet ITSELF, with NO conversation walk. `None` if it cannot be read.

    Exists so a caller can SHOW the user what a post is before ingesting it. That is not a
    nicety for X the way it would be for a blog: a host model can fetch a Verge article or an arXiv
    page and describe it, but x.com serves a JS shell to unauthenticated fetchers, so the model
    pasting a status link genuinely knows nothing but the url. And `x:2086520133909168332` — the
    only thing a bare preview could show — is unverifiable by a human, unlike
    `blog:theverge.com/…/some-cool-article`.

    It uses this machine's own X session. The local-session path is the reason a preview needs no
    separate key: the only earlier keyless read available was
    `TweetDetail`, which returns a CONVERSATION, and `reconstruct_chain` returns [] for a chain of
    one — so a solo post was invisible. `TweetResultsByRestIds` fetches the post itself, which is
    what closed that gap.

    Fail-safe: `None` on a fetch failure or a deleted / protected / suspended post — both of which
    the caller reports rather than guesses."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion.utils import log

    try:
        session = core.x_session("https://x.com/home")
        got = core.fetch_tweets_by_ids(session, session, [str(tid)]) or []
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
    if chain:                               # splice OUR copy of the focal in — same as the walk
        chain = [tweet if str(t.get("id")) == str(tid) else t for t in chain]
    return tweet, chain or None


def x_atom_from_url(conn: sqlite3.Connection, embedder, url: str, *,
                    seen: dict | None = None,
                    img_cache: dict | None = None, sink=None,
                    on_written=None) -> tuple[str, str | None]:
    """Fetch ONE X post by URL → an opinion atom. The single-item twin of `sync_bookmarks`, and the
    adapter Hopper routes an x.com/twitter.com status link to. Returns `(status, atom_id)` where
    status is "present" | "saved" | "failed"; `atom_id` is real for the first two.

    Keys on `x:{tweet_id}` — the SAME id the bookmark walk uses, deliberately. Dumping a post you
    later bookmark (or bookmarked already) collapses to ONE atom instead of a twin. That is the
    opposite of the footprint adapter's `xprofile:` namespace decision, and for the opposite
    reason: footprint renders a thread by a DIFFERENT path so the two are not byte-identical, while
    this path renders exactly what the bookmark walk renders.

    Writes the author entity + a `save` curation signal, identically to a bookmark, because it IS
    one — David personally handed the post over. That grows the entity graph and the Stage-4
    candidate ranking; it does NOT create an Oracle (only `add_oracle` does, and Hopper must never
    route around it).

    Fail-safe: a fetch that fails or a tweet that no longer exists SKIPS — no atom, no signal, no
    mark. Never raises."""
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
    if atom_id in seen:      # already have it → no fetch, no thread call, no VLM, no embed
        # A hand deposit of a post the frontier already crawled IS attestation — promote, then
        # answer exactly as before (RULED 2026-08-25).
        promote_atom(conn, atom_id, "user-saved")
        return "present", atom_id

    try:
        # Match the bulk path: reject a known vector-subspace mismatch before any fetch, vision
        # work, raw archival, or embedding can spend work that `_write_atom` would refuse.
        assert_model(conn, embedder)
        home = opyt_home()
        own_cache = img_cache is None
        cache = load_image_cache(home) if own_cache else img_cache
        norm, chain = _fetch_one_tweet(tid)
        if not norm:
            return "failed", None

        # Read every image, matching the bookmark path — a chart in a post you
        # saved by hand is exactly the image worth reading.
        made = enrich_tweet_media(norm, cache, describe_all=True)
        md = tweet_to_markdown(norm, article=norm.get("article"), thread_tweets=chain,
                               source="x-bookmark", footer_label="Bookmarked")
        decided = snapshot_and_hash("x", atom_id, md, seen)
        if decided is None:                       # unchanged snapshot → nothing to re-embed
            promote_atom(conn, atom_id, "user-saved")
            return "present", atom_id
        raw_ref, raw_hash = decided

        meta = derive.derive_x(norm)
        atom = build_x_atom(norm, atom_id, raw_ref=raw_ref, raw_hash=raw_hash, meta=meta,
                            thread_tweets=chain)
        schema.upsert_entity(conn, meta["who_id"], name=meta.get("who_name"),
                             identity_links=[meta["who_site"]] if meta.get("who_site") else None,
                             profile={"handle": meta["who_handle"]} if meta.get("who_handle") else None)
        schema.add_signal(conn, meta["who_id"], "save", "x")
        submit_atom(conn, embedder, sink, atom=atom, snapshot_text=md, on_written=on_written)
        seen[atom_id] = raw_hash
        if made and own_cache:
            save_image_cache(home, cache)        # persist VLM descriptions for the next run
        return "saved", atom_id
    except Exception as e:                       # fail-safe: a bad post never raises out
        log(f"[hopper] x post ingest failed for {url}: {e}")
        return "failed", None


def sync_bookmarks(conn: sqlite3.Connection, embedder, *, limit: int = 0,
                   since: datetime | None = None,
                   should_stop: Callable[[], bool] | None = None,
                   enrich: bool = False) -> dict:
    """Ingest up to `limit` bookmarks (0 = all) as opinion atoms. Idempotent: unchanged
    bookmarks (same snapshot hash) are skipped, no re-embed. Returns a run summary.

    TWO PASSES, ONE FUNCTION, ONE RENDERER.

    `enrich=False` — the FREE pass. Everything comes off the Bookmarks page, which is 11 requests
    of a 500/15-min bucket for 1,001 bookmarks (measured 2026-09-13): no `TweetDetail`, no VLM. The
    atom is written WITHOUT thread context and WITHOUT image descriptions, and the tweet id stays
    out of `convo_checked`, which is what brings it back for the pass below.

    `enrich=True` — the METERED pass ("Enrichment" to the user). Resolves each conversation through
    `TweetDetail` (exactly 150/15 min) and reads every photo, then re-renders. An atom whose
    markdown changes re-mints at a bumped `version`, keeping its `first_seen`.

    THE ATOM NEVER WAITS ON A METERED FETCH (R3, 2026-09-13), and this inverts what the tests used
    to assert. CLAUDE.md's fail-safe invariant is "a FAILED external call must SKIP — no write, no
    mark-processed". The Bookmarks call SUCCEEDED; an atom written from it is complete state
    derived from a successful read, not partial state from a failed one. The `TweetDetail` call
    that did not answer marks nothing, so it retries — which is the invariant, honoured. The old
    behaviour was not caution: 966 of 1,001 bookmarks need `TweetDetail`, `_ConvoFetcher` disables
    for the whole run on the first refusal, and the re-run needed a resident worker that is not
    installed, so 851 bookmarks got no atom EVER. Not late — never.

    Two passes and not two functions because `tweet_to_markdown` renders from what it is handed. A
    VLM-only second pass calling it with `thread_tweets=None` would render markdown WITHOUT the
    chain, change the hash, and re-mint — deleting thread context from an atom that already had it.

    The parameter is `enrich`, never `fetch_threads`: that name is banned on this function by
    `.guards.py` (`retired-saved-x-options`) and as a literal anywhere in this file
    (`retired-x-adapter-paths`).

    `deferred` counts bookmarks whose atom is written but whose thread context is still owed — the
    honest size of the Enrichment backlog, not of a coverage shortfall.

    `since` filters on when the tweet was WRITTEN, not when you saved it (X exposes no
    bookmark timestamp). It's applied to the ITERATOR, upstream of `_work`, so a skipped bookmark
    costs no thread fetch, VLM read, or embed. It SKIPS rather than STOPS: the walk is ordered by
    save time while the cutoff is on write time, so an old tweet saved recently sits near the top —
    breaking there would silently truncate everything saved after it.
"""
    from pipeline.ingestion.x_graphql import iterate_bookmarks
    from pipeline.ingestion.x_render import _article_shape, tweet_to_markdown, _parse_twitter_date
    from pipeline.ingestion.utils import load_state, log, save_state
    # Per-image latency lives on the seam that makes the calls (ocr_cascade), not describe_images.
    from pipeline import ocr_cascade as _ocr
    from pipeline.image_cache import load_image_cache, save_image_cache
    from opyt_core.paths import opyt_home
    from .vision import _photos_pending, enrich_tweet_media

    # Guard the store's embedding identity BEFORE paying to embed — a model mismatch on an
    # existing store raises here (dim is verified per-atom once discovered), not after work.
    assert_model(conn, embedder)
    seen = schema.load_hashes(conn, "x")
    home = opyt_home()
    # Image descriptions cached by URL (immutable CDN links) → re-runs are free + hash-stable.
    img_cache = load_image_cache(home)
    # Conversation resolved-ledger: tweet ids whose thread was already fetched, so the one-time
    # backfill of all bookmarks doesn't re-pay on later syncs. The on-disk ledger is THREAD-AFFINE
    # (main thread only); snapshot to a plain set for the producer pool and bulk-persist at the end.
    convo_ledger = home / "x_convo_checked.json"
    convo_checked = set(load_state(convo_ledger))
    convo = _ConvoFetcher(convo_checked)
    added = threads = images_new = out_of_window = 0
    # `added`/`threads`/`images_new` are mutated ONLY on the consumer (single-threaded) → no lock.
    # The two SKIP tallies fire from many producer threads (fast-skip + hash-unchanged), so they sit
    # behind a lock — a lost `+=` would misreport how much of the corpus was one-time-skipped.
    counts = {"skipped": 0, "thread_skipped_standalone": 0, "thread_deferred": 0,
              "article_teaser": 0}
    counts_lock = threading.Lock()
    # Presence-hit bookmarks, promoted AFTER the walk on this (main) thread. A skip in `_work` is a
    # bookmark the user holds on an atom the store already has — attestation, so a frontier-lane
    # row becomes user-saved (RULED 2026-08-25). Collected rather than written in place because
    # `_work` runs on the producer pool and the one rule there is that nothing but the consumer
    # touches `conn`. `set.add` is GIL-atomic, the same guarantee `convo_checked` leans on.
    presence_hits: set[str] = set()
    # Funnel counters (ARC-1 Phase-2 prep). Snapshot the per-image VLM stats so the summary reports
    # THIS run's delta, not the process-cumulative total (matters in the long-lived MCP server).
    vlm0 = {k: _ocr.stats_snapshot()[k] for k in ("calls", "failures", "seconds")}

    # Cross-atom embed batching + per-stage timing (ARC-1 Phase 1). The sink defers embedding until
    # ~256 chunks accrue (4×batch_size); `timer` records where the wall-clock actually goes. Phase 2:
    # producers run the network work in parallel; a SINGLE consumer owns this sink + the conn.
    timer = StageTimer()
    # An 8×batch_size flush (not 4×): the flush width is the demand pressing on embed()'s AIMD
    # gate, so wider lets it probe further. A crash re-embeds only the buffered idempotent chunks.
    bs = int(getattr(embedder, "batch_size", 64) or 64)
    sink = AtomSink(conn, embedder, timer=timer, flush_chunks=8 * bs)
    cache_pending = 0            # new descriptions accrued since the last consumer-side cache flush
    ledger_pending = 0          # durable atoms since the last conversation-ledger checkpoint (3a)
    lease_lost = False

    def _mark_written(atom_id: str, raw_hash: str, is_thread: bool) -> None:
        """Fires only AFTER an atom is durably written (on the consumer thread), so the summary counts
        DURABLE atoms — truthful on a crash (an atom lost in the pre-write buffer is never marked)."""
        nonlocal added, threads
        seen[atom_id] = raw_hash
        added += 1
        if is_thread:
            threads += 1

    def _work(norm: dict):
        """PRODUCER (pool thread): resolve context, enrich images, render, hash.
        Touches ONLY per-bookmark locals + thread-safe shared state (the
        AIMD-gated transport seams, the GIL-atomic `convo_checked`/`img_cache`, the locked timer).
        No `conn` writes here. Returns a write-ready result dict, or None to skip."""
        tid = norm.get("id")
        if not tid:
            return None
        atom_id = f"x:{tid}"
        # FULLY RESOLVED — ingested, conversation-checked, AND every photo described → skip BEFORE
        # any fetch. This is what makes the big first backfill one-time.
        #
        # ⚠️ THE THIRD TERM IS LOAD-BEARING and was added with the deferred-VLM pass. Without it:
        # the free pass writes the atom (→ `seen`), Enrichment run 1 fetches the chain and
        # `_ConvoFetcher.chain` adds the tid on ANY successful read (an empty chain included) — and
        # now both of the first two terms are true forever, so the photo is never read. Silent, and
        # it would hit all 258 photo-bearing bookmarks in David's store.
        if (atom_id in seen and str(tid) in convo_checked
                and not _photos_pending(norm, img_cache)):
            presence_hits.add(atom_id)
            with counts_lock:
                counts["skipped"] += 1
            return None

        # ⚠️ NOTHING BELOW WAITS ON A METERED FETCH (R3, 2026-09-13). Everything the atom needs —
        # text, author, entities, urls, media urls, the article node — is on the Bookmarks page,
        # which SUCCEEDED. Thread context and image descriptions are the two metered upgrades, and
        # an upgrade that did not arrive leaves `tid` out of `convo_checked` / the photo out of the
        # image cache, so the next Enrichment run re-reads exactly those and the atom re-mints.
        #
        # Waiting was not caution, it was loss: 966 of 1,001 bookmarks want `TweetDetail`,
        # `_ConvoFetcher` disables itself for the whole run on the first refusal, and the re-run
        # needed a resident worker that is not installed — so 851 bookmarks got no atom EVER, and
        # (until 2026-09-13) no `save` signal either, which is the corroboration evidence the
        # screen scores on. See `sync_bookmarks`' docstring for why this honours the fail-safe
        # invariant rather than bending it.
        meta = derive.derive_x(norm)

        article = norm.get("article")
        teaser = bool(article) and not _article_shape(article)[1]
        if teaser:
            # The title and preview, not the body — X ships that shape with a 200 and a truthy
            # node. Still the post the user saved, so it is written; `tid` stays out of
            # `convo_checked` (below) so the next walk re-reads it, and when the body finally
            # arrives the markdown changes and the atom re-mints.
            log(f"[kb] X article {tid} arrived without its body; teaser atom written, re-minted "
                f"when the body lands.")
            with counts_lock:
                counts["article_teaser"] += 1

        # Thread context — the metered half. `TweetDetail` is EXACTLY 150 per 15 minutes against
        # 966 bookmarks that want it, so the free pass does not ask at all and renders solo.
        thread_tweets = None
        should_fetch = bool(norm.get("isReply")) or (norm.get("replyCount") or 0) > 0
        if teaser:
            pass                                # already counted; `tid` deliberately left unmarked
        elif not should_fetch:
            convo_checked.add(str(tid))         # set.add is atomic → safe across producer threads
            with counts_lock:
                counts["thread_skipped_standalone"] += 1
        elif not enrich:
            with counts_lock:                   # owed, not lost — Enrichment picks up exactly these
                counts["thread_deferred"] += 1
        else:
            with timer.stage("thread_fetch"):
                chain = convo.chain(tid)
            if chain is None:
                with counts_lock:
                    counts["thread_deferred"] += 1
            elif chain:
                thread_tweets = [norm if str(t.get("id")) == str(tid) else t for t in chain]

        # Image work — the other metered half (R5). Deferred with the chain so ONE renderer call
        # produces the final markdown: a VLM-only second pass would have to call
        # `tweet_to_markdown(thread_tweets=None)` and would delete the chain from an atom that
        # already had it. The photo stays out of the image cache until it is read, which is the
        # third term of the skip gate above.
        made = 0
        if enrich:
            with timer.stage("vlm"):
                made = enrich_tweet_media(norm, img_cache, describe_all=True)

        # Pass the article node so the full body — not the teaser — is chunked; thread_tweets renders
        # the whole debate chain (renderer precedence is article > thread).
        with timer.stage("render"):
            md = tweet_to_markdown(norm, article=article, thread_tweets=thread_tweets,
                                   source="x-bookmark", footer_label="Bookmarked")

        decided = snapshot_and_hash("x", atom_id, md, seen)
        if decided is None:                          # snapshot unchanged → skip (no re-embed)
            presence_hits.add(atom_id)
            with counts_lock:
                counts["skipped"] += 1
            return None
        raw_ref, raw_hash = decided

        atom = build_x_atom(norm, atom_id, raw_ref=raw_ref, raw_hash=raw_hash, meta=meta,
                            thread_tweets=thread_tweets)
        who_id = meta["who_id"]
        return {
            "atom": atom, "md": md, "meta": meta, "who_id": who_id,
            "raw_hash": raw_hash, "is_thread": bool(thread_tweets), "img_new": made,
        }

    def _consume(res: dict) -> None:
        """CONSUMER (caller's thread, SERIAL): the sole owner of the write path. Applies the author
        entity + curation signal (MOVED off the producer so nothing but this thread touches `conn`),
        then submits to the batching sink and owns the image-cache flush cadence."""
        nonlocal cache_pending, images_new, ledger_pending, lease_lost
        if should_stop and should_stop():
            lease_lost = True
            return
        meta = res["meta"]
        who_id = res["who_id"]
        site = meta.get("who_site")
        # Author entity + "save" signal, keyed on the SAME x:user:{rest_id} the follow/like/list
        # stampers use so a multi-signal person unifies before Stage-3 (the rest_id join invariant).
        # Was in the loop body; here it rides the single-writer thread with the atom itself.
        schema.upsert_entity(conn, who_id, name=meta.get("who_name"),
                             identity_links=[site] if site else None,
                             profile={"handle": meta["who_handle"]} if meta.get("who_handle") else None)
        # PRESENCE, not counting. `add_signal` until 2026-09-13, which summed 1 per atom into the
        # row the screen renders as "bookmarked 12x" — correct while this was the only writer, and
        # double-counting the moment `sync_bookmark_signals` began setting the true aggregate from
        # the full-set walk. That walk is the counting authority; this guarantees the signal exists
        # even when it never ran (a store whose only bookmark path is the rail).
        schema.ensure_signal(conn, who_id, "save", "x")
        # No `res["atom"] is None` branch any more: `_work` now returns an atom on every path it
        # returns at all (a metered upgrade that did not arrive downgrades the atom, it does not
        # withhold it), and a snapshot that did not change returns None from `_work` and never
        # reaches here. A defensive guard for a shape nothing produces would only hide a regression.
        aid = res["atom"]["atom_id"]
        # Bookkeeping rides on_written so seen/added/threads count DURABLE atoms, not submitted ones.
        sink.submit(res["atom"], res["md"],
                    on_written=(lambda a=aid, rh=res["raw_hash"], it=res["is_thread"]:
                                _mark_written(a, rh, it)))
        # Persist VLM descriptions off the hot path — a whole-dict JSON write only every
        # _CACHE_FLUSH_EVERY new descriptions (bounds a mid-run crash's repeated descriptions); the
        # close-time flush below is the backstop. Now the CONSUMER's job, so no producer iterates
        # the cache concurrently (cache writes go through the guarded cache_put).
        images_new += res["img_new"]
        cache_pending += res["img_new"]
        if cache_pending >= _CACHE_FLUSH_EVERY:
            save_image_cache(home, img_cache)
            cache_pending = 0
        # Checkpoint the resolved-conversation ledger on THIS (consumer / main) thread — the same
        # thread the end-of-run save runs on. `.copy()` is an atomic snapshot under the GIL (the same
        # guarantee the lock-free producer-side `.add()` leans on), so it never trips "set changed
        # size during iteration" against a concurrent producer add. Cheap: add-only bulk upsert.
        ledger_pending += 1
        if ledger_pending >= _LEDGER_FLUSH_EVERY:
            save_state(convo_ledger, convo_checked.copy())
            ledger_pending = 0

    def _in_window(bookmarks):
        """The walk, minus bookmarks whose TWEET predates `since`. Runs on the CALLING thread (the
        submission loop drains it serially), so the counter needs no lock.

        A tweet whose date won't parse is KEPT: an unreadable timestamp is not evidence the post
        is old, and dropping it would waste the user's window on a parser bug."""
        nonlocal out_of_window, lease_lost
        for norm in bookmarks:
            if should_stop and should_stop():
                lease_lost = True
                break
            if since is not None:
                created = _parse_twitter_date(norm.get("createdAt", ""))
                if created and created < since:
                    out_of_window += 1
                    continue
            yield norm

    # `limit` still bounds the WALK (it lives inside iterate_bookmarks), not the post-filter
    # survivors — so `limit` + `since` together mean "look at N bookmarks, ingest the ones in
    # window", never "keep walking until N survive". Nothing calls them together today; the
    # onboarding path passes limit=0.
    report = run_concurrent(_in_window(iterate_bookmarks(limit=limit)),
                            _work, _consume, workers=_INGEST_WORKERS)

    lease_lost = lease_lost or bool(should_stop and should_stop())
    if lease_lost:
        sink.discard()
    else:
        sink.close()                           # flush the final partial buffer (embed + write remainder)
    # Every bookmark the walk skipped as already-present, promoted here on the writer thread. The
    # atoms it WROTE need nothing: `upsert_atom` already overwrites `entry_mode` to 'user-saved'.
    if not lease_lost:
        for aid in presence_hits:
            promote_atom(conn, aid, "user-saved")
    save_image_cache(home, img_cache)          # backstop: persist any descriptions not yet flushed
    save_state(convo_ledger, convo_checked)   # persist the resolved-ledger for the next run

    # Per-call latency shape and the stage funnel + failure rates — measurement-only, doesn't
    # change ingest behavior.
    _vlm_now = _ocr.stats_snapshot()
    vlm = {k: _vlm_now[k] - vlm0[k] for k in ("calls", "failures", "seconds")}
    out = {
        "source": "x", "added": added, "skipped": counts["skipped"], "threads": threads,
        "stopped": "lease_lost" if lease_lost else None,
        # Which pass this was — the free one or the metered one. Two runs over the same corpus
        # report different `deferred` for legitimate reasons, and a reader needs to know which.
        "enrich": enrich,
        # Bookmarks whose ATOM IS WRITTEN but whose thread context is still owed: not asked for on
        # the free pass, or asked for and refused on a metered one. The honest size of the
        # ENRICHMENT backlog — never a coverage shortfall, since the atom is already in the store.
        # Top-level, not buried in `funnel`, because it is what the Enrichment loop terminates on.
        "deferred": counts["thread_deferred"] + counts["article_teaser"],
        # SEPARATE from `skipped`: that counter means "already ingested, unchanged" (a free
        # idempotency win), this one means "you chose not to pay for it". Collapsing them would
        # make a narrow window look like a well-deduped corpus.
        "out_of_window": out_of_window,
        "since": since.isoformat() if since else None,
        "total": schema.count_atoms(conn, "x"),
        "stage_seconds": timer.totals,
        "stage_latency": timer.distribution(),
        "funnel": {
            "thread": {
                "calls": convo.n_calls,
                "with_chain": convo.n_chains,
                "failed": convo.n_failed,
                "skipped_standalone": counts["thread_skipped_standalone"],
                # Bookmarks whose thread read did not answer. `n_failed` counts at most 1 here,
                # because the fetcher disables itself on the first refusal and every later call
                # short-circuits before the counter — so only THIS number grows with the shortfall.
                "deferred": counts["thread_deferred"],
                "stopped_early": not convo.enabled,
            },
            "vlm": {
                "images_new": images_new,
                "describe_calls": vlm["calls"],
                "describe_failures": vlm["failures"],
                "describe_seconds": round(vlm["seconds"], 2),
                "describe_mean_seconds": round(vlm["seconds"] / vlm["calls"], 3) if vlm["calls"] else 0.0,
                "describe_max_seconds": round(_vlm_now["max_seconds"], 2),
            },
        },
    }
    if report["source_error"] is not None:
        # The bookmarks walk died and the drain hid it. Nothing in this dict could carry that
        # before — `added: 0, skipped: 0, total: 0` is byte-identical to "you have not bookmarked
        # anything lately", and a rail that believes it succeeded does not retry, so an expired
        # cookie stopped bookmark imports silently.
        #
        # The SOURCE owns the taxonomy, which is why `run_concurrent` hands back the exception
        # rather than a verdict. `XRateLimited` is x.com's meter: come back later, and
        # `undetermined` alongside `error` is `classify_run`'s BLOCKED. Anything else — an expired
        # cookie above all — needs a person, which is `error` alone. `d7dbcfcf` rejected folding
        # the two together: "collapsing them trains the reader to ignore errors".
        from pipeline.ingestion import x_graphql_core as core   # lazy: the transport is heavy

        e = report["source_error"]
        out["error"] = f"{type(e).__name__}: {e}"
        if isinstance(e, core.XRateLimited):
            out["undetermined"] = 1        # the one source whose extent was not established
    return out
