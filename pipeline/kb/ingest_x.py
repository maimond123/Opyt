"""
pipeline/kb/ingest_x.py — X bookmarks → OPINION atoms (direct-to-atom, free source).

A bookmark is content David deliberately SAVED, so `entry_mode="user-saved"` (saved, not
authored) and `what_kind="opinion"`. This reuses the existing cookie-scrape iterator and
markdown renderer wholesale — the atom-KB adds the routing card + chunk embeddings + the
factual edge graph on top of the SAME snapshot the vault path already knew how to make.

Reuses (do NOT reimplement): `x_graphql.iterate_bookmarks` (+ its `_normalize`) for the
free local scrape; `x_render.tweet_to_markdown` for the snapshot text.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime

from . import derive, schema
from .embed import assert_model
from .ingest_common import (BASIS_OBSERVED, BODY_COMPLETE, BODY_PARTIAL, AtomSink, StageTimer,
                            body_fields, promote_atom, run_concurrent, snapshot_and_hash,
                            submit_atom)

# Producer-thread pool ceiling for the concurrent bookmark backfill. Real fan-out is capped lower
# by AdaptiveSemaphores at the transport seams (the VLM); this sits a touch above those so
# the gates, not the pool, are the binding constraint. Env-overridable; write path stays serial.
_INGEST_WORKERS = int(os.environ.get("OPYT_INGEST_WORKERS", "20"))
# How many new image descriptions may accrue before the consumer persists the cache — bounds the
# re-describe cost of a mid-run crash without a whole-dict JSON write per bookmark.
_CACHE_FLUSH_EVERY = 16
# Checkpoint the resolved-conversation ledger every N durable atoms, not only at end-of-run, so a
# crash mid-backfill re-fetches ≤N threads next run. Safe to checkpoint early: a ledger entry ahead
# of its atom just costs a re-fetch, never a wrong skip.
_LEDGER_FLUSH_EVERY = 64


class _ConvoFetcher:
    """Resolves a bookmark's conversation chain via the free cookie-scrape `TweetDetail`
    (150 per 15 minutes, so it degrades on a limit rather than failing).

    Carries a resolved-LEDGER (`checked`, a set of tweet ids) so re-runs skip already-resolved
    bookmarks — the one-time backfill never re-pays. A transient fetch failure leaves the id
    UNCHECKED (retried next run); a cookie rate-limit DISABLES the fetcher for the rest of the run."""

    def __init__(self, profile: str | None, checked: set):
        self.checked = checked
        self.enabled = True
        self.cookies = self.headers = None
        # Funnel counters (attempts/failures/non-empty chains) are lock-guarded because many
        # producer threads call chain() concurrently and `+=` is not atomic under the GIL.
        # `checked` (set) and `enabled` (bool) need no lock — their ops are already atomic.
        self._lock = threading.Lock()
        self.n_calls = self.n_failed = self.n_chains = 0
        # One backend. There used to be a `backend` switch here choosing twitterapi.io's
        # `walk_thread_context` when a key was present and TweetDetail when it was not; the paid
        # arm was deleted on 2026-08-30 and a switch with one arm is not a choice.
        from pipeline.ingestion.x_graphql import read_x_cookies
        from pipeline.ingestion import x_graphql_core as core
        self.cookies = read_x_cookies(profile=profile)
        self.headers = core.auth_headers(self.cookies, referer="https://x.com/home")

    def _bump(self, attr: str) -> None:
        with self._lock:
            setattr(self, attr, getattr(self, attr) + 1)

    def chain(self, tid: str) -> list[dict]:
        """The conversation chain for `tid` (`[ancestors, focal, self-continuation]`), or []."""
        from pipeline.ingestion import x_graphql_core as core
        from pipeline.ingestion.utils import log, SyncAuthError
        if not self.enabled:
            return []
        self._bump("n_calls")
        try:
            chain = core.fetch_conversation(tid, self.cookies, self.headers)
            self.checked.add(str(tid))      # resolved (thread or genuinely none) → don't re-fetch
            if chain:
                self._bump("n_chains")
            return chain
        except SyncAuthError:
            self.enabled = False
            self._bump("n_failed")
            log("[kb] x session rejected on conversation fetch — disabling thread context this run.")
        except RuntimeError as e:
            self._bump("n_failed")
            if "rate-limit" in str(e) or "429" in str(e):
                self.enabled = False
                log("[kb] conversation endpoint rate-limited — disabling thread context this run "
                    "(remaining bookmarks render solo; retried next run).")
            else:
                self.checked.add(str(tid))  # benign per-tweet failure → don't retry forever
                log(f"[kb] conversation fetch failed for {tid} (rendering solo): {e}")
        except Exception as e:
            self._bump("n_failed")
            self.checked.add(str(tid))
            log(f"[kb] conversation fetch failed for {tid} (rendering solo): {e}")
        return []


def build_x_atom(norm: dict, atom_id: str, *, raw_ref: str, raw_hash: str, meta: dict,
                 thread_tweets: list | None = None, thread_incomplete: bool = False,
                 entry_mode: str = "user-saved") -> dict:
    """A rendered tweet → the atom row. Extracted 2026-08-13 from `sync_bookmarks._work` so the
    walk-my-bookmarks path and the hand-dump-one-post path build the SAME shape rather than two
    that drift — the payload here is the substantiveness signal set, and a field missing from one
    producer would look like a real absence to every later reader.

    `entry_mode` is a parameter but both live callers pass "user-saved", and that is not an
    accident: a bookmark and a hand-dumped post are the same act. David saved it personally. See
    the module docstring and `schema.py`'s entry_mode note — the mode records HOW an atom was
    found, never who blessed it."""
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
            # A tweet's own text arrives whole with the bookmark payload — there is no second
            # fetch that could clip it — so the only way a bookmark atom is short of the whole
            # thing is a thread we could not walk. OBSERVED either way: we know whether the
            # chain fetch was attempted and whether it came back.
            **body_fields(BODY_PARTIAL if thread_incomplete else BODY_COMPLETE, BASIS_OBSERVED),
        },
        "entry_mode": entry_mode,
    }


def peek_tweet(tid: str) -> dict | None:
    """The tweet ITSELF, with NO conversation walk. `None` if it cannot be read.

    Exists so a caller can SHOW the user what a post is before paying to ingest it. That is not a
    nicety for X the way it would be for a blog: a host model can fetch a Verge article or an arXiv
    page and describe it, but x.com serves a JS shell to unauthenticated fetchers, so the model
    pasting a status link genuinely knows nothing but the url. And `x:2086520133909168332` — the
    only thing a free preview could show — is unverifiable by a human, unlike
    `blog:theverge.com/…/some-cool-article`.

    FREE, on this machine's own X session. It used to cost ~$0.00015/tweet through twitterapi.io,
    and it used to be the reason a preview needed a key at all: the only keyless read available was
    `TweetDetail`, which returns a CONVERSATION, and `reconstruct_chain` returns [] for a chain of
    one — so a solo post was invisible. `TweetResultsByRestIds` fetches the post itself, which is
    what closed that gap.

    Fail-safe: `None` on a fetch failure or a deleted / protected / suspended post — both of which
    the caller reports rather than guesses."""
    from pipeline.ingestion import x_graphql_core as core
    from pipeline.ingestion.utils import log

    try:
        cookies = core.read_x_cookies()
        headers = core.auth_headers(cookies, referer="https://x.com/home")
        got = core.fetch_tweets_by_ids(cookies, headers, [str(tid)]) or []
    except Exception as e:
        log(f"[hopper] tweet peek failed for {tid}: {e}")
        return None
    return next((t for t in got if str(t.get("id")) == str(tid)), None)


def _fetch_one_tweet(tid: str, *, profile: str | None = None) -> tuple[dict | None, list | None]:
    """ONE tweet by id → `(tweet, thread_chain)`, both in the shape the renderer consumes.
    `thread_chain` is None when there is no conversation worth rendering.

    Two reads on two independent rate buckets, and the ORDER matters. `TweetResultsByRestIds`
    (500/15min) fetches the post itself, so this works for a SOLO post — which the conversation
    read alone cannot do, since `reconstruct_chain` returns [] for a chain of one. `TweetDetail`
    (150/15min) then adds the ancestors and the self-continuation. This used to be two paid
    twitterapi.io calls (~$0.003-$0.033 per dump); both are free now.

    Fail-safe: a failure to read the POST returns `(None, None)` — no atom, nothing marked
    processed. A failure to read the CHAIN degrades to a solo render (chain None) rather than
    sinking the save, matching `_ConvoFetcher.chain`'s own "render solo" contract."""
    tweet = peek_tweet(tid)
    if not tweet:                           # deleted / protected / suspended, or the fetch failed
        return None, None
    # `_ConvoFetcher` owns the ledger and the failure semantics; a throwaway `checked` set is
    # correct here because one dump has nothing to amortize across.
    chain = _ConvoFetcher(profile, set()).chain(str(tid)) or None
    if chain:                               # splice OUR copy of the focal in — same as the walk
        chain = [tweet if str(t.get("id")) == str(tid) else t for t in chain]
    return tweet, chain


def x_atom_from_url(conn: sqlite3.Connection, embedder, url: str, *,
                    entry_mode: str = "user-saved", seen: dict | None = None,
                    img_cache: dict | None = None, profile: str | None = None,
                    sink=None, on_written=None) -> tuple[str, str | None]:
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
        promote_atom(conn, atom_id, entry_mode)
        return "present", atom_id

    home = opyt_home()
    own_cache = img_cache is None
    cache = load_image_cache(home) if own_cache else img_cache
    try:
        norm, chain = _fetch_one_tweet(tid, profile=profile)
        if not norm:
            return "failed", None

        # Read every image (no cost gate), matching the bookmark path — a chart in a post you
        # saved by hand is exactly the image worth paying to read.
        made = enrich_tweet_media(norm, cache, describe_all=True)
        md = tweet_to_markdown(norm, article=norm.get("article"), thread_tweets=chain,
                               source="x-bookmark", footer_label="Bookmarked")
        decided = snapshot_and_hash("x", atom_id, md, seen)
        if decided is None:                       # unchanged snapshot → nothing to re-embed
            promote_atom(conn, atom_id, entry_mode)
            return "present", atom_id
        raw_ref, raw_hash = decided

        meta = derive.derive_x(norm)
        # `thread_incomplete` stays False: unlike the bulk walk, a chain miss here already
        # degraded to a solo render inside `_fetch_one_tweet`, and we cannot distinguish "no
        # thread" from "we were stopped" without the walk's persistent ledger. Claiming PARTIAL on
        # every solo post would be the louder lie.
        atom = build_x_atom(norm, atom_id, raw_ref=raw_ref, raw_hash=raw_hash, meta=meta,
                            thread_tweets=chain, entry_mode=entry_mode)
        schema.upsert_entity(conn, meta["who_id"], name=meta.get("who_name"),
                             identity_links=[meta["who_site"]] if meta.get("who_site") else None,
                             profile={"handle": meta["who_handle"]} if meta.get("who_handle") else None)
        schema.add_signal(conn, meta["who_id"], "save", "x")
        submit_atom(conn, embedder, sink, atom=atom, snapshot_text=md, on_written=on_written)
        seen[atom_id] = raw_hash
        if made and own_cache:
            save_image_cache(home, cache)        # persist paid VLM descriptions for the next run
        return "saved", atom_id
    except Exception as e:                       # fail-safe: a bad post never raises out
        log(f"[hopper] x post ingest failed for {url}: {e}")
        return "failed", None


def sync_bookmarks(conn: sqlite3.Connection, embedder, *, limit: int = 0,
                   profile: str | None = None, fetch_threads: bool = True,
                   since: datetime | None = None) -> dict:
    """Ingest up to `limit` bookmarks (0 = all) as opinion atoms. Idempotent: unchanged
    bookmarks (same snapshot hash) are skipped, no re-embed. When `fetch_threads`, each bookmark's
    conversation is resolved (free cookie-scrape `TweetDetail`) so replies carry their
    debate and self-thread roots carry their continuation. Returns a run summary.

    `since` is a SPEND filter on when the tweet was WRITTEN, not when you saved it (X exposes no
    bookmark timestamp). It's applied to the ITERATOR, upstream of `_work`, so a skipped bookmark
    costs no thread fetch, VLM read, or embed. It SKIPS rather than STOPS: the walk is ordered by
    save time while the cutoff is on write time, so an old tweet saved recently sits near the top —
    breaking there would silently truncate everything saved after it.
"""
    from pipeline.ingestion.x_graphql import iterate_bookmarks
    from pipeline.ingestion.x_render import tweet_to_markdown, _parse_twitter_date
    from pipeline.ingestion.utils import load_state, save_state
    # Per-image latency lives on the seam that makes the calls (ocr_cascade), not describe_images.
    from pipeline import ocr_cascade as _ocr
    from pipeline.image_cache import load_image_cache, save_image_cache
    from opyt_core.paths import opyt_home
    from .vision import enrich_tweet_media

    # Guard the store's embedding identity BEFORE paying to embed — a model mismatch on an
    # existing store raises here (dim is verified per-atom once discovered), not after spend.
    assert_model(conn, embedder)
    seen = schema.load_hashes(conn, "x")
    home = opyt_home()
    # Image descriptions cached by URL (immutable CDN links) → re-runs are free + hash-stable.
    img_cache = load_image_cache(home)
    # Conversation resolved-ledger: tweet ids whose thread was already fetched, so the one-time
    # backfill of all bookmarks doesn't re-pay on later syncs. The on-disk ledger is THREAD-AFFINE
    # (main thread only); snapshot to a plain set for the producer pool and bulk-persist at the end.
    convo_ledger = home / "x_convo_checked.json"
    convo_checked = set(load_state(convo_ledger)) if fetch_threads else set()
    convo = _ConvoFetcher(profile, convo_checked) if fetch_threads else None
    added = threads = images_new = out_of_window = 0
    # `added`/`threads`/`images_new` are mutated ONLY on the consumer (single-threaded) → no lock.
    # The two SKIP tallies fire from many producer threads (fast-skip + hash-unchanged), so they sit
    # behind a lock — a lost `+=` would misreport how much of the corpus was one-time-skipped.
    counts = {"skipped": 0, "thread_skipped_standalone": 0}
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

    def _mark_written(atom_id: str, raw_hash: str, is_thread: bool) -> None:
        """Fires only AFTER an atom is durably written (on the consumer thread), so the summary counts
        DURABLE atoms — truthful on a crash (an atom lost in the pre-write buffer is never marked)."""
        nonlocal added, threads
        seen[atom_id] = raw_hash
        added += 1
        if is_thread:
            threads += 1

    def _work(norm: dict):
        """PRODUCER (pool thread): the network-bound per-bookmark work — enrich images, fetch
        the thread, render, hash. Touches ONLY per-bookmark locals + thread-safe shared state (the
        AIMD-gated transport seams, the GIL-atomic `convo_checked`/`img_cache`, the locked timer).
        No `conn` writes here. Returns a write-ready result dict, or None to skip."""
        tid = norm.get("id")
        if not tid:
            return None
        atom_id = f"x:{tid}"
        # Fully resolved (already ingested AND conversation-checked) → skip BEFORE any fetch.
        # This is what makes the big first backfill one-time.
        if fetch_threads and atom_id in seen and str(tid) in convo_checked:
            presence_hits.add(atom_id)
            with counts_lock:
                counts["skipped"] += 1
            return None

        # BOOKMARKS read ALL images (no cost-gate) — fan-out is AIMD-capped inside the OCR cascade.
        with timer.stage("vlm"):
            made = enrich_tweet_media(norm, img_cache, describe_all=True)

        # Resolve the conversation. A provably-standalone
        # tweet (no replies AND not a reply) skips the fetch but still marks the ledger, so next run's
        # fast-skip fires instead of re-rendering it forever. Fail-safe: no chain → render solo.
        thread_tweets = None
        # A tweet whose chain fetch failed is stored as a SOLO render but flagged incomplete:
        # `chain()` marks `checked` on success (found or provably none), not on transient failure,
        # so "empty AND unchecked" means "we were stopped".
        thread_incomplete = False
        if convo is not None:
            should_fetch = bool(norm.get("isReply")) or (norm.get("replyCount") or 0) > 0
            if should_fetch:
                with timer.stage("thread_fetch"):
                    chain = convo.chain(tid)
                if chain:
                    thread_tweets = [norm if str(t.get("id")) == str(tid) else t for t in chain]
                elif str(tid) not in convo.checked:
                    thread_incomplete = True
            else:
                convo_checked.add(str(tid))         # set.add is atomic → safe across producer threads
                with counts_lock:
                    counts["thread_skipped_standalone"] += 1

        # Pass the article node so the full body — not the teaser — is chunked; thread_tweets renders
        # the whole debate chain (renderer precedence is article > thread).
        with timer.stage("render"):
            md = tweet_to_markdown(norm, article=norm.get("article"), thread_tweets=thread_tweets,
                                   source="x-bookmark", footer_label="Bookmarked")

        decided = snapshot_and_hash("x", atom_id, md, seen)
        if decided is None:                          # snapshot unchanged → skip (no re-embed)
            presence_hits.add(atom_id)
            with counts_lock:
                counts["skipped"] += 1
            return None
        raw_ref, raw_hash = decided

        meta = derive.derive_x(norm)
        atom = build_x_atom(norm, atom_id, raw_ref=raw_ref, raw_hash=raw_hash, meta=meta,
                            thread_tweets=thread_tweets, thread_incomplete=thread_incomplete)
        who_id = meta["who_id"]
        return {
            "atom": atom, "md": md, "meta": meta, "who_id": who_id,
            "raw_hash": raw_hash, "is_thread": bool(thread_tweets), "img_new": made,
        }

    def _consume(res: dict) -> None:
        """CONSUMER (caller's thread, SERIAL): the sole owner of the write path. Applies the author
        entity + curation signal (MOVED off the producer so nothing but this thread touches `conn`),
        then submits to the batching sink and owns the image-cache flush cadence."""
        nonlocal cache_pending, images_new, ledger_pending
        meta = res["meta"]
        who_id = res["who_id"]
        site = meta.get("who_site")
        # Author entity + "save" signal, keyed on the SAME x:user:{rest_id} the follow/like/list
        # stampers use so a multi-signal person unifies before Stage-3 (the rest_id join invariant).
        # Was in the loop body; here it rides the single-writer thread with the atom itself.
        schema.upsert_entity(conn, who_id, name=meta.get("who_name"),
                             identity_links=[site] if site else None,
                             profile={"handle": meta["who_handle"]} if meta.get("who_handle") else None)
        schema.add_signal(conn, who_id, "save", "x")
        aid = res["atom"]["atom_id"]
        # Bookkeeping rides on_written so seen/added/threads count DURABLE atoms, not merely submitted.
        sink.submit(res["atom"], res["md"],
                    on_written=(lambda a=aid, rh=res["raw_hash"], it=res["is_thread"]:
                                _mark_written(a, rh, it)))
        # Persist paid VLM descriptions off the hot path — a whole-dict JSON write only every
        # _CACHE_FLUSH_EVERY new descriptions (bounds a mid-run crash's re-describe cost); the
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
        if fetch_threads:
            ledger_pending += 1
            if ledger_pending >= _LEDGER_FLUSH_EVERY:
                save_state(convo_ledger, convo_checked.copy())
                ledger_pending = 0

    def _in_window(bookmarks):
        """The walk, minus bookmarks whose TWEET predates `since`. Runs on the CALLING thread (the
        submission loop drains it serially), so the counter needs no lock.

        A tweet whose date won't parse is KEPT: an unreadable timestamp is not evidence the post
        is old, and dropping it would spend the user's window on a parser bug."""
        nonlocal out_of_window
        for norm in bookmarks:
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
    run_concurrent(_in_window(iterate_bookmarks(limit=limit, profile=profile)),
                   _work, _consume, workers=_INGEST_WORKERS)

    sink.close()                               # flush the final partial buffer (embed + write remainder)
    # Every bookmark the walk skipped as already-present, promoted here on the writer thread. The
    # atoms it WROTE need nothing: `upsert_atom` already overwrites `entry_mode` to 'user-saved'.
    for aid in presence_hits:
        promote_atom(conn, aid, "user-saved")
    save_image_cache(home, img_cache)          # backstop: persist any descriptions not yet flushed
    if fetch_threads:
        save_state(convo_ledger, convo_checked)   # persist the resolved-ledger for the next run

    # Per-call latency shape, stage funnel + failure rates, and provider rate-limit ceilings —
    # measurement-only, doesn't change ingest behavior.
    from pipeline import llm_client as _llm
    from .embed import _LAST_RATE_LIMIT as _embed_rl
    _vlm_now = _ocr.stats_snapshot()
    vlm = {k: _vlm_now[k] - vlm0[k] for k in ("calls", "failures", "seconds")}
    return {
        "source": "x", "added": added, "skipped": counts["skipped"], "threads": threads,
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
                "calls": (convo.n_calls if convo else 0),
                "with_chain": (convo.n_chains if convo else 0),
                "failed": (convo.n_failed if convo else 0),
                "skipped_standalone": counts["thread_skipped_standalone"],
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
        "rate_limits": {
            "embed": dict(_embed_rl),
            "vision_llm": dict(_llm._LAST_RATE_LIMIT),
        },
    }
