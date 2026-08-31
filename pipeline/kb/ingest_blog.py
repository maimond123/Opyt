"""
pipeline/kb/ingest_blog.py — a confirmed Oracle's OWN blog archive → OPINION atoms.

Stage-5 footprint ingest (after Substack; YouTube's footprint lives in Radar instead). Fetch is
ported from `pipeline.ingestion.ingest_blog` (`_fetch_sitemap_urls` + trafilatura
`_fetch_article`) rather than copied. who_id = `blog:{canonical_host}` (`derive.blog_entity_id`);
atom_id = `_canon_post_url` = `blog:{host}{path}[?query]`, path-preserving so posts don't collapse
to one id the way `canonical_identity` would.

Dedup/fail-safe mirrors `pipeline.kb.ingest_substack` (policy B: skip if the atom already exists
before the paid fetch+embed; a challenge/bot-check or thin-body fetch is skipped and counted,
never stored; a no-content fetch or embed/write failure skips with no `seen` mark). See
`docs/plans/2026-07-18-stage5-footprint-adapters-plan.md`.

Attribution to the Oracle's canonical happens via a later `resolve_entities` run, which treats a
`blog:` link as `self` and unifies it with the Oracle's X-website→blog-home attested edge.

"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse

from . import content_gate, derive, schema
from .embed import assert_model
from .ingest_common import (AtomSink, BASIS_ASSUMED, BODY_COMPLETE, FETCH_OK,
                            FETCH_UNDETERMINED, PENDING_CLAIM, POST_INFLIGHT,
                            POST_WORKERS, StageTimer, body_fields, classify_fetch,
                            llm_run_marker, llm_run_stats, make_consumer,
                            promote_atom, run_concurrent, snapshot_and_hash, submit_atom)

# Tracking/analytics query params, not post identity. Stripped in `_canon_post_url` so an
# RSS-sourced URL (which bolts on `?utm_*`) and a clean hub-harvested URL for the same post mint
# the same atom_id; a real identity param (WordPress `?p=123`) is kept.
_TRACKING_PARAMS = frozenset({
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "yclid", "twclid",
    "mc_cid", "mc_eid", "igshid", "_hsenc", "_hsmi",
})


def _strip_tracking_query(query: str) -> str:
    """Drop tracking params (`utm_` prefix + `_TRACKING_PARAMS`), keep the rest sorted so param
    order never defeats dedup. Returns the cleaned query without a leading `?`, or `""`."""
    kept = [(k, v) for (k, v) in parse_qsl(query, keep_blank_values=True)
            if not (k.lower().startswith("utm_") or k.lower() in _TRACKING_PARAMS)]
    return urlencode(sorted(kept))

# Challenge/thin-body thresholds + markers live in `ingest_common` so blog and Substack classify
# a fetch the same way; see `classify_fetch` for the three-verdict contract.


def _canon_post_url(url: str) -> str:
    """Per-post atom id: `blog:{host}{path}[?query]`, path-preserving (unlike `canonical_identity`,
    which collapses every post to the bare host). Lowercase host, strip `www.`, drop scheme +
    fragment, strip a trailing path slash. Query is kept for real post identity (`?p=123`) but
    tracking params are stripped via `_strip_tracking_query` so a hub-harvested link and an
    RSS-sourced link for the same post mint the same atom_id."""
    u = (url or "").strip()
    if not u:
        return "blog:"
    if "://" not in u:
        u = "https://" + u
    p = urlparse(u)
    host = (p.netloc or "").lower().split("@")[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    path = p.path.rstrip("/")
    clean_q = _strip_tracking_query(p.query)
    query = f"?{clean_q}" if clean_q else ""
    return f"blog:{host}{path}{query}"


def _classify_article(article: dict | None) -> str:
    """A fetched page → FETCH_OK / FETCH_ABSENT / FETCH_UNDETERMINED. Both non-OK verdicts skip
    identically (no atom, no `seen` mark); split so a blocked host reads apart from a thin stub.
    Checks the response headers' `cf-mitigated` marker before falling back to a body heuristic.
"""
    if not article:
        return FETCH_UNDETERMINED   # the fetch itself failed — we learned nothing about the page
    return classify_fetch(article.get("content"), headers=article.get("headers"),
                          title=article.get("title", "") or "")


def _lastmod_before(lastmod: str, since: datetime) -> bool:
    """True iff the sitemap `lastmod` parses to a moment strictly BEFORE `since` (→ skip it).
    Unparseable lastmod → False (don't drop a post we can't date; `since` is a best-effort listing
    bound, mirroring the Substack archive walk, not a hard per-post filter)."""
    try:
        dt = datetime.fromisoformat((lastmod or "").replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt < since


# ── Date cascade ─────────────────────────────────────────────────────────────
# `_resolve_post_date` ladders free (no-LLM) date sources by decreasing exactness;
# `when_precision` travels with the date so a coarse source is never read as a known day.
# See docs/blog-website-signal-extraction.md ("Date extraction").

def _feed_date_map(blog_url: str) -> dict[str, str]:
    """Map ``_canon_post_url(link) -> "YYYY-MM-DD"`` from the site's RSS/Atom feed, when it has one.
    Fetched once per sync and cross-referenced for real publish dates, since sitemap `lastmod` is
    only a modification time. Keyed by ``_canon_post_url`` so link variants collapse. Fail-safe:
    ``{}`` on any miss — the cascade falls through to the next rung."""
    base = (blog_url or "").rstrip("/")
    if not base:
        return {}
    try:
        import feedparser
    except Exception:
        return {}
    import time as _time
    for feed_path in ("/feed.xml", "/feed", "/atom.xml", "/rss.xml", "/rss", "/index.xml"):
        try:
            feed = feedparser.parse(f"{base}{feed_path}")
        except Exception:
            continue
        out: dict[str, str] = {}
        for entry in getattr(feed, "entries", None) or []:
            link = (entry.get("link") or "").strip()
            parsed = entry.get("published_parsed") or entry.get("updated_parsed")
            if link and parsed:
                out[_canon_post_url(link)] = _time.strftime("%Y-%m-%d", parsed)
        if out:                       # first feed that yields DATED entries wins
            return out
    return {}


def _resolve_post_date(htmldate_date: str, url: str, lastmod: str,
                       feed_dates: dict[str, str]) -> tuple[str, str]:
    """Date cascade → ``(YYYY-MM-DD, when_precision)``. Cheapest+most-exact first; every rung free.

        1. htmldate (on-page / URL-mined)  → ``day``      already computed by ``_fetch_article``
        2. feed ``<pubDate>``              → ``day``      genuine publish date, feed-having sites
        3. sitemap ``lastmod``             → ``day``      modified-date; pre-existing behavior
        4. (nothing)                       → ``unknown``  never a fake day

    Feed OUTRANKS lastmod: a feed pubDate is a real publish date; lastmod is a modification time.
    Every rung is network-only, no LLM. A Wayback first-snapshot rung is deliberately NOT wired
    here"""
    d = (htmldate_date or "")[:10]
    if d:
        return d, "day"
    fd = feed_dates.get(_canon_post_url(url), "")
    if fd:
        return fd[:10], "day"
    if lastmod:
        return lastmod[:10], "day"
    return "", "unknown"


# ── The paid per-article stages, shared by the crawl and by a single hand-dumped url ──────────
# Shared by Hopper's single-article path and the crawl so both get identical treatment. Two
# ordering constraints, load-bearing for every caller: (1) content gate runs BEFORE the VLM pass —
# a rejected page must not bill a VLM fan-out; (2) VLM pass runs BEFORE `snapshot_and_hash` — a
# description added after the hash never reaches the indexed surface.

def _stage(timer, name):
    """`timer.stage(name)`, or a no-op when the caller keeps no timer. A single dumped article has
    no run to profile; the crawl does."""
    from contextlib import nullcontext
    return timer.stage(name) if timer is not None else nullcontext()


def build_article_atom(article: dict, *, url: str, atom_id: str, blog_url: str, author: str,
                       author_name: str | None = None, handle: str | None = None,
                       lastmod: str = "", feed_dates: dict[str, str] | None = None,
                       recover_feed_date: bool = False, img_cache: dict | None = None,
                       seen: dict[str, str] | None = None,
                       entry_mode: str = "oracle-footprint", timer=None,
                       log_tag: str = "footprint") -> dict:
    """A FETCHED article → a write-ready atom, WITHOUT touching `conn`. Returns an outcome dict:

      • `{"outcome": "ok", atom, md_kept, atom_id, raw_hash}` — hand it to a sink / store.
      • `{"outcome": "gate_rejected"}` — every unit was non-knowledge. No atom, by design.
      • `{"outcome": "unchanged"}`     — the snapshot hash matches `seen`. Skip; do not re-embed.

    Never returns None: the crawl runs this on a pool thread and `run_concurrent` DROPS a None
    without calling the consumer, which would lose the outcome and the post with it.

    `recover_feed_date` (on for a hand-dump, off for the crawl) fetches the origin's RSS feed here
    to recover the feed-pubDate rung when htmldate found nothing and no feed map was passed in.
    Fail-safe: no feed / 404 / timeout falls through to `unknown`, never a fake day.
    """
    from pipeline.ingestion.sources.blog import _article_to_markdown
    from pipeline.ingestion.utils import log

    from .vision import enrich_markdown_images

    seen = {} if seen is None else seen
    feed_dates = dict(feed_dates or {})

    # Date cascade (all free, no LLM): htmldate → feed pubDate → sitemap lastmod → unknown.
    # See `_resolve_post_date` (Wayback rung is reserved, off-path).
    with _stage(timer, "date_resolve"):
        htmldate = article.get("date") or ""
        if not htmldate and recover_feed_date and not feed_dates:
            feed_dates = _feed_date_map(blog_url)      # rung-2 recovery, only once rung 1 missed
        date, precision = _resolve_post_date(htmldate, url, lastmod, feed_dates)
    article["date"] = date
    article["when_precision"] = precision

    md = _article_to_markdown(article, author=author, author_name=author_name or author)

    # Content-quality gate (Stage-6): drop nav/promo/boilerplate units BEFORE embedding. None →
    # every unit was non-knowledge → skip, no atom. Degrades to keep-all if the gate model is
    # unavailable, so an outage can only add junk, never delete an author's writing. Runs before
    # the VLM pass so a rejected page doesn't bill a VLM fan-out; see reapply_keep below for how
    # the keep-mask still reaches the VLM-enriched text.
    with _stage(timer, "gate"):
        verdict = content_gate.classify_page(md)
    if verdict.kept_text is None:
        log(f"[{log_tag}] blog atom rejected by content gate (no substantive units): {url}")
        return {"outcome": "gate_rejected"}

    # VLM-describe inline images so charts/diagrams become searchable text. MUST run BEFORE
    # snapshot_and_hash: the description is injected into the body, so it is part of the hashed
    # + chunked surface — after the hash it would never reach the index. `base_url=url` resolves
    # a self-hosted blog's relative `/assets/x.jpeg` refs, which is most of them.
    with _stage(timer, "vlm"):      # per-POST (one post fans out to several describe calls)
        md, _ = enrich_markdown_images(md, img_cache if img_cache is not None else {},
                                       context=article.get("title") or "", base_url=url)
    # Reads `seen` from a pool thread: a dict `.get` is atomic under the GIL, and on the crawl the
    # value found is always the generator's `PENDING_CLAIM`, never a real hash, so "unchanged"
    # stays unreachable there. On a hand-dump `seen` is the real ledger and gives idempotency.
    decided = snapshot_and_hash("blog", atom_id, md, seen)
    if decided is None:                 # unchanged — honor the seam
        return {"outcome": "unchanged"}
    raw_ref, raw_hash = decided

    # Replay the pre-enrichment mask onto the enriched body so descriptions land in the CHUNKS
    # (the snapshot above is still the full page, re-derivable via rechunk-from-raw). Fail-safe:
    # if enrichment changed the unit count, re-grade the enriched body rather than emit mis-sliced
    # text that could silently drop real writing and keep an ad.
    md_kept = content_gate.reapply_keep(md, verdict.keep)
    if md_kept is None:
        log(f"[{log_tag}] keep-mask lost alignment after VLM enrichment, re-grading: {url}")
        with _stage(timer, "gate"):
            md_kept = content_gate.gate(md)
        if md_kept is None:
            return {"outcome": "gate_rejected"}

    meta = derive.derive_blog(article, blog_url=blog_url, handle=handle, author_name=author_name)
    atom = {
        "atom_id": atom_id,
        "source_type": "blog",
        "what_kind": "opinion",
        "who_id": meta["who_id"],
        "when_ts": meta["when_ts"],
        "when_precision": meta["when_precision"],
        "about_entities": meta["about_entities"],
        "source_url": url,
        "raw_ref": raw_ref,
        "raw_hash": raw_hash,
        "description": meta["description"],
        # BASIS_ASSUMED: a truncated "read more" preview returns 200 with real prose and no
        # marker, indistinguishable from a genuinely short post, so `complete` here means "nothing
        # indicated otherwise" — weaker than Substack's STATED or X's OBSERVED. Don't collapse the three.
        "payload": {"lastmod": lastmod, **body_fields(BODY_COMPLETE, BASIS_ASSUMED)},
        "entry_mode": entry_mode,                 # crawl: oracle-footprint; hand-dump: user-saved
    }
    # who_id stays the origin author for every atom, including a hub-harvested one.
    return {"outcome": "ok", "atom_id": atom_id, "raw_hash": raw_hash,
            "atom": atom, "md_kept": md_kept}


def article_atom_from_url(conn: sqlite3.Connection, embedder, url: str, *,
                          entry_mode: str = "user-saved", seen: dict | None = None,
                          img_cache: dict | None = None, sink=None,
                          on_written=None) -> tuple[str, str | None]:
    """Fetch ONE article by URL → a blog atom. The hand-dump twin of `sync_blog_footprint`, and the
    adapter Hopper's bare-link case routes to. Returns `(status, atom_id)` where status is
    "present" | "saved" | "rejected" | "blocked" | "failed"; `atom_id` is real for the first two.

    `entry_mode="oracle-footprint"` is refused: this function attributes by URL host with no
    author verification, so stamping it as a tracked person's writing would be trust-laundering
    (an Oracle's employer blog becoming that Oracle's own writing). A dump from a host OPYT
    already tracks attaches to that entity (`who_id = blog:{host}`); a dump from an unknown host
    creates a bare entity with no `kind`, `name`, or `identity_links` so it can never merge into a
    tracked person by accident — recoverable later if that blog is properly crawled.

    Fail-safe throughout: a failed fetch, a challenge page, or a thin stub SKIPS — no atom, nothing
    marked processed, retried whenever the user tries again. Never raises.
"""
    from pipeline.ingestion.utils import log

    if entry_mode == "oracle-footprint":
        raise ValueError("article_atom_from_url may not stamp 'oracle-footprint' — see its "
                         "docstring; a URL-host attribution is not an authorship claim.")

    from pipeline.ingestion.sources.blog import _fetch_article

    u = (url or "").strip()
    if not u:
        return "failed", None
    atom_id = _canon_post_url(u)
    if seen is None:
        seen = schema.load_hashes(conn, "blog")
    if atom_id in seen:            # policy B: present → skip BEFORE the (paid) fetch + gate + embed
        promote_atom(conn, atom_id, entry_mode)     # human-initiated presence hit → attestation
        return "present", atom_id

    try:
        article = _fetch_article(u)
        verdict = _classify_article(article)
        if verdict != FETCH_OK:
            # Never store a bot-check shell or a thin stub. Reported apart so "the host blocked
            # us" reads differently from "this page has no article on it".
            log(f"[hopper] article not stored ({verdict}): {u}")
            return ("blocked" if verdict == FETCH_UNDETERMINED else "failed"), None

        origin = f"{urlparse(u).scheme}://{urlparse(u).netloc}"
        # trafilatura's byline when found, else the host. Per-atom only, never written onto the
        # entity — see the docstring.
        byline = (article.get("author") or "").strip()
        res = build_article_atom(
            article, url=u, atom_id=atom_id, blog_url=origin,
            author=byline or origin, author_name=byline or None,
            feed_dates=None, recover_feed_date=True, img_cache=img_cache, seen=seen,
            entry_mode=entry_mode, log_tag="hopper")
    except Exception as e:                                   # fail-safe: a bad page never raises out
        log(f"[hopper] article ingest failed for {u}: {e}")
        return "failed", None

    if res["outcome"] == "gate_rejected":
        return "rejected", None
    if res["outcome"] == "unchanged":
        return "present", atom_id

    # `upsert_entity` COALESCEs every field, so an existing tracked blog keeps its name/kind/
    # identity-links untouched and the atom simply joins it.
    schema.upsert_entity(conn, res["atom"]["who_id"])
    submit_atom(conn, embedder, sink, atom=res["atom"], snapshot_text=res["md_kept"],
                on_written=on_written)
    seen[atom_id] = res["raw_hash"]
    return "saved", atom_id


def sync_blog_footprint(conn: sqlite3.Connection, embedder, *, blog_url: str,
                        handle: str | None = None, author_name: str | None = None,
                        since: datetime | None = None, limit: int = 0) -> dict:
    """Ingest a confirmed Oracle's OWN blog archive as opinion atoms (footprint).

    `blog_url` is the blog home; `handle`/`author_name` label the author. `since` bounds the
    sitemap walk to posts whose `lastmod` is on/after it (best-effort). `limit` caps NEW posts
    DISPATCHED per run (0 = all, and idempotent under policy B); a nonzero limit makes a re-run
    ADVANCE to the next batch, a resumable partial backfill. `limit` bounds posts processed, not
    atoms added — a gate-rejected post still spends one of the N. Returns a run summary. Does NOT
    run entity resolution; the caller re-resolves after footprint expansion.

    Fetch (sitemap discovery + trafilatura extraction) is imported from the note ingester. Posts
    are processed CONCURRENTLY (fetch stays serial) — see the run_concurrent call below.
    """
    from pipeline.ingestion.sources.blog import _fetch_article
    from pipeline.ingestion.url_canon import canonical_identity
    from pipeline.ingestion.utils import log
    from pipeline.image_cache import load_image_cache, save_image_cache
    from opyt_core.paths import opyt_home

    from . import link_discovery   # lazy: link_discovery imports _canon_post_url from THIS module

    assert_model(conn, embedder)      # guard the store's embedding identity BEFORE any spend
    # Bookmark LLM latency samples before the first paid call so the summary reports THIS run's
    # calls, not the process-cumulative total (long-lived MCP server). Same precedent as
    # `ingest_x`'s ocr_cascade snapshot/diff.
    llm0 = llm_run_marker()
    # VLM descriptions cached by absolute URL: re-runs are free and the snapshot hash stays
    # stable (an un-cached description would vary run-to-run and force spurious re-embeds).
    img_cache = load_image_cache(opyt_home())
    base = (blog_url or "").rstrip("/")
    if not base:
        return {"source": "blog", "error": "no blog_url"}

    who_id = derive.blog_entity_id(base)
    # Seed the entity with its blog home so a later resolve_entities merges it into the Oracle's
    # canonical via the attested-links graph.
    schema.upsert_entity(conn, who_id, name=author_name, identity_links=[base])

    # UNION discovery: sitemap/rss baseline + hub-harvested extras (homepage/index pages,
    # classified/triaged). Each entry carries `via` (the hub page it was found on, None for
    # baseline) and `source` (sitemap|strong|triage). Fail-safe: hub/triage failure degrades to
    # the baseline. `timer` instruments the serial fetch path to size a future producer pool.
    timer = StageTimer()
    # `seen` is loaded BEFORE discovery because discovery consumes it (a gray candidate already
    # in the store is dropped before the paid LLM triage). `known_urls` excludes body-pending
    # atoms — a post stored without a body because its fetch was BLOCKED must stay a discovery
    # candidate, or the block freezes into a permanent hole.
    seen = schema.load_hashes(conn, "blog")
    known_urls = set(seen) - schema.load_body_pending(conn, "blog")
    with timer.stage("discovery"):       # sitemap ∪ hub-harvest, once per sync
        entries = link_discovery.discover_candidate_urls(base, handle=handle,
                                                         author_name=author_name,
                                                         known_urls=known_urls)
    with timer.stage("feed_dates"):      # once per sync: real pubDates to cross-reference onto URLs
        feed_dates = _feed_date_map(base)

    # Batch the embed across posts: blog posts are LONG, so pooling their chunks per flush is
    # where the 8-way embed gate earns its keep. Posts are processed CONCURRENTLY, fetch stays
    # serial (`run_concurrent` drains the generator on the calling thread), so host request rate
    # is unchanged even though gate/VLM/render run on a pool:
    #   generator (_jobs)     discover + dedup + fetch    SERIAL   calling thread
    #      -> _work           gate + VLM + render         PARALLEL pool threads
    #      -> _consume        embed + DB write            SERIAL   calling thread (single writer)
    bs = int(getattr(embedder, "batch_size", 64) or 64)
    sink = AtomSink(conn, embedder, timer=timer, flush_chunks=8 * bs)
    counts = {"added": 0}
    # Counter ownership is by THREAD, not lock: everything below is touched only on the calling
    # thread (generator + consumer), since `+=` is not atomic under the GIL across threads. That
    # is why `_work` REPORTS an outcome instead of counting one itself.
    dispatched = challenge_skipped = undetermined = 0
    # consumed/submitted/skipped/gate_rejected: a shared dict, not four more `nonlocal` ints — see
    # `make_consumer`'s docstring for why (its `_consume` closure is defined outside this scope).
    counters = {"consumed": 0, "submitted": 0, "skipped": 0, "gate_rejected": 0}
    author = ((f"@{handle.lstrip('@')}" if handle else None)
              or author_name or canonical_identity(base) or "blog")

    def _mark() -> None:                     # post-commit durable-write count (never on a poison-skip)
        counts["added"] += 1

    def _jobs():
        """Discover → dedup → fetch, one post at a time on the CALLING thread.

        `seen` is confined to this generator plus the consumer (same thread), which makes the
        check-then-act safe: pool threads never write to it. A post claimed here and then FAILING
        is not retried within this run — nothing durable was written, so the next run picks it up.
"""
        nonlocal dispatched, challenge_skipped, undetermined
        for entry in entries:
            # `limit` caps DISPATCH, not atoms: the consumer lags the generator by up to `inflight`
            # items, so capping on the consumer-side count would over-fetch paid work before noticing.
            if limit and dispatched >= limit:
                break
            url = (entry.get("url") or "").strip()
            if not url:
                continue
            lastmod = (entry.get("lastmod") or "").strip()
            if since and lastmod and _lastmod_before(lastmod, since):   # best-effort recency bound
                continue

            atom_id = _canon_post_url(url)
            if atom_id in seen:             # policy B: skip BEFORE the (paid) fetch
                counters["skipped"] += 1
                continue
            seen[atom_id] = PENDING_CLAIM        # CLAIM before the fetch — see the docstring

            with timer.stage("article_fetch"):   # serial, as before: one fetch+extract per new post
                article = _fetch_article(url)
            verdict = _classify_article(article)
            if verdict != FETCH_OK:         # thin stub OR Cloudflare shell → never store (either way)
                # `undetermined` means "the host stopped us"; separated from `challenge_skipped` so
                # a blocked run reads differently from a thin archive.
                challenge_skipped += 1      # total not-stored
                if verdict == FETCH_UNDETERMINED:
                    undetermined += 1
                    log(f"[footprint] blog BLOCKED (challenge / fetch failed): {url}")
                else:
                    log(f"[footprint] blog skipped (thin body): {url}")  # never a silent drop
                continue

            dispatched += 1
            yield {"entry": entry, "url": url, "lastmod": lastmod,
                   "atom_id": atom_id, "article": article}

    def _work(job: dict) -> dict:
        """The PAID stages, on a pool thread. Returns an OUTCOME for the consumer to tally rather
        than counting anything itself. Never returns None (see counter-ownership note above).

        Stages live in `build_article_atom` (module level) so a hand-dumped url gets identical
        treatment; only per-run state (`feed_dates`, `img_cache`, `seen`, `timer`) is a closure
        here. `recover_feed_date` stays OFF: the feed map was already built once for the blog."""
        return build_article_atom(
            job["article"], url=job["url"], atom_id=job["atom_id"], blog_url=base,
            author=author, author_name=author_name, handle=handle,
            lastmod=job["lastmod"], feed_dates=feed_dates, recover_feed_date=False,
            img_cache=img_cache, seen=seen, entry_mode="oracle-footprint", timer=timer)

    # Byte-identical to `ingest_substack`'s consumer — see `ingest_common.make_consumer`.
    _consume = make_consumer(sink, seen, counters, _mark)

    run_concurrent(_jobs(), _work, _consume, workers=POST_WORKERS, inflight=POST_INFLIGHT)
    sink.close()
    save_image_cache(opyt_home(), img_cache)   # persist new VLM descriptions for the next run
    # A poison-chunk atom the sink isolates fires no _mark → not counted, not marked durable,
    # retried next run. `failed` = submitted-but-never-durable.
    return {"source": "blog", "added": counts["added"], "skipped": counters["skipped"],
            "challenge_skipped": challenge_skipped, "undetermined": undetermined,
            "failed": counters["submitted"] - counts["added"],
            # `dispatched` = handed to the pool. A gap vs what the consumer saw means a producer
            # RAISED (run_concurrent logs and skips those) — tracked here so it isn't silently lost.
            "dispatched": dispatched, "producer_failed": dispatched - counters["consumed"],
            "gate_rejected": counters["gate_rejected"],
            "stage_seconds": timer.totals, "stage_latency": timer.distribution(),
            # per-CALL (not per-atom) — separates "one unlucky call" from "the provider
            # is slow right now", which decide oppositely on hedging.
            **llm_run_stats(llm0),
            "total": schema.count_atoms(conn, "blog")}
