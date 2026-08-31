"""
pipeline/kb/ingest_substack.py — a confirmed Oracle's OWN Substack archive → OPINION atoms.

Stage-5 footprint ingest: pulls every post from a confirmed Oracle's own publication
(`/api/v1/archive`, not RSS's 20-post cap) and lands each as a full-body, chunked opinion atom.
Distinct from `pipeline.ingestion.ingest_substack` (the note ingester this ports its fetch from)
and `pipeline.kb.ingest_curation` (SAVED posts, a curation signal, `entry_mode="user-saved"`).

The atom key `substack:{post_id}` is shared with the curation path on purpose: a post that is
both user-saved and part of an Oracle's footprint collapses to ONE atom, and Policy B (below)
skips it so the curation `user-saved` provenance stands.

The archive LIST returns metadata only (`body_html` is None even for public posts); the full
body comes from the per-post endpoint `/api/v1/posts/{slug}` (`_fetch_full_post`, cookie-less
for public posts). So the loop lists cheaply, then fetches the body per new public post.

Fetch/dedup policy (`docs/plans/2026-07-18-stage5-footprint-adapters-plan.md`): Policy B skips
any post whose atom already exists BEFORE the paid per-post fetch. Only public posts are
ingested in v1 — paywalled posts are skipped and counted, never stored as a partial preview.

Fail-safe: a post with no usable body, or an embed/write failure, SKIPS (no atom, no `seen`
mark). Attribution to the Oracle's canonical is materialized by a later `resolve_entities` run
(the entity is upserted with its publication URL so the attested-links merge unifies it).
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime

from . import content_gate, derive, schema
from .embed import assert_model
from .ingest_common import (AtomSink, BASIS_STATED, BODY_COMPLETE,
                            PENDING_CLAIM, POST_INFLIGHT, POST_WORKERS, StageTimer,
                            body_fields, llm_run_marker, llm_run_stats,
                            make_consumer, promote_atom, run_concurrent, snapshot_and_hash,
                            store_atom)
# A single Substack POST url: `{scheme}://{host}/p/{slug}`. Covers `*.substack.com` posts only —
# a custom-domain Substack (e.g. noahpinion.blog) needs a fetch to distinguish from a generic blog.
_SUBSTACK_POST_RE = re.compile(r"^(https?://[^/]+)/p/([^/?#]+)", re.I)

# The Substack platform-app share form (`substack.com/.../post/p-{id}`) carries only the global
# post id, not {pub}+{slug}. It 302-redirects to the canonical `{pub}.substack.com/p/{slug}`,
# which we resolve before parsing.
_READER_URL_RE = re.compile(r"^https?://substack\.com/(?:[^/]+/)*post/p-\d+", re.I)


def _post_rec(post: dict, *, publication_url: str, handle: str | None,
              author_name: str | None) -> dict:
    """Map one `/api/v1/archive` post dict onto the `rec` shape `derive.derive_substack`
    expects. Publication identity comes from the Oracle source, not per-post, so `who_id`
    is stable across the whole archive."""
    return {
        "id": post.get("id"),
        "title": post.get("title") or "Untitled",
        "post_date": post.get("post_date") or "",
        "author_handle": handle or "",
        "author_name": author_name or "",
        "publication_name": author_name or "",
        "publication_url": publication_url,
        "slug": post.get("slug") or "",
        "url": post.get("canonical_url") or "",
        "audience": post.get("audience") or "",
    }


def sync_substack_footprint(conn: sqlite3.Connection, embedder, *, publication_url: str,
                            handle: str | None = None, author_name: str | None = None,
                            since: datetime | None = None, limit: int = 0) -> dict:
    """Ingest a confirmed Oracle's OWN Substack archive as opinion atoms (footprint).

    `publication_url` is the Oracle's publication home (e.g. `https://carol.substack.com` or a
    custom domain); `handle`/`author_name` label the author. `since` bounds the archive LISTING
    to recent posts (the archive paginates eagerly, so bound it for a recent-window pull);
    `limit` caps the number of NEW posts DISPATCHED per run (0 = all) — a re-run ADVANCES
    (skips already-ingested, adds the next batch); the unbounded pull is the idempotent one
    (Policy B skips all on re-run). `limit` bounds posts processed, not atoms added — a
    gate-rejected post still spends one of the N. Returns a run summary. Does NOT run entity
    resolution — the caller re-resolves after footprint expansion so the new `substack:` row
    unifies into the Oracle's canonical.

    The archive LIST carries only metadata; the full body comes from the per-post endpoint
    (`_fetch_full_post`, cookie-less for public posts). So, like `ingest_curation`: skip
    seen/paywalled BEFORE the per-post fetch, then process posts CONCURRENTLY (fetch stays
    serial; see the run_concurrent call). Full `limit` rationale:
"""
    from pipeline.ingestion.sources.substack import (SubstackFetchError, SubstackListingError,
                                                     _fetch_all_posts, _fetch_full_post,
                                                     _is_paywalled, _post_to_markdown)
    from pipeline.ingestion.utils import log
    from pipeline.image_cache import load_image_cache, save_image_cache
    from opyt_core.paths import opyt_home

    from .vision import enrich_markdown_images

    assert_model(conn, embedder)      # guard the store's embedding identity BEFORE any spend
    # Bookmark LLM latency samples before the first paid call so the summary reports THIS run's
    # calls, not the process-cumulative total. Taken above the listing-error return, which reports it too.
    llm0 = llm_run_marker()
    base = (publication_url or "").rstrip("/")
    if not base:
        return {"source": "substack-footprint", "error": "no publication_url"}

    # VLM image descriptions cached by URL (immutable CDN links) → re-runs are free + hash-stable.
    img_cache = load_image_cache(opyt_home())

    who_id = derive.substack_entity_id(handle, publication_url)
    # Seed the entity with its publication home so a later resolve_entities merges it into
    # the Oracle's canonical via the attested-links graph.
    schema.upsert_entity(conn, who_id, name=author_name,
                         identity_links=[publication_url])

    # Instruments the serial listing path to size a producer pool if one is later needed.
    timer = StageTimer()
    try:
        with timer.stage("list_fetch"):    # the archive walk: one session, paginated, retried
            posts = _fetch_all_posts(base, since=since)
    except SubstackListingError as e:
        # Fail-safe on an incomplete listing: ingest nothing, mark nothing `seen` — the next
        # run redoes the whole walk. Reported as `undetermined`. Return-shape rationale:
        log(f"[footprint] substack BLOCKED during archive listing: {e}")
        return {"source": "substack-footprint", "added": 0, "skipped": 0, "paywalled": 0,
                "no_body": 0, "undetermined": 1, "failed": 0, "gate_rejected": 0,
                "dispatched": 0, "producer_failed": 0,
                "stage_seconds": timer.totals, "stage_latency": timer.distribution(),
                **llm_run_stats(llm0),
                "total": schema.count_atoms(conn, "substack"),
                "error": f"archive listing incomplete: {e}"}
    seen = schema.load_hashes(conn, "substack")

    # Batches the embed across posts: Substack essays are long, so pooling their chunks per
    # flush is where the 8-way embed gate earns its keep. Posts are processed concurrently
    # (list+fetch serial on the calling thread; gate/VLM/render on pool threads; embed+write
    # serial in the consumer) — body fetch stays serial because substack.com is Cloudflare-fronted
    # and parallel GETs risk a soft-ban.
    bs = int(getattr(embedder, "batch_size", 64) or 64)
    sink = AtomSink(conn, embedder, timer=timer, flush_chunks=8 * bs)
    counts = {"added": 0}
    # Every counter is owned by the CALLING thread — the generator's in `_jobs`, the rest in
    # `_consume`. A pool thread must never tally: `+= 1` is a read-modify-write, which the GIL does
    # NOT make atomic, so `_work` reports an outcome and the consumer counts it.
    dispatched = paywalled = no_body = undetermined = 0
    # consumed/submitted/skipped/gate_rejected: a shared dict, not four more `nonlocal` ints — see
    # `make_consumer`'s docstring for why (its `_consume` closure is defined outside this scope).
    counters = {"consumed": 0, "submitted": 0, "skipped": 0, "gate_rejected": 0}
    author = f"@{handle}" if handle else (author_name or "substack")

    def _mark() -> None:                     # post-commit durable-write count (never on a poison-skip)
        counts["added"] += 1

    def _jobs():
        """List → dedup → body fetch, one post at a time on the calling thread.

        `seen` is confined to this generator and the consumer (same thread). The mark records a
        CLAIM (`PENDING_CLAIM`), not a completion; the consumer upgrades it to the real hash. A
        claimed post that then fails is not retried this run — nothing durable was written, so
        the next run picks it up."""
        nonlocal dispatched, paywalled, no_body, undetermined
        for post in posts:
            # `limit` caps DISPATCH, not atoms: the consumer lags by up to `inflight`, so capping on
            # its count would over-spend a window of PAID work before noticing.
            if limit and dispatched >= limit:
                break
            post_id = post.get("id")
            if not post_id:
                continue
            atom_id = f"substack:{post_id}"
            if atom_id in seen:             # policy B: skip BEFORE the (paid) per-post fetch
                counters["skipped"] += 1
                continue
            if _is_paywalled(post):         # v1 public-only → skip-and-count (no stale preview)
                paywalled += 1
                continue
            seen[atom_id] = PENDING_CLAIM        # CLAIM before the fetch — see the docstring

            slug = post.get("slug") or ""
            try:
                with timer.stage("body_fetch"):   # serial, as before: one GET per new post
                    full = _fetch_full_post(base, slug, {}) if slug else None  # {} = public, cookie-less
            except SubstackFetchError as e:
                # Stopped, not confirmed empty — same skip as a body-less post but counted apart
                # (`undetermined`), so a Cloudflare throttle doesn't read as "no real content".
                undetermined += 1
                log(f"[footprint] substack BLOCKED (fetch undetermined), post {post_id}: {e}")
                continue
            body_html = (full or {}).get("body_html") or ""
            if not body_html.strip():       # paid-but-mislabeled / link / podcast → no full body
                no_body += 1
                continue

            dispatched += 1
            yield {"post": post, "atom_id": atom_id, "post_full": {**post, **full}, "full": full}

    def _work(job: dict) -> dict:
        """The PAID stages, on a pool thread. Returns an OUTCOME for the consumer to tally rather
        than counting anything itself. Never returns None — `run_concurrent` drops a None without
        calling the consumer, which would lose the outcome entirely."""
        atom_id, post, post_full, full = (job["atom_id"], job["post"],
                                          job["post_full"], job["full"])
        md = _post_to_markdown(post_full, author=author, author_name=author_name or author)

        # Content-quality gate: drop nav/promo/subscribe-CTA units before embedding (raw snapshot
        # below keeps the full page as a safety net). Graded before the VLM pass so a page that
        # gets fully rejected never pays for image description first. History:
        with timer.stage("gate"):
            verdict = content_gate.classify_page(md)
        if verdict.kept_text is None:
            log(f"[footprint] substack atom {atom_id} rejected by content gate (no substantive units)")
            return {"outcome": "gate_rejected"}

        # VLM-describe inline images so charts/screenshots become searchable text (injected as
        # `*Image:* …` before hashing, so it's part of the chunked surface).
        with timer.stage("vlm"):            # per-POST (one post fans out to several describe calls)
            md, _ = enrich_markdown_images(md, img_cache, context=post_full.get("title") or "")
        # Reads `seen` from a pool thread — safe: a single dict `.get` is atomic under the GIL, and
        # what it finds for THIS atom_id is the generator's `PENDING_CLAIM` claim, which can never equal
        # a real hash, so the "unchanged" branch stays unreachable exactly as before.
        decided = snapshot_and_hash("substack", atom_id, md, seen)
        if decided is None:                 # unchanged — honor the seam
            return {"outcome": "unchanged"}
        raw_ref, raw_hash = decided

        # Replay the pre-enrichment mask onto the enriched body so descriptions reach the CHUNKS
        # while the SNAPSHOT stays the full page (rechunk-from-raw depends on that). Fail-safe: a
        # unit-count change returns None and we re-grade rather than emit mis-sliced text.
        md_kept = content_gate.reapply_keep(md, verdict.keep)
        if md_kept is None:
            log(f"[footprint] keep-mask lost alignment after VLM enrichment, re-grading: {atom_id}")
            with timer.stage("gate"):
                md_kept = content_gate.gate(md)
            if md_kept is None:
                return {"outcome": "gate_rejected"}

        meta = derive.derive_substack(_post_rec(post_full, publication_url=publication_url,
                                                handle=handle, author_name=author_name))
        atom = {
            "atom_id": atom_id,
            "source_type": "substack",
            "what_kind": "opinion",
            "who_id": who_id,
            "when_ts": meta["when_ts"],
            "when_precision": meta["when_precision"],
            "about_entities": meta["about_entities"],
            "source_url": post.get("canonical_url") or "",
            "raw_ref": raw_ref,
            "raw_hash": raw_hash,
            "description": meta["description"],
            # Constant `complete`/`stated`: paywalled and body-less posts are skipped before an
            # atom is built, so everything reaching here holds a full, declared-public body.
            "payload": {"word_count": post.get("wordcount") or full.get("wordcount") or 0,
                        **body_fields(BODY_COMPLETE, BASIS_STATED)},
            "entry_mode": "oracle-footprint",         # NOT user-saved (curation) / crawled (radar)
        }

        return {"outcome": "ok", "atom_id": atom_id, "raw_hash": raw_hash,
                "atom": atom, "md_kept": md_kept}

    # Byte-identical to `ingest_blog`'s consumer — see `ingest_common.make_consumer`.
    _consume = make_consumer(sink, seen, counters, _mark)

    run_concurrent(_jobs(), _work, _consume, workers=POST_WORKERS, inflight=POST_INFLIGHT)
    sink.close()
    save_image_cache(opyt_home(), img_cache)   # persist new VLM descriptions for the next run
    # A poison-chunk atom the sink isolates fires no _mark → not counted, not marked durable, retried
    # next run (fail-safe). `failed` = submitted-but-never-durable, same meaning as the old counter.
    return {"source": "substack-footprint", "added": counts["added"], "skipped": counters["skipped"],
            "paywalled": paywalled, "no_body": no_body, "undetermined": undetermined,
            "failed": counters["submitted"] - counts["added"],
            # `dispatched` = handed to the pool (what `limit` caps). A gap between it and what the
            # consumer saw is a producer that RAISED — run_concurrent logs and skips those, so
            # without this the post would vanish from every counter.
            "dispatched": dispatched, "producer_failed": dispatched - counters["consumed"],
            "gate_rejected": counters["gate_rejected"],
            "stage_seconds": timer.totals, "stage_latency": timer.distribution(),
            # per-CALL (not per-atom) — separates "one unlucky call" from "the provider
            # is slow right now", which decide oppositely on hedging.
            **llm_run_stats(llm0),
            "total": schema.count_atoms(conn, "substack")}


def _resolve_reader_url(url: str) -> str | None:
    """Follow a Substack reader-url's redirect to its canonical `{pub}.substack.com/p/{slug}` — HEAD
    only, redirect-following, hop-capped. Returns the canonical url IF it lands on a `/p/…` post,
    else None (fail-safe: the caller's `_SUBSTACK_POST_RE` match then fails and it skips). One extra
    network hop, incurred ONLY for the rare reader-url form."""
    try:
        import requests
        resp = requests.head(url, allow_redirects=True, timeout=10,
                             headers={"User-Agent": "Mozilla/5.0"})
        final = resp.url or ""
        return final if _SUBSTACK_POST_RE.match(final) else None
    except Exception:
        return None


def substack_atom_from_url(conn: sqlite3.Connection, embedder, url: str, *,
                           entry_mode: str = "author_referenced", seen: dict | None = None,
                           img_cache: dict | None = None) -> str | None:
    """Fetch ONE public Substack POST by url → an opinion atom, idempotent by post id. The
    link-dispatch twin of the whole-publication footprint (above): the Oracle *referenced* this
    post (who_id = the post's author, entry_mode='author_referenced'), it isn't the Oracle's own
    archive. Accepts both the canonical `{pub}.substack.com/p/{slug}` url and the platform
    reader-url form (resolved to canonical via its redirect first). Returns the canonical
    `substack:{post_id}` whenever the atom exists after the call (fresh or already present), or
    None if the url isn't a Substack post, is paywalled/bodyless, or the fetch/embed failed.
    Never raises (fail-safe).

    Keys on the post's numeric id — same `substack:{id}` the whole-pub footprint and curation-save
    paths use — so a post the Oracle also authored, or the user also saved, collapses to one atom;
    Policy B returns an already-present atom for the vouch without re-rendering. `img_cache` is the
    caller's shared VLM cache; absent → loaded/saved internally. Full docstring:
"""
    from pipeline.ingestion.sources.substack import (_fetch_full_post, _is_paywalled,
                                                     _post_to_markdown)
    from pipeline.ingestion.utils import log
    from pipeline.image_cache import load_image_cache, save_image_cache
    from opyt_core.paths import opyt_home

    from .vision import enrich_markdown_images

    u = (url or "").strip()
    if _READER_URL_RE.match(u):           # reader-url carries only a post id → resolve to {pub}/p/{slug}
        u = _resolve_reader_url(u) or u   # failure → unchanged → the match below fails → skip (fail-safe)
    m = _SUBSTACK_POST_RE.match(u)
    if m is None:
        return None
    base, slug = m.group(1).rstrip("/"), m.group(2)
    try:
        full = _fetch_full_post(base, slug, {})       # {} = public, cookie-less (the verified path)
        if not full:
            return None
        post_id = full.get("id")
        if not post_id:
            return None
        atom_id = f"substack:{post_id}"
        if seen is None:
            seen = schema.load_hashes(conn, "substack")
        if atom_id in seen:                           # policy B — present (any path) → vouch-only, no clobber
            promote_atom(conn, atom_id, entry_mode)   # human-initiated presence hit → attestation
            return atom_id
        if _is_paywalled(full):                       # public-only v1 (never store a preview as full)
            return None
        body_html = (full.get("body_html") or "").strip()
        if not body_html:
            return None

        who_id = derive.substack_entity_id(None, base)   # the POST's publication (subdomain)
        author = who_id.split(":", 1)[1]
        md = _post_to_markdown(full, author=f"@{author}", author_name=author)
        own_cache = img_cache is None
        cache = load_image_cache(opyt_home()) if own_cache else img_cache
        md, _ = enrich_markdown_images(md, cache, context=full.get("title") or "")
        if own_cache:
            save_image_cache(opyt_home(), cache)

        decided = snapshot_and_hash("substack", atom_id, md, seen)
        if decided is None:                           # (unreachable: not in seen) honor the seam
            return atom_id
        raw_ref, raw_hash = decided

        assert_model(conn, embedder)
        meta = derive.derive_substack(_post_rec(full, publication_url=base, handle=None,
                                                author_name=author))
        schema.upsert_entity(conn, who_id, name=author, identity_links=[base])
        atom = {
            "atom_id": atom_id,
            "source_type": "substack",
            "what_kind": "opinion",                   # a substack essay is opinion content (like the whole-pub path)…
            "who_id": who_id,
            "when_ts": meta["when_ts"],
            "when_precision": meta["when_precision"],
            "about_entities": meta["about_entities"],
            "source_url": full.get("canonical_url") or url,
            "raw_ref": raw_ref,
            "raw_hash": raw_hash,
            "description": meta["description"],
            # Constant `complete`/`stated` — `_is_paywalled` returns before this point.
            "payload": {"word_count": full.get("wordcount") or 0,
                        **body_fields(BODY_COMPLETE, BASIS_STATED)},
            "entry_mode": entry_mode,                 # …'author_referenced' — the Oracle pointed at it, didn't write it
        }
        store_atom(conn, embedder, atom=atom, snapshot_text=md)
        seen[atom_id] = raw_hash
        return atom_id
    except Exception as e:                            # fetch/embed/write failure → SKIP (no vouch target)
        log(f"[footprint] substack atom from {url} skipped (fetch/embed failed): {e}")
        return None
