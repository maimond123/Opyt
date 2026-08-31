"""
pipeline/kb/vision.py — attaches VLM descriptions to a tweet's images so image-borne posts become
searchable. A bare CDN URL carries no meaning into a chunk, so a screenshot or chart with a thin
caption would otherwise be an invisible atom.

Every path here reads through `pipeline.ocr_cascade.read_image`
that replaced an earlier split with `describe_image`). Reads are cached by image URL (CDN links
are immutable), so a re-run is a lookup, never a re-charge, and the atom's snapshot hash stays
stable across runs. `from_cache` returns None for an entry it cannot interpret, which every caller
treats as a miss: re-read and overwrite, so the cache upgrades itself.

Gating (David 2026-07-16):
  • BOOKMARKS → read ALL images (curated, bounded, no cost-gate — you already vouched for them).
  • ORACLE ingest → gate on thin-text-OR-image-only (bulk pull → conserve spend). The gate lives here
    so both callers share it; only the `describe_all` flag differs.
"""
from __future__ import annotations

import re

# Below this caption length the image is likely the payload (a chart/screenshot with a "wow" caption),
# not an illustration of the text — so the Oracle gate describes it. Bookmarks ignore the gate entirely.
_THIN_TEXT_CHARS = 80

# Markdown image: ![alt](target). Target is not restricted to `https?://` so self-hosted relative
# paths match too; relative targets resolve against `base_url`, and non-image targets are filtered
# by `looks_like_image_url`.
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")

# Thread ceiling for one post's image fan-out. Matches `describe_images._VLM_GATE`'s max_permits —
# that AIMD gate is the REAL throttle (it is process-wide and shared with the X path), so this only
# avoids parking more threads than the gate could ever admit at once.
_VLM_FANOUT = 12

# A description splices in as `"\n*Image:* {desc}"` — one newline, so it joins its image's existing
# unit rather than starting a new one. `content_gate._split_units` splits on blank lines, so an
# unflattened description could change the unit count between the un-enriched and enriched body,
# which `content_gate.reapply_keep` relies on staying fixed.
_WS_RUN = re.compile(r"\s*\n\s*")


def _one_line(desc: str) -> str:
    """Collapse a description onto a single line (see `_WS_RUN`). Idempotent; "" stays ""."""
    return _WS_RUN.sub(" ", desc or "").strip()


def enrich_markdown_images(md: str, cache: dict, *, context: str = "", base_url: str | None = None,
                           max_images: int = 30) -> tuple[str, int]:
    """VLM-describe the inline images of a MARKDOWN body (Substack/blog), injecting each
    description right after its image so it becomes searchable chunk text. Source-agnostic sibling
    to `enrich_tweet_media`; format mirrors the X renderer (`\\n*Image:* {desc}`).

    Describes all images up to `max_images` (Oracle footprint is cost-agnostic; no thin-text gate
    for long-form). Cached by URL so re-runs never re-charge. Fail-safe: a vision failure leaves
    the image untouched (retried next run); "" is cached and injects nothing. Returns
    `(enriched_md, n_new_descriptions)`."""
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import urljoin

    from pipeline import ocr_cascade
    from pipeline.image_cache import cache_put

    from .ingest_common import looks_like_image_url

    text = md or ""
    matches = list(_MD_IMAGE_RE.finditer(text))
    if not matches:
        return text, 0

    # PLAN → DESCRIBE → RENDER (not one sequential `re.sub` pass) so one post's images can be
    # described concurrently instead of strictly in series. Deduping the work-list is load-bearing:
    # fired concurrently, every occurrence of a URL would miss the cache at once and pay twice.

    # 1. PLAN — resolve every ref once, in document order. Resolving relative refs against the post
    #    URL makes them fetchable and makes the cache key absolute, stable across posts and runs.
    resolved: list[tuple[re.Match, str | None]] = []
    for m in matches:
        url = urljoin(base_url, m.group(1)) if base_url else m.group(1)
        # Image syntax can wrap a non-image (e.g. trafilatura renders `![@handle](twitter.com/handle)`).
        # Sending that to the vision model fails and would retry forever uncached, so skip it here.
        resolved.append((m, url if looks_like_image_url(url) else None))

    todo: list[str] = []
    queued: set[str] = set()
    capped = False
    for _, url in resolved:
        if url is None or url in queued:                   # within-post reuse
            continue
        # A cache hit counts only if `from_cache` can interpret the entry (same rule as the X path).
        if url in cache and ocr_cascade.from_cache(cache[url]) is not None:
            continue                                       # cross-post reuse
        if len(todo) >= max_images:                        # runaway guard (an image-wall post)
            capped = True
            break
        queued.add(url)
        todo.append(url)

    # 2. DESCRIBE — concurrently. `ocr_cascade.read_image` holds `_VLM_GATE` per call (halved on a
    #    429), and `cache_put` is lock-guarded, so the gate stays the single throttle; this pool only
    #    makes its permits reachable, bounded to the gate's own ceiling.
    def _describe(url: str):
        try:
            return url, ocr_cascade.read_image(url, context=context)
        except Exception:
            return url, None                    # never raises — one bad image can't fail the post

    if len(todo) == 1:
        results = [_describe(todo[0])]          # single image: no pool, no thread
    elif todo:
        with ThreadPoolExecutor(max_workers=min(len(todo), _VLM_FANOUT),
                                thread_name_prefix="vlm") as ex:
            results = list(ex.map(_describe, todo))
    else:
        results = []

    n_new = 0
    for url, mr in results:
        if mr is None:                          # FAILED — do NOT cache (poison-value rule), retry next run
            continue
        # Store the raw read as-is; `_one_line` normalization happens at RENDER (below), not here,
        # since the X path needs the OCR's line structure and only markdown splicing needs it flat.
        cache_put(cache, url, mr.to_cache())    # a PHOTO's "" IS a verdict → cache it, don't retry
        n_new += 1

    # 3. RENDER — splice descriptions in, reading the cache the describe pass just filled. A URL
    #    still absent either failed or fell past the cap; either way its image is left untouched.
    out: list[str] = []
    last = 0
    for m, url in resolved:
        out.append(text[last:m.start()])
        whole = m.group(0)
        # Normalize at RENDER: the cache stores raw multi-line OCR text, but the rendered text must
        # not gain a blank line (see `_WS_RUN`) — this is what keeps the unit count stable here.
        mr = ocr_cascade.from_cache(cache.get(url)) if url is not None else None
        desc = _one_line(mr.text) if mr else None
        out.append(f"{whole}\n*Image:* {desc}" if desc else whole)
        last = m.end()
    out.append(text[last:])

    if capped:
        from pipeline.ingestion.utils import log
        log(f"[vision] image cap {max_images} hit — remaining images left undescribed (not silent).")
    return "".join(out), n_new


def _photos(norm: dict) -> list[dict]:
    """The still-image media items of a normalized tweet (photos only — video frames aren't described
    in v1). Reads the twitterapi.io shape `extendedEntities.media` x_graphql._normalize maps to."""
    media = (norm.get("extendedEntities") or {}).get("media") or norm.get("media") or []
    return [m for m in media if isinstance(m, dict) and m.get("type") == "photo"]


def _passes_oracle_gate(norm: dict) -> bool:
    """Thin-text OR image-only — the cheap Oracle-ingest trigger."""
    text = (norm.get("text") or "").strip()
    return (not text) or (len(text) < _THIN_TEXT_CHARS)


def _enrich_node_photos(norm: dict, cache: dict, *, describe_all: bool) -> int:
    """Attach a `description` to each photo of ONE tweet node (in place). `describe_all=True`
    reads every photo (bookmarks); otherwise the Oracle gate (thin-text/image-only) decides.
    Returns the number of NEW reads generated (cache hits don't count). Fail-safe: a read failure
    leaves the photo undescribed (the URL stays), never crashes the ingest.

    Reads through the OCR cascade, not `describe_image` — the cascade transcribes verbatim and
    routes charts to a dedicated quantitative prompt, and the two used to write incompatible cache
    shapes to one key space
    A decorative photo yields text="" and no `*Image:*` line; images whose meaning IS their text
    (slides, notes, screenshots) tag SCREENSHOT or CHART, not PHOTO, so they keep their content."""
    photos = _photos(norm)
    if not photos:
        return 0
    if not describe_all and not _passes_oracle_gate(norm):
        return 0
    # `reads` is telemetry the cascade path collects for the footprint run summary; bookmarks have
    # no such summary, so it is discarded rather than threaded through this signature.
    return _cascade_node_photos(norm, cache, [])


def enrich_tweet_media(norm: dict, cache: dict, *, describe_all: bool) -> int:
    """Describe the tweet's photos AND its quoted tweet's photos (one level). A quote's image is
    often the context the quoter is reacting to, so it earns a description too. The quoted node is
    enriched even when the root has none of its own photos, and its Oracle gate is evaluated on the
    quoted node's own text, independent of the root. Returns total NEW descriptions across both."""
    made = _enrich_node_photos(norm, cache, describe_all=describe_all)
    quoted = norm.get("quoted_tweet")
    if isinstance(quoted, dict):
        made += _enrich_node_photos(quoted, cache, describe_all=describe_all)
    return made


# ── OCR cascade path (footprint) — replaces describe-all-VLM with transcribe→route→chart-VLM ──

def _cascade_node_photos(norm: dict, cache: dict, reads: list) -> int:
    """Read ONE tweet node's photos with the OCR cascade (in place). Stores on each photo:
    `description` (transcription / chart read, for the renderer) AND `media_read` ({kind,
    substance}, for the substance filter). Cached by URL (immutable CDN → free re-run + stable
    snapshot hash). Fail-safe: a read failure leaves the photo undescribed (URL stays), never
    cached (poison-value rule), retried next run. Appends each MediaRead to `reads` for telemetry."""
    from pipeline import ocr_cascade
    from pipeline.image_cache import cache_put

    photos = _photos(norm)
    if not photos:
        return 0
    context = norm.get("text", "")
    made = 0
    for m in photos:
        url = m.get("media_url_https") or m.get("url", "")
        if not url:
            continue
        # ABSENT and UNTRUSTABLE collapse into one branch: `from_cache` returns None for a legacy
        # bare-string entry, and the only correct response to an untrustable entry is the same as to
        # a missing one — read and overwrite, which makes the cache self-healing.
        mr = ocr_cascade.from_cache(cache[url]) if url in cache else None
        if mr is None:
            mr = ocr_cascade.read_image(url, context=context)
            if mr is None:                      # FAILED — do NOT cache, retry next run
                continue
            cache_put(cache, url, mr.to_cache())  # locked write — safe under the footprint producer pool
            made += 1
        if mr.text:
            m["description"] = mr.text          # rendered as `*Image:* {desc}` by _render_media
        m["media_read"] = {"kind": mr.kind, "substance": mr.substance}
        reads.append(mr)
    return made


def _iter_node_photos(norm: dict):
    """(url, context) for a tweet node AND its quoted node — the SAME two levels
    `enrich_tweet_media_cascade` reads. Kept next to it deliberately: if a prefetch walks fewer
    levels than the render does, the missed image silently falls back to an inline read inside a
    producer thread, which is exactly the serialization the prefetch exists to remove."""
    for node in (norm, norm.get("quoted_tweet")):
        if not isinstance(node, dict):
            continue
        ctx = node.get("text", "")
        for m in _photos(node):
            url = m.get("media_url_https") or m.get("url", "")
            if url:
                yield url, ctx


def prefetch_group_media(groups: list, cache: dict, *, workers: int,
                         flush_every: int = 0, on_flush=None) -> dict:
    """Read EVERY image across `groups` up front, one future per IMAGE, filling `cache`.

    Dispatching per-image rather than per-group equalizes work units: a group's images used to be
    read serially inside its own thread, so one oversized group dominated the wall time regardless
    of pool size. Every unit here is exactly one round-trip,
    so units are equal by construction, not by luck.

    Same PLAN → READ → RENDER split as `enrich_markdown_images`. Phase 3 (`run_concurrent`) is
    untouched — `_cascade_node_photos` already checks the cache first, so lookups become dict hits.
    Deduping before dispatch is load-bearing: fired concurrently, two occurrences of one URL would
    both miss and pay twice. A failed read isn't cached, so phase 3 retries it inline; the caller
    counts those late reads instead of assuming zero."""
    from concurrent.futures import ThreadPoolExecutor

    from pipeline import ocr_cascade
    from pipeline.image_cache import cache_put

    todo: dict[str, str] = {}          # url → context, insertion-ordered, deduped
    seen = 0
    for group in groups:
        for tweet in group:
            for url, ctx in _iter_node_photos(tweet):
                seen += 1
                if url in todo:
                    continue
                # A hit only counts if the entry is USABLE — a legacy bare string is a miss.
                if url in cache and ocr_cascade.from_cache(cache[url]) is not None:
                    continue
                todo[url] = ctx

    if not todo:
        return {"images": seen, "dispatched": 0, "read": 0, "failed": 0}

    def _read(item):
        url, ctx = item
        try:
            return url, ocr_cascade.read_image(url, context=ctx)
        except Exception:
            return url, None            # never raises — one bad image can't fail the run

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=min(len(todo), workers),
                            thread_name_prefix="ocr-pre") as ex:
        for url, mr in ex.map(_read, list(todo.items())):
            if mr is None:
                failed += 1
                continue
            cache_put(cache, url, mr.to_cache())
            ok += 1
            # Bound a crash's re-OCR cost — this phase holds all the paid work, so a periodic flush
            # avoids losing every prior read to a late interrupt.
            if flush_every and on_flush and ok % flush_every == 0:
                on_flush()
    return {"images": seen, "dispatched": len(todo), "read": ok, "failed": failed}


def enrich_tweet_media_cascade(norm: dict, cache: dict) -> tuple[int, list]:
    """Footprint media understanding via the OCR cascade (transcribe → route → chart-VLM) instead
    of the single describe-all VLM. Reads the tweet's photos AND its quoted tweet's photos (one
    level, same as `enrich_tweet_media`). Returns (n_new_reads, [MediaRead...] for telemetry). A
    photo's `media_read.substance` is what promotes an image-borne post to an ARTIFACT in the
    substance filter; a decorative photo leaves the post resting on the author's own words."""
    reads: list = []
    made = _cascade_node_photos(norm, cache, reads)
    quoted = norm.get("quoted_tweet")
    if isinstance(quoted, dict):
        made += _cascade_node_photos(quoted, cache, reads)
    return made, reads
