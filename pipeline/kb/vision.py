"""
pipeline/kb/vision.py — attaches chart reads to a post's images so a chart becomes searchable. A
bare CDN URL carries no words into the search index, so a chart with a thin caption would otherwise be
an invisible atom.

Every path here reads through `pipeline.chart_read.read_image`: a chart's extraction, or
`NOT_CHART` (no text). Reads are cached by image URL (CDN links are immutable), so a re-run is a
lookup and the snapshot hash stays stable. `from_cache` returns None for an entry it cannot
interpret, which every caller treats as a miss.

Where images are read: every X path reads — an Oracle's own posts and the posts
they quote, the user's saved posts, hopper links, and the X posts an Oracle links. A long-form
page reads only when someone chose it (`_PAGE_IMAGES_READ`); a page pulled in because an Oracle
linked it does not.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin

from pipeline import chart_read
from pipeline.image_cache import cache_put
from pipeline.ingestion.utils import log

from .ingest_common import looks_like_image_url

# The entry modes whose long-form pages get their images read: an Oracle's own blog or Substack,
# and what the user saved. An `author_referenced` page (an article or Substack post an Oracle
# linked) is pulled in automatically and reads none. An allow-list, so a new entry mode spends
# nothing until it is added here.
_PAGE_IMAGES_READ = frozenset({"oracle-footprint", "user-saved"})

# Markdown image: ![alt](target). Target is not restricted to `https?://` so self-hosted relative
# paths match too; relative targets resolve against `base_url`, and non-image targets are filtered
# by `looks_like_image_url`.
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")

# Thread ceiling for one post's image fan-out, matching `_VLM_GATE`'s max_permits: the gate is the
# real throttle, so this only avoids parking more threads than it could admit.
_VLM_FANOUT = 12

# A description splices in as `"\n*Image:* {desc}"` on one line, so it stays beside its image.
_WS_RUN = re.compile(r"\s*\n\s*")

# An image inlined as base64 (`![alt](data:image/…;base64,…)`): nothing to fetch or describe, and
# its bytes would be stored and indexed as text (seen on vitalik.eth.limo and sakana.ai).
_DATA_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(data:[^)\s]*\)")


def _one_line(desc: str) -> str:
    """Collapse a description onto a single line (see `_WS_RUN`). Idempotent; "" stays ""."""
    return _WS_RUN.sub(" ", desc or "").strip()


def enrich_markdown_images(md: str, cache: dict, *, entry_mode: str, context: str = "",
                           base_url: str | None = None, max_images: int = 30) -> tuple[str, int]:
    """Chart-read the inline images of a markdown body (Substack/blog), injecting each chart's
    extraction right after its image so it becomes searchable text. Sibling to
    `enrich_tweet_media`; format mirrors the X renderer (`\\n*Image:* {desc}`).

    Reads nothing unless `entry_mode` is in `_PAGE_IMAGES_READ`. Otherwise reads up to
    `max_images`, cached by URL so re-runs never re-charge. Fail-safe: a failed read leaves the
    image untouched (retried next run); a `NOT_CHART` is cached and injects nothing. A base64
    `data:` image is replaced by its alt text either way. Returns `(enriched_md, n_new_reads)`."""

    text = _DATA_IMAGE_RE.sub(lambda m: m.group(1), md or "")
    if entry_mode not in _PAGE_IMAGES_READ:
        return text, 0                     # pulled in automatically: no read, `![…](url)` kept
    matches = list(_MD_IMAGE_RE.finditer(text))
    if not matches:
        return text, 0

    # Plan → describe (concurrently) → render. The work-list is deduped so concurrent reads of one
    # URL can't each miss the cache and pay twice.

    # 1. Plan — resolve every ref once, in document order, against the post URL (absolute cache key).
    resolved: list[tuple[re.Match, str | None]] = []
    for m in matches:
        url = urljoin(base_url, m.group(1)) if base_url else m.group(1)
        # Image syntax can wrap a non-image (`![@handle](twitter.com/handle)`); skip it.
        resolved.append((m, url if looks_like_image_url(url) else None))

    todo: list[str] = []
    queued: set[str] = set()
    capped = False
    for _, url in resolved:
        if url is None or url in queued:                   # within-post reuse
            continue
        # A cache hit counts only if `from_cache` can interpret the entry (same rule as the X path).
        if url in cache and chart_read.from_cache(cache[url]) is not None:
            continue                                       # cross-post reuse
        if len(todo) >= max_images:                        # runaway guard (an image-wall post)
            capped = True
            break
        queued.add(url)
        todo.append(url)

    # 2. Describe — concurrently; `read_image` holds `_VLM_GATE` per call, and `cache_put` is locked.
    def _describe(url: str):
        try:
            return url, chart_read.read_image(url, context=context)
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
        if mr is None:                          # failed — not cached (poison-value rule), retried next run
            continue
        # Raw read as-is; flattening happens at render.
        cache_put(cache, url, mr.to_cache())    # NOT_CHART is a verdict → cache it, don't retry
        n_new += 1

    # 3. Render — splice descriptions in, reading the cache the describe pass just filled. A URL
    #    still absent either failed or fell past the cap; either way its image is left untouched.
    out: list[str] = []
    last = 0
    for m, url in resolved:
        out.append(text[last:m.start()])
        whole = m.group(0)
        # Flatten here so the rendered text gains no blank line (see `_WS_RUN`).
        mr = chart_read.from_cache(cache.get(url)) if url is not None else None
        desc = _one_line(mr.text) if mr else None
        out.append(f"{whole}\n*Image:* {desc}" if desc else whole)
        last = m.end()
    out.append(text[last:])

    if capped:
        log(f"[vision] image cap {max_images} hit — remaining images left undescribed (not silent).")
    return "".join(out), n_new


def _photos(norm: dict) -> list[dict]:
    """The still-image media items of a normalized tweet (photos only — video frames aren't described
    in v1). Reads `extendedEntities.media`, the normalized shape `x.records.normalize_tweet` emits."""
    media = (norm.get("extendedEntities") or {}).get("media") or norm.get("media") or []
    return [m for m in media if isinstance(m, dict) and m.get("type") == "photo"]


def enrich_tweet_media(norm: dict, cache: dict) -> int:
    """Read the tweet's photos and its quoted tweet's photos (one level), in place. A quote's
    image is often the context the quoter is reacting to, so it is read too, even when the root
    has no photos of its own. Returns the number of new reads (cache hits don't count)."""
    made = _read_node_photos(norm, cache)
    quoted = norm.get("quoted_tweet")
    if isinstance(quoted, dict):
        made += _read_node_photos(quoted, cache)
    return made


def _read_node_photos(norm: dict, cache: dict) -> int:
    """Read one tweet node's photos (in place). Stores on each photo `description` (for the
    renderer) and `media_read` ({kind, substance}, for the substance filter). A failed read is not
    cached and is retried next run; a `NOT_CHART` attaches no description."""

    photos = _photos(norm)
    if not photos:
        return 0
    context = norm.get("text", "")
    made = 0
    for m in photos:
        url = m.get("media_url_https") or m.get("url", "")
        if not url:
            continue
        # Absent and untrustable entries are both read and overwritten.
        mr = chart_read.from_cache(cache[url]) if url in cache else None
        if mr is None:
            mr = chart_read.read_image(url, context=context)
            if mr is None:                      # failed — not cached, retried next run
                continue
            cache_put(cache, url, mr.to_cache())  # locked write — safe under a reader pool
            made += 1
        _attach(m, mr)
    return made


def _attach(photo: dict, mr) -> None:
    """Put one read on its photo (in place): its text for the renderer, its kind and substance
    for the substance filter."""
    if mr.text:
        photo["description"] = mr.text          # rendered as `*Image:* {desc}` by _render_media
    photo["media_read"] = {"kind": mr.kind, "substance": mr.substance}


def attach_cached(norm: dict, cache: dict) -> None:
    """Attach the reads the cache already holds to a tweet's photos and its quoted tweet's (in
    place), reading nothing. A photo with no trustable read is left as it was, so
    `_photos_pending` still sees it. A saved post's first save and its thread re-save render
    through this, and so does every Oracle X render (`ingest_x_footprint._GroupWriter`); the
    saved-images and Oracle photo sources read what is still missing."""

    for node in (norm, norm.get("quoted_tweet")):
        if not isinstance(node, dict):
            continue
        for m in _photos(node):
            url = m.get("media_url_https") or m.get("url", "")
            mr = chart_read.from_cache(cache[url]) if url and url in cache else None
            if mr is not None:
                _attach(m, mr)


def _iter_node_photos(norm: dict):
    """(url, context) for a tweet node and its quoted node — the same two levels the render
    covers; a check that walked fewer would leave a photo owed forever."""
    for node in (norm, norm.get("quoted_tweet")):
        if not isinstance(node, dict):
            continue
        ctx = node.get("text", "")
        for m in _photos(node):
            url = m.get("media_url_https") or m.get("url", "")
            if url:
                yield url, ctx


def _photos_pending(norm: dict, cache: dict) -> bool:
    """Does this tweet still carry a photo nobody has described?

    A term of `ingest_x`'s skip gate, so a bookmark written without its photo read is revisited.
    Presence is not a hit: an entry `from_cache` cannot interpret counts as pending, and a failed
    read is never cached, so a photo stays pending until a read succeeds."""

    return any(url not in cache or chart_read.from_cache(cache[url]) is None
               for url, _ctx in _iter_node_photos(norm))


def read_group_photos(groups: list, cache: dict, *, workers: int) -> dict:
    """Read every unread photo across `groups` (lists of tweets), one future per image, into
    `cache`. The Oracle photo loop's reader (`oracles/sources/x_photos.py`): a page renders
    from the cache and never waits on a read.

    URLs are deduped before dispatch so concurrent reads can't pay twice. A failed read isn't
    cached, so its photo stays pending."""

    todo: dict[str, str] = {}          # url → context, insertion-ordered, deduped
    seen = 0
    for group in groups:
        for tweet in group:
            for url, ctx in _iter_node_photos(tweet):
                seen += 1
                if url in todo:
                    continue
                # A hit only counts if the entry is usable.
                if url in cache and chart_read.from_cache(cache[url]) is not None:
                    continue
                todo[url] = ctx

    if not todo:
        return {"images": seen, "dispatched": 0, "read": 0, "failed": 0}

    def _read(item):
        url, ctx = item
        try:
            return url, chart_read.read_image(url, context=ctx)
        except Exception:
            return url, None            # never raises — one bad image can't fail the others

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=min(len(todo), workers),
                            thread_name_prefix="chart-read") as ex:
        for url, mr in ex.map(_read, list(todo.items())):
            if mr is None:
                failed += 1
                continue
            cache_put(cache, url, mr.to_cache())
            ok += 1
    return {"images": seen, "dispatched": len(todo), "read": ok, "failed": failed}
