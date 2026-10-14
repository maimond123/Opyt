"""
pipeline/kb/link_discovery.py — blog/website URL discovery: the baseline plus the owned hub links.

The baseline (sitemap ∪ rss) is kept; discovery may only add to it. Hub links (homepage + index
pages, one hop deep into owned index-shaped pages) drop the baseline, hub pages, cross-host
mirrors and not-owned links, then `classify_url` drops confident junk by structure. Fetch list =
baseline ∪ the rest. Structure is a reliable reject filter but not an accept filter: real posts
live at marker-less paths (chrisbarber.co/resonance, gajesh.com/writings/entropic), so every
owned link it cannot reject is kept. A model triaging those kept 57 of 63.

Ownership guard: a hub link is kept only if it is on the origin host (or a subdomain) or carries
one of the author's identity tokens, since everything kept is attributed to the author.

Fail-safe: a hub-harvest failure leaves the baseline only.
"""

from __future__ import annotations

from pipeline.ingestion.utils import log

import re

from pipeline.platforms import identity_tokens
from pipeline.platforms.url_canon import host_of
from pipeline.platforms.web.discover import is_nav_path  # shared nav denylist (single source)
from urllib.parse import urlparse

from .ingest_blog import canon_post_url  # discovery-level dedup key == the atom_id key

# ── structural tiers ─────────────────────────────────────────────────────────
# Assets/media (incl. pdf, which the blog gate does not handle) and feeds: a confident reject.
_ASSET_RE = re.compile(
    r"\.(?:png|jpe?g|gif|svg|webp|avif|ico|bmp|tiff?|css|js|mjs|json|xml|zip|gz|tgz|"
    r"woff2?|ttf|otf|eot|mp4|mov|avi|webm|mkv|mp3|wav|m4a|pdf|rss|atom)(?:[?#]|$)",
    re.I,
)
def classify_url(url: str) -> str:
    """``"drop"`` for confident junk — assets/media, nav/tag/feed pages, mailto/non-http, bare
    homepages — else ``"keep"``. The host is not consulted; the caller applies `_is_owned` first.
    A nav segment anywhere drops, so ``/blog/tags/foo`` is dropped despite ``/blog/``."""
    u = (url or "").strip()
    if not u or u.startswith(("mailto:", "tel:", "javascript:", "#")):
        return "drop"
    p = urlparse(u)
    if p.scheme not in ("http", "https"):
        return "drop"
    path = p.path or "/"
    if _ASSET_RE.search(path):
        return "drop"
    if path in ("", "/"):                 # a bare homepage (internal root or external landing)
        return "drop"
    if is_nav_path(path):                 # about/contact/tag/author/feed/… (shared, whole-segment)
        return "drop"
    return "keep"


# ── composition ──────────────────────────────────────────────────────────────

def hub_page_keys(base: str) -> set[str]:
    """Canonical keys of the hub pages themselves (homepage + every probed index path), so an
    index page like ``/blog`` is never mistaken for an article candidate."""
    from pipeline.platforms.web.discover import HUB_INDEX_PATHS

    pages = [base] + [f"{base}{p}" for p in HUB_INDEX_PATHS]
    return {canon_post_url(p) for p in pages}


def _is_owned(url: str, origin_host: str, tokens=frozenset()) -> bool:
    """True iff the link is the author's own: on the origin host (or a subdomain), or carrying one
    of their identity tokens as a whole host label or path segment.

    The auto-attribute guard: a link that is not the author's must never become their atom. The
    token arm reaches their content on hosts they don't control (`medium.com/@them`). Empty
    `tokens` ⇒ the host rule alone.
    """
    from pipeline.platforms.identity_tokens import url_carries_token
    h = host_of(url)
    if not h:
        return False
    if origin_host and (h == origin_host or h.endswith("." + origin_host)):
        return True
    return url_carries_token(url, tokens)


def _path_key(url: str) -> str:
    """Host-independent post key = path + tracking-stripped query, so two of the author's own
    domains serving the same path collapse onto one entry instead of double-ingesting."""
    from urllib.parse import urlparse

    from .ingest_blog import _strip_tracking_query
    p = urlparse(url if "://" in (url or "") else "https://" + (url or ""))
    q = _strip_tracking_query(p.query)
    return (p.path.rstrip("/") or "/") + (f"?{q}" if q else "")


# ── depth-1 second level (index → its posts) ─────────────────────────────────
_MAX_INDEX_PAGES = 8  # crawl budget: at most this many index pages fetched one hop deep, per site

# Section-index path segments. A shallow path ending in one of these is a list-of-posts landing to
# crawl one hop deeper; a wrong guess costs one bounded fetch.
_INDEX_SEGMENTS = frozenset({
    "blog", "blogs", "post", "posts", "writing", "writings", "essay", "essays",
    "article", "articles", "note", "notes", "archive", "archives", "talk", "talks",
    "project", "projects", "paper", "papers", "newsletter", "newsletters",
    "story", "stories", "thought", "thoughts", "word", "words", "log", "logs",
    "journal", "til", "reading", "writes", "publication", "publications",
})


def _is_index_path(url: str) -> bool:
    """True iff ``url``'s path looks like a section index: ≤2 segments whose last is a known
    section word (``/writings``, ``/blog/archive``; ``.html`` tolerated). ``/writings/entropic``
    is not one."""
    from urllib.parse import urlparse
    p = urlparse(url if "://" in (url or "") else "https://" + (url or ""))
    segs = [s for s in (p.path or "").split("/") if s]
    if not segs or len(segs) > 2:
        return False
    last = segs[-1].lower()
    if last.endswith(".html"):
        last = last[:-5]
    return last in _INDEX_SEGMENTS


def _pick_index_pages(seed_urls: list[str], origin_host: str, exclude_keys: set[str],
                      *, max_pages: int = _MAX_INDEX_PAGES) -> list[str]:
    """From ``seed_urls`` (baseline ∪ level-1 hub), the owned, index-shaped pages to crawl one hop
    deeper, de-duped, excluding ``exclude_keys`` (pages already crawled), capped at ``max_pages``."""
    picked: list[str] = []
    seen: set[str] = set()
    for url in seed_urls:
        if len(picked) >= max_pages:
            break
        url = (url or "").strip()
        if not url:
            continue
        key = canon_post_url(url)
        if key in exclude_keys or key in seen:
            continue
        if not _is_owned(url, origin_host) or not _is_index_path(url):
            continue
        seen.add(key)
        picked.append(url)
    return picked


def discover_candidate_urls(base: str, *,
                            author_name: str | None = None,
                            ) -> list[dict]:
    """Union discovery: the sitemap/rss baseline plus hub-harvested, post-shaped extras.

    Returns entries ``{url, lastmod, via, source}`` where ``source`` is ``sitemap`` (baseline,
    ``via=None``) or ``hub``; ``via`` is the hub page the link was found on."""
    from pipeline.platforms.web.discover import (fetch_sitemap_urls, harvest_hub_links,
                                                 harvest_links_from)

    base = (base or "").rstrip("/")

    baseline = fetch_sitemap_urls(base)
    for e in baseline:                            # tag provenance; the baseline is always kept
        e.setdefault("via", None)
        e.setdefault("source", "sitemap")

    baseline_keys = {canon_post_url(e["url"]) for e in baseline}
    baseline_paths = {_path_key(e["url"]) for e in baseline}   # host-independent, for mirror dedup
    hub_keys = hub_page_keys(base)
    origin_host = host_of(base)
    tokens = identity_tokens.tokens_for(base, author_name)

    try:
        hub_links = harvest_hub_links(base)
    except Exception as e:                        # any hub-harvest bug → baseline survives untouched
        log(f"[link_discovery] hub-harvest failed ({type(e).__name__}: {e}); baseline only")
        hub_links = []

    # ── depth-1 second level ─────────────────────────────────────────────────────────────────
    # Some sites list posts one hop below an index page nothing else reaches (/writings →
    # /writings/{slug}). Owned index pages are crawled one hop deeper; their links join hub_links.
    seed_pages = [e["url"] for e in baseline] + [l.get("url", "") for l in hub_links]
    index_pages = _pick_index_pages(seed_pages, origin_host, hub_keys)
    if index_pages:
        hub_keys = hub_keys | {canon_post_url(u) for u in index_pages}  # index ≠ a post
        try:
            level2 = harvest_links_from(index_pages)
        except Exception as e:                    # level-2 crash → level-1 candidates untouched
            log(f"[link_discovery] level-2 crawl failed ({type(e).__name__}: {e}); level-1 only")
            level2 = []
        if level2:
            hub_links = list(hub_links) + level2
            log(f"[link_discovery] level-2: {len(index_pages)} index page(s) → {len(level2)} link(s)")

    return baseline + select_hub_urls(base, baseline, hub_links, index_pages,
                                      author_name=author_name)


def select_hub_urls(base, baseline, hub_links, index_pages, *, author_name=None):
    """Apply ownership, mirror, navigation and quality rules to already fetched hub links."""
    baseline_keys = {canon_post_url(e["url"]) for e in baseline}
    baseline_paths = {_path_key(e["url"]) for e in baseline}
    hub_keys = hub_page_keys(base) | {canon_post_url(u) for u in index_pages}
    origin_host = host_of(base)
    tokens = identity_tokens.tokens_for(base, author_name)

    # Candidates = new hub links: not in the baseline (by key or as a cross-host mirror), not hub
    # pages, de-duped, and owned by the author.
    candidates: list[dict] = []
    seen_keys: set[str] = set()
    mirror_dropped = external_dropped = 0
    for link in hub_links:
        url = (link.get("url") or "").strip()
        if not url:
            continue
        key = canon_post_url(url)
        if key in baseline_keys or key in hub_keys or key in seen_keys:
            continue
        if _path_key(url) in baseline_paths:      # same path, different host = mirror of a baseline post
            mirror_dropped += 1
            continue
        if not _is_owned(url, origin_host, tokens):   # not the author's → not their atom
            external_dropped += 1
            continue
        seen_keys.add(key)
        candidates.append(link)

    extras = [{"url": c["url"], "lastmod": "", "via": c.get("via"), "source": "hub"}
              for c in candidates if classify_url(c["url"]) == "keep"]
    if extras or mirror_dropped or external_dropped:
        log(f"[link_discovery] {base}: {len(baseline)} baseline + {len(extras)} hub extras; "
            f"dropped {external_dropped} external + {mirror_dropped} mirror")
    return extras
