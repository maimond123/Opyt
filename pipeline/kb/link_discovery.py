"""
pipeline/kb/link_discovery.py — blog/website URL discovery: UNION + a two-tier funnel.

Drop-in replacement for the old sitemap-only discovery, which missed real articles by being
first-source-wins and same-domain-only

The funnel — the baseline is preserved, discovery may only ADD to it. Baseline (sitemap ∪ rss)
is fetched as today; hub links (homepage + index pages, any domain, one hop deep into owned
index-shaped pages) are filtered to candidates by dropping the baseline, hub pages, cross-host
mirrors, and not-owned hosts, then classified `drop` (never fetched) / `strong` (fetched
directly) / `gray` (sent to one batched LLM triage call). Fetch list = baseline ∪ strong ∪
triage-approved.

Ownership guard (auto-attribute is only safe for the author's OWN content): a hub link is kept
only if its host is the origin or a subdomain of it (`_is_owned`). Cross-host MIRRORS of a
baseline post (`_path_key` match) collapse onto the baseline instead of double-ingesting.

URL structure is a reliable REJECT filter but an unreliable ACCEPT filter (empirically measured
across 5 sites), so structure only drops confident junk; the LLM decides the gray zone, and the
content gate is the final judge on the page actually fetched.

Fail-safe cascade (mirrors content_gate): a failure here may only ADD fetches, never drop an
article path. Hub-harvest failure → baseline only. Triage failure → approve all gray.
"""

from __future__ import annotations

import json
import re

from pipeline.ingestion.sources.blog import is_nav_path  # shared nav denylist (single source)
from urllib.parse import urlparse

from .ingest_blog import _canon_post_url  # discovery-level dedup key == the atom_id key

_TRIAGE_ROLE = "url_triage"

# ── structural tiers ─────────────────────────────────────────────────────────
# Assets/media (incl. pdf — arXiv/PDF routing is parked; a pdf is not a blog article the gate
# handles) and feeds. A confident REJECT.
_ASSET_RE = re.compile(
    r"\.(?:png|jpe?g|gif|svg|webp|avif|ico|bmp|tiff?|css|js|mjs|json|xml|zip|gz|tgz|"
    r"woff2?|ttf|otf|eot|mp4|mov|avi|webm|mkv|mp3|wav|m4a|pdf|rss|atom)(?:[?#]|$)",
    re.I,
)
# Social / media / commerce homes — a link to a profile, not an article. Host-anchored so
# `github.io` custom-domain blogs are NOT caught (only bare github.com/linkedin/etc.).
_SOCIAL_HOST_RE = re.compile(
    r"(?:^|\.)(?:twitter\.com|x\.com|t\.co|github\.com|gitlab\.com|linkedin\.com|"
    r"youtube\.com|youtu\.be|instagram\.com|facebook\.com|threads\.net|mastodon\.|"
    r"bsky\.app|bluesky|discord\.(?:gg|com)|patreon\.com|reddit\.com|ko-fi\.com|"
    r"buymeacoffee\.com|paypal\.(?:com|me)|amazon\.|amzn\.|goodreads\.com|producthunt\.com)",
    re.I,
)
# Confident ARTICLE markers: a dated path segment, or a known post/section segment.
_STRONG_PATH_RE = re.compile(
    r"/(?:19|20)\d\d(?:[/-]\d|/)|"                                        # /2024/…  /2024-…
    r"/(?:blog|posts?|writing|notes?|essays?|articles?|p|abs|story|pub|newsletter|til)/",
    re.I,
)


def classify_url(url: str, anchor: str = "") -> str:
    """Structural tier for a candidate URL: ``"drop" | "strong" | "gray"``.

    URL structure is a reliable REJECT filter and an unreliable ACCEPT filter (empirical), so:
      • ``drop``   — confident junk: assets/media, social/commerce homes, nav/tag/feed pages,
                     mailto/non-http, bare homepages. Never fetched.
      • ``strong`` — confident article: a dated path or a post/section segment. Fetched directly.
      • ``gray``   — everything else (marker-less slugs, section pages). Sent to LLM triage.

    DROP is checked before STRONG so a mixed path (e.g. ``/blog/tags/foo`` — a tag index) is
    correctly dropped rather than accepted on the ``/blog/`` marker. ``anchor`` is unused here
    (structure only); it feeds the LLM triage of the gray zone."""
    u = (url or "").strip()
    if not u or u.startswith(("mailto:", "tel:", "javascript:", "#")):
        return "drop"
    p = urlparse(u)
    if p.scheme not in ("http", "https"):
        return "drop"
    host = (p.netloc or "").lower()
    path = p.path or "/"
    if _ASSET_RE.search(path):
        return "drop"
    if _SOCIAL_HOST_RE.search(host):
        return "drop"
    if path in ("", "/"):                 # a bare homepage (internal root or external landing)
        return "drop"
    if is_nav_path(path):                 # about/contact/tag/author/feed/… (shared, whole-segment)
        return "drop"
    if _STRONG_PATH_RE.search(path):
        return "strong"
    return "gray"


# ── LLM url-triage (gray zone only) ──────────────────────────────────────────
_TRIAGE_SYSTEM = """You triage candidate URLs found on a person's blog or personal website for a \
knowledge base that indexes that person's writing. You see each link's URL and (when present) its \
anchor text — NOT the page contents. For EVERY index shown, output "keep" if the link most likely \
leads to a substantive readable article, essay, blog post, talk/podcast transcript, or a real \
piece of writing; output "drop" if it most likely leads to site navigation, a \
tag/category/archive index, a login/subscribe page, a product/course/landing/pricing page, a \
social-media profile, or another non-article destination.

RECALL FIRST: when a link is plausibly an article, prefer "keep" — a wrongly kept link is cheaply \
discarded by a later content check, but a wrongly dropped article is lost for good. Only "drop" a \
link that is clearly NOT an article.

Return ONLY a JSON object mapping every index shown to "keep" or "drop", e.g. \
{"0":"keep","1":"drop"}."""

# Tolerant `"<idx>": "keep"|"drop"` scrape — json_object mode should give valid JSON, but a stray
# token shouldn't waste a whole batch (mirrors content_gate._PAIR_RE).
_PAIR_RE = re.compile(r'["\']?(\d+)["\']?\s*:\s*["\'](keep|drop)["\']', re.I)


def _parse_triage(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k, v in obj.items():
                try:
                    out[int(str(k).strip())] = str(v).strip().lower()
                except (ValueError, TypeError):
                    continue
    except (json.JSONDecodeError, TypeError):
        pass
    if not out:                                   # fallback: scrape pairs out of the raw text
        for m in _PAIR_RE.finditer(text or ""):
            out[int(m.group(1))] = m.group(2).lower()
    return out


def _triage_prompt(candidates: list[dict], author_name: str | None) -> str:
    who = f" by {author_name}" if author_name else ""
    lines = []
    for i, c in enumerate(candidates):
        anchor = (c.get("anchor") or "").strip().replace("\n", " ")[:160]
        lines.append(f'[{i}] {c["url"]}' + (f'  — "{anchor}"' if anchor else ""))
    return (f"These links were found on a personal site/blog{who}. For each index, decide whether "
            f"the link points to a readable article/essay/post/transcript worth ingesting.\n\n"
            + "\n".join(lines))


def _triage_gray(candidates: list[dict], *, author_name: str | None = None) -> list[dict]:
    """One batched LLM call classifying the GRAY candidates as article vs not, by url + anchor.

    Returns the approved subset (order preserved). FAIL-SAFE: any failure (import / preflight /
    call error / unparseable / empty verdicts) approves all gray rather than silently dropping an
    article. On a valid response, a candidate the model omits from the keep-mask defaults to
    drop."""
    if not candidates:
        return []
    from pipeline.ingestion.utils import log

    try:
        from pipeline import llm_client
    except Exception as e:                        # import failure → approve-all
        log(f"[url_triage] llm_client import failed (approve-all gray): {e}")
        return list(candidates)

    try:
        reason = llm_client.preflight(_TRIAGE_ROLE)
    except Exception as e:
        reason = f"role {_TRIAGE_ROLE!r} unavailable: {e}"
    if reason:                                    # missing role/key → approve-all (degrade)
        log(f"[url_triage] approve-all gray (degrade): {reason}")
        return list(candidates)

    try:
        resp = llm_client.call(_TRIAGE_ROLE, system=_TRIAGE_SYSTEM,
                               user=_triage_prompt(candidates, author_name))
        verdicts = _parse_triage(resp.text)
    except Exception as e:                        # call error → approve-all
        log(f"[url_triage] call failed (approve-all gray): {type(e).__name__}: {e}")
        return list(candidates)

    if not verdicts:                              # unparseable / empty → approve-all
        log(f"[url_triage] no usable verdicts (approve-all gray, {len(candidates)} urls)")
        return list(candidates)

    approved = [c for i, c in enumerate(candidates) if verdicts.get(i) == "keep"]
    log(f"[url_triage] {len(approved)}/{len(candidates)} gray urls approved")
    return approved


# ── composition ──────────────────────────────────────────────────────────────

def _hub_page_keys(base: str) -> set[str]:
    """Canonical keys of the hub pages themselves (homepage + every probed index path), so an
    index page like ``/blog`` is never mistaken for an article candidate. Computed for ALL
    candidate paths regardless of whether they 200 — subtracting a key nothing links to is free."""
    from pipeline.ingestion.sources.blog import HUB_INDEX_PATHS

    pages = [base] + [f"{base}{p}" for p in HUB_INDEX_PATHS]
    return {_canon_post_url(p) for p in pages}


def _host(url: str) -> str:
    """Lowercased host: drop scheme/userinfo/port and a leading ``www.``."""
    from urllib.parse import urlparse
    u = url if "://" in (url or "") else "https://" + (url or "")
    h = (urlparse(u).netloc or "").lower().split("@")[-1].split(":")[0]
    return h[4:] if h.startswith("www.") else h


def _is_owned(url: str, origin_host: str) -> bool:
    """True iff ``url``'s host IS the origin host or a SUBDOMAIN of it — i.e. the link lives on
    the author's OWN site. The ownership guard for auto-attribute: a NOT-owned link (press about
    the author, another person's site) must never be minted as the author's atom. Conservative —
    also drops the author's own content on a SHARED platform domain (e.g. ``substack.com/@them``);
"""
    h = _host(url)
    return bool(h) and bool(origin_host) and (h == origin_host or h.endswith("." + origin_host))


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

# Section-index path segments (singular + plural). A SHALLOW path ending in one of these is a
# "list of posts" landing to crawl one hop deeper, not an article. Deliberately generous — a
# wrong guess only costs one bounded fetch, still gated by the full funnel below.
_INDEX_SEGMENTS = frozenset({
    "blog", "blogs", "post", "posts", "writing", "writings", "essay", "essays",
    "article", "articles", "note", "notes", "archive", "archives", "talk", "talks",
    "project", "projects", "paper", "papers", "newsletter", "newsletters",
    "story", "stories", "thought", "thoughts", "word", "words", "log", "logs",
    "journal", "til", "reading", "writes", "publication", "publications",
})


def _is_index_path(url: str) -> bool:
    """True iff ``url``'s path looks like a section INDEX (a list-of-posts landing) rather than an
    article: a SHALLOW path (≤2 segments) whose LAST segment is a known section word
    (``/writings``, ``/blog/archive``), plural-tolerant, ``.html`` tolerated. Structural only — used
    to pick which OWNED discovered pages to crawl one hop deeper. ``/writings/entropic`` (a real
    essay) is NOT an index (its last segment isn't a section word); ``/writings`` (its index) is."""
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
    """From ``seed_urls`` (baseline ∪ level-1 hub), the OWNED, index-shaped pages to crawl one hop
    deeper — de-duped on the canonical key, EXCLUDING pages already crawled at level 1
    (``exclude_keys`` = the hub index pages), capped at ``max_pages`` (bounded fan-out)."""
    picked: list[str] = []
    seen: set[str] = set()
    for url in seed_urls:
        if len(picked) >= max_pages:
            break
        url = (url or "").strip()
        if not url:
            continue
        key = _canon_post_url(url)
        if key in exclude_keys or key in seen:
            continue
        if not _is_owned(url, origin_host) or not _is_index_path(url):
            continue
        seen.add(key)
        picked.append(url)
    return picked


def discover_candidate_urls(base: str, *, handle: str | None = None,
                            author_name: str | None = None,
                            known_urls: set[str] | None = None) -> list[dict]:
    """UNION discovery: the sitemap/rss baseline PLUS hub-harvested, triaged extras.

    ``known_urls`` is the REFRESH seam: canonical post keys already in the store. Matching GRAY
    candidates are dropped before ``_triage_gray`` so a re-crawl doesn't re-pay an LLM call for
    urls already ingested; the CALLER must pass ``seen − body_pending`` so atoms rescued by
    ``schema.load_body_pending`` still get a chance to self-heal

    Drop-in for the old ``_fetch_sitemap_urls(base)`` call. Returns entries
    ``{url, lastmod, via, source}`` where ``source`` is ``sitemap`` (baseline, untriaged,
    ``via=None``), ``strong`` (confident structural match), or ``triage`` (LLM-approved gray);
    ``via`` is the hub page a non-baseline link was found on.

    Discovery can only ADD to the baseline, never subtract from it. FAIL-SAFE: hub-harvest failure
    → baseline only; triage failure → all gray approved (see module docstring)."""
    from pipeline.ingestion.sources.blog import (_fetch_sitemap_urls, harvest_hub_links,
                                                 harvest_links_from)
    from pipeline.ingestion.utils import log

    base = (base or "").rstrip("/")

    baseline = _fetch_sitemap_urls(base)
    for e in baseline:                            # tag provenance; baseline is never triaged
        e.setdefault("via", None)
        e.setdefault("source", "sitemap")

    baseline_keys = {_canon_post_url(e["url"]) for e in baseline}
    baseline_paths = {_path_key(e["url"]) for e in baseline}   # host-independent, for mirror dedup
    hub_page_keys = _hub_page_keys(base)
    origin_host = _host(base)

    try:
        hub_links = harvest_hub_links(base)
    except Exception as e:                        # any hub-harvest bug → baseline survives untouched
        log(f"[link_discovery] hub-harvest failed ({type(e).__name__}: {e}); baseline only")
        hub_links = []

    # ── depth-1 second level ─────────────────────────────────────────────────────────────────
    # Some sites list real posts one hop below an index page neither the sitemap nor the level-1
    # hub reaches (e.g. /writings → /writings/{slug}). Crawl OWNED, index-shaped pages one hop
    # deeper; harvested links append to hub_links and run the same candidate funnel below, so
    # this only enlarges the candidate set. Bounded (_MAX_INDEX_PAGES) + fail-safe.
    seed_pages = [e["url"] for e in baseline] + [l.get("url", "") for l in hub_links]
    index_pages = _pick_index_pages(seed_pages, origin_host, hub_page_keys)
    if index_pages:
        hub_page_keys = hub_page_keys | {_canon_post_url(u) for u in index_pages}  # index ≠ a post
        try:
            level2 = harvest_links_from(index_pages)
        except Exception as e:                    # level-2 crash → level-1 candidates untouched
            log(f"[link_discovery] level-2 crawl failed ({type(e).__name__}: {e}); level-1 only")
            level2 = []
        if level2:
            hub_links = list(hub_links) + level2
            log(f"[link_discovery] level-2: {len(index_pages)} index page(s) → {len(level2)} link(s)")

    # Candidates = NEW hub links: not already in the baseline (exact key OR cross-host mirror of a
    # baseline post), not the hub index pages, de-duped on the canonical key, and OWNED by the author
    # (their domain / a subdomain) — press-about-them and others' writing are dropped, not attributed.
    candidates: list[dict] = []
    seen_keys: set[str] = set()
    mirror_dropped = external_dropped = 0
    for link in hub_links:
        url = (link.get("url") or "").strip()
        if not url:
            continue
        key = _canon_post_url(url)
        if key in baseline_keys or key in hub_page_keys or key in seen_keys:
            continue
        if _path_key(url) in baseline_paths:      # same path, different host = mirror of a baseline post
            mirror_dropped += 1
            continue
        if not _is_owned(url, origin_host):       # not the author's domain → NOT their atom (auto-attribute guard)
            external_dropped += 1
            continue
        seen_keys.add(key)
        candidates.append(link)

    strong: list[dict] = []
    gray: list[dict] = []
    known = known_urls or set()
    known_gray = 0
    for c in candidates:
        tier = classify_url(c["url"], c.get("anchor", ""))
        if tier == "drop":
            continue
        if tier == "gray" and _canon_post_url(c["url"]) in known:
            known_gray += 1          # already ingested → never re-triage it (see the docstring)
            continue
        (strong if tier == "strong" else gray).append(c)

    approved_gray = _triage_gray(gray, author_name=author_name)

    extras = (
        [{"url": c["url"], "lastmod": "", "via": c.get("via"), "source": "strong"} for c in strong]
        + [{"url": c["url"], "lastmod": "", "via": c.get("via"), "source": "triage"}
           for c in approved_gray]
    )
    if extras or mirror_dropped or external_dropped or known_gray:
        log(f"[link_discovery] {base}: {len(baseline)} baseline + {len(extras)} hub extras "
            f"({len(strong)} strong, {len(approved_gray)} triaged); dropped "
            f"{external_dropped} external + {mirror_dropped} mirror + {known_gray} already-known gray")
    return baseline + extras
