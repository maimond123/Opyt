"""
pipeline/ingestion/sources/blog.py
Blog URL discovery (sitemap / RSS / hub crawl), article fetch via trafilatura, and
article→markdown rendering.

Layer 1 only — see ``pipeline/ingestion/sources/__init__.py``. The Layer-2 note-writing walk that
used to sit above this (``ingest_blog.sync_blog``, vault-writing) was DELETED 2026-08-14 with the
``raw/`` rail. This module was deliberately NOT deleted with it: ``pipeline/kb/ingest_blog.py``
(the atom-rail ingester of the same name, a different layer) imports ``_fetch_article`` and
``_article_to_markdown`` from here to land ATOMS.
"""

import re
import time
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests
import trafilatura

from pipeline.ingestion.utils import log

# Non-post path SEGMENTS: nav / feed / admin / commerce. Whole-SEGMENT membership, NOT a substring
# match — a real slug like "feedback-loops" must not be caught by the stop-word "feed". Add a
# stop-word by editing this SET; genuine PATTERNS (extensions, numbered pages) use the regex below.
_NAV_SEGMENTS = frozenset({
    "about", "contact", "privacy", "terms", "legal", "tag", "tags", "category", "author",
    "authors", "feed", "feeds", "rss", "atom", "login", "register", "search", "cart",
    "checkout", "sitemap", "wp-content", "wp-admin", "wp-includes",
})
# Genuinely pattern-shaped rejects (a real regex is the right tool here): asset/data extensions at a
# segment end, robots.txt, and WordPress numbered pagination (/page/2). Anchored so ".js" can't eat
# "node.js-guide" and "/page/2" can't eat "/mypage/2".
_NAV_PATTERN_RE = re.compile(r"\.(?:xml|json|css|js)(?:[?#]|$)|/robots\.txt(?:[?#]|$)|/page/\d", re.I)


def is_nav_path(path: str) -> bool:
    """True iff a URL path is site navigation / a feed / admin / an asset — i.e. NOT a readable post.

    Whole-SEGMENT membership for stop-words (so ``/feedback-loops`` is not caught by ``feed``), plus
    a small regex for genuine patterns (extensions, robots.txt, numbered pages). Replaces the old
    substring match, which matched a stop-word as a PREFIX of a real slug and silently dropped posts."""
    segs = [s.lower() for s in (path or "").split("/") if s]
    if any(s in _NAV_SEGMENTS for s in segs):
        return True
    return bool(_NAV_PATTERN_RE.search(path or ""))

SM_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


# ── Sitemap discovery ────────────────────────────────────────────────────────

def _fetch_sitemap_urls(blog_url: str) -> list[dict]:
    """Discover all blog post URLs from sitemap.xml.

    Returns list of {url, lastmod} dicts.
    Handles sitemap index files (nested sitemaps).
    """
    base = blog_url.rstrip("/")
    sitemap_candidates = [
        f"{base}/sitemap.xml",
        f"{base}/sitemap_index.xml",
        f"{base}/wp-sitemap.xml",
        f"{base}/post-sitemap.xml",
        f"{base}/sitemap-posts.xml",
    ]

    urls = []

    for sitemap_url in sitemap_candidates:
        try:
            resp = requests.get(sitemap_url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (compatible; OPYT/1.0)"
            })
            if resp.status_code != 200:
                continue
            if "<urlset" not in resp.text[:500] and "<sitemapindex" not in resp.text[:500]:
                continue

            root = ET.fromstring(resp.text)

            # Check if this is a sitemap index (links to other sitemaps)
            if root.tag.endswith("sitemapindex"):
                for sitemap in root.findall(".//sm:sitemap/sm:loc", SM_NS):
                    child_url = sitemap.text.strip()
                    try:
                        child_resp = requests.get(child_url, timeout=15, headers={
                            "User-Agent": "Mozilla/5.0 (compatible; OPYT/1.0)"
                        })
                        if child_resp.status_code == 200:
                            child_root = ET.fromstring(child_resp.text)
                            for url_elem in child_root.findall(".//sm:url", SM_NS):
                                loc = url_elem.find("sm:loc", SM_NS)
                                lastmod = url_elem.find("sm:lastmod", SM_NS)
                                if loc is not None and loc.text:
                                    urls.append({
                                        # resolve relative <loc> (e.g. guzey.com's "/balls/") to
                                        # absolute against the sitemap URL, else the fetch dies with
                                        # MissingSchema and the whole archive is unfetchable.
                                        "url": urljoin(child_url, loc.text.strip()),
                                        "lastmod": lastmod.text.strip() if lastmod is not None and lastmod.text else "",
                                    })
                        time.sleep(0.5)
                    except Exception:
                        continue
            else:
                # Direct URL set
                for url_elem in root.findall(".//sm:url", SM_NS):
                    loc = url_elem.find("sm:loc", SM_NS)
                    lastmod = url_elem.find("sm:lastmod", SM_NS)
                    if loc is not None and loc.text:
                        urls.append({
                            # resolve relative <loc> to absolute (see the sitemap-index branch above)
                            "url": urljoin(sitemap_url, loc.text.strip()),
                            "lastmod": lastmod.text.strip() if lastmod is not None and lastmod.text else "",
                        })

            if urls:
                log(f"  Found {len(urls)} URLs from {sitemap_url}")
                break

        except Exception as e:
            continue

    # Fallback 1: trafilatura sitemap/link discovery
    if not urls:
        log(f"  No sitemap found, trying trafilatura discovery...")
        try:
            discovered = trafilatura.sitemaps.sitemap_search(blog_url)
            if discovered:
                urls = [{"url": u, "lastmod": ""} for u in discovered]
                log(f"  Trafilatura found {len(urls)} URLs")
        except Exception:
            pass

    # Fallback 2: RSS/Atom feed discovery
    if not urls:
        import feedparser
        base = blog_url.rstrip("/")
        for feed_path in ["/feed.xml", "/feed", "/atom.xml", "/rss.xml", "/rss", "/index.xml"]:
            feed_url = f"{base}{feed_path}"
            try:
                feed = feedparser.parse(feed_url)
                if feed.entries:
                    for entry in feed.entries:
                        link = entry.get("link", "")
                        if link:
                            date = ""
                            if entry.get("published_parsed"):
                                import time as _time
                                date = _time.strftime("%Y-%m-%d", entry.published_parsed)
                            urls.append({"url": link, "lastmod": date})
                    log(f"  RSS feed found {len(urls)} URLs via {feed_path}")
                    break
            except Exception:
                continue

    # Fallback 3: HTML link crawling (scrape homepage for internal links)
    if not urls:
        log(f"  No sitemap or RSS, trying HTML link crawling...")
        try:
            resp = requests.get(blog_url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            if resp.status_code == 200:
                base_domain = urlparse(blog_url).netloc.lower()
                # Reuse the shared scraper (captures anchors + ALL domains, urljoin-resolved),
                # then keep this fallback's original SAME-DOMAIN-only behavior. Following external
                # links is the hub/union layer's job (link_discovery), not this last-resort crawl.
                found_links = [
                    link["url"] for link in extract_links(resp.text, blog_url)
                    if urlparse(link["url"]).netloc.lower() == base_domain
                ]
                if found_links:
                    urls = [{"url": u, "lastmod": ""} for u in found_links]
                    log(f"  HTML crawl found {len(urls)} internal links")
        except Exception as e:
            log(f"  [warn] HTML crawl failed: {e}")

    # Filter out non-blog-post URLs
    filtered = []
    for entry in urls:
        url = entry["url"]
        path = urlparse(url).path
        # Skip the root/homepage
        if path in ("", "/", "/index.html"):
            continue
        # Skip known non-post nav / feed / admin paths (whole-segment match)
        if is_nav_path(path):
            continue
        filtered.append(entry)

    log(f"  After filtering: {len(filtered)} blog post URLs (from {len(urls)} total)")
    return filtered


# ── Link harvesting (shared scraper + hub crawl) ─────────────────────────────

class _LinkExtractor(HTMLParser):
    """Pull every ``<a href>`` off a page WITH its visible anchor text. Keeps ALL domains and
    resolves relative/protocol-relative hrefs against the page URL (``urljoin``) — so the hub
    layer sees the external and relative links the old same-domain-only crawl silently dropped.
    Anchor text is the accumulated inner text of the ``<a>`` (nested tags flow through
    ``handle_data``), which the LLM URL-triage uses as its main signal."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.links: list[dict] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        href = next((v for (k, v) in attrs if k == "href" and v), None)
        # Skip in-page anchors and non-navigational schemes (mailto/tel/js) up front.
        if not href or href.startswith("#") or href.startswith(("mailto:", "tel:", "javascript:")):
            return
        self._href = urljoin(self.base_url, href)
        self._text = []

    def handle_data(self, data):
        if self._href is not None and data.strip():
            self._text.append(data.strip())

    def handle_endtag(self, tag):
        if tag == "a" and self._href is not None:
            self.links.append({"url": self._href, "anchor": " ".join(self._text)[:200]})
            self._href = None
            self._text = []


def extract_links(html: str, base_url: str) -> list[dict]:
    """Every ``<a href>`` on a page as ``{url, anchor}`` — ALL domains, relative hrefs resolved
    against ``base_url``, de-duped on url (first anchor wins). The reusable scraper behind both
    the fallback-3 homepage crawl (which then filters to same-domain) and ``harvest_hub_links``.
    A malformed-HTML parse error must not sink discovery — it returns whatever it parsed."""
    lx = _LinkExtractor(base_url)
    try:
        lx.feed(html or "")
    except Exception:
        pass
    seen: set[str] = set()
    out: list[dict] = []
    for link in lx.links:
        if link["url"] in seen:
            continue
        seen.add(link["url"])
        out.append(link)
    return out


# Index pages to probe beyond the homepage — the common "list of posts" landing paths. A hub is
# any of these that returns 200; its links are candidate posts (the union layer subtracts the hub
# pages themselves so /blog is never mistaken for an article).
HUB_INDEX_PATHS = ("/archive", "/blog", "/posts", "/writing", "/articles", "/essays", "/notes")


def harvest_links_from(pages: list[str]) -> list[dict]:
    """Fetch each URL in ``pages`` and harvest every ``<a href>`` + anchor off it, tagged with
    ``via`` = the page it was found on. De-duped on url across all pages (first page wins, so an
    earlier page is preferred as the provenance source). Keeps ALL domains — the ownership guard in
    the discovery layer decides what to attribute.

    The shared fetch/extract mechanism behind BOTH ``harvest_hub_links`` (homepage + probed index
    paths) and the discovery layer's depth-1 SECOND-LEVEL crawl (``link_discovery`` hands it the
    OWNED, index-shaped pages it found — e.g. a ``/writings`` index whose ``/writings/{slug}`` essays
    the sitemap omits). FAIL-SAFE: a page that won't fetch is skipped; a total failure returns
    whatever was harvested so far, never raises out."""
    seen: set[str] = set()
    out: list[dict] = []
    for page in pages:
        try:
            resp = requests.get(page, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            if resp.status_code != 200 or not resp.text:
                continue
            for link in extract_links(resp.text, page):
                if link["url"] in seen:
                    continue
                seen.add(link["url"])
                out.append({**link, "via": page})
        except Exception:
            continue
    return out


def harvest_hub_links(base: str) -> list[dict]:
    """Scrape ``<a href>`` + anchor text off the homepage AND each probed index page
    (``HUB_INDEX_PATHS``) that returns 200 — the level-1 recall arm that catches posts featured on a
    hub but ABSENT from the sitemap (marker-less internal posts, the author's own work hosted
    elsewhere). Thin wrapper over ``harvest_links_from`` for the fixed homepage+probe page list;
    returns the same ``{url, anchor, via}`` dicts, de-duped across all hub pages (homepage wins as
    provenance). FAIL-SAFE via ``harvest_links_from``."""
    base = (base or "").rstrip("/")
    if not base:
        return []
    return harvest_links_from([base] + [f"{base}{p}" for p in HUB_INDEX_PATHS])


# ── Content extraction ───────────────────────────────────────────────────────

# ── Rung 0 of the date cascade: the publisher's OWN explicit assertion ────────────────────────
# Keys that mean "this is when the thing was PUBLISHED". Deliberately published-only — no
# `article:modified_time`, no `dateModified`, because a 2019 essay re-touched in 2026 must read
# 2019. Matched case-insensitively against a meta tag's `property`/`name`, and against JSON-LD keys.
_PUBLISHED_KEYS = frozenset({
    "article:published_time", "article:published", "og:article:published_time",
    "datepublished", "publish-date", "publication-date", "pubdate",
    "parsely-pub-date", "sailthru.date", "dc.date.issued",
})
_META_TAG_RE = re.compile(r"<meta\s[^>]*>", re.I)
_ATTR_RE = re.compile(r"""(\w[\w.:-]*)\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""")
_LD_JSON_RE = re.compile(
    r"""<script[^>]*type\s*=\s*["']application/ld\+json["'][^>]*>(.*?)</script>""", re.I | re.S)
_TIME_TAG_RE = re.compile(r"<time\s[^>]*>", re.I)
_ISO_DAY_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def _attrs(tag_text: str) -> dict:
    """`<meta property="x" content="y">` → `{"property": "x", "content": "y"}`, lowercased keys,
    quotes stripped. Order-agnostic, because half the web writes `content` before `property`."""
    return {k.lower(): v.strip("\"'") for k, v in _ATTR_RE.findall(tag_text)}


def _as_day(value) -> str:
    """An asserted date value → ``YYYY-MM-DD``, or ``""`` if it is not a real, non-future day.

    Accepts the whole ISO family a publisher might emit — a bare ``2025-04-21`` and a full
    ``2025-04-21T00:00:00.000Z`` both reduce to the same day. Rejects a FUTURE date: a publish
    date that has not happened yet is a templating bug, not a publication."""
    from datetime import date

    m = _ISO_DAY_RE.match(str(value or "").strip())
    if not m:
        return ""
    try:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:                      # 2025-13-45 and friends
        return ""
    return "" if d > date.today() else d.isoformat()


def _ld_published(node, out: list) -> None:
    """Collect every ``datePublished`` anywhere in a parsed JSON-LD tree. Recursive because
    schema.org nests: publishers routinely wrap the Article inside an ``@graph`` list."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k.lower() == "datepublished" and isinstance(v, (str, int)):
                out.append(v)
            else:
                _ld_published(v, out)
    elif isinstance(node, list):
        for item in node:
            _ld_published(item, out)


def _asserted_publish_dates(html: str) -> list:
    """Every explicit publish-date the page itself declares — OpenGraph/meta tags, JSON-LD
    ``datePublished``, and ``<time itemprop="datePublished">``. Never raises."""
    import json

    found: list = []
    try:
        for tag in _META_TAG_RE.findall(html or ""):
            a = _attrs(tag)
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key in _PUBLISHED_KEYS and a.get("content"):
                found.append(a["content"])
        for block in _LD_JSON_RE.findall(html or ""):
            try:
                _ld_published(json.loads(block.strip()), found)
            except Exception:               # one malformed block must not lose the others
                continue
        for tag in _TIME_TAG_RE.findall(html or ""):
            a = _attrs(tag)
            if (a.get("itemprop") or "").lower() in _PUBLISHED_KEYS and a.get("datetime"):
                found.append(a["datetime"])
    except Exception:
        return []
    return [d for d in (_as_day(v) for v in found) if d]


def _publish_date(html: str, url: str, meta=None) -> str:
    """Best-effort PUBLICATION (creation) date as ``YYYY-MM-DD``, or ``""``.

    We want the ORIGINAL publish date, not a later edit:

      0. the page's OWN asserted publish date (og/meta, JSON-LD, <time itemprop>)  ← authoritative
      1. ``htmldate.find_date(original_date=True, url=…)``                          ← inferred
      2. trafilatura's metadata date
      3. ``""``

    Rung 0 exists because htmldate can silently pick up the wrong date from the full document even
    when the page's own tag is correct.

    When a page asserts SEVERAL publish dates that disagree, the OLDEST wins — the same rule
    ``original_date=True`` encodes.

    Never raises — a date miss must not sink the fetch (fail-safe)."""
    asserted = _asserted_publish_dates(html)
    if asserted:
        return min(asserted)                # oldest = the original publication
    try:
        from htmldate import find_date
        d = find_date(html, original_date=True, url=url, outputformat="%Y-%m-%d")
        if d:
            return d[:10]
    except Exception:
        pass
    if meta is not None and getattr(meta, "date", None):
        return str(meta.date)[:10]
    return ""


def _fetch_article(url: str) -> dict | None:
    """Fetch a blog post and extract content + metadata with trafilatura.

    Carries the response HEADERS through on the returned dict — trafilatura only returns extracted
    text, so without this the one machine-readable challenge signal (`cf-mitigated`) was
    unavailable, and `classify_fetch` had to lean on body heuristics a long challenge page defeats.
"""
    try:
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        if resp.status_code != 200:
            return None
        html = resp.text
        resp_headers = dict(resp.headers)
    except Exception:
        return None

    content = trafilatura.extract(
        html,
        output_format="markdown",
        include_links=True,
        include_images=True,
        no_fallback=False,
    )

    if not content:
        return None

    # Title from trafilatura; date = the PUBLICATION (creation) date, preferred over 'modified'.
    meta = trafilatura.metadata.extract_metadata(html)
    title = (meta.title or "") if meta else ""
    date = _publish_date(html, url, meta)

    return {
        "url": url,
        "title": title,
        "date": date,
        # BYLINE trafilatura already extracted, carried instead of discarded: a hand-dumped article
        # has no owning Oracle to fall back to, so without this the display name is the bare host.
        # Free (metadata object already parsed); absent on most pages → "", never a guess.
        "author": (meta.author or "") if meta else "",
        "content": content,
        # Response headers, for `classify_fetch`'s explicit-marker check. Consumers that only
        # read url/title/date/content are unaffected; a caller that never sets it (a test double)
        # simply falls back to the body heuristic, which is the pre-existing behavior.
        "headers": resp_headers,
    }


# ── Markdown ─────────────────────────────────────────────────────────────────

def _article_to_markdown(article: dict, author: str, author_name: str) -> str:
    """Convert extracted article to markdown with frontmatter."""
    title = article.get("title", "Untitled")
    date_str = article.get("date", "unknown")
    url = article.get("url", "")
    content = article.get("content", "")

    fm = (
        f"---\n"
        f"source: blog\n"
        f'author: "{author}"\n'
        f'author_name: "{author_name}"\n'
        f"url: {url}\n"
        f"date: {date_str}\n"
        f"type: article\n"
        f"tags: []\n"
        f"---\n\n"
    )

    body = f"# {title}\n\n"
    body += f"{content}\n\n"
    body += f"---\n*Blog · [Original post]({url})*\n"

    return fm + body
