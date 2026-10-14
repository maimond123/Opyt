"""Discover article URLs on a site: its sitemap (index files included), then its RSS/Atom feed,
then a same-domain crawl of its homepage; and harvest every link off a hub's index pages for the
discovery layer to attribute. Every read is fail-safe: a page that will not fetch is skipped.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse

import feedparser
import requests

from pipeline.host_access import feed_input, request as host_request
from pipeline.ingestion.utils import log
from pipeline.platforms.web.links import extract_links

# Non-post path segments: nav / feed / admin / commerce. Whole-segment membership, not a substring
# match, so a slug like "feedback-loops" is not caught by "feed". Patterns use the regex below.
_NAV_SEGMENTS = frozenset({
    "about", "contact", "privacy", "terms", "legal", "tag", "tags", "category", "author",
    "authors", "feed", "feeds", "rss", "atom", "login", "register", "search", "cart",
    "checkout", "sitemap", "wp-content", "wp-admin", "wp-includes",
})
# Pattern-shaped rejects: asset/data extensions at a segment end, robots.txt, and WordPress
# numbered pagination (/page/2). Anchored so ".js" can't eat "node.js-guide".
_NAV_PATTERN_RE = re.compile(r"\.(?:xml|json|css|js)(?:[?#]|$)|/robots\.txt(?:[?#]|$)|/page/\d", re.I)


def is_nav_path(path: str) -> bool:
    """True iff a URL path is site navigation / a feed / admin / an asset — not a readable post."""
    segs = [s.lower() for s in (path or "").split("/") if s]
    if any(s in _NAV_SEGMENTS for s in segs):
        return True
    return bool(_NAV_PATTERN_RE.search(path or ""))

SM_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# Two identities on purpose: sitemaps are fetched as a self-identified crawler, while the
# homepage crawl and hub pages are fetched as a browser.
_SITEMAP_UA = "Mozilla/5.0 (compatible; OPYT/1.0)"
_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


# ── Sitemap discovery ────────────────────────────────────────────────────────

def _sitemap_entries(root: ET.Element, sitemap_url: str) -> list[dict]:
    """The {url, lastmod} entries of one parsed `<urlset>`. A relative `<loc>` (e.g. guzey.com's
    "/balls/") resolves against the sitemap URL, else the fetch dies with MissingSchema and the
    whole archive is unfetchable."""
    out = []
    for url_elem in root.findall(".//sm:url", SM_NS):
        loc = url_elem.find("sm:loc", SM_NS)
        lastmod = url_elem.find("sm:lastmod", SM_NS)
        if loc is not None and loc.text:
            out.append({
                "url": urljoin(sitemap_url, loc.text.strip()),
                "lastmod": lastmod.text.strip() if lastmod is not None and lastmod.text else "",
            })
    return out


def fetch_sitemap_urls(blog_url: str) -> list[dict]:
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
            resp = host_request(requests.get, sitemap_url, timeout=15,
                                headers={"User-Agent": _SITEMAP_UA})
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
                        child_resp = host_request(requests.get, child_url, timeout=15,
                                                  headers={"User-Agent": _SITEMAP_UA})
                        if child_resp.status_code == 200:
                            urls.extend(_sitemap_entries(ET.fromstring(child_resp.text),
                                                         child_url))
                        time.sleep(0.5)
                    except Exception:
                        continue
            else:
                urls.extend(_sitemap_entries(root, sitemap_url))   # direct URL set

            if urls:
                log(f"  Found {len(urls)} URLs from {sitemap_url}")
                break

        except Exception:
            continue

    # Fallback: RSS/Atom feed discovery
    if not urls:
        base = blog_url.rstrip("/")
        for feed_path in ["/feed.xml", "/feed", "/atom.xml", "/rss.xml", "/rss", "/index.xml"]:
            feed_url = f"{base}{feed_path}"
            try:
                feed = feedparser.parse(feed_input(feed_url))
                if feed.entries:
                    for entry in feed.entries:
                        link = entry.get("link", "")
                        if link:
                            date = ""
                            if entry.get("published_parsed"):
                                date = time.strftime("%Y-%m-%d", entry.published_parsed)
                            urls.append({"url": link, "lastmod": date})
                    log(f"  RSS feed found {len(urls)} URLs via {feed_path}")
                    break
            except Exception:
                continue

    # Fallback 2: HTML link crawling (scrape homepage for internal links)
    if not urls:
        log(f"  No sitemap or RSS, trying HTML link crawling...")
        try:
            resp = host_request(requests.get, blog_url, timeout=15,
                                headers={"User-Agent": _BROWSER_UA})
            if resp.status_code == 200:
                base_domain = urlparse(blog_url).netloc.lower()
                # Same-domain only: following external links is link_discovery's job.
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


# ── Link harvesting (hub crawl) ──────────────────────────────────────────────

# Index pages to probe beyond the homepage — the common "list of posts" landing paths. A hub is
# any of these that returns 200; its links are candidate posts (the union layer subtracts the hub
# pages themselves so /blog is never mistaken for an article).
HUB_INDEX_PATHS = ("/archive", "/blog", "/posts", "/writing", "/articles", "/essays", "/notes")


def harvest_links_from(pages: list[str]) -> list[dict]:
    """Fetch each URL in ``pages`` and harvest every ``<a href>`` + anchor off it, tagged with
    ``via`` = the page it was found on. De-duped on url across all pages (first page wins). Keeps
    all domains; the discovery layer decides what to attribute. A page that won't fetch is
    skipped; never raises."""
    seen: set[str] = set()
    out: list[dict] = []
    for page in pages:
        try:
            resp = host_request(requests.get, page, timeout=15,
                                headers={"User-Agent": _BROWSER_UA})
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
    """``harvest_links_from`` over the homepage and each ``HUB_INDEX_PATHS`` page, to catch posts
    featured on a hub but absent from the sitemap (homepage wins as provenance)."""
    base = (base or "").rstrip("/")
    if not base:
        return []
    return harvest_links_from([base] + [f"{base}{p}" for p in HUB_INDEX_PATHS])
