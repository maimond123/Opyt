"""Extract article content and publication-date evidence from one web page. `fetch_article` reads
the page and returns its extracted Markdown, title, byline, response headers, final URL,
publication day, and what the page declares itself to be (`declares`). The day comes from a cascade whose precedence is `publication_date`: the page's
own asserted publish date, then htmldate's inference, then trafilatura's metadata date.

No domain decisions here: the library decides what a fetched article becomes.
"""
from __future__ import annotations

import json
import re
from datetime import date

import requests
import trafilatura

from pipeline.host_access import request as host_request
from pipeline.platforms.browser import render


# ── Rung 0 of the date cascade: the publisher's own explicit assertion ────────────────────────
# Published-only keys (no `dateModified`): an old essay re-touched later must read its original
# date. Matched case-insensitively against meta `property`/`name` and JSON-LD keys.
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


def publication_day(value, today: date) -> str:
    """An asserted date value → ``YYYY-MM-DD``, or ``""`` if it is not a real day on or before
    ``today``.

    Accepts the whole ISO family a publisher might emit. Rejects a future date: that is a
    templating bug, not a publication."""
    m = _ISO_DAY_RE.match(str(value or "").strip())
    if not m:
        return ""
    try:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:                      # 2025-13-45 and friends
        return ""
    return "" if d > today else d.isoformat()


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
    today = date.today()
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
    return [d for d in (publication_day(v, today) for v in found) if d]


# ── What the page says it is ──────────────────────────────────────────────────────────────────
_ARTICLE_TYPES = frozenset({
    "article", "blogposting", "newsarticle", "report", "scholarlyarticle", "techarticle",
    "opinionnewsarticle", "analysisnewsarticle", "reportagenewsarticle", "reviewnewsarticle",
    "backgroundnewsarticle", "socialmediaposting", "posting",
})
_PUBLISHED_TIME_RE = re.compile(
    r"""(?:property|name)\s*=\s*["']article:published_time["']|itemprop\s*=\s*["']datePublished["']""",
    re.I)


def _ld_types(node, out: set) -> None:
    """Every JSON-LD ``@type`` anywhere in a parsed tree, lowercased."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "@type":
                out.update(t.lower() for t in ([v] if isinstance(v, str) else v)
                           if isinstance(t, str))
            else:
                _ld_types(v, out)
    elif isinstance(node, list):
        for item in node:
            _ld_types(item, out)


def declares(html: str) -> dict:
    """What the page says it is: `article` when its OpenGraph type or a JSON-LD type names an
    article, `published` when a meta tag or an itemprop states its publish time. A WebPage's
    JSON-LD `datePublished` does not count: a CMS stamps one on every page. Never raises."""
    article = False
    try:
        for tag in _META_TAG_RE.findall(html or ""):
            a = _attrs(tag)
            if (a.get("property") or a.get("name") or "").lower() == "og:type":
                article = article or (a.get("content") or "").strip().lower() == "article"
        types: set = set()
        for block in _LD_JSON_RE.findall(html or ""):
            try:
                _ld_types(json.loads(block.strip()), types)
            except Exception:               # one malformed block must not lose the others
                continue
        article = article or bool(types & _ARTICLE_TYPES)
        published = bool(_PUBLISHED_TIME_RE.search(html or ""))
    except Exception:
        return {"article": False, "published": False}
    return {"article": article, "published": published}


def _publish_date(html: str, url: str, meta=None) -> str:
    """Best-effort publication (creation) date as ``YYYY-MM-DD``, or ``""``, never a later edit:

      0. the page's own asserted publish date (og/meta, JSON-LD, <time itemprop>)  ← authoritative
      1. ``htmldate.find_date(original_date=True, url=…)``                          ← inferred
      2. trafilatura's metadata date
      3. ``""``

    Rung 0 exists because htmldate can pick up the wrong date even when the page's own tag is
    correct. Among several asserted dates the oldest wins. Never raises."""
    asserted = _asserted_publish_dates(html)
    # htmldate's full-document parse runs only when the page asserts nothing; it cannot win then.
    inferred = "" if asserted else _inferred_date(html, url)
    metadata = str(meta.date) if meta is not None and getattr(meta, "date", None) else ""
    return publication_date(asserted, inferred, metadata)


def _inferred_date(html: str, url: str) -> str:
    """htmldate's original-date guess for the page, or ``""``. Never raises."""
    try:
        # Imported at call time so a missing htmldate degrades to the metadata rung (pinned).
        from htmldate import find_date
        return find_date(html, original_date=True, url=url, outputformat="%Y-%m-%d") or ""
    except Exception:
        return ""


def publication_date(asserted: list[str], inferred: str, metadata: str) -> str:
    """The publication day from the cascade's three rungs, most authoritative first."""
    if asserted:
        return min(asserted)                # oldest = the original publication
    if inferred:
        return inferred[:10]
    if metadata:
        return metadata[:10]
    return ""


def fetch_article(url: str) -> dict | None:
    """Fetch a blog post and extract content + metadata with trafilatura.

    Carries the response headers through, for `classify_fetch`'s `cf-mitigated` challenge check.
"""
    try:
        resp = host_request(requests.get, url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        if resp.status_code != 200:
            return None
        resp_headers = dict(resp.headers)
        final_url = str(getattr(resp, "url", "") or url)
        html = render.filled(resp.text, final_url)
    except Exception:
        return None

    # A parser failure is an unreadable page, the same None as a failed fetch.
    try:
        content = trafilatura.extract(
            html,
            output_format="markdown",
            include_links=True,
            include_images=True,
            no_fallback=False,
        )
        if not content:
            return None
        meta = trafilatura.metadata.extract_metadata(html)
    except Exception:
        return None
    title = (meta.title or "") if meta else ""
    date = _publish_date(html, url, meta)

    return {
        "url": url,
        "title": title,
        "date": date,
        # The byline: a hand-dumped article has no owning Oracle to name it. "" when absent.
        "author": (meta.author or "") if meta else "",
        "content": content,
        "headers": resp_headers,
        # Where the request landed after redirects; tells a shortener apart from its article.
        "final_url": final_url,
        "declares": declares(html),
    }
