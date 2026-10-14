"""
pipeline/oracles/discovery/backlink.py — what one candidate source says about itself: whom it
links, where it redirects, whether it publishes, and which accounts it declares. Discovery's
precision step: a candidate counts when it links back to the Oracle.

| candidate                                   | read                                                    |
|---------------------------------------------|---------------------------------------------------------|
| a Substack, by subdomain or custom domain   | `/api/v1/publication/users/ranked` → `public_profile`: `twitterAccount`, `userLinks` |
| a GitHub owner                              | the user API: `twitter_username`, `blog`                 |
| any other site                              | its home page: `<a href>`, the URL it redirects to, an RSS/Atom `<link rel=alternate>`; a page served by Substack also gets the Substack read |

A fetch that fails gives `reached=False`, which `verdict` reads as unknown, never as "does not link
back". The parsers are pure and take what was fetched; `facts_for` is the only network entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from pipeline.host_access import request as host_request
from pipeline.platforms.browser import render
from pipeline.ingestion.utils import log
from pipeline.platforms.github import api as github_api
from pipeline.platforms.source_classify import classify_source
from pipeline.platforms.url_canon import canonical_identity
from pipeline.platforms.url_canon import parse_url

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OPYT/1.0)"}
_FEED_TYPES = ("application/rss+xml", "application/atom+xml", "application/feed+json")


@dataclass(frozen=True)
class Facts:
    key: str                                       # the candidate's canonical identity
    reached: bool = False                          # False: nothing could be read (unknown)
    links_to: frozenset = frozenset()              # canonical identities it links or names
    final_key: Optional[str] = None                # where its home redirected, if elsewhere
    has_feed: Optional[bool] = None                # publishes posts; None = could not tell
    declared: tuple = field(default_factory=tuple)  # accounts it declares as its own (one hop)


def _ids(urls) -> frozenset:
    return frozenset(c for c in (canonical_identity(u) for u in urls if u) if c)


def _final(key: str, final_url: str | None) -> str | None:
    final = canonical_identity(final_url or "") if final_url else None
    return final if final and final != key else None


# ── parsers (pure) ───────────────────────────────────────────────────────────────

def facts_from_substack_profile(key: str, data: dict, final_url: str | None = None) -> Facts:
    """A Substack author's `public_profile` → facts. The X account and every `userLinks` entry
    are both what it links and what it declares."""
    tw = (data or {}).get("twitterAccount") or {}
    urls = [f"https://x.com/{tw['screen_name']}"] if tw.get("screen_name") else []
    urls += [(link or {}).get("url") for link in (data or {}).get("userLinks") or []]
    urls = [u for u in urls if u]
    return Facts(key, True, _ids(urls), _final(key, final_url), True, tuple(urls))


def facts_from_github_user(key: str, user: dict) -> Facts:
    """A GitHub user record → facts. Its X username and `blog` field are what it declares; an
    account with public repositories has something to pull."""
    urls = []
    if (user or {}).get("twitter_username"):
        urls.append(f"https://x.com/{user['twitter_username']}")
    blog = ((user or {}).get("blog") or "").strip()
    if blog:
        urls.append(blog if blog.startswith("http") else f"https://{blog}")
    return Facts(key, True, _ids(urls), None, bool((user or {}).get("public_repos")), tuple(urls))


def facts_from_html(key: str, html: str, final_url: str | None = None) -> Facts:
    """A home page → facts: every absolute `<a href>` it carries, and whether it advertises a
    feed. A home page declares nothing for the one-hop rule: its links are only evidence.

    An advertised feed means it publishes; no advertised feed means unknown, not "publishes
    nothing". Measured: chrisbarber.co and sonnetx.github.io, both personal sites with
    writing, carry no feed tag, the same as the projects pavlovslist.com and slashquiz.org."""
    soup = BeautifulSoup(html or "", "html.parser")
    hrefs = [a["href"] for a in soup.find_all("a", href=True)
             if a["href"].lower().startswith(("http://", "https://"))]
    feed = any((link.get("type") or "").lower() in _FEED_TYPES
               for link in soup.find_all("link", href=True)
               if "alternate" in [r.lower() for r in (link.get("rel") or [])])
    return Facts(key, True, _ids(hrefs), _final(key, final_url), True if feed else None, ())


def served_by_substack(html: str) -> bool:
    """A custom-domain page Substack serves carries its CDN in the markup."""
    return "substackcdn.com" in (html or "")


# ── fetchers (doing) ─────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 15):
    """One GET, following redirects → the response whatever its status, or None when nothing
    answered (a network error or a timeout)."""
    try:
        return host_request(requests.get, url, timeout=timeout, allow_redirects=True,
                            headers=_HEADERS)
    except requests.RequestException:
        return None


def _ok(resp) -> bool:
    return resp is not None and resp.status_code == 200


def fetch_page(url: str, timeout: int = 12) -> tuple[str | None, str | None, bool]:
    """A page's `(html, final_url, answered)`: the html only on a 200; `answered` False when
    nothing answered at all."""
    resp = _get(url, timeout)
    if not _ok(resp):
        return None, None, resp is not None
    final = getattr(resp, "url", None) or url
    return render.filled(resp.text, final), final, True


def _substack_read(publication_url: str) -> tuple[dict | None, str | None, bool]:
    """`(owner's public_profile, the URL the publication's API answered from, answered)`."""
    parsed = parse_url(publication_url)
    if parsed is None:
        return None, None, False
    ranked = _get(f"https://{parsed.netloc}/api/v1/publication/users/ranked?public=true")
    try:
        users = ranked.json() if _ok(ranked) else []
    except ValueError:
        users = []
    handle = next((str(u["handle"]) for u in users or [] if (u or {}).get("handle")), None)
    if not handle:
        return None, None, ranked is not None
    resp = _get(f"https://substack.com/api/v1/user/{handle}/public_profile", timeout=20)
    try:
        data = resp.json() if _ok(resp) else None
    except ValueError:
        data = None
    answered = getattr(ranked, "url", None)
    final = f"https://{urlparse(answered).netloc}" if answered else None
    return (data if isinstance(data, dict) else None), final, resp is not None


def substack_profile(publication_url: str) -> tuple[dict | None, str | None]:
    """A publication's owner's `public_profile`, and the URL its API answered from (a subdomain
    that moved to a custom domain answers from there). `(None, None)` when it cannot be read.

    The top-ranked author is taken as the owner; `public_profile` is the public route, not the
    Cloudflare-hostile subscriber-lists API."""
    data, final, _answered = _substack_read(publication_url)
    return (data, final) if data else (None, None)


def facts_for(key: str, url: str) -> Facts:
    """Read one candidate. Never raises.

    | read                                            | facts                         |
    |-------------------------------------------------|-------------------------------|
    | answered 200                                    | what it says                  |
    | answered with a refusal or a miss (403, 404, …) | reached, links nothing        |
    | nothing answered (network error, timeout), or GitHub rate-limited | unknown     |

    A refusal is an answer: news sites refuse a bot (nytimes, bloomberg, ft),
    and reading them as unknown would ask the user about every newspaper an Oracle cites."""
    pl = classify_source(url)
    try:
        if pl is not None and pl.type == "github" and pl.handle:
            user = github_api.fetch_user(pl.handle)       # None: no such account (or a failed read)
            return facts_from_github_user(key, user) if user else Facts(key, True)
        if pl is not None and pl.type == "substack":
            data, final, answered = _substack_read(url)
            return (facts_from_substack_profile(key, data, final) if data else
                    Facts(key, answered))
        html, final, answered = fetch_page(url)
        if html is None:
            return Facts(key, answered)
        page = facts_from_html(key, html, final)
        if not served_by_substack(html):
            return page
        data, _ = substack_profile(final or url)
        if not data:
            return page
        sub = facts_from_substack_profile(key, data, final)
        return Facts(key, True, page.links_to | sub.links_to, page.final_key, True, sub.declared)
    except github_api.GitHubRateLimited:
        log(f"  [backlink] GitHub rate-limited; {key} is unknown this pass")
        return Facts(key)
    except Exception as e:                      # one candidate never ends the pass
        log(f"  [backlink] {key}: {type(e).__name__}: {e}")
        return Facts(key)

