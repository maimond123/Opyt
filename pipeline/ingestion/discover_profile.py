"""
pipeline/ingestion/discover_profile.py
Discover all content sources for a credible person from their X handle.

Runs 4 probes to find blog, Substack, GitHub, academic papers:
  1. Twitter bio extraction (free, on the user's own x.com session)
  2. Substack RSS probe (free HTTP check)
  3. GitHub user lookup (free API)
  4. Semantic Scholar author search (free API)
Probe 5 (open-web search for a blog/YouTube/podcasts) does not exist in any form — see

No outbound work here is PAID at all since 2026-08-30 — the X profile probe moved to the user's
own session. Trust is computed entirely from
free signals (bio-declared links, landing-page fetches, propagation).

Results cached in state/discovered_profiles.json for reuse.

Usage:
  python pipeline/ingestion/discover_profile.py --username someuser
  python pipeline/ingestion/discover_profile.py --username someuser --skip-web-search
  python pipeline/ingestion/discover_profile.py --username someuser --ingest --since 2025-06-01
  python pipeline/ingestion/discover_profile.py --username someuser --ingest --dry-run
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pipeline.timeparse import utc_now
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from opyt_core.paths import opyt_path

# Load .env — check standard locations
for _env_candidate in [
    Path(__file__).parent.parent.parent / ".env",
    opyt_path(".env"),
]:
    if _env_candidate.exists():
        load_dotenv(_env_candidate)
        break

from pipeline.config import state_paths, StatePaths
from pipeline.ingestion.utils import log
from pipeline.ingestion.trust_graph import propagate
from pipeline.ingestion.trust_types import Edge, TrustEvidence
from pipeline.ingestion.trust_edges import (
    PROFILE_SCOPE,
    edges_from_github,
    fetch_landing_edges,
    fetch_profile_links,
    fetch_substack_twitter_handle,
)
from pipeline.ingestion.source_classify import classify_source
from pipeline.ingestion.handle_match import match_known_handle, normalize_handle
from pipeline.ingestion.url_canon import canonical_identity

S2_BASE = "https://api.semanticscholar.org/graph/v1"


def _s2_headers() -> dict:
    """S2 headers incl. `S2_API_KEY` when set. Imported lazily and called per-request so a key
    written to ~/.opyt/.env mid-session takes effect without a restart."""
    from pipeline.credentials import s2_headers
    return s2_headers()


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DiscoveredSource:
    source_type: str                           # blog, substack, github, scholar, youtube, podcast
    url: str
    feed_url: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    confidence: str = "high"                   # high | medium | low
    trust: Optional[TrustEvidence] = None      # owner-validation verdict (asdict-serializable)
    evidence_edges: list = field(default_factory=list)  # supporting edges for the verdict


@dataclass
class DiscoveredProfile:
    username: str
    display_name: str = ""
    bio: str = ""
    website: str = ""
    sources: list = field(default_factory=list)
    discovered_at: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Normalize a URL for deduplication."""
    if not url:
        return url
    url = url.rstrip("/")
    if url.startswith("http://"):
        url = "https://" + url[7:]
    # Fix double slashes in path (but not after protocol)
    parsed = urlparse(url)
    if "//" in parsed.path:
        clean_path = parsed.path.replace("//", "/")
        url = parsed._replace(path=clean_path).geturl()
    return url


def _load_discovered(state_file: Path | None = None) -> dict:
    sf = state_file or state_paths().state_file("discovered_profiles")
    if sf.exists():
        return json.loads(sf.read_text())
    return {}


def _save_discovered(profiles: dict, state_file: Path | None = None) -> None:
    sf = state_file or state_paths().state_file("discovered_profiles")
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps(profiles, indent=2, default=str))


def _detect_rss_feed(blog_url: str) -> Optional[str]:
    """Try common RSS feed paths for a blog URL. Returns feed URL or None."""
    parsed = urlparse(blog_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    candidates = [
        f"{blog_url}/feed",
        f"{blog_url}/feed.xml",
        f"{blog_url}/rss",
        f"{blog_url}/atom.xml",
        f"{blog_url}/rss.xml",
        f"{base}/feed",
        f"{base}/feed.xml",
        f"{base}/atom.xml",
    ]
    # Dedupe while preserving order
    seen = set()
    unique = []
    for c in candidates:
        c = _normalize_url(c)
        if c not in seen:
            seen.add(c)
            unique.append(c)

    for url in unique:
        try:
            resp = requests.get(url, timeout=10, allow_redirects=True, headers={
                "User-Agent": "Mozilla/5.0 (compatible; OPYT/1.0)"
            })
            if resp.status_code == 200:
                text = resp.text[:500]
                if "<rss" in text or "<feed" in text or "<channel" in text:
                    log(f"    Found RSS feed: {url}")
                    return url
        except requests.RequestException:
            continue
    return None


def _classify_url(url: str) -> Optional[str]:
    """Classify a URL into a source type based on domain."""
    if not url:
        return None
    domain = urlparse(url).netloc.lower()
    host = domain[4:] if domain.startswith("www.") else domain
    # Exact-host match (not substring) so a site like "max.com" can't false-positive on "x.com".
    if host in ("x.com", "twitter.com", "mobile.twitter.com"):
        return "x"
    if "substack.com" in domain:
        return "substack"
    if "github.com" in domain:
        return "github"
    if "youtube.com" in domain or "youtu.be" in domain:
        return "youtube"
    if "scholar.google" in domain:
        return "scholar"
    if "linkedin.com" in domain:
        return "linkedin"
    if "spotify.com" in domain:
        return "podcast"
    if "podcasts.apple.com" in domain or "apple.co" in domain:
        return "podcast"
    if "medium.com" in domain:
        return "blog"
    if "mirror.xyz" in domain:
        return "blog"
    return "blog"  # default: treat unknown URLs as blogs


# ── Probes ───────────────────────────────────────────────────────────────────

def _probe_twitter_bio(username: str) -> tuple[dict, list]:
    """Probe 1: bio, website and the other homes an X profile links to.

    Reads the free cookie-scrape `UserByScreenName`. This used to carry its own `requests.get`
    against twitterapi.io — which is why it was easy to miss when the provider was removed:
    nothing here imported the client module, so grepping for `x_render` did not find it.
    t.co expansion now happens once inside `fetch_user_profile`, beside the entities block.

    The x.com self-link SKIP below stays here. It is a consumer policy and the other consumer of
    the same call (`oracles._fetch_x_identity`) disagrees — it blanks the field instead."""
    from pipeline.ingestion import x_graphql_core as core

    log(f"  [probe] Twitter bio for @{username}")
    try:
        cookies = core.read_x_cookies()
        headers = core.auth_headers(cookies, f"https://x.com/{username}")
        data = core.fetch_user_profile(cookies, headers, username)
        if not data:
            log(f"    [error] No profile data returned for @{username}")
            return {}, []
    except Exception as e:
        log(f"    [error] Twitter bio probe failed: {e}")
        return {}, []

    profile_info = {
        "display_name": data["display_name"],
        "bio": data["bio"],
        "website": data["website"],
    }

    sources = []

    website = data["website"]
    if website:
        # Skip X/Twitter self-links
        website_domain = urlparse(website).netloc.lower()
        if "twitter.com" not in website_domain and "x.com" not in website_domain:
            stype = _classify_url(website)
            feed_url = None
            if stype == "substack":
                parsed = urlparse(website)
                feed_url = f"{parsed.scheme}://{parsed.netloc}/feed"
            elif stype == "blog":
                feed_url = _detect_rss_feed(website)
            sources.append(DiscoveredSource(
                source_type=stype,
                url=_normalize_url(website),
                feed_url=feed_url,
            ))

    # The OTHER homes the bio links to — a Substack, a podcast, a personal site. Already expanded
    # past t.co by `fetch_user_profile`.
    for expanded in data["bio_urls"]:
        if not expanded:
            continue
        expanded = _normalize_url(expanded)
        # Skip X/Twitter self-links
        parsed = urlparse(expanded)
        if parsed.netloc and ("twitter.com" in parsed.netloc or "x.com" in parsed.netloc):
            continue
        stype = _classify_url(expanded)
        feed_url = None
        if stype == "substack":
            parsed = urlparse(expanded)
            feed_url = f"{parsed.scheme}://{parsed.netloc}/feed"
        elif stype == "blog":
            feed_url = _detect_rss_feed(expanded)
        sources.append(DiscoveredSource(
            source_type=stype,
            url=expanded,
            feed_url=feed_url,
        ))

    log(f"    Found {len(sources)} URLs from Twitter bio")
    return profile_info, sources


def _probe_substack(username: str) -> list:
    """Probe 2: Check if {username}.substack.com exists."""
    log(f"  [probe] Substack for {username}")
    url = f"https://{username.lower()}.substack.com/feed"
    try:
        resp = requests.get(url, timeout=10, allow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (compatible; OPYT/1.0)"
        })
        if resp.status_code == 200:
            text = resp.text[:500]
            if "<rss" in text or "<feed" in text or "<channel" in text:
                canonical = f"https://{username.lower()}.substack.com"
                log(f"    Found Substack: {canonical}")
                return [DiscoveredSource(
                    source_type="substack",
                    url=canonical,
                    feed_url=url,
                )]
    except requests.RequestException:
        pass
    log(f"    No Substack found")
    return []


def _resolve_substack_user_slug(seed: str) -> Optional[str]:
    """A Substack seed → the user slug the public_profile API keys on.

    A bare label is already a user slug (used as-is). A host/URL (custom-domain mint) is
    resolved to its primary author's slug via the publication's ranked-users API. Returns
    None on any miss — caller skips gracefully.
    """
    seed = (seed or "").strip()
    if not seed:
        return None
    if "." not in seed and "/" not in seed:
        return seed                                      # already a user slug
    host = urlparse(seed if "://" in seed else f"https://{seed}").netloc or seed
    host = host.split("/")[0]
    try:
        resp = requests.get(
            f"https://{host}/api/v1/publication/users/ranked?public=true",
            headers={"User-Agent": "Mozilla/5.0 (compatible; OPYT/1.0)"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        users = resp.json() or []
    except Exception as e:
        log(f"    [error] Substack slug resolution failed for {host}: {e}")
        return None
    for u in users:                                      # top-ranked author = the owner
        h = (u or {}).get("handle")
        if h:
            return str(h)
    return None


def _probe_substack_profile(seed: str) -> tuple[dict, list, list]:
    """ROOT probe for a Substack-seeded person — the non-X analog of _probe_twitter_bio.

    GETs the public profile (`/api/v1/user/{handle}/public_profile`, not the
    Cloudflare-hostile subscriber-lists API). `userLinks` are the person's own typed,
    declared links (site / X / YouTube / …).

    Returns (profile_info, sources, identity_targets):
      profile_info:     {display_name, bio, website, root_url}
      sources:          the ROOT Substack itself + a DiscoveredSource per external userLink.
      identity_targets: [(canonical_id, verified)] — declared accounts that become typed
                        identity edges from the Substack root (Rule 5, T1). `verified` =
                        OAuth-connected account (is_connected_account).
    """
    log(f"  [probe] Substack profile for {seed}")
    profile_info = {"display_name": "", "bio": "", "website": "", "root_url": ""}
    sources: list = []
    identity_targets: list = []
    handle = _resolve_substack_user_slug(seed)
    if not handle:
        log(f"    Could not resolve a Substack user slug for {seed!r}")
        return profile_info, sources, identity_targets
    try:
        resp = requests.get(
            f"https://substack.com/api/v1/user/{handle}/public_profile",
            headers={"User-Agent": "Mozilla/5.0 (compatible; OPYT/1.0)"},
            timeout=20,
        )
        if resp.status_code != 200:
            log(f"    No Substack profile for {handle} (status {resp.status_code})")
            return profile_info, sources, identity_targets
        data = resp.json()
    except Exception as e:
        log(f"    [error] Substack profile probe failed: {e}")
        return profile_info, sources, identity_targets

    tw = data.get("twitterAccount") or {}
    profile_info["display_name"] = data.get("name") or tw.get("display_name") or handle
    profile_info["bio"] = data.get("bio") or ""
    # Use primaryPublication's {subdomain}.substack.com as the root (not the user-level
    # subdomainUrl, which is often null, or "substack.com/@{handle}", which collides across
    # users) — this canonicalizes to a unique node matching the Oracle's substack:{subdomain} id.
    sub = (data.get("primaryPublication") or {}).get("subdomain") or handle
    root_url = f"https://{sub}.substack.com"
    profile_info["root_url"] = root_url

    seen: set = set()

    def _add(url: str, verified: bool) -> None:
        url = _normalize_url(url)
        cid = canonical_identity(url)
        if not cid or cid in seen:
            return
        seen.add(cid)
        identity_targets.append((cid, verified))
        stype = _classify_url(url)
        feed_url = None
        if stype == "substack":
            parsed = urlparse(url)
            feed_url = f"{parsed.scheme}://{parsed.netloc}/feed"
        elif stype == "blog":
            feed_url = _detect_rss_feed(url)
        if not profile_info["website"] and stype == "blog":
            profile_info["website"] = url
        sources.append(DiscoveredSource(source_type=stype, url=url, feed_url=feed_url))

    # Add the root Substack as a source FIRST — its canonical id == root_id, so propagate
    # Rule-1-trusts it and it routes through the normal Substack footprint adapter.
    root_cid = canonical_identity(root_url)
    if root_cid:
        seen.add(root_cid)
        parsed = urlparse(root_url)
        sources.append(DiscoveredSource(
            source_type="substack", url=_normalize_url(root_url),
            feed_url=f"{parsed.scheme}://{parsed.netloc}/feed"))

    # A connected twitterAccount is a platform-VERIFIED cross-link (strongest).
    if tw.get("screen_name"):
        _add(f"https://x.com/{tw['screen_name']}",
             verified=bool(tw.get("is_connected_account") or tw.get("verified")))
    for link in data.get("userLinks") or []:
        url = (link or {}).get("url")
        if url:
            _add(url, verified=bool(link.get("is_connected_account")))

    log(f"    Substack profile: {len(identity_targets)} declared links for {handle}")
    return profile_info, sources, identity_targets


# Anchor links on a blog home that count as a declared own account (a typed identity edge → Rule 5).
# Everything else (`_classify_url` defaults unknown hosts to "blog") is ignored as a random
# outbound link, not an identity claim.
_BLOG_IDENTITY_TYPES = frozenset({"x", "github", "substack", "linkedin", "youtube"})


def _probe_blog_profile(seed: str) -> tuple[dict, list, list]:
    """ROOT probe for a blog-seeded person — the blog analog of `_probe_substack_profile`.

    `seed` is the blog home URL. GETs the home HTML: `<title>` is the display name, and any
    anchor to a known identity platform (x / github / substack / linkedin / youtube) is a
    self-declared own account → a typed IDENTITY edge (Rule 5, T1). The blog itself is added
    first as the self-rooted source. A JS-rendered home yields no server-side anchors — the
    blog still roots + ingests, just with no cross-platform links surfaced.

    Returns (profile_info, sources, identity_targets), platform-agnostic across x/substack/blog
    roots — the shape `discover_profile`'s dispatch feeds `_compute_trust`."""
    seed_url = _normalize_url(seed if "://" in seed else f"https://{seed}")
    host = urlparse(seed_url).netloc
    profile_info = {"display_name": "", "bio": "", "website": seed_url, "root_url": seed_url}
    sources: list = []
    identity_targets: list = []
    log(f"  [probe] Blog profile for {seed_url}")

    html = ""
    try:
        resp = requests.get(seed_url, timeout=20, allow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; OPYT/1.0)"})
        if resp.status_code == 200:
            html = resp.text or ""
        else:
            log(f"    Blog home returned status {resp.status_code} for {seed_url}")
    except Exception as e:
        log(f"    [error] Blog home probe failed: {e}")

    # display_name from <title>, trimming a trailing " | Site" tail; falls back to the host.
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        title = re.split(r"\s+[|–—-]\s+", title)[0].strip()
        profile_info["display_name"] = title or host
    else:
        profile_info["display_name"] = host

    # Add the blog itself first, as the self-rooted source.
    root_cid = canonical_identity(seed_url)
    seen: set = {root_cid} if root_cid else set()
    sources.append(DiscoveredSource(source_type="blog", url=seed_url,
                                    feed_url=_detect_rss_feed(seed_url)))

    # Outbound anchors to KNOWN identity platforms → declared OWN accounts (Rule 5).
    for href in set(re.findall(r'href=[\'"](https?://[^\'"]+)[\'"]', html, re.I)):
        url = _normalize_url(href)
        cid = canonical_identity(url)
        if not cid or cid in seen:
            continue
        stype = _classify_url(url)
        if stype not in _BLOG_IDENTITY_TYPES:            # skip the page's random outbound links
            continue
        seen.add(cid)
        identity_targets.append((cid, False))            # declared, not platform-verified
        feed_url = None
        if stype == "substack":
            parsed = urlparse(url)
            feed_url = f"{parsed.scheme}://{parsed.netloc}/feed"
        sources.append(DiscoveredSource(source_type=stype, url=url, feed_url=feed_url))

    log(f"    Blog profile: {len(identity_targets)} declared links for {host}")
    return profile_info, sources, identity_targets


def _probe_github(username: str) -> list:
    """Probe 3: Check GitHub for user profile and blog URL."""
    log(f"  [probe] GitHub for {username}")
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(
            f"https://api.github.com/users/{username}",
            headers=headers,
            timeout=15,
        )
        if resp.status_code == 404:
            log(f"    No GitHub user found")
            return []
        resp.raise_for_status()
        user = resp.json()
    except Exception as e:
        log(f"    [error] GitHub probe failed: {e}")
        return []

    sources = []

    # GitHub profile itself
    sources.append(DiscoveredSource(
        source_type="github",
        url=f"https://github.com/{user.get('login', username)}",
        metadata={
            "name": user.get("name"),
            "bio": user.get("bio"),
            "public_repos": user.get("public_repos", 0),
            "followers": user.get("followers", 0),
        },
    ))

    # Blog field — often a personal website
    blog = user.get("blog", "")
    if blog:
        if not blog.startswith("http"):
            blog = f"https://{blog}"
        blog = _normalize_url(blog)
        # Skip X/Twitter self-links
        blog_domain = urlparse(blog).netloc.lower()
        if "twitter.com" not in blog_domain and "x.com" not in blog_domain:
            stype = _classify_url(blog)
            feed_url = None
            if stype == "substack":
                parsed = urlparse(blog)
                feed_url = f"{parsed.scheme}://{parsed.netloc}/feed"
            elif stype == "blog":
                feed_url = _detect_rss_feed(blog)
            sources.append(DiscoveredSource(
                source_type=stype,
                url=blog,
                feed_url=feed_url,
                metadata={"discovered_via": "github_blog_field"},
            ))

    log(f"    Found GitHub profile" + (f" + blog: {blog}" if blog else ""))
    return sources


def _probe_semantic_scholar(display_name: str) -> list:
    """Probe 4: Search Semantic Scholar for author by display name.

    Semantic Scholar has a 100 req/5min rate limit. Retries on 429 with backoff.
    """
    if not display_name:
        return []
    log(f"  [probe] Semantic Scholar for '{display_name}'")

    authors = None
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{S2_BASE}/author/search",
                params={
                    "query": display_name,
                    "fields": "name,url,paperCount,citationCount,hIndex",
                    "limit": 5,
                },
                headers=_s2_headers(),     # S2_API_KEY when set — raises the rate limit
                timeout=15,
            )
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                log(f"    [rate-limit] 429 from Semantic Scholar, retrying in {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            authors = resp.json().get("data", [])
            break
        except Exception as e:
            log(f"    [error] Semantic Scholar probe failed: {e}")
            return []

    if authors is None:
        log(f"    [error] Semantic Scholar rate limit exceeded after 3 retries")
        return []

    if not authors:
        log(f"    No authors found")
        return []

    # Pick the best match — prefer exact name match, then highest paper count
    best = None
    for a in authors:
        if a.get("name", "").lower() == display_name.lower():
            best = a
            break
    if not best:
        best = max(authors, key=lambda a: a.get("paperCount", 0))

    confidence = "high" if best.get("name", "").lower() == display_name.lower() else "medium"

    log(f"    Found: {best.get('name')} — {best.get('paperCount', 0)} papers (confidence: {confidence})")
    return [DiscoveredSource(
        source_type="scholar",
        url=best.get("url", f"https://www.semanticscholar.org/author/{best.get('authorId', '')}"),
        metadata={
            "author_id": best.get("authorId"),
            "name": best.get("name"),
            "paper_count": best.get("paperCount", 0),
            "citation_count": best.get("citationCount", 0),
            "h_index": best.get("hIndex"),
        },
        confidence=confidence,
    )]


def _sources_from_urls(urls) -> list:
    """Turn host-supplied URLs into typed CANDIDATE sources. Never raises.

    The host model does the FINDING (it has web search); the trust graph does the JUDGING — a
    host guess is a claim, never a trust root. `confidence="low"` and appended after every
    deterministic probe, so `_dedupe_sources` lets a probe-found source always win a collision.
    Fail-safe: unusable input (prose, mailto:, empty, None) is dropped silently.
    """
    out = []
    for raw in (urls or []):
        if not isinstance(raw, str) or not raw.strip():
            continue
        raw = raw.strip()
        # classify_source prepends "https://" to a scheme-less string, so it is NOT a URL
        # validator — this caller's input is generated text, so filter obvious non-URLs first.
        if any(c.isspace() for c in raw) or "." not in raw:
            continue
        pl = classify_source(raw)
        # is_profile filters ARTIFACTS (e.g. a /watch link) — sources are accounts/homes only.
        if not pl or not pl.is_profile:
            continue
        # X/GitHub are already covered by probes 1 and 3; papers aren't footprint.
        if pl.type in ("x", "github", "scholar", "orcid", "paper"):
            continue
        out.append(DiscoveredSource(
            source_type=pl.type,
            url=pl.url,
            confidence="low",
            metadata={"found_by": "host_web_search", "shape": pl.shape, "handle": pl.handle},
        ))
    return out


# Open-web discovery does not happen here — a known gap, not an oversight; see
# docs/plans/2026-08-16-cold-start-anchor-as-built.md for why it stays open.


def _dedupe_sources(sources: list) -> list:
    """Deduplicate sources by (source_type, normalized URL)."""
    seen = set()
    deduped = []
    for s in sources:
        key = (s.source_type, _normalize_url(s.url))
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped


# ── Trust cache: skip expensive re-discovery when the X profile is unchanged ──

TRUST_CACHE_TTL_DAYS = 30


def _x_snapshot_hash(display_name: str, bio_url_ids: list[str]) -> str:
    """Fingerprint the trust-relevant X profile state (display name + declared URLs) so a
    change forces re-discovery and an unchanged profile can reuse the cached verdict.
    """
    raw = (display_name or "") + "||" + "|".join(sorted(bio_url_ids))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _get_cached_trust(username: str, snapshot: str, cfg) -> Optional[dict]:
    """Return the cached profile result if fresh (hash matches + within TTL), else None."""
    sf = cfg.state_file("trust_cache")
    if not sf.exists():
        return None
    try:
        cache = json.loads(sf.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    entry = cache.get(username)
    if not entry or entry.get("x_snapshot_hash") != snapshot:
        return None
    ts = entry.get("evaluated_at")
    if ts:
        try:
            age = utc_now() - datetime.fromisoformat(ts)
            if age.days >= TRUST_CACHE_TTL_DAYS:
                return None
        except (ValueError, TypeError):
            return None  # unparseable timestamp → treat as stale
    return entry.get("result")


def _save_cached_trust(username: str, snapshot: str, result: dict, cfg) -> None:
    sf = cfg.state_file("trust_cache")
    cache = {}
    if sf.exists():
        try:
            cache = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}
    cache[username] = {
        "x_snapshot_hash": snapshot,
        "evaluated_at": utc_now().isoformat(),
        "result": result,
    }
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps(cache, indent=2, default=str))


# ── Hub expansion: let a TRUSTED blog/Substack surface the Oracle's own profiles ──

# A trusted, self-curated hub page can surface at most this many NEW candidate
# profiles — a hard bound so a link-heavy page can't blow up the graph.
MAX_NEW_HUB_CANDIDATES = 15


def _known_handles(username: str, all_sources: list, verdicts: dict) -> set[str]:
    """Normalized handles the Oracle demonstrably owns: their X username + the
    handle of every ALREADY-TRUSTED source. This is the set a blog-surfaced
    namesake must match to earn auto-trust (Decision 1)."""
    known = {normalize_handle(username)}
    for s in all_sources:
        ev = verdicts.get(canonical_identity(s.url))
        if not (ev and ev.trusted):
            continue
        pl = classify_source(s.url)
        if pl and pl.handle:
            known.add(normalize_handle(pl.handle))
    known.discard("")
    return known


def _expand_from_trusted_hubs(
    username: str,
    all_sources: list,
    verdicts: dict,
    relevant: set,
    x_attested: set,
    edges: list,
) -> dict:
    """Grow the trust graph from ALREADY-TRUSTED hubs, then re-propagate.

    A trusted, self-curated blog/Substack surfaces the Oracle's other accounts as NEW candidate
    nodes + `hub→id` edges; re-runs all 4 rules over the expanded graph, then applies an
    ADDITIVE handle-match layer that auto-trusts a surfaced namesake the graph itself didn't
    reach (Decision 1, 4).

    Bounded: trusted hubs only, single hop, one-profile-per-platform, dedup,
    MAX_NEW_HUB_CANDIDATES. Fail-safe: any fetch/parse error makes this a no-op.
    """
    def _trusted(u: str) -> bool:
        ev = verdicts.get(canonical_identity(u))
        return bool(ev and ev.trusted)

    hubs = [s for s in all_sources
            if s.source_type in ("blog", "substack") and _trusted(s.url)]
    if not hubs:
        return verdicts

    known = _known_handles(username, all_sources, verdicts)
    # Decision 5 — one profile per platform. Seed with platforms already trusted.
    seen_types = {s.source_type for s in all_sources if _trusted(s.url)}

    added = 0
    n_edges_before = len(edges)
    surfaced: list = []

    for hub in hubs:
        hub_cid = canonical_identity(hub.url)

        # Substack renders identity via JS → recover the X back-edge from _preloads.
        if hub.source_type == "substack":
            th = fetch_substack_twitter_handle(hub.url)
            if th:
                x_cid = canonical_identity(f"x.com/{th}")
                if x_cid:
                    relevant.add(x_cid)
                    edges.append(Edge(hub_cid, x_cid, via="substack_preload", found_by="substack"))

        for pl in fetch_profile_links(hub.url):
            if pl.type not in PROFILE_SCOPE or not pl.is_profile:
                continue
            cid = canonical_identity(pl.url)
            if not cid or cid in relevant or pl.type in seen_types:
                continue                                # dedup / stop-when-found
            if added >= MAX_NEW_HUB_CANDIDATES:
                break
            relevant.add(cid)
            edges.append(Edge(hub_cid, cid, via="hub_link", found_by="hub"))
            src = DiscoveredSource(
                source_type=pl.type,
                url=pl.url,
                confidence="medium",
                metadata={"discovered_via": "blog_hub", "hub": hub.url,
                          "shape": pl.shape, "handle": pl.handle},
            )
            all_sources.append(src)      # flows to discovery output + Phase D ingest
            surfaced.append(src)
            seen_types.add(pl.type)      # one profile per platform across the whole pass
            added += 1

    if added == 0 and len(edges) == n_edges_before:
        return verdicts                  # nothing surfaced → unchanged

    # Re-propagate over the EXPANDED graph — a surfaced profile that also links
    # back graduates by Rule 2/3 with no special-casing.
    verdicts = propagate(edges, x_attested, candidates=relevant)
    log(f"  [expand] surfaced {added} candidate(s) from {len(hubs)} trusted hub(s); re-propagated")

    # Additive auto-trust for surfaced candidates the graph can't reach. Only ever
    # PROMOTES a not-yet-trusted, non-org surfaced candidate — two layers:
    #   • research profiles (scholar/orcid) → trust on from-hub provenance alone;
    #   • everything else → tight-fuzzy handle-match against the Oracle's handles.
    for s in surfaced:
        cid = canonical_identity(s.url)
        ev = verdicts.get(cid)
        if ev and ev.trusted:
            continue                     # already graduated by the graph — keep its reason
        if (s.metadata or {}).get("shape") == "org":
            continue                     # org-shaped → affiliation later, never "is the Oracle"
        hub_url = (s.metadata or {}).get("hub", "")

        # Research profiles can't graduate via the graph (an academic page doesn't link back
        # to socials), so one linked from the Oracle's own trusted hub is trusted on that
        # provenance alone.
        if s.source_type in ("scholar", "orcid"):
            verdicts[cid] = TrustEvidence(
                trusted=True,
                reasons=[f"Research profile ({s.source_type}) linked from the "
                         f"Oracle's trusted hub {hub_url}"],
                edges=[{"source": canonical_identity(hub_url), "target": cid,
                        "via": "research_hub_link", "found_by": "hub"}],
            )
            log(f"  [expand] auto-trusted {s.source_type} {s.url} — research profile on a trusted hub")
            continue

        matched = match_known_handle(s.metadata.get("handle"), known)
        if matched:
            verdicts[cid] = TrustEvidence(
                trusted=True,
                reasons=[f"Handle '{s.metadata.get('handle')}' matches the Oracle's "
                         f"'{matched}' on their trusted hub {hub_url}"],
                edges=[{"source": canonical_identity(hub_url), "target": cid,
                        "via": "handle_match", "found_by": "hub"}],
            )
            log(f"  [expand] auto-trusted {s.source_type} {s.url} — handle matches '{matched}'")

    return verdicts


def _x_identity_targets(profile_info: dict, twitter_sources: list) -> list:
    """The X seed's declared accounts → typed identity targets (→ Rule 5 → T1).

    Every bio/website link becomes an identity edge from the X handle, `verified=False`
    (an X-bio link is declared, not platform-verified). Returns [(canonical_id, verified)] —
    the X-side analog of `_probe_substack_profile`'s identity_targets, feeding the same
    platform-agnostic `_compute_trust` contract.
    """
    targets: list = []
    seen: set = set()
    for s in twitter_sources:
        cid = canonical_identity(s.url)
        if cid and cid not in seen:
            seen.add(cid)
            targets.append((cid, False))
    wcid = canonical_identity(profile_info.get("website", ""))
    if wcid and wcid not in seen:
        seen.add(wcid)
        targets.append((wcid, False))
    return targets


def _compute_trust(
    seed_label: str,
    root_id: str,
    identity_targets: list,
    profile_info: dict,
    github_sources: list,
    all_sources: list,
    skip_edge_fetch: bool = False,
) -> dict:
    """Build the trust edge graph for a person and run propagation.

    PLATFORM-AGNOSTIC: the root is the CONFIRMED seed profile (`root_id`), never assumed to
    be X. `identity_targets` = [(canonical_id, verified)] — the seed's own declared accounts,
    each becoming a TYPED IDENTITY edge that graduates the target to T1 via Rule 5 (no
    bidirectional back-link needed). Generic edges (GitHub bio/blog, landing-page footers)
    still feed Rules 2/3 for sources the seed doesn't name directly. Bounded to `relevant`
    (the person's own candidate set) so a content-heavy page can't blow up the graph.
    """
    roots: set[str] = {root_id} if root_id else set()

    relevant: set[str] = set(roots)
    for tid, _ in identity_targets:
        if tid:
            relevant.add(tid)
    for s in all_sources:
        cid = canonical_identity(s.url)
        if cid:
            relevant.add(cid)

    edges: list[Edge] = []

    # 1. Typed IDENTITY edges — the seed profile's own declared accounts. Each
    #    graduates its target to T1 via Rule 5 (no bidirectional needed).
    for tid, verified in identity_targets:
        if tid and tid != root_id:
            edges.append(Edge(
                root_id, tid,
                via="identity_verified" if verified else "identity_declared",
                found_by="root_profile",
            ))

    # 2. GitHub bio mentions + blog field (data already in metadata).
    gh_login, gh_bio, gh_blog = None, "", ""
    for s in github_sources:
        meta = s.metadata or {}
        if s.source_type == "github" and meta.get("discovered_via") != "github_blog_field":
            gh_login = s.url.rstrip("/").split("/")[-1]
            gh_bio = meta.get("bio") or ""
        elif meta.get("discovered_via") == "github_blog_field":
            gh_blog = s.url
    if gh_login:
        edges += edges_from_github(gh_login, gh_bio, gh_blog, relevant)

    # 3. Landing-page footers for substack/blog (one fetch each).
    if not skip_edge_fetch:
        for s in all_sources:
            if s.source_type in ("substack", "blog"):
                sid = canonical_identity(s.url)
                if sid:
                    edges += fetch_landing_edges(sid, s.url, relevant)

    # Dedupe edges by (source, target); keep first occurrence's metadata.
    edges = list({(e.source, e.target): e for e in edges}.values())
    if edges:
        log(f"  [trust] {len(edges)} edges, root {root_id}")
    # Pass `relevant` as candidates so EVERY discovered source is evaluated and
    # gets near-miss/no-corroboration evidence (not a caller-side fallback).
    verdicts = propagate(edges, roots, candidates=relevant)

    # ── Hub expansion: mine ALREADY-TRUSTED blog/Substack hubs for the person's
    # other profiles, grow the graph, re-propagate + additive handle-match. Gated
    # by skip_edge_fetch (it fetches hub pages), same as the landing-page edges. ──
    if not skip_edge_fetch:
        verdicts = _expand_from_trusted_hubs(
            seed_label, all_sources, verdicts, relevant, roots, edges,
        )

    return verdicts


def discover_profile(
    username: str,
    skip_trust_cache_write: bool = False,
    config: StatePaths | None = None,
    skip_edge_fetch: bool = False,
    reverify: bool = False,
    probe_scholar: bool = True,
    seed_type: str = "x",
    extra_source_urls: list[str] | None = None,
) -> dict:
    """
    Discover all content sources for a credible person, ROOTED at a confirmed seed
    profile — an X handle (`seed_type="x"`) OR a Substack handle (`seed_type="substack"`).
    The trust graph is platform-agnostic: a Substack-only person is rooted at their
    Substack, with no X anywhere in the graph.

    Probe 1 resolves the ROOT profile + its declared identity links; the remaining
    deterministic probes (Substack-convention, GitHub, Scholar) run on the discovered
    name/handle. There is NO open-web step — see the module docstring.

    Args:
        username: the seed handle — X handle (no @) or Substack user handle.
        seed_type: "x" or "substack" — which platform the seed profile lives on.
        skip_trust_cache_write: suppress the invalidation-keyed trust-cache WRITE. Reads are
            unaffected; only `reverify` skips those.
        extra_source_urls: Probe 5's return leg — URLs the HOST model found by web search.
            They enter as LOW-confidence candidates and are judged by `_compute_trust` like
            any other source; none is trusted on the host's say-so. Junk is dropped silently.
            See `_sources_from_urls`.
        reverify: force re-discovery, ignoring the trust cache
        probe_scholar: run Probe 4 (Semantic Scholar author lookup). Default True. The atom-KB
            footprint expansion (pipeline.kb.expand) sets this False — it never ingests papers as
            Oracle footprint (papers aren't footprint), so a `scholar` source would only be
            discovered to be discarded, at the cost of a live API call per person. Scholar feeds
            NO trust edges (only identity/github/substack/blog do), so skipping it can't weaken
            trust-rooting. Every OTHER caller (vault add_person, radar, setup) keeps it on.

    Returns:
        dict with: username, display_name, bio, website, sources[], discovered_at
    """
    seed_type = (seed_type or "x").lower()
    log(f"Discovering sources for {seed_type}:{username}")

    cfg = config or state_paths()
    all_sources = []

    # ── Probe 1 — the ROOT profile (platform-agnostic). Yields profile_info, the
    #    root's own sources, its canonical root_id, and identity_targets = the
    #    person's declared accounts (→ typed identity edges → Rule 5 → T1).
    if seed_type == "substack":
        profile_info, root_sources, identity_targets = _probe_substack_profile(username)
        root_id = canonical_identity(
            profile_info.get("root_url") or f"https://{username}.substack.com")
    elif seed_type == "blog":
        # `username` here is the blog home URL (not a handle); the probe roots on it + mines its
        # home-page anchors for declared accounts.
        profile_info, root_sources, identity_targets = _probe_blog_profile(username)
        root_id = canonical_identity(profile_info.get("root_url") or username)
    else:  # "x"
        profile_info, root_sources = _probe_twitter_bio(username)
        root_id = canonical_identity(f"x.com/{username}")
        identity_targets = _x_identity_targets(profile_info, root_sources)

    # If Probe 1 returned nothing, the seed profile likely doesn't exist.
    if not profile_info.get("display_name"):
        log(f"  [error] {seed_type}:{username} is not a resolvable profile")
        return asdict(DiscoveredProfile(
            username=username,
            discovered_at=utc_now().isoformat(),
        ))

    all_sources.extend(root_sources)

    display_name = profile_info.get("display_name", "")
    bio = profile_info.get("bio", "")

    # Trust cache: if the seed profile (name + declared links) is unchanged and fresh, reuse
    # the cached result and skip the expensive probes. Key namespaced for non-X seeds so an X
    # handle and a Substack handle sharing a string don't collide.
    cache_key = username if seed_type == "x" else f"{seed_type}:{username}"
    root_link_ids = [t for t, _ in identity_targets]
    snapshot = _x_snapshot_hash(display_name, root_link_ids)
    # extra_source_urls bypasses the cache: the snapshot key fingerprints the profile, not
    # URLs the host just found, so a hit would silently discard the host's whole search.
    if not reverify and not extra_source_urls:
        cached = _get_cached_trust(cache_key, snapshot, cfg)
        if cached is not None:
            log(f"  [cache] hit for {seed_type}:{username} — profile unchanged, skipping probes")
            # Marked on the way OUT, never stored, so the disk entry stays clean. Callers (e.g.
            # oracles.add_oracle) use this flag to gate a redundant open-web followup.
            return {**cached, "from_cache": True}

    # Probe 2/3 guess accounts by treating the seed as a HANDLE — meaningful only for an
    # x-rooted seed. blog/substack roots discover via their own page instead.
    if seed_type not in ("substack", "blog"):
        all_sources.extend(_probe_substack(username))

    # Probe 3: GitHub (handle-rooted only).
    github_sources = [] if seed_type == "blog" else _probe_github(username)
    all_sources.extend(github_sources)

    # Probe 4: Semantic Scholar — try display_name, then GitHub name as fallback.
    if probe_scholar:
        scholar_name_candidates = []
        if display_name:
            # Strip emoji and special chars for Scholar search
            import re as _re
            clean_name = _re.sub(r'[^\w\s\'-]', '', display_name).strip()
            if clean_name:
                scholar_name_candidates.append(clean_name)
        # GitHub often has a cleaner real name than X display name
        for s in github_sources:
            gh_name = s.metadata.get("name")
            if gh_name and gh_name not in scholar_name_candidates:
                scholar_name_candidates.append(gh_name)

        scholar_found = False
        for name_candidate in scholar_name_candidates:
            # Skip names that look like handles/aliases (contain dots, @, are single words)
            if "." in name_candidate or "@" in name_candidate:
                continue
            results = _probe_semantic_scholar(name_candidate)
            if results:
                all_sources.extend(results)
                scholar_found = True
                break

        if not scholar_found and not scholar_name_candidates:
            log(f"  [skip] Semantic Scholar — no display name available")
    else:
        log("  [skip] Semantic Scholar — disabled (probe_scholar=False; papers aren't footprint)")

    # ── Probe 5 — open-web discovery, CO-ROUTED to the host model ────────────────────────────
    # The host model does the FINDING (web search) via `extra_source_urls`; this module does the
    # JUDGING. Appended last: `_dedupe_sources` keeps the first (type, url) occurrence, so a
    # deterministic probe always wins a collision against a host guess.
    if extra_source_urls:
        host_found = _sources_from_urls(extra_source_urls)
        if host_found:
            log(f"  [probe 5] {len(host_found)} host-supplied candidate(s) "
                f"from {len(extra_source_urls)} suggested")
        all_sources.extend(host_found)

    # Dedupe and build profile
    all_sources = _dedupe_sources(all_sources)

    # Break ① — a discovered YouTube URL is usually a watch/playlist link, but the
    # ingester (and trust identity) need the CHANNEL. Resolve before trust runs.
    if not skip_edge_fetch:
        from pipeline.ingestion.url_canon import resolve_channel_url
        for s in all_sources:
            if s.source_type == "youtube":
                channel = resolve_channel_url(s.url)
                if channel:
                    s.url = _normalize_url(channel)

    # ── Owner validation: trust via graph reachability ───────────────────────
    verdicts = _compute_trust(
        username, root_id, identity_targets, profile_info, github_sources, all_sources,
        skip_edge_fetch=skip_edge_fetch,
    )
    _NEEDS_REVIEW = TrustEvidence(
        trusted=False, reasons=["No trust path from a confirmed root"], edges=[]
    )
    for s in all_sources:
        ev = verdicts.get(canonical_identity(s.url), _NEEDS_REVIEW)
        s.trust = ev
        s.evidence_edges = ev.edges

    profile = DiscoveredProfile(
        username=username,
        display_name=display_name,
        bio=bio,
        website=profile_info.get("website", ""),
        sources=[asdict(s) for s in all_sources],
        discovered_at=utc_now().isoformat(),
    )

    # Save to state (discovery output + the invalidation-keyed trust cache).
    sf = cfg.state_file("discovered_profiles")
    profiles = _load_discovered(sf)
    profiles[cache_key] = asdict(profile)
    _save_discovered(profiles, sf)

    result = asdict(profile)
    # A degraded run (skip_edge_fetch → fewer edges, weaker verdicts) must not poison the cache
    # for later full runs. skip_trust_cache_write stays True for the ingest path deliberately:
    # letting ingest populate the cache would flip re-ingest to skip discovery, an out-of-scope
    # behavior change.
    if not skip_trust_cache_write and not skip_edge_fetch:
        _save_cached_trust(cache_key, snapshot, result, cfg)

    trusted = [s for s in all_sources if s.trust and s.trust.trusted]
    review = [s for s in all_sources if not (s.trust and s.trust.trusted)]
    log(f"Discovery complete: {len(all_sources)} sources for @{username} "
        f"({len(trusted)} trusted, {len(review)} need review)")
    for s in trusted:
        feed_info = f" (feed: {s.feed_url})" if s.feed_url else ""
        why = s.trust.reasons[0] if s.trust and s.trust.reasons else ""
        log(f"  [trusted] {s.source_type}: {s.url}{feed_info}  — {why}")
    for s in review:
        feed_info = f" (feed: {s.feed_url})" if s.feed_url else ""
        log(f"  [review]  {s.source_type}: {s.url}{feed_info}")

    return result


# This module only discovers accounts; it does not pull content. Routing a discovered source
# to its atom adapter is `pipeline/kb/expand.py` + `onboard_footprint.py`. Keep the halves apart.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover content sources for a credible person"
    )
    parser.add_argument("--username", required=True, help="X handle (no @)")
    parser.add_argument("--skip-trust-cache-write", action="store_true",
                        help="Do not write the trust cache for this run (reads are unaffected; "
                             "use --reverify to ignore an existing entry)")
    parser.add_argument("--show-untrusted", action="store_true",
                        help="Print the needs-review partition with trust reasons")
    parser.add_argument("--skip-edge-fetch", action="store_true",
                        help="Skip landing-page fetches for trust edges (faster, fewer edges)")
    parser.add_argument("--reverify", action="store_true",
                        help="Force re-discovery, ignoring the trust cache")

    args = parser.parse_args()

    # This CLI only DISCOVERS. Ingesting a discovered person is the atom rail's job
    # (`pipeline/kb/expand.py`) — do not add ingest-side flags here.
    result = discover_profile(
        username=args.username,
        skip_trust_cache_write=args.skip_trust_cache_write,
        skip_edge_fetch=args.skip_edge_fetch,
        reverify=args.reverify,
    )

    if args.show_untrusted:
        review = [s for s in result.get("sources", [])
                  if not (s.get("trust") or {}).get("trusted")]
        print(f"\n── Needs review ({len(review)}) " + "─" * 40)
        for s in review:
            reasons = (s.get("trust") or {}).get("reasons") or []
            why = reasons[0] if reasons else ""
            print(f"  {s['source_type']:10} {s['url']}  — {why}")
        print()

    print(json.dumps(result, indent=2, default=str))
