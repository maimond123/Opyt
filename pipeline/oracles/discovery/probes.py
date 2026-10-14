"""
pipeline/oracles/discovery/probes.py — one probe per profile page: the profile and the accounts it
declares. The root is one of them; the links stage reads every other profile of theirs too (X,
Substack, GitHub, ORCID). Each reads the network and returns what it found as
sources; none writes. No probe guesses an account from a handle.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import requests

from pipeline.host_access import request as host_request
from pipeline.ingestion.utils import log
from pipeline.oracles.discovery import backlink
from pipeline.oracles.discovery import shaping
from pipeline.platforms import identity_tokens
from pipeline.platforms import sessions
from pipeline.platforms.github import api as github_api
from pipeline.platforms.papers import openalex
from pipeline.platforms.url_canon import canonical_identity
from pipeline.platforms.x import profile as x_profile


def probe_twitter_bio(username: str) -> tuple[dict, list]:
    """Probe 1: bio, website and the other homes an X profile links to.

    Reads the free cookie-scrape `UserByScreenName`. t.co expansion happens once inside
    `fetch_user_profile`, beside the entities block.

    The x.com self-link SKIP below stays here. It is a consumer policy and the other consumer of
    the same call (`identify.fetch_x_identity`) disagrees — it blanks the field instead."""

    log(f"  [probe] Twitter bio for @{username}")
    try:
        session = sessions.x_session(f"https://x.com/{username}")
        data = x_profile.fetch_user_profile(session, username)
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
    stype = shaping.classify_url(website)
    # Skip X/Twitter self-links.
    if stype and stype != "x":
        sources.append(shaping.DiscoveredSource(
            source_type=stype,
            url=shaping.normalize_url(website),
        ))

    # The OTHER homes the bio links to — a Substack or a personal site. Already expanded
    # past t.co by `fetch_user_profile`.
    for expanded in data["bio_urls"]:
        if not expanded:
            continue
        expanded = shaping.normalize_url(expanded)
        # Skip X/Twitter self-links
        stype = shaping.classify_url(expanded)
        if not stype or stype == "x":
            continue
        sources.append(shaping.DiscoveredSource(
            source_type=stype,
            url=expanded,
        ))

    log(f"    Found {len(sources)} URLs from Twitter bio")
    return profile_info, sources


def probe_substack_profile(seed: str) -> tuple[dict, list, list]:
    """ROOT probe for a Substack-seeded person — the non-X analog of `probe_twitter_bio`.

    Reads the owner's public profile (`backlink.substack_profile`). `userLinks` are the person's
    own typed, declared links (site / X / GitHub / …).

    Returns (profile_info, sources, identity_targets):
      profile_info:     {display_name, bio, website, root_url}
      sources:          the ROOT Substack itself + a DiscoveredSource per external userLink.
      identity_targets: [(canonical_id, verified)] — the accounts the Substack declares;
                        `verified` = OAuth-connected account (is_connected_account).
    """
    log(f"  [probe] Substack profile for {seed}")
    profile_info = {"display_name": "", "bio": "", "website": "", "root_url": ""}
    sources: list = []
    identity_targets: list = []
    publication_url = shaping.normalize_url(
        seed if "." in seed or "/" in seed else f"{seed}.substack.com")
    data, _final = backlink.substack_profile(publication_url)
    if not data:
        log(f"    No Substack profile for {seed!r}")
        return profile_info, sources, identity_targets
    handle = data.get("handle") or seed

    tw = data.get("twitterAccount") or {}
    profile_info["display_name"] = data.get("name") or tw.get("display_name") or handle
    profile_info["bio"] = data.get("bio") or ""
    # Use primaryPublication's {subdomain}.substack.com as the root (not the user-level
    # subdomainUrl, which is often null, or "substack.com/@{handle}", which collides across
    # users) — this canonicalizes to a unique node matching the Oracle's substack:{subdomain} id.
    sub = (data.get("primaryPublication") or {}).get("subdomain")
    root_url = f"https://{sub}.substack.com" if sub else publication_url
    profile_info["root_url"] = root_url

    seen: set = set()

    def _add(url: str, verified: bool) -> None:
        url = shaping.normalize_url(url)
        cid = canonical_identity(url)
        if not cid or cid in seen:
            return
        seen.add(cid)
        identity_targets.append((cid, verified))
        stype = shaping.classify_url(url)
        if not profile_info["website"] and stype == "blog":
            profile_info["website"] = url
        sources.append(shaping.DiscoveredSource(source_type=stype, url=url))

    # The root Substack itself comes first: its canonical id is the root's.
    root_cid = canonical_identity(root_url)
    if root_cid:
        seen.add(root_cid)
        sources.append(shaping.DiscoveredSource(
            source_type="substack", url=shaping.normalize_url(root_url)))

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


def probe_github_profile(owner: str) -> tuple[dict, list, list]:
    """A GitHub account's profile: its name, and the accounts it declares as theirs, its X and
    its website (`blog`), as `backlink` reads them. Never a root: read beside the root at the
    links stage. The same shape as the other probes, and empty when it could not be read, a rate
    limit included, so the stage retries instead of losing what it declares.

    An organization's account declares nothing as theirs: its X and site are the organization's,
    their employer's, never pulled as their writing (live: Gajesh's github:Layr-Labs names
    Eigen Labs' X and site)."""
    log(f"  [probe] GitHub profile for {owner}")
    try:
        user = github_api.fetch_user(owner)          # None: no such account, or a failed read
    except github_api.GitHubRateLimited as e:
        log(f"    [error] GitHub profile probe: {e}")
        return {}, [], []
    if not user:
        return {}, [], []
    login = user.get("login") or owner
    root_url = f"https://github.com/{login}"
    declared = [] if user.get("type") == "Organization" else [
        shaping.normalize_url(u)
        for u in backlink.facts_from_github_user(canonical_identity(root_url), user).declared]
    profile_info = {"display_name": user.get("name") or login, "bio": user.get("bio") or "",
                    "website": (user.get("blog") or "").strip(), "root_url": root_url}
    sources = [shaping.DiscoveredSource(source_type="github", url=root_url)]
    sources += [shaping.DiscoveredSource(source_type=shaping.classify_url(u) or "blog", url=u)
                for u in declared]
    return profile_info, sources, [(canonical_identity(u), False) for u in declared]


# Blog links to known account platforms remain ownership declarations.
# The occasional mention of another person is an accepted tradeoff.
#
# A non-platform host is not ignored as a random outbound link: `shaping.classify_url` defaults
# unknown hosts to "blog", so a person's own `karpathy.github.io` or `karpathy.medium.com` would be
# missed. It is admitted when it carries one of the person's identity tokens — see
# `identity_tokens`, which holds the measurement. This set is the first of two arms, not the whole
# rule.
_BLOG_IDENTITY_TYPES = frozenset({"x", "github", "substack"})


def probe_blog_profile(seed: str) -> tuple[dict, list, list]:
    """ROOT probe for a blog-seeded person — the blog analog of `probe_substack_profile`.

    `seed` is the blog home URL. GETs the home HTML: `<title>` is the display name, and any
    anchor to a known identity platform (x / github / substack) is a
    self-declared own account. The blog itself is added
    first as the self-rooted source. A JS-rendered home yields no server-side anchors — the
    blog still roots + ingests, just with no cross-platform links surfaced.

    Returns (profile_info, sources, identity_targets), the same shape as the other root probes."""
    seed_url = shaping.normalize_url(seed)
    root_cid = canonical_identity(seed_url)
    if not root_cid:
        return {}, [], []
    host = urlparse(seed_url).netloc
    profile_info = {"display_name": "", "bio": "", "website": seed_url, "root_url": seed_url}
    sources: list = []
    identity_targets: list = []
    log(f"  [probe] Blog profile for {seed_url}")

    html = ""
    try:
        resp = host_request(requests.get, seed_url, timeout=20, allow_redirects=True,
                            headers={"User-Agent": "Mozilla/5.0 (compatible; OPYT/1.0)"})
        if resp.status_code == 200:
            html = resp.text or ""
        else:
            log(f"    Blog home returned status {resp.status_code} for {seed_url}")
            return {}, [], []
    except Exception as e:
        log(f"    [error] Blog home probe failed: {e}")
        return {}, [], []

    if not html:
        return {}, [], []

    # display_name from <title>, trimming a trailing " | Site" tail; falls back to the host.
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        title = re.split(r"\s+[|–—-]\s+", title)[0].strip()
        profile_info["display_name"] = title or host
    else:
        profile_info["display_name"] = host

    # Add the blog itself first, as the self-rooted source.
    seen: set = {root_cid}
    sources.append(shaping.DiscoveredSource(source_type="blog", url=seed_url))

    # Outbound anchors to a declared OWN account — a known identity platform, or any
    # other host that carries one of this person's identity tokens.
    tokens = identity_tokens.tokens_for(seed_url, profile_info.get("display_name"))
    for href in set(re.findall(r'href=[\'"](https?://[^\'"]+)[\'"]', html, re.I)):
        url = shaping.normalize_url(href)
        stype = shaping.classify_url(url)
        if stype not in _BLOG_IDENTITY_TYPES:
            # HOST match only, never a path match. `karpathy.github.io` is a whole site of his and
            # becomes one source whose feed yields the posts; `cs.stanford.edu/people/karpathy/…`
            # carries the token in its PATH, and promoting that would register all of Stanford CS
            # as his site. Path-scoped work is `link_discovery`'s job, as a hub candidate.
            if not identity_tokens.host_carries_token(url, tokens):
                continue
            # The HOME, not the deep link that revealed it: his 14 essays are one source, and the
            # blog adapter discovers the rest of them from its sitemap/RSS.
            parsed_home = urlparse(url)
            url = f"{parsed_home.scheme}://{parsed_home.netloc}"
        cid = canonical_identity(url)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        identity_targets.append((cid, False))            # declared, not platform-verified
        sources.append(shaping.DiscoveredSource(source_type=stype, url=url))

    log(f"    Blog profile: {len(identity_targets)} declared links for {host}")
    return profile_info, sources, identity_targets


# ── ORCID: a researcher's own record lists their sites ──────────
_ORCID_PERSON = "https://pub.orcid.org/v3.0/{orcid}/person"


def orcid_person(data: dict) -> tuple[str, list[str]]:
    """An ORCID `/person` record → `(display name, the sites it lists)`. Pure.

    `researcher-urls` is what the researcher entered about themselves, so each one is declared."""
    name = (data or {}).get("name") or {}
    credit = (name.get("credit-name") or {}).get("value") if name.get("credit-name") else ""
    given = (name.get("given-names") or {}).get("value") or ""
    family = (name.get("family-name") or {}).get("value") or ""
    display = credit or " ".join(p for p in (given, family) if p)
    urls = [((u or {}).get("url") or {}).get("value")
            for u in ((data or {}).get("researcher-urls") or {}).get("researcher-url") or []]
    return display, [u for u in urls if u]


def probe_orcid_profile(seed: str) -> tuple[dict, list, list]:
    """ROOT probe for a researcher: their ORCID record's name and listed sites.

    `seed` is an ORCID iD, or an OpenAlex author id (`A…`) whose author record carries one; an
    OpenAlex author with no ORCID has nothing to read. Free, no key."""
    orcid = seed
    if seed.upper().startswith("A") and seed[1:].isdigit():
        try:
            rec = openalex.OpenAlexClient().author(seed) or {}
        except Exception as e:
            log(f"    [error] OpenAlex author {seed}: {type(e).__name__}: {e}")
            rec = {}
        orcid = str(rec.get("orcid") or "").rstrip("/").rsplit("/", 1)[-1]
        if not orcid:
            return {}, [], []
    log(f"  [probe] ORCID record for {orcid}")
    try:
        resp = host_request(requests.get, _ORCID_PERSON.format(orcid=orcid), timeout=20,
                            headers={"Accept": "application/json"})
        data = resp.json() if resp.status_code == 200 else None
    except (requests.RequestException, ValueError) as e:
        log(f"    [error] ORCID probe failed: {e}")
        data = None
    if not isinstance(data, dict):
        return {}, [], []
    display, urls = orcid_person(data)
    root_url = f"https://orcid.org/{orcid}"
    profile_info = {"display_name": display or orcid, "bio": "", "website": "",
                    "root_url": root_url}
    sources, identity_targets, seen = [], [], {canonical_identity(root_url)}
    for url in urls:
        url = shaping.normalize_url(url)
        cid = canonical_identity(url)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        identity_targets.append((cid, False))
        sources.append(shaping.DiscoveredSource(source_type=shaping.classify_url(url) or "blog",
                                                url=url))
    return profile_info, sources, identity_targets
