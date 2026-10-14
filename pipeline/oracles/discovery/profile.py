"""
pipeline/oracles/discovery/profile.py — flow: find a person's other sources, rooted at a
confirmed X, Substack, blog or ORCID profile:

  1. the root profile, every other profile page of theirs (another X bio, a Substack or GitHub
     profile, an ORCID record), and the accounts each declares   (`probes`)
  2. the sites their own X posts link, top `POST_CANDIDATES`   (`post_links`)
  3. one read of each candidate: does it link back?   (`backlink`)
  4. one hop: a source that links back may declare more of theirs, read the same way
  5. a verdict for each   (`verdict.judge`)

Nothing is guessed from a handle and no search engine is asked. Nothing here writes: `register`
in `sources/web/discovery.py` records what the verdicts say. No outbound work here is paid — the
X profile probe runs on the user's own session.
"""

from __future__ import annotations

from dataclasses import asdict

from pipeline.ingestion.utils import log
from pipeline.kb import resolve
from pipeline.oracles.discovery import backlink
from pipeline.oracles.discovery import post_links
from pipeline.oracles.discovery import probes
from pipeline.oracles.discovery import shaping
from pipeline.oracles.discovery import verdict
from pipeline.oracles.discovery.verdict import Candidate
from pipeline.platforms.source_classify import classify_source
from pipeline.platforms.url_canon import canonical_identity
from pipeline.timeparse import utc_now

# How many of the sites their posts link get a back-link read: the most-linked first. Measured
# over 11 Oracles: every known find ranked in the top 24.
POST_CANDIDATES = 40

# The reason a source declared on a profile page of theirs carries, by the page's type.
_DECLARED_ON = {"x": "in their X bio", "substack": "on their Substack profile",
                "blog": "on their site", "github": "on their GitHub profile",
                "orcid": "on their ORCID record", "openalex": "on their ORCID record"}
# What a back-link read can tell nothing about: their own accounts by declaration only.
_NOT_READ = ("x", "orcid", "scholar")


def _label(seed_type: str) -> str:
    return _DECLARED_ON.get(seed_type, "on their profile")


def _read_profile(username: str, seed_type: str) -> tuple[dict, list, list, str]:
    """One profile page → `(profile_info, its own sources, the sources it declares, its id)`."""
    if seed_type == "substack":
        info, sources, _targets = probes.probe_substack_profile(username)
        root_id = canonical_identity(info.get("root_url") or f"https://{username}.substack.com")
    elif seed_type == "blog":
        info, sources, _targets = probes.probe_blog_profile(username)
        root_id = canonical_identity(info.get("root_url") or username)
    elif seed_type in ("orcid", "openalex"):
        info, sources, _targets = probes.probe_orcid_profile(username)
        root_id = canonical_identity(info.get("root_url") or "")
    elif seed_type == "github":
        info, sources, _targets = probes.probe_github_profile(username)
        root_id = canonical_identity(info.get("root_url") or f"github.com/{username}")
    else:
        info, sources = probes.probe_twitter_bio(username)
        root_id = canonical_identity(f"x.com/{username}")
    own = [s for s in sources if canonical_identity(s.url) == root_id]
    declared = [s for s in sources if canonical_identity(s.url) != root_id]
    return info, own, declared, root_id


def _candidates(declared: list, posted: list) -> list[Candidate]:
    """What their profiles declare and the sites their posts link, one candidate per key.
    `declared` is `[(reason, sources)]`, the root's first, so a key two profiles declare keeps
    the root's reason."""
    by_key: dict[str, Candidate] = {}
    for reason, sources in declared:
        for s in sources:
            key = canonical_identity(s.url)
            if key:
                by_key.setdefault(key, Candidate(key, s.url, declared=reason))
    for p in posted:
        c = by_key.get(p.key)
        by_key[p.key] = (Candidate(p.key, c.url, p.post_count, declared=c.declared) if c else
                         Candidate(p.key, p.url, p.post_count))
    return list(by_key.values())


def _read(c: Candidate) -> bool:
    pl = classify_source(c.url)
    return not (pl and pl.type in _NOT_READ)


def _one_hop(cands: list[Candidate], facts: dict, oracle_ids: frozenset) -> list[Candidate]:
    """What the sources that link back declare as theirs, not already a candidate."""
    have = {c.key for c in cands}
    out = []
    for c in cands:
        f = facts.get(c.key)
        if not f or not (f.links_to & oracle_ids):
            continue
        for url in f.declared:
            key = canonical_identity(url)
            if key and key not in have:
                have.add(key)
                out.append(Candidate(key, shaping.normalize_url(url), via=c.key))
    return out


# What the root stage registers before any pull: the accounts a pull of their own work needs.
ROOT_TYPES = ("x", "orcid")


def discover_root(username: str, seed_type: str = "x") -> dict:
    """The root stage: their root profile alone, and what it declares that we are sure of and
    that is pulled as a starting source, their X and their ORCID. No candidate is read.

    Returns `{display_name, root_id, verdicts}`; `display_name` is empty when the root profile
    did not resolve."""
    seed_type = (seed_type or "x").lower()
    info, own, declared, root_id = _read_profile(username, seed_type)
    if not info.get("display_name"):
        return {"display_name": "", "root_id": root_id, "verdicts": []}
    cands = _candidates([(_label(seed_type), declared)], [])
    known = frozenset({root_id, *(canonical_identity(s.url) for s in own)} - {None, ""})
    verdicts = verdict.judge(cands, {}, oracle_ids=frozenset(known | {c.key for c in cands}),
                             known=known)
    return {"display_name": info.get("display_name", ""), "root_id": root_id,
            "verdicts": [asdict(v) for v in verdicts
                         if v.verdict == verdict.SURE and v.source_type in ROOT_TYPES]}


def _unread(username: str, page: str) -> dict:
    """What a pass returns when a profile page did not answer: nothing judged, and which page."""
    log(f"  [error] {page} is not a resolvable profile")
    return {**asdict(shaping.DiscoveredProfile(username=username,
                                               discovered_at=utc_now().isoformat())),
            "unread": page}


def discover_profile(username: str, seed_type: str = "x", *, others: list | tuple = (),
                     conn=None, oracle_id: str | None = None,
                     skip_edge_fetch: bool = False) -> dict:
    """One Oracle's other sources, as verdicts. `username` is the seed handle (X, Substack), the
    blog home URL, or an ORCID / OpenAlex author id. `others` are their other profile pages,
    `[{seed, seed_type}]` (`expand.member_profiles` after the root): what each declares is
    theirs as what the root declares is, with that page's reason. `conn` and `oracle_id` let the
    flow read their own X posts and know every account already theirs, whichever member is the
    root (`resolve.cluster_self_urls`); without them only the profiles' declarations are
    candidates. `skip_edge_fetch` reads no candidate (every fact unknown).

    Returns `{username, display_name, bio, website, root_id, verdicts, discovered_at}`. When a
    profile page did not answer, nothing is judged: `display_name` is empty and `unread` names
    the page, so the stage retries instead of losing what that page declares."""
    seed_type = (seed_type or "x").lower()
    log(f"Discovering sources for {seed_type}:{username}")
    info, own, declared, root_id = _read_profile(username, seed_type)
    if not info.get("display_name"):
        return _unread(username, f"{seed_type}:{username}")
    mine = {root_id, *(canonical_identity(s.url) for s in own)}
    declared_on = [(_label(seed_type), declared)]
    for page in others:
        p_info, p_own, p_declared, p_id = _read_profile(page["seed"], page["seed_type"])
        if not p_info.get("display_name"):
            return _unread(username, f"{page['seed_type']}:{page['seed']}")
        mine |= {p_id, *(canonical_identity(s.url) for s in p_own)}
        declared_on.append((_label(page["seed_type"]), p_declared))

    posted, members = [], frozenset()
    if conn is not None and oracle_id:
        posted = post_links.own_post_candidates(conn, oracle_id)[:POST_CANDIDATES]
        members = resolve.cluster_self_urls(conn, oracle_id)
    cands = _candidates(declared_on, posted)
    known = frozenset((mine | members) - {None, ""})
    oracle_ids = frozenset(known | {c.key for c in cands if c.declared})
    facts: dict = {}
    if not skip_edge_fetch:
        for c in cands:
            if _read(c) and c.key not in known:
                facts[c.key] = backlink.facts_for(c.key, c.url)
        hop = _one_hop(cands, facts, oracle_ids)
        for c in hop:
            if _read(c) and c.key not in known:
                facts[c.key] = backlink.facts_for(c.key, c.url)
        cands += hop
    verdicts = verdict.judge(cands, facts, oracle_ids=oracle_ids, known=known)

    counts = {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    log(f"Discovery complete for {seed_type}:{username}: {counts}")
    return {"username": username, "display_name": info.get("display_name", ""),
            "bio": info.get("bio", ""), "website": info.get("website", ""), "root_id": root_id,
            "verdicts": [asdict(v) for v in verdicts],
            "discovered_at": utc_now().isoformat()}


# This module only discovers accounts; it does not pull content. `register` in
# `sources/web/discovery.py` records the verdicts; `first_pull.onboard_footprint` pulls them.
