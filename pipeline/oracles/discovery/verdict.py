"""
pipeline/oracles/discovery/verdict.py — what discovery does with each candidate source. Pure: the
candidates and what `backlink` read about them in, verdicts out. The rule is one for every source
type: a source we are sure is theirs is added, a project is mentioned, the rest drop. Nothing is
asked about.

| candidate                                                                    | verdict |
|------------------------------------------------------------------------------|---------|
| already known (one of their profiles, an added source)                       | none    |
| redirects to a known source                                                  | merge   |
| a research profile (Scholar, arXiv author page, DBLP), declared              | also    |
| an ORCID, declared on one of their profiles or by a source that links back   | sure    |
| an X account declared on one of their profiles                               | sure    |
| an X account, a research profile or an ORCID otherwise                       | drop    |
| theirs: declared on one of their profiles, or one hop from a source that links back and itself links back, or linked from their posts and links back | below |
|   ↳ org-shaped, or read and publishes nothing                                | also    |
|   ↳ otherwise                                                                | sure    |
| otherwise                                                                    | drop    |

Their profiles are the profile pages the Oracle owns: the root (X bio, Substack profile, blog home,
ORCID record) and, at the links stage, every other X bio, Substack profile, GitHub profile and
ORCID record of theirs. A blog's home page is one only as the root.

A link back is a link to one of `oracle_ids`: their profiles and the accounts those declare. Only
a declared X counts as theirs: a handle that appears inside a post is someone they mention.
Reasons are plain, at most two, and name no pronoun the code cannot know.

A site that is not theirs and could not be read is dropped, however many of their posts link it.
Across nine real stores, such sites were news, journals and link shorteners (wsj.com five times, cell.com six times from one person,
science.org, lnkd.in): sites that refuse automated reads. OPYT could not pull one if it were added.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.oracles.discovery.backlink import Facts
from pipeline.platforms.source_classify import classify_source

MERGE, SURE, ALSO, DROP = "merge", "sure", "also", "drop"

_RANK = {SURE: 0, ALSO: 1, MERGE: 2, DROP: 3}


@dataclass(frozen=True)
class Candidate:
    key: str                  # canonical identity
    url: str                  # the page to read
    post_count: int = 0       # their own posts that link it
    declared: str = ""        # the reason one of their profiles names it ("in their X bio")
    via: str | None = None    # one hop: the key of the source of theirs that declares it


@dataclass(frozen=True)
class Verdict:
    key: str
    url: str
    source_type: str
    verdict: str
    reasons: tuple = ()
    post_count: int = 0


def _posts(n: int) -> str:
    return f"in {n} of their posts"


def _theirs(c: Candidate, links_back: bool) -> bool:
    return bool(c.declared) or (links_back and (c.via is not None or c.post_count > 0))


def _one(c: Candidate, f: Facts, oracle_ids: frozenset, known: frozenset) -> Verdict:
    pl = classify_source(c.url)
    stype = pl.type if pl else "blog"
    shape = pl.shape if pl else "unknown"
    back = f.links_to & oracle_ids
    links_back = bool(back)

    def out(verdict: str, *reasons: str) -> Verdict:
        return Verdict(c.key, c.url, stype, verdict, tuple(r for r in reasons if r)[:2],
                       c.post_count)

    declared_reason = c.declared or (f"listed on {c.via}" if c.via else "")
    if f.final_key and f.final_key in known:
        return out(MERGE)
    if stype == "scholar":
        return out(ALSO, c.key.split("/", 1)[0]) if c.declared or c.via else out(DROP)
    if stype == "orcid":
        return out(SURE, "their ORCID", declared_reason) if c.declared or c.via else out(DROP)
    if stype == "x":
        return out(SURE, declared_reason) if c.declared else out(DROP)
    if _theirs(c, links_back):
        if shape == "org" or (f.reached and f.has_feed is False):
            return out(ALSO, c.key.split("/", 1)[0])
        back_reason = ("links back to their X" if any(k.startswith("x.com/") for k in back)
                       else "links back to their other sources") if links_back else ""
        return out(SURE, declared_reason, c.post_count and _posts(c.post_count), back_reason)
    return out(DROP)


def judge(candidates: list[Candidate], facts: dict, *, oracle_ids: frozenset,
          known: frozenset) -> list[Verdict]:
    """Every candidate not already known → its verdict, sure first, then the most-linked, then by
    key. `facts` maps a key to what `backlink` read; a key with no entry was not read (unknown)."""
    seen: set[str] = set()
    out = []
    for c in candidates:
        if c.key in known or c.key in seen:
            continue
        seen.add(c.key)
        out.append(_one(c, facts.get(c.key) or Facts(c.key), oracle_ids, known))
    return sorted(out, key=lambda v: (_RANK[v.verdict], -v.post_count, v.key))
