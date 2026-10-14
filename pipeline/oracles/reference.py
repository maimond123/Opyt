"""
pipeline/oracles/reference.py — what a free-form reference names, and whether it can be an
Oracle's root.

A reference is what the user or the host typed: an X @handle, a Substack or blog URL, a canonical
id, an academic identity URL. `classify_reference` says which; `unsupported_root` refuses the
platforms OPYT cannot root a person on, network-free; `match_local_roster` finds a reference the
store already knows.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from pipeline.kb import derive
from pipeline.kb import schema
from pipeline.oracles import roster
from pipeline.platforms.url_canon import canonical_identity


# `openalex:` is the prefix live scholar candidates carry; without it an `openalex:A…` id would be
# taken for an X handle.
_CANONICAL_PREFIX = re.compile(
    r"^(x:user:|substack:|blog:|github:|org:|scholar:|openalex:|paper-authors:)", re.I)


def classify_reference(reference: str) -> str:
    """What kind of reference is this: 'canonical' (an existing per-cluster id → Mode C),
    'url' (a Substack/blog/site home), or 'handle' (a bare X @handle)? Mirrors `identify.resolve_handle`'s
    URL test (http… or a substack.com string). Anything else is a handle the host resolved from a
    name — a bare blog domain (no scheme) reads as a handle, so blog refs must arrive as full URLs."""
    r = (reference or "").strip()
    if _CANONICAL_PREFIX.match(r):
        return "canonical"
    if r.startswith("http") or "substack.com" in r:
        return "url"
    return "handle"


# Hosts whose URLs are profiles on a platform, not a personal site; keyed as `blog:{host}` they
# would mint a phantom entity. X is separate because it has a right answer: the bare @handle.
_PLATFORM_HOSTS: frozenset[str] = frozenset({"github.com"})
_X_HOSTS: frozenset[str] = frozenset({"x.com", "www.x.com", "twitter.com", "www.twitter.com"})
# Academic identity URLs, keyed by the prefix `canonical_identity` normalizes each host to.
# The value says whether OPYT can turn that URL into an id it can pull work by: OPYT pulls papers
# through OpenAlex only, so the pullable roots are the ids OpenAlex is keyed by.
SCHOLAR_ROOTS: dict[str, bool] = {
    "orcid.org/": True,                     # → an OpenAlex author id, one free lookup
    "semanticscholar.org/author/": False,
    "scholar.google.com/": False,
    "dblp.org/": False,
    "researchgate.net/": False,
    "academia.edu/": False,
    "arxiv.org/a/": False,
}


def scholar_root(reference: str) -> tuple[str, str] | None:
    """An academic identity URL → `(registry_prefix, identifier)`, or None if it is not one.

    Positive classification off `canonical_identity`, not a host deny-list: the canonicalizer
    already knows these are scholarly identity URLs and normalizes each to a stable prefix."""
    canon = canonical_identity((reference or "").strip())
    for prefix in SCHOLAR_ROOTS:
        if canon.startswith(prefix):
            return prefix, canon[len(prefix):]
    return None


def unsupported_root(reference: str) -> str | None:
    """Why this reference cannot be an Oracle root, or None if it can.

    Rootable: an X @handle, a Substack URL, a personal blog/site URL, a canonical_id, an academic
    identity URL OPYT can turn into a pullable author id (an ORCID, or an OpenAlex author URL
    carrying the id itself), or a research venue. Every other platform URL is refused rather than
    minted as a personal blog. A venue is admitted by the site-URL fallthrough; `identify` tells it
    from a blog, keeping this function network-free."""
    r = (reference or "").strip()
    if not r.startswith("http"):
        return None
    if not canonical_identity(r):
        return "that URL is unusable or belongs to an excluded platform"

    scholar = scholar_root(r)
    if scholar:
        prefix, _ = scholar
        if SCHOLAR_ROOTS[prefix]:
            return None                     # rootable — `identify.resolve_handle` mints a scholar entity
        # Refused with the shape that would work: any other registry would mint an Oracle with
        # nothing to pull.
        return (f"{prefix.rstrip('/')} has no author→works feed OPYT can pull — add them by their "
                "ORCID (https://orcid.org/0000-…) or their OpenAlex author page "
                "(https://openalex.org/A…), either of which resolves to a publication record "
                "OPYT can pull")

    host = (urlparse(r).hostname or "").lower()
    if host in _X_HOSTS:
        return ("that is an X profile URL — pass the bare @handle instead, so it resolves to the "
                "person's x:user id rather than minting a second, blog-shaped identity for them")
    if host in _PLATFORM_HOSTS:
        return (f"{host} is a platform profile, not a personal site — add the person by their X "
                "@handle, Substack, or blog")
    return None


def url_entity_id(url: str) -> str:
    """A home URL → its per-platform entity id: `substack:{host}` for substack.com, else
    `blog:{host}`. Same split as the `identify.resolve_handle` blog branch — one rule, two callers."""
    u = url if url.startswith("http") else f"https://{url}"
    return derive.substack_entity_id(None, u) if "substack.com" in u else derive.blog_entity_id(u)


def match_local_roster(conn, reference: str) -> dict | None:
    """Network-free dedup / Mode-C promote: does `reference` already map to a known entity? Returns
    {canonical_id, name, members} when it does (so we reflect the user's existing curation signals
    and never mint a duplicate Oracle), else None. Structural match only — an X handle via the
    stored `profile.handle`, a URL via its derived `substack:`/`blog:` id, a canonical_id directly.
    Never a fuzzy name match."""
    kind = classify_reference(reference)
    ref = reference.strip()
    if kind == "canonical":
        eid = ref
    elif kind == "url":
        eid = url_entity_id(ref)
    else:                                   # handle — match the stored profile.handle, no network
        row = conn.execute(
            "SELECT entity_id FROM entities WHERE entity_id LIKE 'x:user:%' "
            "AND profile IS NOT NULL "
            "AND lower(json_extract(profile, '$.handle')) = lower(?) LIMIT 1",
            (ref.lstrip("@"),),
        ).fetchone()
        eid = row[0] if row else None
    if not eid or schema.get_entity(conn, eid) is None:
        return None
    cid = schema.current_canonical(conn, eid)
    members = [{"entity_id": r["entity_id"], "name": r["name"],
                "identity_links": r["identity_links"]}
               for r in schema.entities_for_canonical(conn, cid)]
    return {"canonical_id": cid, "name": roster.name_for(conn, cid), "members": members}
