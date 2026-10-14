"""
pipeline/oracles/identify.py — resolve a reference to a minted root entity.

An X handle becomes an `x:user:{id}` from its profile, an ORCID or OpenAlex author URL a scholar
entity, a research venue's URL an OpenAlex source, a Substack or blog URL its site entity; then the
resolve runs, so the reference lands on its cluster's current head. Network calls live here.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from pipeline.ingestion.utils import log
from pipeline.kb import derive
from pipeline.kb import resolve
from pipeline.kb import schema
from pipeline.oracles import reference
from pipeline.platforms import sessions
from pipeline.platforms.papers import openalex
from pipeline.platforms.x import profile as x_profile


def fetch_x_identity(handle: str) -> dict | None:
    """One X handle → {user_id, display_name, bio, site, verified, followers, handle} via the free
    cookie-scrape `UserByScreenName`. The numeric `rest_id` is the load-bearing field — it is what
    `x:user:{id}` entities key on, and `_probe_twitter_bio` does not surface it. Returns None on any
    failure, which the caller reports as unresolved.

    The x.com self-link blanking stays here, not in the transport: the other consumer of the same
    call skips the source instead."""
    try:
        session = sessions.x_session(f"https://x.com/{handle.lstrip('@')}")
        p = x_profile.fetch_user_profile(session, handle)
    except Exception:
        return None
    if not p:
        return None
    site = p["website"]
    return {
        "user_id": p["user_id"],
        "display_name": p["display_name"],
        "bio": p["bio"],
        "site": "" if ("twitter.com" in site or "x.com" in site) else site,
        "verified": p["verified"],
        "followers": p["followers"],
        "handle": p["handle"],
    }


def resolve_handle(conn, raw: str) -> str | None:
    """A pasted handle/URL → a resolved `canonical_id`, minting + resolving the entity along the
    way. X handle → `fetch_x_identity`; an academic identity URL → a scholar entity; a research
    venue → an `openalex:S…` entity; a substack.com (or any other home) URL → a substack or blog
    entity keyed on its subdomain/host. Returns None when unresolvable (fail-safe). Runs the full
    resolution after minting so the new entity merges into an existing cluster it cross-links.

    Order matters between the OpenAlex check and the blog fallthrough, and only in that direction:
    a venue that is not recognised is still a usable blog, while a blog wrongly taken for a venue
    would pull someone else's corpus. `openalex_root` carries the uniqueness rule that makes the
    first case the only one that happens."""
    raw = (raw or "").strip()
    if not raw:
        return None

    scholar = reference.scholar_root(raw) if raw.startswith("http") else None
    venue = openalex_root(raw) if (scholar is None and raw.startswith("http")) else None
    if scholar:
        prefix, identifier = scholar
        if not reference.SCHOLAR_ROOTS[prefix]:
            # Refused here too: `confirm(add_handles=[…])` reaches this without passing
            # `reference.unsupported_root`.
            return None
        eid = _mint_orcid_entity(conn, identifier)
        if not eid:
            return None
    elif venue:
        # A venue mints the same `openalex:` entity shape a researcher does, with a source id,
        # so it inherits the scholar pull, DOI dedup and the topic filter.
        eid = f"openalex:{venue['openalex_id']}"
        schema.upsert_entity(conn, eid, name=venue.get("name"), identity_links=[raw])
    elif raw.startswith("http") or "substack.com" in raw:
        url = raw if raw.startswith("http") else f"https://{raw}"
        # A substack.com URL keys on the Substack subdomain/host; any other http home is a
        # generic blog keyed on `blog:{host}`.
        eid = derive.substack_entity_id(None, url) if "substack.com" in url \
            else derive.blog_entity_id(url)
        schema.upsert_entity(conn, eid, identity_links=[url])
    else:
        ident = fetch_x_identity(raw)
        if not ident:
            return None
        eid = f"x:user:{ident['user_id']}"
        prof = {k: ident[k] for k in ("bio", "verified", "followers", "handle") if ident.get(k)}
        schema.upsert_entity(conn, eid, name=ident.get("display_name"),
                             identity_links=[ident["site"]] if ident.get("site") else None,
                             profile=prof or None)

    resolve.resolve_entities(conn)                 # recompute canonical (merge if it cross-links)
    row = schema.get_entity(conn, eid)
    return (row["canonical_id"] if row and row["canonical_id"] else eid)


def _mint_orcid_entity(conn, orcid: str) -> str | None:
    """An ORCID → the `openalex:` entity that has a works feed, or None when unresolvable.

    The ORCID is resolved to an OpenAlex author id before minting, because only that id has a works
    feed: `openalex:{id}` is a pullable Oracle and `orcid:{id}` would be an Oracle whose every
    refresh found nothing. The ORCID is still stored as the entity's identity link, which is what
    merges it with a `scholar:` entity for the same person.

    ORCID is the only prefix `reference.SCHOLAR_ROOTS` marks pullable, so this takes the identifier and not
    a `(prefix, id)` pair. A second pullable registry brings the parameter back with it.
    """

    # One free lookup; a failure refuses rather than falling back to a `blog:` entity.
    try:
        rec = openalex.OpenAlexClient().author_by_orcid(orcid)
    except Exception as e:
        log(f"[oracles] ORCID {orcid} lookup failed: {type(e).__name__}: {e}")
        rec = None
    author_id = str((rec or {}).get("id") or "").rsplit("/", 1)[-1]
    if not author_id:
        return None
    eid = f"openalex:{author_id}"
    schema.upsert_entity(conn, eid, name=(rec or {}).get("display_name"),
                         identity_links=[f"https://orcid.org/{orcid}"])
    return eid


# An OpenAlex id URL pasted directly — `https://openalex.org/A5009290031` (an author) or
# `https://openalex.org/S4393918830` (a source). Both letters, because both refusal messages name
# one of these as the shape that works.
_OPENALEX_ID_RE = re.compile(r"openalex\.org/([AS]\d+)\b", re.I)


def openalex_root(reference: str) -> dict | None:
    """A URL that names an OpenAlex id → `{openalex_id, name, works}`, or None if it is not one.

    An `A…` is an author and an `S…` a venue; both become `openalex:{id}`, and every later stage is
    identical. A venue is a filter on OpenAlex, never a new ingester.

    Two accepted forms: an OpenAlex id URL, matched as a string; or a host whose name resolves to
    exactly one OpenAlex source. That name search is admissible only because of the uniqueness
    rule: personal-blog hosts match zero sources, and ambiguous names like `medium` match many.
    Any lookup failure returns None, so the URL falls through to `blog:{host}`.
    """

    r = (reference or "").strip()
    direct = _OPENALEX_ID_RE.search(r)
    if direct:
        oid = direct.group(1).upper()
        adapter = openalex.OpenAlexClient()
        try:
            rec = (adapter.author(oid) if oid.startswith("A") else adapter.source(oid)) or {}
        except Exception:
            rec = {}                        # the id itself is already enough to root on
        return {"openalex_id": oid, "name": rec.get("display_name"),
                "works": rec.get("works_count")}

    # The label immediately before the TLD names the publisher (`journals.sagepub.com` →
    # sagepub). Blog-platform hosts reduce to their platform name, which matches no source.
    host = (urlparse(r).hostname or "").lower().removeprefix("www.")
    parts = [x for x in host.split(".") if x]
    label = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")
    if len(label) < 3:
        return None
    adapter = openalex.OpenAlexClient()
    if not adapter.available():
        return None                         # breaker open: skip the lookup, do not pay for it
    try:
        hits = adapter.sources_by_name(label)
    except Exception:
        return None                         # fail-safe: unchanged, so it stays a blog
    if len(hits) != 1:
        # Zero (a personal blog) or several (an ambiguous name) — both stay blogs. The refusal
        # copy in `reference.unsupported_root` only fires for the ambiguous half.
        return None
    rec = hits[0]
    sid = str(rec.get("id") or "").rstrip("/").rsplit("/", 1)[-1]
    if not sid.startswith("S"):
        return None
    return {"openalex_id": sid, "name": rec.get("display_name"),
            "works": rec.get("works_count")}
