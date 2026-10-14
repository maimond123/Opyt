"""
pipeline/kb/expand.py — root an Oracle for footprint discovery, and list the profile pages of
theirs it reads.

Shared helpers with no loop of their own; the callers (background discovery, the Oracle review
flow) own orchestration.
"""

from __future__ import annotations

import json
import re
import sqlite3

from pipeline.platforms.url_canon import canonical_identity

from . import schema

_X_HANDLE = re.compile(r"(?:x|twitter)\.com/([^/?#]+)", re.I)


def _x_handles(conn: sqlite3.Connection, canonical_id: str) -> list[str]:
    """Every X member's @handle, from its `profile.handle` (entities are keyed by numeric id).
    Resolves a stale canonical_id to the current head first."""
    head = schema.current_canonical(conn, canonical_id)
    rows = conn.execute(
        "SELECT profile FROM entities WHERE COALESCE(canonical_id, entity_id)=? "
        "AND entity_id LIKE 'x:user:%' AND profile IS NOT NULL",
        (head,),
    ).fetchall()
    return [h for h in ((json.loads(r[0]) or {}).get("handle") for r in rows if r[0]) if h]


def _x_handle(conn: sqlite3.Connection, canonical_id: str) -> str | None:
    """The Oracle's X @handle: the first X member's."""
    return next(iter(_x_handles(conn, canonical_id)), None)


def _x_handle_from_url(url: str | None) -> str | None:
    m = _X_HANDLE.search(url or "")
    return m.group(1) if m else None


def _substack_handle(oracle: dict) -> str | None:
    """A Substack member's handle (or subdomain) from its `substack:` entity id, or None."""
    for m in oracle.get("members", []):
        eid = m.get("entity_id", "")
        if eid.startswith("substack:"):
            h = eid.split("substack:", 1)[1].strip()
            if h and h != "unknown":
                return h
    return None


def _first_url(links) -> str | None:
    """The first http URL in an entity's `identity_links` (a JSON string or a list), or None."""
    if not links:
        return None
    if isinstance(links, str):
        try:
            links = json.loads(links)
        except (ValueError, TypeError):
            return links if links.startswith("http") else None
    if isinstance(links, list):
        for u in links:
            if isinstance(u, str) and u.startswith("http"):
                return u
    return None


def _blog_home(oracle: dict) -> str | None:
    """A blog member's home URL: its stored link, else rebuilt from the `blog:{host}` id."""
    for m in oracle.get("members", []):
        eid = m.get("entity_id", "")
        if eid.startswith("blog:"):
            url = _first_url(m.get("identity_links"))
            if url:
                return url
            host = eid.split("blog:", 1)[1].strip()
            if host and host != "unknown":
                return f"https://{host}"
    return None


def _orcid(member: dict) -> str | None:
    """The ORCID iD a member's `identity_links` carries, or None."""
    links = member.get("identity_links")
    if isinstance(links, str):
        try:
            links = json.loads(links)
        except (ValueError, TypeError):
            links = []
    return next((u.rstrip("/").rsplit("/", 1)[-1] for u in links or []
                 if isinstance(u, str) and "orcid.org/" in u), None)


def _researcher(oracle: dict) -> dict | None:
    """A researcher member's root: its ORCID iD when the `openalex:A…` entity carries one, else
    the OpenAlex author id, whose record the probe reads for it. A venue (`openalex:S…`) is not a
    person and roots nothing."""
    for m in oracle.get("members", []):
        eid = m.get("entity_id", "")
        if not eid.startswith("openalex:A"):
            continue
        orcid = _orcid(m)
        return ({"seed": orcid, "seed_type": "orcid"} if orcid else
                {"seed": eid.split(":", 1)[1], "seed_type": "openalex"})
    return None


def _root_profile(conn: sqlite3.Connection, oracle: dict) -> dict | None:
    """The Oracle's best confirmed root profile for discovery, as {seed, seed_type}.

    Preference: X (richest probe), then Substack, then blog, then a researcher's ORCID. None for
    a GitHub-only Oracle or a venue.
    """
    x = _x_handle(conn, oracle["canonical_id"])
    if x:
        return {"seed": x, "seed_type": "x"}
    sub = _substack_handle(oracle)
    if sub:
        return {"seed": sub, "seed_type": "substack"}
    blog = _blog_home(oracle)
    if blog:
        return {"seed": blog, "seed_type": "blog"}
    return _researcher(oracle)


def profile_url(page: dict) -> str:
    """The URL of one profile page, `{seed, seed_type}` as `_root_profile` gives it."""
    seed, kind = page["seed"], page["seed_type"]
    if kind == "x":
        return f"https://x.com/{seed}"
    if kind == "substack":
        return seed if "." in seed or "/" in seed else f"https://{seed}.substack.com"
    if kind == "github":
        return f"https://github.com/{seed}"
    if kind == "orcid":
        return f"https://orcid.org/{seed}"
    if kind == "openalex":
        return f"https://openalex.org/{seed}"
    return seed


def _profile_page(member: dict) -> dict | None:
    """A non-X member's profile page, the one that declares accounts as theirs, or None.

    A Substack is read by the publication it is: an id can be its author's handle
    (`substack:mraginsky`), which names no publication. A blog's home page is no profile: it links
    employers, friends and blogrolls, and a declaration counts as theirs with no link back. An
    OpenAlex author with no ORCID has no page of its own."""
    eid = member.get("entity_id", "")
    kind, _, rest = eid.partition(":")
    if kind == "substack" and rest and rest != "unknown":
        return {"seed": _first_url(member.get("identity_links")) or rest, "seed_type": "substack"}
    if kind == "github" and rest and "/" not in rest:
        return {"seed": rest, "seed_type": "github"}
    if eid.startswith("openalex:A") and (orcid := _orcid(member)):
        return {"seed": orcid, "seed_type": "orcid"}
    return None


def member_profiles(conn: sqlite3.Connection, oracle: dict) -> list[dict]:
    """Every profile page of theirs discovery reads, as `{seed, seed_type}`: the root
    (`_root_profile`) first, then each other X account, Substack, GitHub account and ORCID record
    of the Oracle's cluster, each page once. Empty when nothing roots
    discovery: a GitHub-only Oracle or a venue."""
    root = _root_profile(conn, oracle)
    if root is None:
        return []
    others = [*({"seed": h, "seed_type": "x"} for h in _x_handles(conn, oracle["canonical_id"])),
              *filter(None, map(_profile_page, oracle.get("members", [])))]
    pages, seen = [root], {canonical_identity(profile_url(root))}
    for page in others:
        key = canonical_identity(profile_url(page))
        if key not in seen:
            seen.add(key)
            pages.append(page)
    return pages
