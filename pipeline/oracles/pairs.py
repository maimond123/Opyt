"""
pipeline/oracles/pairs.py — which pullable pairs an Oracle's entities name, and seeding them into
the registry.

One cluster member names at most one `(source_type, source_key)` pair (`pair_from_member`); an
Oracle's pairs are its members' pairs, one adapter per publication. `seed_from_entities` registers
them, `adopt_orphaned_sources` moves rows left under a pre-merge head (`resolve_entities` calls
it whenever it writes heads), and `retire_superseded` drops a pair a better one replaced. What a
pair has covered is `registry`'s.
"""

from __future__ import annotations

import json
import re
import sqlite3

from pipeline.kb import schema
from pipeline.kb.writes import write_tx
from pipeline.oracles import registry, review_items, roster
from pipeline.platforms.url_canon import host_of

_GITHUB_OWNER_RE = re.compile(r"github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/?(?:[?#]|$)",
                              re.I)


def _links(raw) -> list:
    """An entity's `identity_links` (a JSON string, a list, or a bare URL) → a list of strings."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, str)]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return [raw] if raw.startswith("http") else []
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, str)]
        if isinstance(parsed, str):
            return [parsed]
    return []


def _first_http(links) -> str | None:
    for u in _links(links):
        if u.startswith("http"):
            return u
    return None


def github_owners_from_links(links) -> list[str]:
    """GitHub owner logins named in an entity's `identity_links`. A GitHub node can carry an
    attested link without folding into the Oracle's cluster, so seeding from `github:*` members
    alone would miss it."""
    out: list[str] = []
    for u in _links(links):
        m = _GITHUB_OWNER_RE.search(u)
        if m and m.group(1).lower() not in ("orgs", "settings", "about", "features"):
            out.append(m.group(1))
    return out


def pair_from_member(member: sqlite3.Row) -> tuple[str, str] | None:
    """One cluster member entity → its `(source_type, source_key)` pair, or None when the member
    carries no pullable root (an `org:`/`paper-authors:` node, a `scholar:` node — a Semantic
    Scholar author id has no works feed OPYT can pull — or an X entity whose handle we never
    stored, since the adapter pulls `from:handle` and a bare numeric id is not enough).

    A pair carries no topic filter: that is a user decision and lives on the `oracle_sources` row.
    `source_key` is what the pull takes: an X handle, a publication or blog URL, a GitHub owner,
    an OpenAlex id."""
    eid = member["entity_id"] or ""
    links = member["identity_links"]

    if eid.startswith("x:user:"):
        profile = member["profile"]
        try:
            parsed = json.loads(profile) if isinstance(profile, str) else (profile or {})
        except (ValueError, TypeError):
            parsed = {}
        handle = ((parsed or {}).get("handle") or "").strip().lstrip("@")
        return ("x", handle) if handle else None

    if eid.startswith("substack:"):
        url = _first_http(links)
        if not url:
            tail = eid.split("substack:", 1)[1].strip()
            if not tail or tail == "unknown":
                return None
            # `substack_entity_id` keys on the author handle when it has one and on the host
            # otherwise, so a dot is the only signal telling the two apart.
            url = f"https://{tail}" if "." in tail else f"https://{tail}.substack.com"
        return ("substack", url)

    if eid.startswith("blog:"):
        url = _first_http(links)
        if not url:
            host = eid.split("blog:", 1)[1].strip()
            if not host or host == "unknown":
                return None
            url = f"https://{host}"
        return ("blog", url)

    if eid.startswith("openalex:"):
        # The bare OpenAlex id, which `works_filter` takes. An `A…` is an author and an `S…` a
        # source (a venue); one pair shape covers both because the id's prefix picks the field.
        openalex_id = eid.split("openalex:", 1)[1].strip()
        return ("openalex", openalex_id) if openalex_id else None

    if eid.startswith("github:"):
        owner = eid.split("github:", 1)[1].strip()
        # Entity ids are `github:{owner}`; `github:{owner}/{name}` is an atom id. Never register a
        # repo as if it were a feed.
        return ("github", owner) if owner and "/" not in owner else None

    return None


def _pairs_with_entities(conn: sqlite3.Connection,
                         canonical_id: str) -> tuple[list[tuple[str, tuple[str, str]]], list[str]]:
    """The one walk of an Oracle's cluster: `[(entity_id, (source_type, source_key))]` plus every
    `who_id` its atoms may carry. `pairs_for_oracle` and `pairs_by_entity` are two views of this,
    so the GitHub-from-links widening cannot be present in one and missing from the other."""
    members = schema.entities_for_canonical(conn, canonical_id)
    found: list[tuple[str, tuple[str, str]]] = []
    who_ids: list[str] = []
    for m in members:
        eid = m["entity_id"]
        who_ids.append(eid)
        p = pair_from_member(m)
        if p and p[0] in registry.SUPPORTED_SOURCES and p[1]:
            found.append((eid, p))
        for owner in github_owners_from_links(m["identity_links"]):
            # Keyed by the owner's own entity id, not the linking member's — the same GitHub can be
            # linked from an X member and a blog member, and both must resolve to one pair.
            found.append((f"github:{owner}", ("github", owner)))
            who_ids.append(f"github:{owner}")
    return found, list(dict.fromkeys(who_ids))


# What a footprint pull writes; other entry modes (a saved post) say nothing about the adapter.
FOOTPRINT_ENTRY_MODE = "oracle-footprint"


def _holds_atoms(conn: sqlite3.Connection, source_type: str, who_ids) -> bool:
    """Has a footprint pull of `source_type` ever landed atoms for this cluster?

    `entry_mode` is the whole question: a saved post from the user's bookmarks is evidence about
    their reading, not about whether the adapter works for this publication.
    """
    ids = [i for i in (who_ids or []) if i]
    if not ids:
        return False
    placeholders = ", ".join("?" for _ in ids)
    return conn.execute(
        f"SELECT 1 FROM atoms WHERE source_type=? AND entry_mode=? "
        f"AND who_id IN ({placeholders}) LIMIT 1",
        [source_type, FOOTPRINT_ENTRY_MODE, *ids]).fetchone() is not None


def _one_adapter_per_publication(conn: sqlite3.Connection, pairs: list[tuple[str, str]],
                                 who_ids) -> list[tuple[str, str]]:
    """One publication, one adapter — drop the losing pair when `substack` and `blog` both cover
    the same host.

    A Substack on a custom domain is two cluster members (`substack:{host}` and `blog:{host}`);
    pulling both duplicates the corpus under different atom ids. The adapter this cluster already
    holds footprint atoms from wins, and substack wins only when neither has pulled yet. The rule
    is stable and self-correcting: a source the gate refuses never accumulates atoms.
    """
    hosts = {}
    for stype, key in pairs:
        if stype in ("substack", "blog"):
            hosts.setdefault(host_of(key), set()).add(stype)
    contested = {h for h, kinds in hosts.items() if {"substack", "blog"} <= kinds}
    if not contested:
        return pairs

    # One query per adapter, not per host — a cluster's atoms are keyed by who_id, not by url.
    blog_has = _holds_atoms(conn, "blog", who_ids)
    substack_has = _holds_atoms(conn, "substack", who_ids)
    loser = "substack" if (blog_has and not substack_has) else "blog"
    return [(stype, key) for stype, key in pairs
            if not (stype == loser and host_of(key) in contested)]


def pairs_for_oracle(conn: sqlite3.Connection, canonical_id: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Every pullable `(source_type, source_key)` for one Oracle, plus the `who_id`s its atoms may
    carry (the cluster members, widened by any GitHub owner found only in a member's links). One
    adapter per publication (`_one_adapter_per_publication`)."""
    found, who_ids = _pairs_with_entities(conn, canonical_id)
    # De-dupe, preserving order — an X member and a blog member can both link the same GitHub.
    seen: set = set()
    deduped = [p for _eid, p in found if not (p in seen or seen.add(p))]
    return _one_adapter_per_publication(conn, deduped, who_ids), who_ids


def pair_url(source_type: str, source_key: str) -> str:
    """A pair as the URL a person would name it by: an X handle, a GitHub owner and an OpenAlex
    id become their profile pages; a Substack or blog key already is its URL."""
    if source_type == "x":
        return f"https://x.com/{source_key}"
    if source_type == "github":
        return f"https://github.com/{source_key}"
    if source_type == "openalex":
        return f"https://openalex.org/{source_key}"
    return source_key


def pairs_by_entity(conn: sqlite3.Connection, canonical_id: str) -> dict[str, tuple[str, str]]:
    """`entity_id -> (source_type, source_key)` for one Oracle's registered pairs.

    The join a caller needs to attribute an adapter run to the row that records it: both sides
    meet on a derived entity id rather than on two spellings of a URL."""
    found, _who_ids = _pairs_with_entities(conn, canonical_id)
    return dict(found)


def seed_from_entities(conn: sqlite3.Connection, canonical_ids=None, *, include_x: bool = True) -> dict:
    """Register every confirmed Oracle's enabled pullable sources. Idempotent after every ingest.

    `include_x` is false while OPYT has no managed X session. Discovery can still identify an
    Oracle's X profile, but that identity is not permission to schedule a session-backed pull.
    Existing X rows are intentionally preserved: they are the record of a past, connected pull
    and return to the registry when the user reconnects.

    Seeding registers rows; it does not claim coverage. A new pair gets `last_pulled_at = NULL`
    and only a corpus-derived `cursor_ts`; an existing pair keeps what the loop has recorded
    (`upsert_source`). The pull that happened is recorded by whoever performed it (`record_pull`)."""
    registry.init_state_schema(conn)
    rows = roster.list_oracles(conn)
    if canonical_ids:
        want = {schema.current_canonical(conn, c) for c in canonical_ids}
        rows = [o for o in rows if schema.current_canonical(conn, o["canonical_id"]) in want]

    # A row some writer registered under a stale id: `resolve_entities` moves the rest itself.
    adopted = adopt_orphaned_sources(conn)
    seeded = retired = 0
    for o in rows:
        cid = schema.current_canonical(conn, o["canonical_id"])
        pairs, who_ids = pairs_for_oracle(conn, cid)
        # A source the user removed (`forget(source=…)`) can still sit in the cluster: an X bio
        # that names a site attests it on its own. Membership is not permission to pull it.
        removed = review_items.dismissed_keys(conn, cid)
        for stype, key in pairs:
            if stype == "x" and not include_x:
                continue
            if (stype, review_items.source_key(pair_url(stype, key))) in removed:
                continue
            registry.upsert_source(conn, registry.SourceRow(
                canonical_id=cid, source_type=stype, source_key=key, status="trusted",
                cursor_ts=registry.latest_atom_ts(conn, stype, who_ids),
            ))
            seeded += 1
        retired += retire_superseded(conn, cid, pairs)
    return {"oracles": len(rows), "pairs": seeded, "retired": retired, "adopted": adopted}


def adopt_orphaned_sources(conn: sqlite3.Connection) -> int:
    """Re-point every row stranded under an id that is no longer its Oracle's head, preserving
    pull history.

    A merge moves an Oracle's head, and a row keeps the id it was written under. The scrapers and
    `progress` read pairs by head, so a stranded row is neither walked nor reported: this runs in
    the merge itself (`resolve.resolve_entities`). When both ids carry the pair, the row with the
    later `last_pulled_at` survives: the orphan is usually the one that did the pulling.
    """
    registry.init_state_schema(conn)
    with write_tx(conn, site="pairs.adopt_orphans"):
        adopted = 0
        for row in conn.execute(
                "SELECT s.*, e.canonical_id AS head FROM oracle_sources s "
                "JOIN entities e ON e.entity_id = s.canonical_id "
                "WHERE e.canonical_id IS NOT NULL AND e.canonical_id != s.canonical_id"
                ).fetchall():
            old = (row["canonical_id"], row["source_type"], row["source_key"])
            key = (row["head"], row["source_type"], row["source_key"])
            here = conn.execute(
                "SELECT last_pulled_at FROM oracle_sources "
                "WHERE canonical_id=? AND source_type=? AND source_key=?", key).fetchone()
            if here is None:
                registry.rehome(conn, old, row["head"])
            else:
                if (row["last_pulled_at"] or "") > (here["last_pulled_at"] or ""):
                    registry.take_pull(conn, key, row)
                registry.keep_coverage(conn, key, row)
                registry.delete_pair(conn, old)
            adopted += 1
    return adopted


def retire_superseded(conn: sqlite3.Connection, canonical_id: str,
                      pairs: list[tuple[str, str]]) -> int:
    """Delete this Oracle's registered website rows that `pairs_for_oracle` no longer produces,
    for the one reason it stops producing one: a `substack`/`blog` pair lost the contest for a
    host the other one also covers (`_one_adapter_per_publication`). The scrapers read rows, not
    `pairs_for_oracle`, so without this a losing row keeps being walked.

    Narrow on purpose: only a contested host qualifies, so preserved X rows and a pair that
    momentarily fails to derive keep their history. Atoms are not deleted; that is `forget`'s.
    """
    live = {(stype, host_of(key)) for stype, key in pairs if stype in ("substack", "blog")}
    with write_tx(conn, site="pairs.retire_superseded"):
        rows = conn.execute(
            "SELECT source_type, source_key FROM oracle_sources "
            "WHERE canonical_id=? AND source_type IN ('substack','blog')",
            (canonical_id,)).fetchall()

        registered_kinds: dict = {}
        for r in rows:
            registered_kinds.setdefault(host_of(r["source_key"]), set()).add(r["source_type"])

        retired = 0
        for r in rows:
            host = host_of(r["source_key"])
            contested = {"substack", "blog"} <= (registered_kinds.get(host) or set())
            if not contested or (r["source_type"], host) in live:
                continue
            registry.delete_pair(conn, (canonical_id, r["source_type"], r["source_key"]))
            retired += 1
    return retired
