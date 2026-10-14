"""
pipeline/kb/resolve.py — Stage-3 entity resolution (attested links only).

Step-2 mints one entity per platform (e.g. `x:user:{rest_id}`, `substack:{handle}`) for what may
be the same human. Stage-3 links same-person rows into one canonical entity by writing
`canonical_id` (the cluster head; itself when unmerged).

Merge rule: two entities merge iff one's `attests` URL set hits the other's `self` set, or they
share a `self` (union-find). Never merge on shared `attests` alone — that's the squatter defense.
`canonical_id` is fully recomputed from `identity_links` on every run. The one other input is
the user's own word: a source added for an Oracle (`oracle_review_items`, status `added`) joins
that Oracle's cluster however the Oracle's own profile is refreshed, and leaves it again once
dismissed (`_join_added`). The entity that joins is the one that IS the added URL (`own_urls`):
a site its home, a scholar its ORCID, an X account its profile (`x.com/{handle}`, from the stored
handle, since its stored links are its bio's), and a GitHub account `github.com/{owner}`, which
counts against added sources only. A source several Oracles added joins the one that added it
first, and never makes two Oracles one person.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

from pipeline.oracles import pairs
from pipeline.platforms.url_canon import canonical_identity

from . import schema
from .writes import write_tx


# ── URL-set extraction ───────────────────────────────────────────────────────────

# Platforms whose stored `identity_links` is the entity's own home, so the link counts as `self`
# rather than `attests`.
_SELF_PLATFORMS = frozenset({"substack", "blog", "site", "scholar", "openalex"})

# For the scholar platforms only an ORCID counts as `self`: it is unique per person, while an
# institution URL is shared by colleagues and would false-merge them.
_SCHOLAR_PLATFORMS = frozenset({"scholar", "openalex"})
_ORCID_PREFIX = "orcid.org/"


def _url_sets(entity_id: str, identity_links,
              handle: str | None = None) -> tuple[frozenset, frozenset]:
    """(`self`, `attests`) canonical-URL sets for one entity, per `_SELF_PLATFORMS`. `handle` is
    an X account's stored @handle: its profile is its `self`."""
    platform = (entity_id.split(":", 1)[0] or "").lower()
    canon = frozenset(
        c for c in (canonical_identity(u) for u in (identity_links or [])) if c
    )
    if platform in _SCHOLAR_PLATFORMS:
        selfs = frozenset(c for c in canon if c.startswith(_ORCID_PREFIX))
        return selfs, canon - selfs
    if platform in _SELF_PLATFORMS:
        return canon, frozenset()      # a footprint source's stored link is its home
    if platform == "x" and handle:
        return frozenset({canonical_identity(f"https://x.com/{handle}")}), canon
    return frozenset(), canon          # an outbound-storing platform: attests only


def own_urls(entity_id: str, identity_links, handle: str | None = None) -> frozenset:
    """The canonical URLs an entity IS, as a source added for an Oracle names it: its `self` set,
    and a GitHub account's `github.com/{owner}`, which is never matched against anyone's links."""
    selfs = _url_sets(entity_id, identity_links, handle)[0]
    if entity_id.startswith("github:"):
        return selfs | {"github.com/" + entity_id.split(":", 1)[1].lower()}
    return selfs


# ── union-find over the self-anchored URL graph ──────────────────────────────────

class _DSU:
    """Disjoint-set with path compression, unioning toward the lexicographically smallest id so
    `find` returns a deterministic, stable representative — the `canonical_id`."""

    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:              # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            hi, lo = (ra, rb) if ra > rb else (rb, ra)
            self.parent[hi] = lo                   # smaller id wins → stable min-representative


def _components(entities: list, added: dict) -> dict:
    """`{entity_id: canonical_id}` over the attested graph, then the sources added for Oracles.
    `entities` is a list of (entity_id, identity_links, handle), the stored links only; `added`
    is `_added_sources`. canonical_id is the component's min entity_id (its own id when
    unmerged)."""
    dsu = _DSU()
    self_index: dict[str, set] = defaultdict(set)    # url → entities that are it
    attest_index: dict[str, set] = defaultdict(set)  # url → entities that link to it

    for eid, links, handle in entities:
        dsu.add(eid)                                 # every entity present (singletons too)
        selfs, attests = _url_sets(eid, links, handle)
        for u in selfs:
            self_index[u].add(eid)
        for u in attests:
            attest_index[u].add(eid)

    # Union only along self-URLs, so co-attesters of a third-party link never merge.
    for url, owners in self_index.items():
        group = owners | attest_index.get(url, set())
        it = iter(group)
        first = next(it)
        for other in it:
            dsu.union(first, other)

    _join_added(dsu, entities, added)
    return {eid: dsu.find(eid) for eid in dsu.parent}


def _join_added(dsu: _DSU, entities: list, added: dict) -> None:
    """A source added for an Oracle joins it: the cluster of the entity that IS the added URL
    (`own_urls`) joins the Oracle's. Decided once, on the clusters the stored links made, and only
    then joined, so no join can turn a claim into a different one:

    | the cluster an added URL names                   | joins                          |
    |--------------------------------------------------|--------------------------------|
    | named by one Oracle                              | that Oracle                    |
    | named by several Oracles                         | the one whose add is earliest  |
    | is one of the naming Oracles' own cluster        | nothing: it stays theirs       |

    The last row is why one source never makes two Oracles one person. A later adder's claim
    waits: it wins on the resolve after the earlier add is dismissed or its Oracle forgotten."""
    owners: dict[str, set] = defaultdict(set)       # url → entities that are it
    for eid, links, handle in entities:
        for url in own_urls(eid, links, handle):
            owners[url].add(eid)
    claims: dict[str, list] = defaultdict(list)     # cluster → [(add key, Oracle cluster)]
    for oracle, rows in added.items():
        if oracle in dsu.parent:
            for url, key in rows:
                for owner in owners.get(canonical_identity(url), ()):
                    claims[dsu.find(owner)].append((key, dsu.find(oracle)))
    joins = [(cluster, min(by)[1]) for cluster, by in claims.items()
             if cluster not in {oracle for _key, oracle in by}]
    for cluster, oracle in joins:
        dsu.union(cluster, oracle)


# ── stats + orchestration ────────────────────────────────────────────────────────

@dataclass
class ResolveStats:
    """A run's outcome."""

    total_entities: int = 0
    components: int = 0            # distinct canonical_ids (== distinct people/orgs)
    merged_entities: int = 0      # entities that landed in a component of size > 1
    cross_platform: int = 0       # components spanning more than one platform
    merges: list = field(default_factory=list)   # [(canonical_id, [member ids])] size > 1

    def as_dict(self) -> dict:
        return {
            "total_entities": self.total_entities,
            "components": self.components,
            "merged_entities": self.merged_entities,
            "cross_platform": self.cross_platform,
            "duplicate_rows_collapsed": self.total_entities - self.components,
            "merges": self.merges,
        }


def _platform(entity_id: str) -> str:
    return (entity_id.split(":", 1)[0] or "").lower()


def _stats(mapping: dict) -> ResolveStats:
    members: dict[str, list] = defaultdict(list)
    for eid, cid in mapping.items():
        members[cid].append(eid)
    st = ResolveStats(total_entities=len(mapping), components=len(members))
    for cid, ms in members.items():
        if len(ms) > 1:
            st.merged_entities += len(ms)
            st.merges.append((cid, sorted(ms)))
            if len({_platform(m) for m in ms}) > 1:
                st.cross_platform += 1
    st.merges.sort()
    return st


def _handle(profile) -> str | None:
    """The @handle an entity's stored profile carries (every X writer stores one), or None."""
    return json.loads(profile).get("handle") if profile else None


def _loads(links) -> list:
    """`identity_links` is stored as a JSON string; tolerate None/str/already-list."""
    if not links:
        return []
    if isinstance(links, list):
        return links
    try:
        v = json.loads(links)
        return v if isinstance(v, list) else [v]
    except (ValueError, TypeError):
        return []


# ── resolution ─────────────────────────────────────────────────────────────────
def backfill_entities_from_atoms(conn) -> int:
    """Create a minimal `entities` row for any atom author missing one; returns the count."""
    with write_tx(conn, site="resolve.backfill_entities"):
        missing = [r[0] for r in conn.execute(
            "SELECT DISTINCT a.who_id FROM atoms a "
            "LEFT JOIN entities e ON e.entity_id = a.who_id "
            "WHERE a.who_id IS NOT NULL AND e.entity_id IS NULL"
        )]
        for who_id in missing:
            conn.execute("INSERT OR IGNORE INTO entities (entity_id) VALUES (?)", (who_id,))
    if missing:
        from pipeline.ingestion.utils import log
        log(f"[resolve] backfilled {len(missing)} entity row(s) missing for atom authors")
    return len(missing)


def _added_sources(conn) -> dict[str, list[tuple[str, tuple]]]:
    """`{entity_id: [(source url, add key)]}` for the sources added to an Oracle: discovery was
    sure of them or the user added them (`oracle_review_items`). The add key orders adds by when
    each row became `added` (`updated_at`: set on insert, and again when the user adds a pending
    row; `created_at` is when it was found), then by `review_id`. A store without the table adds
    nothing."""
    try:
        rows = conn.execute(
            "SELECT canonical_id, source_url, updated_at, review_id FROM oracle_review_items "
            "WHERE status='added' AND kind='source'").fetchall()
    except sqlite3.OperationalError:
        return {}
    out: dict[str, list[tuple[str, tuple]]] = defaultdict(list)
    for cid, url, added_at, review_id in rows:
        out[cid].append((url, (added_at, review_id)))
    return out


def claimed_elsewhere(conn, entity_ids: list[str], head: str) -> set[str]:
    """Of `entity_ids`, those a source added for an Oracle other than `head` names: forgetting
    `head`, or one of its sources, leaves what another Oracle still collects."""
    urls = {canonical_identity(url) for cid, rows in _added_sources(conn).items()
            if schema.current_canonical(conn, cid) != head for url, _key in rows}
    out = set()
    for eid in entity_ids:
        row = schema.get_entity(conn, eid)
        if row and own_urls(eid, _loads(row["identity_links"]), _handle(row["profile"])) & urls:
            out.add(eid)
    return out


def resolve_entities(conn, *, dry_run: bool = False) -> ResolveStats:
    """Recompute every entity's `canonical_id` from the `identity_links` graph and, unless
    `dry_run` (pure read), write it back. Idempotent."""
    if not dry_run:
        backfill_entities_from_atoms(conn)
    rows = schema.all_entities(conn)
    added = _added_sources(conn)
    entities = [(r["entity_id"], _loads(r["identity_links"]), _handle(r["profile"]))
                for r in rows]
    mapping = _components(entities, added)
    if not dry_run and mapping:
        # A head that moves leaves `oracles` rows on the id each was confirmed under: every
        # roster read folds them (`schema.ROSTER_HEADS_SQL`, `oracles/roster.py`).
        schema.set_canonical_ids(conn, mapping)
        # `oracle_sources` rows are keyed by head and read by head, so they move with it here,
        # in the one place a head changes.
        pairs.adopt_orphaned_sources(conn)
    return _stats(mapping)


def cluster_self_urls(conn, canonical_id: str) -> frozenset:
    """The canonical URLs one cluster's members ARE (`own_urls`: a site's home, a scholar's ORCID,
    an X account's profile, a GitHub account). Discovery treats them as already known."""
    return frozenset().union(*(
        own_urls(r["entity_id"], _loads(r["identity_links"]), _handle(r["profile"]))
        for r in schema.entities_for_canonical(conn, canonical_id)))
