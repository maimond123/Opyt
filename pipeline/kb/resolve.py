"""
pipeline/kb/resolve.py — Stage-3 entity resolution (attested-only, locked 2026-07-16).

Step-2 mints one entity per platform (e.g. `x:user:{rest_id}`, `substack:{handle}`) for what may
be the same human. Stage-3 links same-person rows into one canonical entity by writing
`canonical_id` (the cluster head; itself when unmerged).

Merge rule: two entities merge iff one's `attests` URL set hits the other's `self` set, or they
share a `self` (union-find). Never merge on shared `attests` alone — that's the squatter defense.
`canonical_id` is fully recomputed from `identity_links` on every run; no evidence table.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from pipeline.ingestion.url_canon import canonical_identity

from . import schema


# ── URL-set extraction ───────────────────────────────────────────────────────────

# Platforms whose stored `identity_links` IS the entity's own home, so the link counts as `self`
# rather than `attests`.
_SELF_PLATFORMS = frozenset({"substack", "blog", "site", "scholar", "openalex"})

# The scholar platforms are in that set with a RESTRICTION, applied in `_url_sets`: only an ORCID
# counts as their `self`. Nobody can claim another person's ORCID — the same uniqueness property
# `blog_entity_id`'s docstring requires of a blog host — while two researchers in one department
# share an institution URL and would false-merge on it. Without the restriction the safety of the
# merge would depend on every distant writer being disciplined about what it puts in
# `identity_links`, which is the kind of implicit invariant that breaks the first time someone
# adds a lab page.
_SCHOLAR_PLATFORMS = frozenset({"scholar", "openalex"})
_ORCID_PREFIX = "orcid.org/"


def _url_sets(entity_id: str, identity_links) -> tuple[frozenset, frozenset]:
    """(`self`, `attests`) canonical-URL sets for one entity — see the module docstring for
    the per-platform `who_site` semantics this encodes. A footprint entity's stored link IS its
    home (self, see `_SELF_PLATFORMS`); every other platform's stored link is an outbound
    attestation. Empty/garbage links drop out via `canonical_identity`."""
    platform = (entity_id.split(":", 1)[0] or "").lower()
    canon = frozenset(
        c for c in (canonical_identity(u) for u in (identity_links or [])) if c
    )
    if platform in _SCHOLAR_PLATFORMS:
        # ORCID only as `self`; anything else the entity carries is an outbound attestation. This
        # is what lets `openalex:A5043841592` and `scholar:2081297` merge into one person while an
        # institution page shared by two researchers merges nobody.
        selfs = frozenset(c for c in canon if c.startswith(_ORCID_PREFIX))
        return selfs, canon - selfs
    if platform in _SELF_PLATFORMS:
        return canon, frozenset()      # a footprint source's stored link IS its home
    return frozenset(), canon          # X (and any outbound-storing platform): attests only


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


def _components(entities) -> dict:
    """`{entity_id: canonical_id}` over the attested graph. `entities` is an iterable of
    (entity_id, identity_links); canonical_id is the component's min entity_id (its own id
    when unmerged)."""
    dsu = _DSU()
    self_index: dict[str, set] = defaultdict(set)    # url → entities that ARE it
    attest_index: dict[str, set] = defaultdict(set)  # url → entities that LINK to it

    for eid, links in entities:
        dsu.add(eid)                                 # every entity present (singletons too)
        selfs, attests = _url_sets(eid, links)
        for u in selfs:
            self_index[u].add(eid)
        for u in attests:
            attest_index[u].add(eid)

    # Union only along self-URLs — a shared third-party link never gets iterated here, so
    # co-attesters never merge. This is the squatter defense.
    for url, owners in self_index.items():
        group = owners | attest_index.get(url, set())
        it = iter(group)
        first = next(it)
        for other in it:
            dsu.union(first, other)

    return {eid: dsu.find(eid) for eid in dsu.parent}


# ── stats + orchestration ────────────────────────────────────────────────────────

@dataclass
class ResolveStats:
    """A run's outcome. `duplicate_rows_collapsed` (total − components) is the recall signal a
    deferred fuzzy-match sweep would improve."""

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


def backfill_entities_from_atoms(conn) -> int:
    """Ensure every atom AUTHOR (`atoms.who_id`) has an `entities` row, so resolution + Stage-4
    see all authors even if an ingest path wrote an atom without upserting its entity. Creates
    a minimal row (name/kind/links NULL) for any missing who_id; returns the count created
    (logged when > 0). Idempotent."""
    missing = [r[0] for r in conn.execute(
        "SELECT DISTINCT a.who_id FROM atoms a "
        "LEFT JOIN entities e ON e.entity_id = a.who_id "
        "WHERE a.who_id IS NOT NULL AND e.entity_id IS NULL"
    )]
    for who_id in missing:
        conn.execute("INSERT OR IGNORE INTO entities (entity_id) VALUES (?)", (who_id,))
    if missing:
        conn.commit()
        from pipeline.ingestion.utils import log
        log(f"[resolve] backfilled {len(missing)} entity row(s) missing for atom authors")
    return len(missing)


def resolve_entities(conn, *, dry_run: bool = False) -> ResolveStats:
    """Recompute every entity's `canonical_id` from the attested `identity_links` graph and
    (unless `dry_run`) write it back. Idempotent + safe to re-run; degrades to zeroed stats on
    an empty `entities` table. `dry_run` is pure-read (skips the atom-author backfill too)."""
    if not dry_run:
        backfill_entities_from_atoms(conn)
    rows = schema.all_entities(conn)
    entities = [(r["entity_id"], _loads(r["identity_links"])) for r in rows]
    mapping = _components(entities)
    if not dry_run and mapping:
        schema.set_canonical_ids(conn, mapping)
        # THE ONE PLACE A HEAD MOVES, so the one place the `oracles` rows pointing at the old one
        # are repaired. Leaving them stale is what `current_canonical` exists to absorb on read —
        # and every join that forgot to call it returned a confidently wrong answer instead
        # (`trusted_atoms` off by 10×, an Oracle's display name resolving to NULL, a second
        # `oracle_sources` row under the dead id). `reanchor_oracles` says what each cost.
        schema.reanchor_oracles(conn)
    return _stats(mapping)
