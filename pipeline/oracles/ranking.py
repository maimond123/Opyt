"""
pipeline/oracles/ranking.py — who the candidates are, and in what order.

Every signal-bearing canonical entity is a candidate; `rank` groups the gathered signal rows by
canonical id, marks the retired, and sorts by `Candidate.sort_key`; `stronger_source` applies the
same order per source, to say which one source a person is listed under on the screen.
`rank_candidates(conn)` gathers the rows and calls `rank`.

Invariants:
  • Group by canonical_id, never per-platform entity_id, or a cross-platform person is
    double-counted.
  • Endorsement first: a person the user endorsed outranks anyone they only ever read. Paper
    registries have no follow primitive, so they are one source (`PAPERS`) with their own group.
  • `count` is a soft tiebreak only, third behind endorsement and distinct-signal count; never a
    count-weighted score.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from pipeline.curation import clock
from pipeline.curation.x_collectors import Following
from pipeline.kb import derive, schema
from pipeline.oracles import roster
from pipeline.timeparse import parse_ts

# The person-endorsement signals — the primary rank tier: a follow/list/subscribe is the user
# vouching for the person directly; a save/like vouches for a piece of content (the author is
# inferred). Subscribe and list stay grouped with follow because all three are person-level acts.
#
# `coauthor` and `recommended` must never join this set: the user performed no act. Ranking
# someone else's vouch above a follow the user made by hand would invert what the tier means.
_ENDORSEMENT = frozenset({"follow", "list", "subscribe"})

# The registries a researcher's entity id comes from. They have no follow primitive — a person
# cannot follow an author on OpenAlex — so a scholar candidate can never earn an endorsement
# signal and would sort below every X follow permanently under `sort_key` alone. They are one
# source (`PAPERS`) with its own group, which fixes that without weakening the endorsement key.
_SCHOLAR_PLATFORMS = frozenset({"scholar", "openalex"})

# A source is a platform as the user meets it, so the two paper registries are one source.
PAPERS = "papers"
# Print order of the screen's source groups. A source not named here (none exists today) comes
# after these.
SOURCE_ORDER = ("x", "substack", PAPERS)


# ── Candidate ────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    canonical_id: str
    name: str | None = None
    handle: str | None = None
    distinct_signals: int = 0               # # of distinct (signal_type, platform) — rank key b
    has_endorsement: bool = False           # any follow/list/subscribe — the primary rank key
    total_count: int = 0                    # Σ action count (soft tiebreak only)
    signals: list = field(default_factory=list)      # [{signal_type, platform, count, extra}]
    members: list = field(default_factory=list)      # every per-platform entity_id in the cluster
    identity_links: list = field(default_factory=list)
    profile: dict = field(default_factory=dict)      # {bio, verified, followers, …}
    retired: bool = False                            # unfollowed — see `rank_candidates`

    @property
    def is_scholar(self) -> bool:
        """Every signal this person carries came from a paper registry.

        All, not any: someone the user follows on X who also wrote a paper they saved has a real
        endorsement and is listed by the source that endorsement is on (`stronger_source`). Only a
        person the user knows purely as an author is listed under their papers alone."""
        return bool(self.signals) and all(
            s["platform"] in _SCHOLAR_PLATFORMS for s in self.signals)

    def sort_key(self) -> tuple:
        # Descending priority, tiered not weighted: endorsement (a person-level act) wins
        # categorically, then distinct signals (variety beats volume), then count. canonical_id
        # last so the order is stable across renders.
        return (not self.has_endorsement, -self.distinct_signals,
                -self.total_count, self.canonical_id)


# ── (a) ranking ────────────────────────────────────────────────────────────────

def _loads(v) -> list | dict:
    if not v:
        return []
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return []


def _as_dict(v) -> dict:
    """A stored JSON field that must be a dict (profile/extra) → a dict, defaulting to {} for
    null/empty/malformed. `_loads` defaults to [] (right for list fields like identity_links);
    calling .get/.items on that [] is the trap this closes."""
    d = _loads(v)
    return d if isinstance(d, dict) else {}


# The one collector whose walk is trusted to prove an absence: X following. Named as a collector,
# not a signal type, because `substack_follows` lands `follow` rows too. The Substack follow walk
# is not trusted for an absence: Cloudflare refusals make "not seen" routinely a block.
MAINTAINED = Following
MAINTAINED_COLLECTOR = MAINTAINED.collector


def _retired_ids(conn) -> set[str]:
    """Entity ids whose maintained signal a healthy full walk failed to re-confirm.

    Two gates, and both must pass before absence counts as evidence. The collector's last walk has
    to be trustworthy (`clock.walk_is_trustworthy` — an `ok` run whose `found` did not
    collapse), and the signal's own `last_confirmed_at` has to predate that walk.

    Fail-safe is the empty set, and every failure path returns it: no clock, an unreadable
    clock, an untrustworthy walk, an unparseable stamp. Retiring nobody costs a cycle; retiring
    someone a broken walk failed to see costs a signal no later run brings back. Only store errors
    are caught; a programming error must not look like a healthy, empty retirement."""
    try:
        run = clock.get_run(conn, MAINTAINED.collector)
        if not clock.walk_is_trustworthy(run):
            return set()
        # The walk's start, not its finish. A collector confirms people as it goes and stamps the
        # clock at the end, so comparing against `last_ok_at` reads every person a healthy walk saw
        # as unconfirmed — it retires the whole list. A row with no `started_at` is pre-upgrade
        # state and retires nobody, which is the fail-safe direction.
        walked_at = parse_ts(run.started_at)
        if walked_at is None:
            return set()
        # `set_signal` stamps through SQLite's `datetime('now')`, which truncates down to the
        # second. A person confirmed 0.4s into the walk therefore records a whole-second stamp that
        # can precede the walk's own sub-second start. Truncating the boundary the same way makes
        # the two directly comparable, instead of papering over it with a magic margin.
        walked_at = walked_at.replace(microsecond=0)
        out = set()
        for r in conn.execute(
                "SELECT entity_id, last_confirmed_at FROM curation_signals "
                " WHERE signal_type=? AND platform=?", (MAINTAINED.signal, MAINTAINED.platform)):
            seen = parse_ts(r["last_confirmed_at"])
            if seen is not None and seen < walked_at:
                out.add(r["entity_id"])
        return out
    except sqlite3.Error:          # a store without the retirement tables retires nobody
        return set()


def _is_maintained(row) -> bool:
    """Is this signal row the one the maintained collector writes?

    Matches on signal type and platform. Type alone would let a Substack `follow` — which no walk
    is ever compared against, so it can never be retired — permanently rescue a cluster whose X
    follow went stale."""
    return row["signal_type"] == MAINTAINED.signal and row["platform"] == MAINTAINED.platform


def rank_candidates(conn, *, include_retired: bool = False) -> list[Candidate]:
    """Group every curation signal by canonical_id and rank the resulting people structurally.
    Candidates are signal-bearing canonical entities only (an atom-author with no curation signal
    is corpus, not a candidate). Returns the ordered list; empty when no signals.

    Confirmed Oracles are not candidates: someone already promoted is never waiting to be
    screened. This is the one place that rule lives; the screen, the waiting count, candidate
    search and scouting all read through it.

    Retired people are dropped by default — see `MAINTAINED_COLLECTOR`. `include_retired=True`
    returns them with `.retired` set, so a caller can report the count."""
    cands = rank(schema.signals_with_canonical(conn), _retired_ids(conn), roster.heads(conn),
                 schema.cluster_members(conn))
    return cands if include_retired else [c for c in cands if not c.retired]


def rank(rows, retired_ids: set[str], oracles: set[str] = frozenset(),
         clusters: dict[str, list[str]] | None = None) -> list[Candidate]:
    """The ranking decision over gathered rows: one `Candidate` per canonical id, each marked
    retired or not, in `Candidate.sort_key` order. `rows` are `schema.signals_with_canonical`'s
    (anything indexable by its column names); `retired_ids` are the entity ids `_retired_ids`
    found stale; `oracles` are the roster's heads, never candidates; `clusters` are
    `schema.cluster_members`', so `members` also lists the cluster's entities that carry no
    signal (an X account merged into a Substack-only person). Retired candidates stay in the
    list, marked."""
    groups: dict[str, list] = {}
    for r in rows:
        if r["canonical_id"] not in oracles:
            groups.setdefault(r["canonical_id"], []).append(r)

    cands: list[Candidate] = []
    for cid, grp in groups.items():
        distinct = {(r["signal_type"], r["platform"]) for r in grp}
        # One member row per entity_id (its name/links/profile), deduped.
        members: dict[str, dict] = {}
        for r in grp:
            members.setdefault(r["entity_id"], {
                "entity_id": r["entity_id"], "name": r["name"],
                "identity_links": r["identity_links"], "profile": r["profile"]})
        member_rows = list(members.values())
        name, handle = derive.best_name(member_rows)
        links: list = []
        for m in member_rows:
            for u in _loads(m["identity_links"]):
                if u and u not in links:
                    links.append(u)
        # Merge the member profiles.
        prof: dict = {}
        for m in member_rows:
            prof.update({k: v for k, v in _as_dict(m["profile"]).items() if v is not None})

        cands.append(Candidate(
            canonical_id=cid, name=name, handle=handle,
            distinct_signals=len(distinct),
            has_endorsement=any(st in _ENDORSEMENT for st, _ in distinct),
            total_count=sum(int(r["count"] or 0) for r in grp),
            signals=[{"signal_type": r["signal_type"], "platform": r["platform"],
                      "count": int(r["count"] or 0), "extra": _loads(r["extra"])} for r in grp],
            members=[m["entity_id"] for m in member_rows] + [
                e for e in (clusters or {}).get(cid, []) if e not in members],
            identity_links=links, profile=prof,
            # A cluster is retired when every one of its members' maintained signals is. A person
            # resolved across two platforms whose X follow is stale but who was re-confirmed under
            # another member id is still followed.
            retired=bool(retired_ids) and all(
                m["entity_id"] in retired_ids for m in member_rows
                if any(_is_maintained(r) and r["entity_id"] == m["entity_id"]
                       for r in grp)) and any(_is_maintained(r) for r in grp),
        ))

    cands.sort(key=Candidate.sort_key)
    return cands


def source_of(platform: str) -> str:
    """A signal's platform → the source it is listed under."""
    return PAPERS if platform in _SCHOLAR_PLATFORMS else platform


def _strength(signals: list[dict]) -> tuple:
    return (any(s["signal_type"] in _ENDORSEMENT for s in signals),
            len({s["signal_type"] for s in signals}),
            sum(s["count"] for s in signals))


def source_rank(source: str) -> int:
    """Where a source prints among the screen's groups (`SOURCE_ORDER`, unknown ones last)."""
    return SOURCE_ORDER.index(source) if source in SOURCE_ORDER else len(SOURCE_ORDER)


def stronger_source(signals: list[dict]) -> str:
    """The one source a person is listed under, from their signals on each — `sort_key`'s own
    order, applied per source:

      | on source A                 | on source B   | listed under                           |
      |-----------------------------|---------------|----------------------------------------|
      | a follow / list / subscribe | 40 bookmarks  | A — an endorsement wins categorically  |
      | bookmarked + liked          | 40 saves      | A — more distinct signal types         |
      | 5 bookmarks                 | 2 saves       | A — higher total count                 |
      | identical                   | identical     | whichever comes first in SOURCE_ORDER  |

    Never a sum across sources: someone with a follow on X and a subscribe on Substack is listed
    once, under X, and the other source is named on their line ("also on Substack")."""
    per: dict[str, list[dict]] = {}
    for s in signals:
        per.setdefault(source_of(s["platform"]), []).append(s)
    return min(per, key=lambda src: (tuple(-int(v) for v in _strength(per[src])),
                                     source_rank(src), src))


def known_platforms(conn) -> list[str]:
    """Every platform this store actually holds a signal from — the vocabulary `source=` accepts.
    Read from the data, not a constant, so a new collector needs no edit here."""
    return sorted(r[0] for r in conn.execute(
        "SELECT DISTINCT platform FROM curation_signals WHERE platform IS NOT NULL") if r[0])


def waiting_count(conn) -> int:
    """How many people are waiting to be screened: the one count the server's suggestions and
    onboarding's handoff both say."""
    return len(rank_candidates(conn))
