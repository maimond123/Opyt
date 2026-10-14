"""
pipeline/arrivals/fresh.py — which arrivals are new in time, and how the older ones are counted.

What's New means recently published, not recently stored. When an Oracle's archive lands, years of
their posts arrive at once: one look found 13,248 arrivals, 174 of them published in
the week before. An arrival is new when it was published no more than a week before it arrived,
which keeps a post Opyt pulled a few days after it went up. One with no date of its own takes the
date of the earliest post that linked it, and one with no date at all is new: nothing says it is
old. The rest are counted, one line per Oracle or per kind, and never listed.

Decisions over the rows `window.arrivals` reads.
"""
from __future__ import annotations

import collections
from datetime import date, timedelta

RECENT = timedelta(days=7)

# What an older arrival is, by how it entered, when it is not an Oracle's own post.
_KIND = {
    "user-saved": "Posts you saved earlier",
    "author_referenced": "Pages your Oracles linked in older posts",
}


def _day(value: str | None) -> date | None:
    try:
        return date.fromisoformat((value or "")[:10])
    except ValueError:
        return None


def published(row: dict) -> date | None:
    """The day the item was published: its own date, else the earliest post that linked it."""
    return _day(row["when_ts"]) or min(filter(None, map(_day, row["linked_on"])), default=None)


def is_new(row: dict) -> bool:
    day, arrived = published(row), _day(row["arrived"])
    return day is None or arrived is None or day >= arrived - RECENT


def oracle_names(members) -> dict[str, tuple[str, str]]:
    """`{author id: (cluster id, name)}` from every member of the authors' clusters, given as
    `(entity_id, canonical_id, name)`. A cluster is named by its X account when it has a name, the
    name people know; else by any member's name; else by its site's address."""
    members, clusters = list(members), collections.defaultdict(list)
    for entity_id, canonical_id, name in members:
        clusters[canonical_id or entity_id].append((entity_id, name or ""))
    named = {}
    for cluster, people in clusters.items():
        ranked = sorted(people, key=lambda p: (not p[0].startswith("x:"), not p[1], p[0]))
        named[cluster] = ranked[0][1] or cluster.split(":", 1)[-1]
    return {entity_id: (canonical_id or entity_id, named[canonical_id or entity_id])
            for entity_id, canonical_id, _name in members}


def older_lines(rows: list[dict], who: dict[str, tuple[str, str]]) -> list[str]:
    """One sentence per Oracle, for their older posts, and one per kind for the rest, largest
    first. `who` maps an author id to its Oracle's `(cluster id, name)`, so one person's X posts
    and blog posts share a line."""
    counts, years = collections.Counter(), collections.defaultdict(list)
    for row in rows:
        if row["entry_mode"] == "oracle-footprint":
            key, name = who.get(row["who_id"], (row["who_id"], row["who_id"]))
            label = (key, name, True)
        else:
            label = (row["entry_mode"], _KIND.get(row["entry_mode"], "Other older items"), False)
        counts[label] += 1
        if day := published(row):
            years[label].append(day.year)
    lines = []
    for (_key, name, oracle), n in counts.most_common():
        line = f"{name}: {n:,}" + (f" older post{'s' if n != 1 else ''}" if oracle else "")
        if span := years[(_key, name, oracle)]:
            line += f", {min(span)}" + (f"–{max(span)}" if max(span) > min(span) else "")
        lines.append(line)
    return lines
