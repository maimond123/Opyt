"""
pipeline/kb/substack_recommendations.py
The publications a confirmed Substack Oracle recommends → one entity + one `recommended` signal
each. Signals only, never Oracles.

This module owns exactly one thing: turning someone ELSE'S published endorsements into candidates
the user can screen. Every other producer of `curation_signals` records something the USER did —
a follow, a save, a subscription. Two do not, and they are the two whose input grows with the
Oracle roster rather than with the user's own activity: `paper_authors.sync_coauthor_signals` and
this. That shared property is why this follows that module's shape rather than
`ingest_curation`'s: it is not a collector, it is on no full-set clock the user's own lists share,
and its absence from a walk means nothing about anybody.

WHY IT IS NOT AN AUTO-ORACLE PATH. The same discipline that disabled the second-degree follow
scout. An Oracle's recommendation is evidence, and the user remains the only thing that turns
evidence into trust. Nothing here calls `oracles.confirm`, and nothing here should learn to.
"""

from __future__ import annotations

import sqlite3
import time

from pipeline.ingestion.sources import substack
from pipeline.ingestion.utils import log

from . import derive, oracle_refresh_state, schema

# The signal this module writes. Deliberately NOT in `screen._ENDORSEMENT`, which is the tier for
# acts the USER performed — a follow, a List, a paid subscription. `coauthor` sits outside it for
# exactly this reason and this joins it there: the user did nothing, somebody they trust did.
# `screen.reflect` carries a phrase that says so out loud, because a candidate the user cannot
# place is a candidate they will guess about.
SIGNAL_TYPE = "recommended"
PLATFORM = "substack"

# The clock row. NOT a member of `ingest_curation.COLLECTOR_SPECS`: `curation_catchup` iterates
# that registry, and this read belongs to the Oracle rail, whose input is the roster. A row in
# `collector_runs` that no spec names is written and read here and nowhere else.
COLLECTOR = "substack_recommendations"

# How long a pass is worth repeating. LOAD-BEARING, and the number comes from the rail, not from
# taste: `rail_worker.RAILS["oracle_refresh"].cadence` is 600 seconds, so an ungated read would go
# out 144 times a day per Oracle against a host whose Cloudflare 403 window is the whole
# constraint on every other Substack read OPYT makes. A publication's recommendation list is
# edited by hand, on the order of months; 24 hours is already far finer than the thing it watches.
FLOOR_HOURS = 24.0

# Pace between Oracles, matching the archive walk's own delay. Cloudflare is hostile to bursts and
# this loop makes two requests per Oracle back to back.
_PACE_SECONDS = substack.FETCH_DELAY


def _substack_oracles(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Every confirmed Oracle's `substack` pair, as `[(canonical_id, publication_url)]`.

    Reads `oracle_sources` rather than walking `entities` directly, because that registry is
    already the answer to "what can be pulled for this Oracle" and `pair_from_member` is the one
    place that maps a cluster member to a source key. A second walk here would be a second
    opinion about the same question.
    """
    return [(r.canonical_id, r.source_key)
            for r in oracle_refresh_state.list_sources(conn)
            if r.source_type == "substack" and r.status != "paused" and r.source_key]


def sync_recommendation_signals(conn: sqlite3.Connection) -> dict:
    """Read each confirmed Substack Oracle's recommendations and land one signal per publication.

    `count` is HOW MANY of the user's Oracles recommend that publication, which is the whole
    corroboration story for this signal. There is no `min_papers` analog and deliberately so: a
    recommendation is a deliberate editorial act a writer publishes under their own name, so one
    is already meaningful, where one shared byline is not. Two Oracles converging is stronger, and
    the count carries that without a threshold that would discard the singletons.

    Excluded: the Oracles themselves and anyone already confirmed — a person is not their own
    candidate, and a confirmed Oracle is past the screen. Same rule, same shape, as
    `sync_coauthor_signals`.

    PER-ORACLE FAILURE ISOLATION, and it is what replaces a fan-out bound. One Oracle's refused
    read costs that Oracle's recommendations this pass and nothing else; the walk continues, the
    signals already gathered are written, and the next pass re-reads everyone, because
    `set_signal` is a full-set write and nothing removes a signal for being absent. That makes a
    Cloudflare 403 mid-pass a delay rather than a hole, which is why no `max_oracles` slice exists
    here. Add one when a measured pass is dominated by requests rather than by Oracles — at the
    three confirmed on 2026-09-08 it would be a parameter with one value.

    Returns the counts, including `skipped_oracles`, so a pass that read nobody is visible as a
    pass that read nobody rather than as a store with no recommendations in it.
    """
    pairs = _substack_oracles(conn)
    oracle_ids = {r[0] for r in conn.execute("SELECT canonical_id FROM oracles")}

    by_entity: dict[str, dict] = {}
    read = skipped = 0
    for i, (cid, pub_url) in enumerate(pairs):
        if i:
            time.sleep(_PACE_SECONDS)
        try:
            pub_id = substack.fetch_publication_id(pub_url)
            recs = substack.fetch_recommendations(pub_id)
        except Exception as e:
            # Never sinks the pass — see the docstring. The Oracle is reported, not silently
            # dropped, because "read nobody" and "nobody recommends anybody" look identical in
            # the signal table and only this count tells them apart.
            log(f"[substack-recs] {cid} skipped: {type(e).__name__}: {e}")
            skipped += 1
            continue
        read += 1
        for rec in recs:
            eid = derive.substack_entity_id(None, rec["url"])
            slot = by_entity.setdefault(eid, {"rec": rec, "oracles": []})
            if cid not in slot["oracles"]:
                slot["oracles"].append(cid)

    written = self_recommended = 0
    for eid, slot in by_entity.items():
        if eid in oracle_ids or schema.current_canonical(conn, eid) in oracle_ids:
            self_recommended += 1
            continue
        rec = slot["rec"]
        schema.upsert_entity(conn, eid, name=rec["name"] or None,
                             identity_links=[rec["url"]],
                             profile={"bio": rec["bio"]} if rec["bio"] else None)
        # `count` and `extra.oracles` are two projections of ONE list, written in one call from
        # one variable, so they cannot drift. The list is the audit trail for a signal the user
        # did not create: `reflect()` reads the count, and a person asking "why is this candidate
        # here" needs the names.
        schema.set_signal(conn, eid, SIGNAL_TYPE, PLATFORM,
                          count=len(slot["oracles"]),
                          extra={"oracles": slot["oracles"]})
        written += 1
    conn.commit()

    return {"source": "substack-recommendations",
            "oracles": len(pairs), "oracles_read": read, "skipped_oracles": skipped,
            "publications": len(by_entity), "signalled": written,
            "already_oracles": self_recommended}
