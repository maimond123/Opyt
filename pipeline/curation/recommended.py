"""The publications a confirmed Substack Oracle recommends → one entity + one `recommended` signal
each. Signals only, never Oracles: nothing here calls `oracles.confirm`.
"""

from __future__ import annotations

from pipeline.curation import signals
from pipeline.kb import derive

# Not in `screen._ENDORSEMENT`: the user did nothing, somebody they trust did.
SIGNAL_TYPE = "recommended"
PLATFORM = "substack"

# Not one of `collectors.CLOCKS`: this `collector_runs` row is written only by
# `oracles/sources/web/recommendations.py`, which owns the read.
COLLECTOR = "substack_recommendations"


def store_recommendations(conn, results):
    """Set corroboration counts from a completed pass's durable per-publication observations."""
    oracles = signals.oracle_ids(conn)
    by_entity = {}
    for result in results:
        cid = result["canonical_id"]
        if result["status"] != "ok":
            continue
        for rec in result["recommendations"]:
            eid = derive.substack_entity_id(None, rec["url"])
            slot = by_entity.setdefault(eid, {"rec": rec, "oracles": []})
            if cid not in slot["oracles"]:
                slot["oracles"].append(cid)

    written = self_recommended = 0
    for eid, slot in by_entity.items():
        if signals.is_oracle(conn, eid, oracles):
            self_recommended += 1
            continue
        rec = slot["rec"]
        person = signals.Person(eid, name=rec["name"] or None, site=rec["url"],
                                profile={"bio": rec["bio"]} if rec["bio"] else None)
        # `count` and `extra.oracles` come from one list in one call; the names are the audit
        # trail for a signal the user did not create.
        signals.counts(conn, [signals.Tally(person, len(slot["oracles"]),
                                            {"oracles": slot["oracles"]})], SIGNAL_TYPE, PLATFORM)
        written += 1


    return {"source": "substack-recommendations",
            "oracles": len(results), "oracles_read": sum(r["status"] == "ok" for r in results),
            "skipped_oracles": sum(r["status"] != "ok" for r in results),
            "publications": len(by_entity), "signalled": written,
            "already_oracles": self_recommended}
