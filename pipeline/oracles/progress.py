"""
pipeline/oracles/progress.py — `oracle(action='progress')`: where each pick stands, by milestone.

Waits until a milestone moves, or `PROGRESS_WAIT_SECONDS`, whichever comes first, then stamps what
it showed as told. The scrapers run in another process, so the wait is a poll of the store.

Each pick also carries `other_sources` when discovery found something for it (`other_sources`):
sources it added are said once, and projects on an "Also:" line once.
"""

from __future__ import annotations

import time as _time

from pipeline.oracles import milestones
from pipeline.oracles import review_items
from pipeline.oracles import told

# How long one progress question waits before it answers. Not a budget on work: nothing is cut
# when it expires, and no pull may read it. Kept under the client's ~60 s call wall.
PROGRESS_WAIT_SECONDS = 40.0


_PROGRESS_POLL_SECONDS = 0.5


def progress(conn, *, canonical_ids=None) -> dict:
    """action='progress' — where each pick stands, by milestone. See `oracle()`.

    Waits until a milestone moves, or `PROGRESS_WAIT_SECONDS`, whichever comes first, and not at
    all when nothing is left that could move (everything in, or only stuck picks). Moving means a
    milestone turned, or a pick became stuck or unstuck; X reaching a little further back is
    reported, but is not worth returning early for.

    A DB poll, not a condition variable: the scrapers run in another process.

    There is no "complete". The work goes on whether or not anyone calls this, and returning
    rather than raising on every absence keeps a host from reasoning its way into a retry."""

    def moved(picks):
        return [(p["canonical_id"], p["newest_posts"], p["archives"],
                 (p["reach"] or {}).get("reached"), (p.get("stuck") or {}).get("cause"))
                for p in picks]

    def movable(picks):
        return any(milestones.owes(p) and not p.get("stuck") for p in picks)

    deadline = _time.time() + PROGRESS_WAIT_SECONDS
    picks = milestones.for_picks(conn, canonical_ids)
    if not picks and canonical_ids:
        return {"picks": [],
                "message": (f"None of {', '.join(canonical_ids)} is a confirmed Oracle. "
                            "`oracle(action='progress')` with no ids lists every Oracle.")}
    if not picks:
        return {"picks": [],
                "message": ("No Oracles are confirmed on this store yet. `oracle(action='screen')` "
                            "then `oracle(action='confirm')` is how someone is added.")}
    first = moved(picks)
    while movable(picks) and moved(picks) == first and _time.time() < deadline:
        _time.sleep(_PROGRESS_POLL_SECONDS)
        picks = milestones.for_picks(conn, canonical_ids)
    told.mark_shown(conn, picks)
    found = review_items.for_oracles(conn, [p["canonical_id"] for p in picks])
    shown, said = told.found_shown(conn), []
    for pick in picks:
        cid = pick["canonical_id"]
        items = found.get(cid, [])
        other = other_sources(items, {rid for c, rid in shown if c == cid})
        if other:
            pick["other_sources"] = other
            said += [(cid, it["review_id"]) for it in items
                     if it["status"] == review_items.ADDED]
    told.mark_found_shown(conn, said)
    out: dict = {"picks": picks}
    if (waiting := milestones.waiting_on(conn, picks)) is not None:
        out["waiting_on"] = waiting
    return out


def _shown_as(url: str) -> str:
    return url.split("://", 1)[-1].rstrip("/")


def other_sources(items: list[dict], shown: set[int]) -> dict | None:
    """What to tell the user about the sources discovery found for one pick, or None. Pure.

    | item                         | told                                                 |
    |------------------------------|------------------------------------------------------|
    | added source, not yet told   | "Also pulling …", once                               |
    | also (a project), not told   | on the "Also:" line, once                            |
    | added or also, already told  | nothing                                              |
    """
    def reasons(it):
        return list(it.get("reasons") or [])[:2]

    added = [{"source": _shown_as(it["source_url"]), "reasons": reasons(it)} for it in items
             if it["status"] == review_items.ADDED and it["kind"] == review_items.SOURCE
             and it["review_id"] not in shown]
    also = [_shown_as(it["source_url"]) for it in items
            if it["kind"] == review_items.ALSO and it["review_id"] not in shown]
    if not (added or also):
        return None

    def line(entry):
        why = " · ".join(entry["reasons"])
        return f"{entry['source']} ({why})" if why else entry["source"]

    said = [f"Also pulling {line(e)}." for e in added]
    if also:
        said.append("Also: " + ", ".join(also) + ".")
    return {"added": added, "also": also, "message": " ".join(said)}
