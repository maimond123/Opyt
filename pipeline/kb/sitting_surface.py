"""
pipeline/kb/sitting_surface.py — what a region looks like before anyone pays to read it.

Delegate layer behind `mcp_server/sitting_tools.py`, importable/testable without the MCP server.

`scope()` reports a built region's size, time span, and author spread before a read. It is a
scope check, not a spend gate — it informs a read, never gates one, and its warnings only warn,
never refuse. Every signal is already computed at build time, so it costs no extra query.

"""

from __future__ import annotations

from datetime import date

from . import sitting_builder as sb
from . import sitting_render as sre

# ── When a region is a poor fit for the question ────────────────────────────────
# ARC_MIN_DAYS and SINGLE_AUTHOR_SHARE are judgement calls, not calibrations, unlike most other
# thresholds in this rail.
ARC_MIN_DAYS = 14  # below this span, reading in publication order shows a snapshot, not an arc.

SINGLE_AUTHOR_SHARE = 0.60  # above this author share, queries skew self-referential (measured).

SAMPLE_ATOMS = 5  # atoms named back to the caller to confirm scope, without dumping full membership.


def _span(dates: list[str]) -> tuple[str | None, str | None, int | None, int]:
    """`(first, last, days, undated)` over a region's `when_ts` values.

    Undated atoms are counted separately, never folded in — treating a missing date as any
    particular date would stretch or crush the span. Compared as ISO strings for ordering, parsed
    only for the day count, so a year-precision date ("2026") still orders correctly.
    """
    have = sorted(d for d in dates if d)
    undated = len(dates) - len(have)
    if not have:
        return None, None, None, undated
    lo, hi = have[0], have[-1]

    def _d(s: str):
        parts = (s[:10].split("-") + ["01", "01"])[:3]
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            return None

    a, b = _d(lo), _d(hi)
    return lo, hi, ((b - a).days if a and b else None), undated


# Per-lens wording for "this region is too thin" — the tier check is shared, but the reason text
# must fit what each lens actually does (a lens that emits no queries can't say "not enough to
# generate standing queries").
_TIER_REASON = {
    "queries": "there may not be enough here to generate standing queries that a whole month of "
              "new publishing could answer",
    "briefing": "there may not be enough material here to say more than one atom already would",
    "trajectory": "tracing how a conversation MOVED needs more than a couple of posts to move "
                 "between",
    "disconfirmation": "there is not enough material here for one atom to plausibly contradict "
                       "another",
    "gaps": "the nearest-miss framing needs enough material for there to BE a near miss, not just "
           "an absence",
    "claims": "the strongest claims join a date to a later one — a thin region has too little "
              "material spread over time for a claim to connect",
}


def lens_warnings(scope: dict, lens: str = "queries") -> list[str]:
    """Ways this region is a poor fit for `lens`. Empty means nothing stands out.

    One function covers every lens because the same region property spoils different lenses
    differently (a three-day span kills `trajectory` but barely touches `briefing`). `sprouts` is
    never a valid `lens` — it has no seed/floor/sitting_id, so there's no region to warn about.
    Every message names the observed value (e.g. "spans 3 days"), not just the rule it broke.
    """
    if lens not in _TIER_REASON:
        # Named explicitly — an unknown lens returning no warnings would look like a clean bill of health.
        return [f"no warnings are defined for lens {lens!r} yet — this is silence, not approval"]

    out: list[str] = []
    if scope["tier"] != "standalone":
        out.append(
            f"{scope['atoms']} atoms is {scope['tier']} tier (standalone starts at "
            f"{sb.TIER_STANDALONE_MIN}) — {_TIER_REASON[lens]}")
    # A narrow-span region is a snapshot, not an arc — relevant to queries, trajectory, and claims.
    if lens in ("queries", "trajectory", "claims") and scope["days"] is not None \
            and scope["days"] < ARC_MIN_DAYS:
        out.append(
            f"this region spans {scope['days']} days — reading it in publication order is meant to "
            f"show how a conversation MOVED, and a window this narrow holds a snapshot, not an arc")
    # queries-specific: a single-author region tends to generate self-referential queries.
    if lens == "queries" and scope["top_author_share"] >= SINGLE_AUTHOR_SHARE:
        out.append(
            f"{scope['top_author']} wrote {scope['top_author_share']:.0%} of these atoms — a "
            f"single-author region tends to generate queries pointing back at that person's own "
            f"work, which the user already follows")
    # Universal: every lens needs to know it's reading part 1, not the whole region.
    if scope["stop"] == "budget":
        out.append(
            f"the token budget stopped this build with {scope['region_atoms']} atoms still "
            f"admissible — this is part 1 of a region bigger than one read, not the whole thing")
    if scope["undated"]:
        out.append(
            f"{scope['undated']} of {scope['atoms']} atoms carry no date — they trail the "
            f"chronology and contribute nothing to the arc")
    return out


def scope(conn, rec: dict, *, lens: str = "queries") -> dict:
    """A built record -> what it would cost and what is in it. Reads text for no atom.

    `rec` is what `build_sitting` returns. Adds one metadata query over the region's atoms and no
    chunk text, keeping a scope check cheap enough that nobody skips it.
    """
    ids = [a["atom_id"] for a in rec["admissions"]]
    meta, _text = ({}, {})
    if ids:
        meta, _text = sre._atom_bodies(conn, ids, with_text=False)
    whos = [meta.get(a, ("?", ""))[0] for a in ids]
    top_share, top_who = sre._concentration(whos)
    first, last, days, undated = _span([meta.get(a, ("?", ""))[1] for a in ids])

    out = {
        "sitting_id": rec["sitting_id"],
        "seed_kind": rec["seed_kind"],
        "seed_ref": rec["seed_ref"],
        "atoms": rec["atoms"],
        "tokens": rec["tokens"],
        # Still admissible, not the whole region — the number that says whether another part is
        # worth building. `stop` is what tells the caller which of those two they are looking at.
        "region_atoms": rec["region_atoms"],
        "region_tokens": rec["region_tokens"],
        "stop": rec["stop"],
        "skipped_dupes": rec["skipped_dupes"],
        "tier": sb.tier_for_reading(rec["atoms"]),
        "first": first, "last": last, "days": days, "undated": undated,
        "authors": len({w for w in whos if w}),
        "top_author": top_who, "top_author_share": round(top_share, 3),
        # Named so the caller can recognize whether this is the region they meant. Chronological,
        # matching the order a read would see them in.
        "sample": [{"atom_id": a, "who": meta.get(a, ("?", ""))[0],
                    "when": meta.get(a, ("?", ""))[1]}
                   for a in sorted(ids, key=lambda a: (not (meta.get(a, ("?", ""))[1] or ""),
                                                       meta.get(a, ("?", ""))[1] or "", a)
                                   )[:SAMPLE_ATOMS]],
    }
    out["warnings"] = lens_warnings(out, lens)
    return out
