"""
pipeline/kb/sitting_surface.py — what a region looks like before anyone pays to read it.

Delegate layer behind `mcp_server/sitting_tools.py`, importable/testable without the MCP server.

`scope()` reports a built region's size, time span, and author spread before a read. It is a
scope check, not a spend gate — it informs a read, never gates one, and its warnings only warn,
never refuse. It calls no model and reads no chunk text, but it does run `ceil(atoms / 900)`
metadata queries (CORRECTED 2026-09-05: this claimed zero).

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
        # Named explicitly — an unknown lens returning no warnings would look like a clean bill of
        # health. "yet" until 2026-09-05 (S10), which promised a gap someone would fill; for
        # `sprouts` — reachable from `sitting_tools.py`'s `lens or "queries"` — the docstring above
        # says there is no region to warn about at all, so there is nothing to fill.
        return [f"no warnings are defined for lens {lens!r} — this is silence, not approval"]

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

    `rec` is what `build_sitting` returns. Adds `ceil(len(admissions) / sitting_vectors.SQL_VARS)`
    metadata queries and no chunk text — `_atom_bodies` batches ids under SQLite's parameter
    ceiling, so it is one query per 900 atoms, not one flat. Cheap enough that nobody skips it.
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


# ── the same fit check, without paying to build a region ────────────────────────
# `scope()` above answers "is THIS BUILT REGION a good fit for this lens", which is the right
# question once someone has named a topic and paid an embedding for it. It is the wrong question
# for a caller that has not named anything yet and is deciding what to SUGGEST — there, building a
# region per candidate topic would cost an embedding each to answer a question nobody asked.
#
# So these two measure the same properties with pure SQL over a tag, and feed them to the SAME
# `lens_warnings` with the SAME thresholds. That reuse is the whole point: the moment a suggester
# carries its own idea of "too thin" or "too one-voiced", it drifts from what `sitting` will
# actually tell the user one call later, and the product contradicts itself.

# A tag scope is an approximation of a region and says so. `build_sitting` grows a region by
# similarity from a seed, so it can pull in an atom nobody tagged and drop a tagged one that sits
# far from the seed. Close enough to decide WHETHER to suggest a sitting; never reported as the
# region a read would actually assemble.
def store_shape(conn) -> dict:
    """Properties of the WHOLE store, shaped exactly like `scope()`'s return so `lens_warnings`
    consumes it unchanged.

    Author names resolve through `sitting_render._atom_bodies`, the same path `scope()` uses, so a
    warning naming an author reads identically whichever way the shape was measured. That costs
    one metadata query per 900 atoms and reads no chunk text.

    `stop` is always "region" and `region_atoms` equals `atoms`: nothing was built, so there is no
    budget that could have stopped a build part-way. That keeps the budget warning silent here
    rather than firing on a build that never happened.

    WAS `shape_by_tag(conn, tag=None)` UNTIL 2026-09-16. The tag branch filtered on
    `payload.source_tags` so `suggest` could measure "the shape of your top topic" — and when
    that seed was deleted (a 5-atom-of-1,801 tag space presented as the store's subject matter),
    its only caller started passing `None` on every call. A parameter with no live argument is a
    second code path nothing exercises; the tag SQL went with it. `scope={"tags": …}` on
    `search`/`aggregate` is untouched and is where tag filtering lives.

    ⚠️ WHAT A STORE-WIDE SHAPE DOES NOT MEAN. `lens_warnings` was written about a BUILT REGION,
    and several of its rules fire on a store for reasons unrelated to fitness — one undated atom
    out of 1,801 is enough to spoil every lens. Read this shape for SIZE (`tier`), and let
    `sitting` judge fitness against the region a real query actually builds. See
    `opyt_core/suggest.py`, which learned this the expensive way.
    """
    ids = [r[0] for r in conn.execute("SELECT a.atom_id FROM atoms a")]

    meta, _text = ({}, {})
    if ids:
        meta, _text = sre._atom_bodies(conn, ids, with_text=False)
    whos = [meta.get(a, ("?", ""))[0] for a in ids]
    top_share, top_who = sre._concentration(whos)
    first, last, days, undated = _span([meta.get(a, ("?", ""))[1] for a in ids])

    return {
        "atoms": len(ids),
        "region_atoms": len(ids),
        "stop": "region",
        "tier": sb.tier_for_reading(len(ids)),
        "first": first, "last": last, "days": days, "undated": undated,
        "authors": len({w for w in whos if w}),
        "top_author": top_who, "top_author_share": round(top_share, 3),
    }


# Tried in order, first clean one wins. Two orders because one prolific author and a crowd are
# opposite situations, not degrees of the same one: a single voice has an arc to trace and nothing
# to argue with, so `trajectory` leads and `disconfirmation` is dropped entirely — a lens that
# hunts for what contradicts the material cannot work when every atom shares an author.
# `lens_warnings` does not catch that case (its single-author rule is scoped to `queries`, where it
# was measured), so it is encoded here rather than widened there on an unmeasured hunch.
_ORDER_ONE_VOICE = ("trajectory", "briefing", "gaps")
_ORDER_MANY_VOICES = ("briefing", "disconfirmation", "trajectory", "gaps")


def best_lens(shape: dict) -> tuple[str | None, list[str]]:
    """`(lens, warnings)` — the first lens this shape supports cleanly, or `(None, why_not)`.

    `None` is a real answer and the caller must be able to say it: a four-atom region supports no
    reading at all, and offering one anyway is how a suggestion becomes a promise the next call
    breaks. The returned warnings then belong to the least-bad lens, so the caller can say what is
    missing and by how much rather than going silent.
    """
    order = (_ORDER_ONE_VOICE if shape.get("top_author_share", 0) >= SINGLE_AUTHOR_SHARE
             else _ORDER_MANY_VOICES)
    scored = [(lens, lens_warnings(shape, lens)) for lens in order]
    for lens, warns in scored:
        if not warns:
            return lens, []
    return None, min(scored, key=lambda lw: len(lw[1]))[1]
