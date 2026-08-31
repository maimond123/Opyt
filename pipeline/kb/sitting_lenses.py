"""
pipeline/kb/sitting_lenses.py — the host-side lenses over a sitting (Job L; D7, D17, D21).

Four of these read ONE region: `briefing`, `trajectory`, `disconfirmation`, `gaps`. The fifth,
`sprouts`, reads the material nothing else covers — `sitting_render.render_sprouts_digest` — and
folds in what was going to be a separate Job C until David ruled 2026-08-16 that the sprouts digest
is a lens, not a rail (docs/plans/2026-08-16-sitting-rail-handoff.md).

⚠️ THIS FILE SPENDS NOW (amended 2026-08-24). It used to call no model at all: a lens handed the
host one region's document and the host read it in-session. That could not survive chained parts —
David judged the claims notebook TOO LOSSY as the sole carrier of old parts for lens questions
(trajectory gutted, disconfirmation and sprouts likewise; the notebook is the paid-read chain's
memory, and a lens is a different consumer).

So a sitting-scoped lens is now MAP-REDUCE:

  MAP (here, code-driven) — a for loop over the region's part chain, one model call per part with
  the lens's map instruction. Never model discretion: a host walking a cursor protocol answers
  early from partial material, and that was rejected on exactly that ground. Outputs are cached per
  (sitting, lens) in `sitting_lens_outputs`; closed parts are FROZEN, so a part is mapped once per
  lens EVER and steady state is one call for the open part.

  REDUCE (host, in-session) — the host receives ONE compact document, the per-part outputs labeled
  with their date ranges, plus the lens's JOIN rule. It holds the search tools, so the reconcile is
  not a sealed call: it can search the region, open atoms, or point-lens an old part when the join
  surfaces a gap. Its output is never persisted — the input is the live region.

`sprouts` is untouched and still calls nothing: it has no chain, no `sitting_id`, and no part to
cache against.

"""
from __future__ import annotations

from datetime import datetime
from pipeline.timeparse import utc_now

from pipeline.ingestion.utils import log

from . import reader_core as core
from . import sitting_reader as sr
from . import sitting_render as sre
from . import sitting_store as sst

SITTING_LENSES = ("briefing", "trajectory", "disconfirmation", "gaps")
LENSES = SITTING_LENSES + ("sprouts",)


class LensError(ValueError):
    """A caller mistake — an unknown lens name, or a sitting-scoped lens with no `sitting_id`.

    Raised rather than degraded, the same call `sitting_builder.SeedError` makes: there is no
    honest fallback for "I do not know what you are asking for."
    """


# ── Instructions — no tuning history to carry, unlike `sitting_reader._SYSTEM` ──────────────────
# That prompt is calibrated for a WEAKER model reached over the API (D18: redundancy first, then
# price) and every rule in it was bought by a measured failure against a real window, run three
# times. These run on the host model in the session — "frontier-grade" is D17's own word for it —
# so the instructions below trust general reading competence and are precise only about FRAMING:
# what question this lens is answering, and the one failure mode particular to it.

_BRIEFING = """Read this sitting and say what it actually SAYS — not a table of contents, not "a
range of topics". State the material as knowledge: what is established, what is claimed but
unresolved, what the strongest or most-repeated position is. Name specific systems, people,
benchmarks and numbers rather than categories. Cite the atom (its date and id) for any claim you
attribute to the material, the way a briefing cites its sources."""

_TRAJECTORY = """Read this sitting in the chronological order it is given and trace how the
thinking on this topic MOVED. What was claimed early, what happened to that claim later, what
reversed, what got quietly abandoned, and what is unresolved right now. A trajectory with no
movement in it — nothing changed, nothing reversed — is a legitimate finding; say so plainly rather
than manufacturing an arc the material does not support. Cite the atoms that mark each turn."""

_SPROUTS = """This is NOT one sitting — it is every atom no sitting has covered yet: true orphans,
built-but-unread regions, and fracture leftovers too small to stand alone. It bundles unrelated
material together by construction. Read it and throw out whatever is noise or a grab-bag; for
whatever DOES cohere into something worth surfacing, say what it is and cite the atoms. The person
asked what is in their blind spots — material they saved but nothing has ever actually read — so
answer as "here is what has been sitting unread", not as a forced summary of everything in it."""


# ── Map / join ──────────────────────────────────────────────────────────────────
# ONE RECORD PER LENS, and the variation lives in prompt STRINGS, never in two code paths — the map
# loop and the reduce hand-off are identical for every lens. A record is `map` (what one part is
# read for), `join` (how the parts are reconciled), and, for the two lenses that aim at something,
# `default_target` — the stand-in when the caller passes no `claim`. The map is deliberately
# CLAIM-INDEPENDENT: a part's map output must be reusable for any later question, or the cache key
# would have to carry the claim and a part would be re-mapped per question forever. The target
# enters at the JOIN, which the host performs in-session with the claim in hand.

_MAP_PREFIX = ("You are reading ONE PART of a longer region — one contiguous stretch of its "
               "timeline. Do not treat it as the whole topic and do not summarize what a reader "
               "outside this stretch would need; another pass is reading the other parts.\n\n")

_LENS: dict[str, dict[str, str]] = {
    "briefing": {
        "map": _BRIEFING,
        "join": (
            "Below are per-part briefings of one region, oldest stretch first. JOIN them into one "
            "briefing: where two parts conflict, the LATER one supersedes the earlier, and say so "
            "explicitly rather than silently dropping the earlier claim."),
    },
    "trajectory": {
        "map": _TRAJECTORY,
        "join": (
            "Below are per-part trajectories of one region, oldest stretch first. JOIN them END TO "
            "END into one arc: what each stretch inherited from the one before, what it changed, "
            "and where the thinking stands now. A stretch with no movement is a real finding — say "
            "so."),
    },
    "disconfirmation": {
        "map": (
            "Read this part and inventory, WITHOUT arguing against anything yet: (1) the positions "
            "this stretch treats as settled, and (2) the strongest concrete evidence it carries, "
            "for or against each. Name systems, numbers, dates and cite the atoms. Do not attack a "
            "position here — a position abandoned later in the region would be attacked for "
            "nothing."),
        "join": (
            "Below are per-part inventories of one region's settled positions and its strongest "
            "evidence, oldest stretch first. Now PERFORM the attack against {target}: consensus is "
            "a property of the WHOLE region, so read across the parts first and attack a position "
            "the region still holds — never one it abandoned partway through. Name the tension and "
            "cite the atom. If nothing here disconfirms it, say so plainly rather than "
            "manufacturing a doubt the material does not support."),
        "default_target": "the position this material argues for most strongly",
    },
    "gaps": {
        "map": (
            "Read this part and inventory what it COVERS: the questions this stretch actually "
            "answers, and, for each, how directly. Then name what it gestures at but leaves open. "
            "Cite atoms throughout. Do not conclude that anything is missing from the region — a "
            "later part may fill it, and only the join can tell."),
        # The nearest-miss framing below is MANDATORY (D7/D21), not stylistic: it is what makes a
        # false "nothing here" distinguishable from a true one. Never weaken it to "or say nothing
        # was found".
        # failure mode it prevents.
        "join": (
            "Below are per-part coverage inventories of one region, oldest stretch first. Answer "
            "{target} using ONLY what is here, and INTERSECT the gaps: something is a real gap "
            "only if NO part fills it. But never answer with a bare absence. If nothing in the "
            "material directly answers it, find and cite the CLOSEST this region comes: the atom "
            "that comes nearest to addressing it, and say specifically why it falls short (too "
            "vague, too old, addresses a related but different question, etc). \"Nothing here "
            "establishes X\" is not an acceptable answer on its own — there is always a nearest "
            "miss, even when the miss is wide, and naming it is what makes this checkable instead "
            "of a guess about what was skimmed."),
        "default_target": "what this material's own apparent conclusion leaves most unclear",
    },
}


def _map_instruction(lens: str) -> str:
    """What ONE part is read for. No `claim` parameter, deliberately: that absence is what keeps
    the cache key (sitting_id, lens) honest — a map output must serve any later question."""
    return _MAP_PREFIX + _LENS[lens]["map"]


def _instruction(lens: str, *, claim: str | None) -> str:
    """The JOIN rule the host reduces with — the only place `claim` enters."""
    rec = _LENS[lens]
    target = rec.get("default_target")
    if target is None:
        return rec["join"]
    return rec["join"].format(target=f'"{claim}"' if claim else target)


def _map_part(conn, sitting_id: str, lens: str, *, ref: datetime) -> dict | None:
    """One part's map output — a cache hit, or one paid call. None when the call could not be made.

    Follows the canonical call sequence the two API lenses use (resolve backend → preflight
    degrade-open → call → the 402 branch → usage), and re-applies `MAX_INPUT_CHARS` because
    `render_sitting` is used directly here rather than through `sitting_reader.render_prompt`,
    which is where that cap normally lives.

    Fail-safe: a part that cannot be mapped returns None and the reconcile proceeds without it. A
    lens is prose for a person, so a region missing one stretch is a degraded answer; refusing the
    whole lens over one bad call would be a worse one.
    """
    hit = sst.get_lens_output(conn, sitting_id, lens)
    if hit:
        return {**hit, "cached": True}

    document = sre.render_sitting(conn, sitting_id)
    if len(document) > sr.MAX_INPUT_CHARS:
        document = document[:sr.MAX_INPUT_CHARS] + "\n\n[TRUNCATED — part exceeds the input budget]"

    backend = core.resolve_backend()
    reason = core.preflight(backend)
    if reason:
        log(f"[sitting-lenses] {lens} map skipped (degrade-open) on {sitting_id}: {reason}")
        return None
    try:
        resp = core.call(backend, _map_instruction(lens), document)
    except Exception as e:
        if getattr(e, "status", None) == 402:
            log(f"[sitting-lenses] OUT OF CREDITS (HTTP 402) — {lens} map on {sitting_id} rejected "
                f"before inference, nothing spent. Upstream: {e}")
        else:
            log(f"[sitting-lenses] {lens} map failed on {sitting_id}: {type(e).__name__}: {e}")
        return None

    usage = {"model": resp.model, "in_tokens": resp.input_tokens,
             "out_tokens": resp.output_tokens, "cost_usd": resp.cost_usd}
    text = (resp.text or "").strip()
    if not text:
        log(f"[sitting-lenses] {lens} map returned nothing for {sitting_id} — not cached")
        return None
    sst.record_lens_output(conn, sitting_id, lens, text, usage=usage, at=ref)
    sr.record_lens_run(conn, sitting_id, lens, ref=ref, usage=usage)
    return {"output": text, **usage, "cached": False}


def read_lens(conn, lens: str, *, sitting_id: str | None = None, claim: str | None = None,
             ref: datetime | None = None) -> dict:
    """`{status, lens, instruction, document, parts, spent_usd, ...}` — the MAP, plus the JOIN rule
    the host reduces with.

    ⚠️ SPENDS on a cache miss, which is a change from this function's original contract (it called
    no model at all). One call per uncached (part, lens): a closed part is frozen so it is mapped
    once per lens EVER, and steady state on any region is one call for the open tail.

    `document` is the per-part outputs labeled with the stretch each covers — NOT the region's raw
    text. That compaction is the point: the claims notebook is the paid-read chain's memory and was
    judged too lossy to be a lens's memory too.

    The loop is CODE, never model discretion. A host handed a cursor protocol and told to walk the
    chain itself answers early from partial material, and that was rejected on exactly that ground.

    `claim` enters at the JOIN only, never the map — so a part's map output is reusable for any
    later question and the cache key stays (sitting_id, lens).

    Raises `LensError` for an unknown lens or a sitting-scoped lens with no `sitting_id`, and
    `KeyError` for a `sitting_id` that does not exist — both caller mistakes the tool layer turns
    into an `error` status.
    """
    if lens not in LENSES:
        raise LensError(f"unknown lens {lens!r} — must be one of {', '.join(LENSES)}")
    ref = ref or utc_now()

    if lens == "sprouts":
        # No chain, no part, nothing frozen to cache against — and no model call, which is what
        # this whole file used to be.
        rep = sre.render_sprouts_digest(conn)
        return {"status": "ok", "lens": lens, "instruction": _SPROUTS,
                "document": rep["document"], "atoms": rep["atoms"], "truncated": rep["truncated"]}

    if not sitting_id:
        raise LensError(f"lens {lens!r} needs a sitting_id — preview or build a region first")
    if sst.get_sitting(conn, sitting_id) is None:
        raise KeyError(f"no sitting {sitting_id!r}")

    chain = sst.ancestors(conn, sitting_id) + [sitting_id]
    parts, blocks, spent, missing = [], [], 0.0, []
    for i, sid in enumerate(chain, start=1):
        got = _map_part(conn, sid, lens, ref=ref)
        if got is None:
            missing.append(i)
            continue
        lo, hi = sre.part_span(conn, sid)
        parts.append({"part": i, "sitting_id": sid, "covering": f"{lo}–{hi}",
                      "cached": got["cached"]})
        blocks.append(f"## Part {i} of {len(chain)} — covering {lo}–{hi}  (`{sid}`)\n\n"
                      f"{got['output']}")
        if not got["cached"]:
            spent += float(got.get("cost_usd") or 0.0)

    out = {"status": "ok", "lens": lens, "sitting_id": sitting_id,
           "instruction": _instruction(lens, claim=claim),
           "document": "\n\n".join(blocks), "parts": parts, "spent_usd": round(spent, 6)}
    if missing:
        # Never silent: a reconcile over a region with a hole in it must say which stretch is
        # absent, or the join reads as complete.
        out["missing_parts"] = missing
        out["note"] = (f"{len(missing)} part(s) of this region could not be read this time — the "
                       f"answer below is missing that stretch")
    return out
