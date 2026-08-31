"""
pipeline/kb/sitting_builder.py — assemble a reading context by growing it from a seed.

A sitting is one context an agent reads end to end. Start from a point in embedding space, admit
related atoms one at a time, and stop when the qualifying pool runs dry or the token budget binds.
Sittings take their natural size — the corpus decides, not a fixed-window partition.

Four rules:
  membership  rel(a) = MAX over a's chunks of cos(chunk, seed centroid) >= FLOOR (chunk-grain,
              not a pooled atom vector — pooling blurs topic with boilerplate).
  order       chronological, oldest first, fixed before the first admission. Governs read
              order and where a part is cut; it never decides membership. Redundancy is still
              computed, but only to skip a near-duplicate above the ceiling.
  anchor      relevance is measured against the SEED forever, never against the growing region.
  budget      at most FRONTIER_REGION_BUDGET members whose CURRENT entry_mode is 'frontier',
              per region ACROSS ALL PARTS, top-rel first. Occupancy is counted from the chain
              at build time, so promoting a member frees its slot with no bookkeeping.

A region bigger than the budget is read in PARTS (continuations). Every part's redundancy
baseline includes every prior part's atoms, not just the seeds, so the same content can't
re-enter under a different atom id (e.g. the same essay as both an X-article and a blog post).

The floor is a calibrated number: the 99th percentile of random chunk-pair cosine in this corpus
with this embedder, plus a margin. It is per-seed, not global, so a coarse coverage pass and a
finer `zoom` re-seed can coexist without a rebuild. `zoom` re-seeds inside a region via k-means to
fracture it into sub-conversations, at a finer floor, over the whole corpus.

Building is free (no LLM, no network) and reading is what spends, which is why sittings are
appended rather than overwritten. One exception: resolving a typed phrase costs one metered embed
call ($0.01/M) — see `_vector_seed_atoms`.

The budget bills RENDERED tokens, not stored ones: a full-text paper renders as its head plus the
sections that cleared the floor, so `sitting_render.projection` — the single function that decides
both the projection and its cost — is what this loop measures with.

Split into five files 2026-08-16 (pure move). This module keeps the seeds, build loop, and floor
calibration. Siblings: `sitting_vectors.py`, `sitting_store.py`, `sitting_zoom.py`,
`sitting_render.py`. The CLI (`main`) stays here and lazily imports the other three.

Full design history — what this replaced, every rejected alternative, and the measurements behind
each rule and constant:
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pipeline.timeparse import utc_iso, utc_now

import numpy as np

from pipeline.ingestion.utils import log

from . import schema
from . import sitting_render as sre
from . import sitting_store as sst
from . import sitting_vectors as sv
from .embed import assert_model, get_kb_embedder, stored_dtype

# ── Dials, with the defaults the simulations picked ─────────────────────────────
# Calibrated against this corpus' pairwise-cosine distribution and re-measured whenever the embed
# surface changes. Full derivation and re-measurement history:
FLOOR_DEFAULT = 0.67        # the coverage pass: big regions, ~84% of the corpus reachable
FLOOR_ZOOM = 0.71           # re-seeding inside a region to fracture it
CEILING_DEFAULT = 0.95      # redundancy above this = a near-duplicate repost; skip it outright
BUDGET_TOKENS_DEFAULT = 120_000

# Cap on members whose CURRENT entry_mode is 'frontier', per region ACROSS ALL PARTS (RULED
# 2026-08-27). Relevance alone cannot gate frontier admission — the candidates were produced by
# the region's own standing queries, so the rel test has weak discriminating power there and
# admissions scale as pass-rate × machine-paced arrival. A cap, not a fill target: slots the pool
# cannot earn stay empty, and a per-part cap was rejected as a rate limiter rather than a cap.
# Start small, raise on evidence.
FRONTIER_REGION_BUDGET = 5

# Region sizes that decide what a region IS. Small regions are batched into a sprouts digest
# rather than discarded, since emergent threads often start small. See doc for the measurement
# behind the thresholds:
TIER_STANDALONE_MIN = 10    # enough trajectory to narrate on its own
TIER_SPROUT_MIN = 3

# ── Floor calibration ───────────────────────────────────────────────────────────
CALIBRATION_CHUNKS = 2_000  # sampled chunk vectors held in RAM at once (~33MB at 4096 float32)
CALIBRATION_PAIRS = 20_000  # random pairs drawn from that sample
CALIBRATION_MARGIN = 0.03   # sit just ABOVE the noise ceiling, not on it
CALIBRATION_SEED = 7        # fixed, so the floor a run reports is reproducible
# Below this many DISTINCT chunks the calibration is not measured enough to enforce. Derived, not
# picked: n chunks give n(n-1)/2 distinct pairs, and 200 is where that first reaches
# CALIBRATION_PAIRS — under it the 20,000 draws are resampling a handful of values, so the p99 is
# an accident of which few atoms happen to exist. Clamping on that would push seeds into
# `SeedError` on a store's first day, which is the one day nobody can debug it.
CALIBRATION_MIN_CHUNKS = 200


class SeedError(ValueError):
    """The seed could not be resolved to a point in embedding space.

    Raised, not degraded: a seed that matched nothing has no honest empty-sitting fallback — that
    would report "this corner of the corpus is empty" when the truth is "not in the corpus at all".
    """

# ── Floor calibration ───────────────────────────────────────────────────────────
def calibrate_floor(conn, *, sample: int = CALIBRATION_CHUNKS,
                    pairs: int = CALIBRATION_PAIRS) -> dict:
    """The corpus' own noise ceiling: `{p50, p90, p99, floor, n_chunks}`.

    Draws random chunk pairs and puts the floor just above their 99th-percentile cosine, so at
    most ~1% of truly unrelated pairs can sneak in.

    ENFORCED, not advisory, since 2026-08-25: `build_sitting` raises a below-ceiling floor to this
    number when it creates a region (see the clamp there for why creation only, and for the
    min-sample guard that keeps a young store out of it). This function still only MEASURES — the
    `calibrate` CLI subcommand prints the raw numbers and changes nothing.

    Samples HUMAN_ATTESTED chunks only, and did NOT widen with the union (RULED 2026-08-24). The
    number this returns is "how alike are two unrelated things in David's corpus" — the bar every
    region is then judged against. Frontier atoms are pulled by queries derived from that same
    corpus, so they sit closer together than random, and folding them in would drag the p99 UP and
    silently raise the floor on every region. Conscious residue: the floor describes a slightly
    different population than membership now scans.
    """
    dt = sv._dtype(conn)
    rng = np.random.default_rng(CALIBRATION_SEED)
    # Sample across the corpus, not the first N by id — `chunk_id` is insertion order, and a
    # prefix would skew toward the earliest-ingested source/period, dragging the floor down.
    ids = [r[0] for r in conn.execute(
        f"SELECT c.chunk_id FROM chunks c JOIN atoms a ON a.atom_id = c.atom_id "
        f"WHERE c.vector IS NOT NULL AND {sv._human_clause()}", sv.HUMAN_ATTESTED)]
    if len(ids) < 2:
        return {"p50": None, "p90": None, "p99": None, "floor": None, "n_chunks": len(ids)}
    if len(ids) > sample:
        ids = [ids[i] for i in rng.choice(len(ids), size=sample, replace=False)]
    blobs = []
    for i in range(0, len(ids), sv.SQL_VARS):
        part = ids[i:i + sv.SQL_VARS]
        blobs += [r[0] for r in conn.execute(
            f"SELECT vector FROM chunks WHERE chunk_id IN ({sv._in_clause(len(part))})", part)]
    V = sv._decode(blobs, dt)
    i = rng.integers(0, len(V), pairs)
    j = rng.integers(0, len(V), pairs)
    keep = i != j                                   # a chunk paired with itself scores 1.0
    cos = np.einsum("ij,ij->i", V[i[keep]], V[j[keep]])
    p50, p90, p99 = (float(np.percentile(cos, q)) for q in (50, 90, 99))
    return {"p50": round(p50, 3), "p90": round(p90, 3), "p99": round(p99, 3),
            "floor": round(p99 + CALIBRATION_MARGIN, 2), "n_chunks": len(V)}

# ── Seeds ───────────────────────────────────────────────────────────────────────
# Vector retrieval always returns something, so a meaningless phrase needs a bar or it would
# quietly build a plausible-looking region out of the least-unrelated atoms in the store.
# NOT `FLOOR_DEFAULT`: that is calibrated on chunk-to-chunk cosine; this lives on query-to-chunk
# cosine (the phrase carries the Qwen instruction prefix, the chunk does not) — a different scale.
# Measured, not calibrated; re-run before trusting on a different store or embedder.
SEED_MATCH_FLOOR = 0.40


def _vector_seed_atoms(conn, phrase: str, embedder, *, limit: int) -> list:
    """The human-attested atoms nearest the phrase in embedding space. Raises if none is near enough.

    The phrase vector is never the seed itself — it only picks atoms; the seed is the centroid of
    their content chunks (computed by `resolve_seed`, same as an `atoms` seed). So the anchor stays
    in chunk space and `FLOOR_DEFAULT` stays valid; `SEED_MATCH_FLOOR` is the only new number, and
    it lives on the query scale. Ranked by cosine, not recency, since these atoms become the
    centroid anchoring the whole sitting.
    """
    # Same-subspace guard: a seed centroid built from one model's query vector compared against
    # another model's chunks would make every downstream cosine meaningless while still looking
    # calibrated. This function only embeds and discards a phrase — it never writes a chunk vector,
    # so it doesn't need `assert_model`'s stricter write-hazard check.
    assert_model(conn, embedder, storage_dtype=stored_dtype(conn))
    q = np.asarray(embedder.embed([phrase], role="query")[0], dtype=np.float32)
    # `_relevance`, not `retrieve.atom_semantic_search`: it enforces `_human_clause` internally,
    # so the anti-narrowing restriction is structural rather than a candidate set a caller passes.
    # (Both stream now — that stopped being the distinguishing reason on 2026-08-26.)
    #
    # NARROW on purpose, and it stays narrow after the union widened membership: this picks the
    # atoms whose centroid BECOMES the anchor the whole region is measured against forever. Seeding
    # a region on machine finds would let the frontier choose what a typed phrase means — the
    # anti-narrowing loop, arriving one level up from where the guard watches for it.
    scored = sv._relevance(conn, q)
    ranked = sorted(((float(v[0]), a) for a, v in scored.items()), reverse=True)
    if not ranked or ranked[0][0] < SEED_MATCH_FLOOR:
        best = f"{ranked[0][0]:.3f}" if ranked else "nothing embedded"
        raise SeedError(
            f"nothing in the trusted corpus is close enough to {phrase!r} "
            f"(best atom {best}, floor {SEED_MATCH_FLOOR}) — this phrase names something the "
            f"corpus does not hold")
    # The bar prunes as well as gates: a phrase with one strong match and three mediocre ones gets
    # a one-atom seed rather than being padded up to `limit`.
    #
    # This ranking is not reproducible for a near-floor phrase: the hosted embedder is
    # non-deterministic (measured — repeated embeds of the same phrase return different vectors,
    # cosine ~0.9999 apart, never bit-identical), which can reorder or swap near-floor matches. A
    # continuation therefore reuses the STORED seed vector (D2) rather than re-resolving the phrase.
    # `region_key` (D1) excludes resolved atom ids for the same reason: it identifies "the same
    # question," not "the same atoms."
    return [a for s, a in ranked[:limit] if s >= SEED_MATCH_FLOOR]

# Held at 4 by a k-sweep (docs/plans/2026-08-16-sitting-rail-k-sweep.md): region size is monotone
# in k, k=1 is unstable (single-author regions), and k=8 buys depth on rich phrases but drift on
# thin ones (the majority). A per-author seed cap would fix the remaining author-voice-clustering
# failure but is not shipped — see doc for why.
SEED_ATOMS_DEFAULT = 4      # enough to average out one atypical post, few enough to stay one topic

def resolve_seed(conn, *, query: str | None = None, atom_ids=None, vector=None,
                 label: str | None = None, limit: int = SEED_ATOMS_DEFAULT,
                 embedder=None) -> dict:
    """One of three seed shapes -> `{kind, ref, atom_ids, vector}`.

    The three sources — a typed phrase (`query`), an atom or person pointed at (`atom_ids`), a
    system-picked centroid (`vector`) — all reduce to a point in chunk space.

    `query` is the one shape that spends: `embedder` is a required parameter rather than built
    internally, so cost stays visible at the call site and the seed path stays testable offline.
    """
    if vector is not None:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        return {"kind": "vector", "ref": label or "centroid", "atom_ids": [],
                "vector": v / (np.linalg.norm(v) + 1e-9)}
    if atom_ids:
        seeds = list(atom_ids)
        ref = label or ",".join(seeds[:4])
        kind = "atoms"
    elif query:
        if embedder is None:
            raise SeedError("a query seed needs an embedder — see resolve_seed's docstring")
        seeds = _vector_seed_atoms(conn, query, embedder, limit=limit)
        ref, kind = label or query, "query"
    else:
        raise SeedError("a seed needs one of: query, atom_ids, vector")

    vecs = sv._atom_chunk_vectors(conn, seeds)
    stack = [m for a in seeds if (m := vecs.get(a)) is not None and len(m)]
    if not stack:
        raise SeedError(f"seed atoms have no embedded chunks: {seeds}")
    c = np.vstack(stack).mean(axis=0)
    return {"kind": kind, "ref": ref, "atom_ids": [a for a in seeds if a in vecs],
            "vector": c / (np.linalg.norm(c) + 1e-9)}

# ── Continuations ───────────────────────────────────────────────────────────────
class ChainError(KeyError):
    """A continuation named a parent that cannot be resolved.

    Raised, never degraded: silently treating an unresolvable parent as "no parent" would produce
    a fresh sitting wearing a continuation's name, re-admitting everything earlier parts read while
    reporting `skipped_dupes=0` — indistinguishable from a continuation with no real duplicates.
    """


def chain_atom_ids(conn, sitting_id: str | None) -> list[str]:
    """Every atom read by `sitting_id` and by all of its ancestors, nearest part first.

    Walks the `continues` links rather than taking an atom list from the caller, so a hand-assembled
    list can't silently drop a part and shrink the redundancy baseline.
    """
    out: list[str] = []
    seen: set[str] = set()
    cur = sitting_id
    while cur:
        if cur in seen:
            # Only reachable from a corrupted store; bound the walk rather than hang on it.
            raise ChainError(f"continuation chain loops at {cur!r}")
        seen.add(cur)
        row = conn.execute("SELECT continues FROM sittings WHERE sitting_id = ?", (cur,)).fetchone()
        if row is None:
            raise ChainError(f"no sitting {cur!r} to continue from")
        out += [r["atom_id"] for r in conn.execute(
            "SELECT atom_id FROM sitting_atoms WHERE sitting_id = ? ORDER BY rank", (cur,))]
        cur = row["continues"]
    return list(dict.fromkeys(out))

# ── The frontier budget ─────────────────────────────────────────────────────────
def _apply_frontier_budget(conn, pool: list, rel: dict, prior: list) -> tuple[list, list]:
    """Enforce FRONTIER_REGION_BUDGET over `pool`: `(kept_pool, dropped)`.

    Occupancy is DERIVED — prior chain members counted by their CURRENT `entry_mode` at build
    time, never a stored counter, which would be a second source of truth with a decrement hook.
    So a member `ingest_common.promote_atom` flipped since the last part frees its slot with no
    extra logic here, and the promotion code is reused untouched (decision 5, 2026-08-27).

    Frontier pool members compete for the remaining slots on `rel` — the score admission already
    computed — highest first. The losers are recorded with a reason, never silently vanished; a
    budget drop must not look like a floor rejection. Human-attested pool members pass through
    untouched whatever the budget says.
    """
    if not pool:
        return pool, []
    modes: dict[str, str] = {}
    ids = list(dict.fromkeys(pool + prior))
    for i in range(0, len(ids), sv.SQL_VARS):
        part = ids[i:i + sv.SQL_VARS]
        modes.update((r["atom_id"], r["entry_mode"]) for r in conn.execute(
            f"SELECT atom_id, entry_mode FROM atoms "
            f"WHERE atom_id IN ({sv._in_clause(len(part))})", part))
    spent = sum(1 for a in prior if modes.get(a) == "frontier")
    slots = max(0, FRONTIER_REGION_BUDGET - spent)
    ranked = sorted((a for a in pool if modes.get(a) == "frontier"),
                    key=lambda a: rel[a], reverse=True)
    over = set(ranked[slots:])
    dropped = [{"atom_id": a, "rel": round(rel[a], 4), "reason": "frontier_budget"}
               for a in ranked[slots:]]
    return [a for a in pool if a not in over], dropped

# ── The loop ────────────────────────────────────────────────────────────────────
def build_sitting(conn, seed: dict, *, floor: float = FLOOR_DEFAULT,
                  ceiling: float = CEILING_DEFAULT,
                  budget_tokens: int = BUDGET_TOKENS_DEFAULT, persist: bool = True,
                  continues: str | None = None, parent_sitting_id: str | None = None,
                  now: datetime | None = None) -> dict:
    """Grow one sitting from a resolved seed and (by default) record it.

    States, in order:
        seed        -> the anchor vector is fixed and never moves again
        region      -> every REGION_VISIBLE atom (human-attested ∪ frontier) with rel >= floor,
                       frontier members capped at FRONTIER_REGION_BUDGET across the whole chain
                       (membership decided ONCE)
        admission   -> chronological, oldest first, skipping near-duplicates above the ceiling
        stop        -> pool exhausted (saturation) | budget reached (budget) | nothing there (empty)

    `rel` is computed once before the first admission and never recomputed — that is the anchor.
    Only `red` (redundancy) moves during the loop.

    `continues` makes this part N of a chain: earlier parts' atoms leave the pool and enter the
    redundancy baseline, so an atom isn't read twice and the same content can't recur under a
    different id. `parent_sitting_id` records a `zoom` fracture as provenance; it changes no
    behavior — a sub-sitting grows exactly like any other.

    `floor` is RAISED to the corpus noise ceiling when this creates a region and the requested
    value sits below it. Never lowered, and never re-applied to a chained part — see the clamp.
    """
    ref = now or utc_now()
    prior = chain_atom_ids(conn, continues)
    cal = calibrate_floor(conn)
    if (continues is None and cal["floor"] is not None
            and cal["n_chunks"] >= CALIBRATION_MIN_CHUNKS and floor < cal["floor"]):
        # ENFORCED, not warned (RULED 2026-08-25). Below the noise ceiling an admission is not
        # distinguishable from unrelated text, and warn-only meant the build logged a line nobody
        # reads and admitted the noise anyway. Raise only, never lower: there is no named caller
        # for a deliberate below-noise pass, and the `calibrate` CLI still prints the raw numbers
        # for anyone who wants to look.
        #
        # AT REGION CREATION ONLY, which is what `continues is None` means. The measured ceiling
        # drifts with the corpus; a region's floor is frozen into its `region_key`. Re-clamping a
        # chained part would change that key, orphan the notebook chain and the read stamp, and
        # make the scheduler re-read and re-pay for the whole region. A fracture child arrives here
        # with `continues=None` and clamps like any other new region, which is correct — it IS one.
        log(f"[sitting] floor {floor} raised to this corpus' noise ceiling {cal['floor']} "
            f"(random-pair p99={cal['p99']} over {cal['n_chunks']} chunks)")
        floor = cal["floor"]

    # THE UNION (RULED 2026-08-24): a region contains Frontier's finds alongside David's own
    # material. `sre.projection` below measures whatever this scan admits, so the two cannot
    # disagree about which atoms exist.
    rel_all = sv._relevance(conn, seed["vector"], entry_modes=sv.REGION_VISIBLE)
    rel = {a: float(v[0]) for a, v in rel_all.items()}
    seed_atoms = [a for a in seed["atom_ids"] if a in rel]
    # Seed atoms stay in every part (even a continuation): a fresh agent reading part N standalone
    # needs to see what the region is anchored to.
    read_before = set(prior) - set(seed_atoms)
    pool = [a for a, r in rel.items()
            if r >= floor and a not in seed_atoms and a not in read_before]
    # THE BUDGET (RULED 2026-08-27), cut before vectors or projections are loaded: a dropped
    # frontier atom costs nothing downstream and stays out of the remaining-mass numbers — a next
    # part could not read it either, unless promotion frees a slot first.
    pool, budget_dropped = _apply_frontier_budget(conn, pool, rel, prior)

    # Chunk vectors for the region, loaded ONCE and used twice: the redundancy baseline below, and
    # the render projection here. Both need per-chunk grain, and decoding the region twice is the
    # only cost of keeping them apart.
    vecs = sv._atom_chunk_vectors(conn, pool + seed_atoms + prior)
    # THE BUDGET BILLS RENDERED TOKENS, NOT STORED ONES (RULED 2026-08-24). A full-text paper
    # renders as head + the sections that cleared the floor, so billing its stored size would cut
    # the part at the wrong boundary — permanently, because a closed part is read,
    # claims-extracted and lens-cached and never repartitioned. `sre.projection` is the single
    # place that decides both the cost and the text, so the two cannot drift.
    proj = sre.projection(conn, pool + seed_atoms, vectors=vecs,
                          seed_vector=seed["vector"], floor=floor)
    toks = {a: p["tokens"] for a, p in proj.items()}
    # REMAINING, not the whole region: full region = region_atoms + prior_atoms + |seeds|.
    region_atoms, region_tokens = len(pool), sum(toks.get(a, 0) for a in pool)

    admitted: list[dict] = [
        {"atom_id": a, "rank": i, "is_seed": 1, "rel": round(rel[a], 4),
         "red": None, "tokens": toks.get(a, 0)}
        for i, a in enumerate(seed_atoms)]
    used = sum(toks.get(a, 0) for a in seed_atoms)
    skipped: list[dict] = []

    if not rel:
        stop = "empty"
    else:
        pool = [a for a in pool if a in vecs]
        # Redundancy baseline = seeds ∪ every prior part, so the same content can't recur under a
        # different atom id across continuation parts.
        baseline = [a for a in dict.fromkeys(seed_atoms + prior) if a in vecs]
        # No prior part and a raw-vector seed -> nothing admitted yet, so the first pick is pure relevance.
        red = ({a: float((vecs[a] @ np.vstack([vecs[b] for b in baseline]).T).max()) for a in pool}
               if baseline else {a: 0.0 for a in pool})
        rank = len(admitted)
        # THE PART CUT (RULED 2026-08-24): oldest first, take until the budget binds. The order is
        # fixed before the first admission and never re-sorted — that is the difference from the
        # MMR loop this replaces, and the reason it replaces it. MMR is ANTI-CLUSTERING by design:
        # it spreads similar items, so repurposed as a partitioner it maximally severs threads. A
        # rebuttal scores redundant against the claim it answers, which put the original in part 1
        # and the response in part 2 systematically — the split was adversarial, not random.
        # Chronology cuts on the one axis the material actually has.
        order = sre.chronological_ids(conn, pool)
        remaining = False
        for i, a in enumerate(order):
            if used >= budget_tokens:
                # Atoms left AND admissible-looking: this is a real part boundary, not the end of
                # the region. `_remainder_claims` reads `stop='budget'` as exactly that.
                remaining = True
                break
            if red[a] > ceiling:
                # Near-duplicate (same thread reposted). Skipped AND recorded, so a reader that
                # dropped material doesn't look the same as one that read everything.
                skipped.append({"atom_id": a, "red": round(red[a], 4)})
                continue
            admitted.append({"atom_id": a, "rank": rank, "is_seed": 0,
                             "rel": round(rel[a], 4), "red": round(red[a], 4),
                             "tokens": toks.get(a, 0)})
            used += toks.get(a, 0)
            rank += 1
            nb = vecs[a]
            for b in order[i + 1:]:                         # incremental redundancy update
                m = float((vecs[b] @ nb.T).max())
                if m > red[b]:
                    red[b] = m
        # EXACT, and the exactness is what `_remainder_claims` rests on: 'budget' means A NEXT PART
        # WOULD READ SOMETHING. Two ways that is false. A region that ran out of material saturated,
        # even if its last atom happened to cross the budget line — reporting 'budget' there claims
        # a remainder read forever against an empty pool. And a part that admitted only its SEEDS
        # read nothing new: seeds are re-admitted into every part, so its successor would render the
        # identical document and the scheduler would pay for it on every pass. Reachable by hand
        # when `budget_tokens` is set below the seed mass (measured 2026-08-25 while building a
        # 3-part chain at budget=300); unreachable at the 120k default, where the tiered render caps
        # any single atom near 2k.
        stop = "budget" if remaining and len(admitted) > len(seed_atoms) else "saturation"
        if not admitted:
            stop = "empty"

    rec = {
        "sitting_id": _sitting_id(seed, floor, ceiling, budget_tokens, continues,
                                  parent_sitting_id, ref),
        "built_at": utc_iso(ref), "seed_kind": seed["kind"], "seed_ref": seed["ref"],
        "seed_atom_ids": seed_atoms, "floor": floor, "calibrated_floor": cal["floor"],
        "ceiling": ceiling, "budget_tokens": budget_tokens,
        "region_atoms": region_atoms, "region_tokens": region_tokens,
        "atoms": len(admitted), "tokens": used, "stop": stop,
        # `skipped` carries both drop kinds, each row naming its own shape; `skipped_dupes` stays
        # the NEAR-DUPLICATE count alone — the render and the surface report it under that name.
        "skipped_dupes": len(skipped), "admissions": admitted,
        "skipped": budget_dropped + skipped,
        "continues": continues, "prior_atoms": len(read_before), "calibration": cal,
        "parent_sitting_id": parent_sitting_id,
        # Travels with the record so `zoom` stores the centroid actually used, not one re-derived
        # from a sub-sitting's admitted atoms (they differ).
        "seed_vector": seed["vector"],
    }
    if persist:
        sst.record_sitting(conn, rec)
    return rec


def _sitting_id(seed: dict, floor, ceiling, budget, continues, parent, ref: datetime) -> str:
    """Identity = the seed, the dials, the PARENTS, and the moment (an event, not a region).

    Time is in the key so a rebuild doesn't erase the `read_at` of the sitting it replaces.
    `continues`/`parent` are in the key because `_iso` is second-resolution: building two parts (or
    two zoom fractures) back to back can land in the same second and collide without them.
    Not `schema.region_key`, which identifies the region an event reads, not the event itself —
    the two keys are deliberately different and must not be collapsed.
    """
    key = "|".join([seed["kind"], str(seed["ref"]), ",".join(sorted(seed["atom_ids"])),
                    f"{floor}", f"{ceiling}", f"{budget}", continues or "",
                    parent or "", utc_iso(ref)])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

# Two questions, one bucket table: "worth building" (free, local) and "worth reading" (spends) are
# split into two named functions so their thresholds can move apart later without touching call sites.
def _bucket(n: int) -> str:
    """`standalone` | `sprout` | `ledger` for a count of `n` atoms.

    No hard existence bar: small regions are batched into a digest rather than discarded, since
    emergent threads often start small.
    """
    if n >= TIER_STANDALONE_MIN:
        return "standalone"
    return "sprout" if n >= TIER_SPROUT_MIN else "ledger"


def tier_for_material(region_atoms: int) -> str:
    """What a region holding this much MATERIAL is worth BUILDING as.

    Input is the candidate pool before admission (`region_atoms`) — a forecast of whether growing
    a sitting is worth it, valid even at `next_seed` where nothing has been built yet.
    """
    return _bucket(region_atoms)


def tier_for_reading(admitted_atoms: int) -> str:
    """What a sitting holding this many ADMITTED atoms is worth READING as.

    Input is what a reader would actually be handed, after the budget and near-duplicate skips —
    this gates a paid read, so it must never count material nobody will see.
    """
    return _bucket(admitted_atoms)

# ── CLI ─────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    # Lazy: only the CLI needs these three modules, and a module-level import would cycle back
    # through sitting_zoom/sitting_render -> this module. Imported early since the `zoom`
    # subcommand's help text reads `sitting_zoom.ZOOM_TARGET_ATOMS`.
    from . import sitting_render as sr, sitting_store as sst, sitting_zoom as sz

    ap = argparse.ArgumentParser(
        description="Assemble a reading context from a seed (local; --query costs one embed call)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="grow one sitting from a seed")
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--query", help="a phrase; the atoms nearest it in embedding space become the "
                                   "seed. Costs one metered embed call")
    g.add_argument("--atoms", help="comma-separated atom_ids to seed from ($0, local)")
    b.add_argument("--floor", type=float, default=FLOOR_DEFAULT)
    b.add_argument("--seed-atoms", type=int, default=SEED_ATOMS_DEFAULT,
                   help="cap on how many nearest atoms form the seed centroid; matches below "
                        f"{SEED_MATCH_FLOOR} are dropped even under the cap")
    b.add_argument("--ceiling", type=float, default=CEILING_DEFAULT)
    b.add_argument("--budget", type=int, default=BUDGET_TOKENS_DEFAULT)
    b.add_argument("--continues", metavar="SITTING_ID",
                   help="build the NEXT part of a budget-stopped region: the named sitting and all "
                        "of its ancestors leave the pool and enter the redundancy baseline")
    b.add_argument("--out", help="also write <slug>.md + <slug>.manifest.json here")
    b.add_argument("--dry-run", action="store_true", help="score it, record nothing")

    z = sub.add_parser("zoom", help="fracture a recorded sitting into sub-conversations ($0, local)")
    z.add_argument("--sitting", required=True, metavar="SITTING_ID",
                   help="the coarse region to fracture")
    z.add_argument("--k", type=int, help="how many centroids to place (default: derived from the "
                                         f"parent's size, ~1 per {sz.ZOOM_TARGET_ATOMS} atoms)")
    z.add_argument("--floor", type=float, default=FLOOR_ZOOM,
                   help="membership threshold for the SUB-regions; finer than the parent's")
    z.add_argument("--sweep", metavar="K,K,...",
                   help="try several k and print the comparison instead — persists NOTHING. Read "
                        "the `distinct` column: where it stops growing is how many separable "
                        "sub-conversations the region actually holds")
    z.add_argument("--dry-run", action="store_true", help="fracture and report, record nothing")

    sub.add_parser("calibrate", help="report this corpus' random-pair cosine distribution")
    sub.add_parser("coverage", help="read / never-read mass, and the densest unread region")

    r = sub.add_parser("render", help="re-render a recorded sitting")
    r.add_argument("sitting_id")
    r.add_argument("--out", help="write to this directory instead of stdout")

    args = ap.parse_args(argv)
    conn = schema.connect()
    try:
        if args.cmd == "calibrate":
            print(json.dumps(calibrate_floor(conn), indent=2))
            return 0
        if args.cmd == "coverage":
            # Reports the unread debt and stops there — deliberately proposes no seed.
            print(json.dumps(sst.coverage(conn), indent=2))
            return 0
        if args.cmd == "render":
            if args.out:
                print(json.dumps(sr.write_artifacts(conn, args.sitting_id, args.out), indent=2))
            else:
                print(sr.render_sitting(conn, args.sitting_id))
            return 0
        if args.cmd == "zoom":
            if args.sweep:
                ks = [int(x) for x in args.sweep.split(",") if x.strip()]
                print(sz.render_sweep(sz.sweep_k(conn, args.sitting, ks=ks, floor=args.floor)))
                return 0
            print(sz.render_zoom(sz.zoom(conn, args.sitting, k=args.k, floor=args.floor,
                                     persist=not args.dry_run)))
            return 0

        # Built only for the arm that needs it: constructing an embedder loads a model client, and
        # an atom-seeded build must not pay for that.
        seeder = get_kb_embedder() if args.query else None
        seed = resolve_seed(conn, query=args.query, limit=args.seed_atoms, embedder=seeder,
                            atom_ids=[s.strip() for s in args.atoms.split(",")] if args.atoms
                            else None)
        rec = build_sitting(conn, seed, floor=args.floor, ceiling=args.ceiling,
                            budget_tokens=args.budget, persist=not args.dry_run,
                            continues=args.continues)
        if args.out and not args.dry_run:
            rec["artifacts"] = sr.write_artifacts(conn, rec["sitting_id"], args.out)
        # seed_vector excluded: it's a numpy array and this dump has no `default=`, so it would raise.
        print(json.dumps({k: v for k, v in rec.items()
                          if k not in ("admissions", "seed_vector")}, indent=2))
        print(f"\n{rec['atoms']} atoms / ~{rec['tokens']:,} tok · stop={rec['stop']} · "
              f"remaining {rec['region_atoms']}a/~{rec['region_tokens']:,}t · "
              f"prior {rec['prior_atoms']}a · "
              f"tier={tier_for_material(rec['region_atoms'])}", file=sys.stderr)
        return 0
    except SeedError as e:
        print(f"seed error: {e}", file=sys.stderr)
        return 1
    except ChainError as e:
        print(f"continuation error: {e}", file=sys.stderr)
        return 1
    except KeyError as e:
        # `render`/`zoom` name a sitting that may not exist — not a crash, just "no such id".
        print(f"not found: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
