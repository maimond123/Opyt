"""
pipeline/kb/sitting_zoom.py — fracture a recorded sitting into its sub-conversations. $0, local.

Split out of `sitting_builder.py` 2026-08-16 (pure move, no behavior change) — see that module's
docstring for WHY re-seeding beats partitioning and what zoom costs. `zoom`/`sweep_k` re-run the
SAME membership rule `sitting_builder.build_sitting` uses (via `sitting_vectors._relevance`), over
k-means centroids of a parent region's own content chunks — zoom manufactures SEEDS, it never
changes a rule, so every property the builder proves holds unchanged for a sub-sitting.

Depends on `sitting_builder` (`resolve_seed`, `build_sitting`, `tier_for_reading`,
`CALIBRATION_SEED`), `sitting_store` (`get_sitting`, `record_sitting`) and `sitting_vectors`
(`_relevance`, `_atom_chunk_vectors`) and `sitting_render` (`projection`, `_spans`,
`whole_tokens`) — all four import cleanly at module scope, since none of them depends back on this
module.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from pipeline.timeparse import utc_now

from . import sitting_builder as sb
from . import sitting_render as sre
from . import sitting_store as sst
from . import sitting_vectors as sv

# ── Zoom dials ──────────────────────────────────────────────────────────────────
# k is DERIVED: k = clamp(ceil(region_atoms / ZOOM_TARGET_ATOMS), 2, 12) — a READING size, not a
# geometric optimum. Over-splitting is self-correcting: near-duplicate centroids get collapsed by
# the D3 merge below.
ZOOM_TARGET_ATOMS = 60
ZOOM_K_MIN, ZOOM_K_MAX = 2, 12
# Two sub-regions sharing this much of their atom sets are ONE read, not two — logged with Jaccard.
ZOOM_MERGE_J = 0.30
KMEANS_MAX_ITERS = 50

# ── Zoom: fracture one region into its sub-conversations ────────────────────────
def _kmeans(V: np.ndarray, k: int, seed: int = sb.CALIBRATION_SEED) -> tuple:
    """`(centroids (k, dim) unit-normalized, labels (n,))` — k-means++ init, then Lloyd.

    Written here instead of imported (scikit-learn is not a project dependency). Spherical: rows
    and centroids stay unit-normalized so "nearest centroid" is cosine `argmax`, matching the
    floor's geometry. Deterministic via fixed `default_rng(seed)`, so drop counts are reproducible.
    """
    V = np.asarray(V, dtype=np.float32)
    n = len(V)
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)

    # k-means++ : first center uniform, each next one drawn with probability proportional to its
    # squared distance from the nearest center already taken. On unit rows, ||u-v||^2 = 2 - 2*u·v.
    idx = [int(rng.integers(0, n))]
    d2 = np.maximum(0.0, 2.0 - 2.0 * (V @ V[idx[0]]))
    while len(idx) < k:
        total = float(d2.sum())
        if total <= 1e-12:
            # All remaining points coincide with a taken center; fill in index order instead of
            # dividing by zero — duplicate centroids are harmless, the D3 merge collapses them.
            taken = set(idx)
            idx += [i for i in range(n) if i not in taken][:k - len(idx)]
            break
        nxt = int(rng.choice(n, p=d2 / total))
        idx.append(nxt)
        d2 = np.minimum(d2, np.maximum(0.0, 2.0 - 2.0 * (V @ V[nxt])))

    C = V[idx].copy()
    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(KMEANS_MAX_ITERS):
        new = np.argmax(V @ C.T, axis=1).astype(np.int64)
        if np.array_equal(new, labels):
            break                                   # assignment is stable; the means cannot move
        labels = new
        for j in range(k):
            members = V[labels == j]
            if len(members):
                C[j] = members.mean(axis=0)
            else:
                # An empty cluster re-seeds to its worst-fit point, guaranteeing k centroids back.
                worst = int(np.argmin(np.einsum("ij,ij->i", V, C[labels])))
                C[j] = V[worst]
                labels[worst] = j
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return C, labels


def _parent_chunks(conn, atom_ids: list) -> np.ndarray | None:
    """The parent's chunk vectors as one `(n_chunks, dim)` matrix, or None if there are none.

    Uses every chunk, deliberately not `content_only=True` — a length-based short-chunk filter was
    measured to do nothing useful here and was removed (see doc). RAM is not a concern: the largest
    real region is 419 chunks (~7MB at 4096 float32).
    """
    vecs = sv._atom_chunk_vectors(conn, atom_ids)
    stack = [m for a in atom_ids if (m := vecs.get(a)) is not None and len(m)]
    return np.vstack(stack) if stack else None


def _fit_k(k: int, n_chunks: int) -> int:
    """k, reduced to what the parent can actually support: at least 2 chunks per cluster.

    Reduced rather than refused, per the fail-safe rule. A thin parent still has a fracture worth
    looking at; erroring on it would make zoom unusable on exactly the small regions where a
    grab-bag most needs splitting apart.
    """
    return max(1, min(int(k), n_chunks // 2))


def derived_k(region_atoms: int) -> int:
    """D2: `clamp(ceil(atoms / ZOOM_TARGET_ATOMS), 2, 12)`. On the 362-atom mlx region this is 6."""
    want = -(-int(region_atoms) // ZOOM_TARGET_ATOMS)          # ceil division, no float rounding
    return max(ZOOM_K_MIN, min(ZOOM_K_MAX, want))


def _merge_overlaps(sets: list) -> tuple:
    """D3, greedy: `(kept indices, [{i, into, jaccard}])`. Keep the largest, drop what overlaps it.

    Deterministic — candidates are visited largest-first with the index as tie-break, so the same
    fracture merges the same way twice.
    """
    kept: list = []
    merged: list = []
    for i in sorted(range(len(sets)), key=lambda i: (-len(sets[i]), i)):
        hit = None
        for j in kept:
            union = len(sets[i] | sets[j])
            jac = (len(sets[i] & sets[j]) / union) if union else 0.0
            if jac >= ZOOM_MERGE_J:
                hit = (j, jac)
                break
        if hit is None:
            kept.append(i)
        else:
            merged.append({"i": i, "into": hit[0], "jaccard": round(hit[1], 3)})
    return sorted(kept), merged


def zoom(conn, sitting_id: str, *, k: int | None = None, floor: float = sb.FLOOR_ZOOM,
         persist: bool = True, now: datetime | None = None) -> dict:
    """Fracture a recorded sitting into k sub-sittings at a finer floor. $0 — no LLM, no network.

    States, in order:
        parent      -> its admitted atoms' content chunks become one matrix
        centroids   -> k-means over that matrix; k is derived from the parent's size unless given
        sub-regions -> each centroid is an ORDINARY vector seed; `build_sitting` re-runs membership
                       over the whole corpus at `floor` (re-seed, not partition — see the module
                       docstring for why, and for what it costs)
        merge       -> sub-regions overlapping a kept one at J >= 0.30 are dropped, each with its J
        persist     -> only STANDALONE-tier keepers become rows; smaller ones are reported as sprout
                       mass, whose consumer is the sprouts digest, not a sitting of their own

    A zoom is not a continuation. `continues` stays NULL: a continuation is the next PART of one
    region and inherits its predecessors' atoms into the redundancy baseline, while a sub-sitting is
    a fresh region at a finer floor that is meant to overlap its siblings. Only `parent_sitting_id`
    links them, and it is a provenance record, not a reading order.

    The returned report is the honest accounting D1 requires. `parent_dropped` is the pure geometry:
    parent atoms below `floor` from EVERY sub-centroid. `parent_uncovered` adds what the merge and
    the token budget left behind. Both are >= 0 by construction and neither is a bug — they are the
    price of re-seeding, and the decision to keep paying it is re-made from these numbers.
    """
    ref = now or utc_now()
    parent = sst.get_sitting(conn, sitting_id)
    if parent is None:
        raise KeyError(f"no sitting {sitting_id!r}")
    patoms = [a["atom_id"] for a in parent["admissions"]]
    rep = {"parent_sitting_id": sitting_id, "parent_ref": parent["seed_ref"],
           "parent_atoms": len(patoms), "parent_floor": parent["floor"], "floor": floor,
           "k": 0, "k_derived": derived_k(len(patoms)), "k_requested": k, "chunks": 0,
           "sub": [], "kept": 0, "merged": [], "persisted": 0,
           "parent_dropped": len(patoms), "parent_uncovered": len(patoms), "reason": None}
    if not patoms:
        return {**rep, "reason": "parent sitting has no atoms"}
    V = _parent_chunks(conn, patoms)
    if V is None:
        # Fail-safe: an unembedded parent (embed pass hasn't run) returns an empty fracture with
        # a reason, not a crash.
        return {**rep, "reason": "parent atoms have no embedded chunks"}
    rep["chunks"] = len(V)

    C, _ = _kmeans(V, _fit_k(rep["k_derived"] if k is None else k, len(V)))
    rep["k"] = len(C)

    # One pass over ALL of an atom's chunks, matching build_sitting's membership exactly — which
    # means the SAME entry-mode tuple. A narrow scan here against a wide parent would report every
    # frontier atom in the parent as "dropped by the fracture" when the fracture never saw it.
    pscore = sv._relevance(conn, C, restrict=set(patoms), entry_modes=sv.REGION_VISIBLE)
    rep["parent_dropped"] = sum(
        1 for a in patoms if a not in pscore or float(pscore[a].max()) < floor)

    subs = []
    for i, c in enumerate(C):
        seed = sb.resolve_seed(conn, vector=c, label=f"{parent['seed_ref']}/{i}")
        subs.append(sb.build_sitting(conn, seed, floor=floor, persist=False, now=ref,
                                     parent_sitting_id=sitting_id))

    # Merge before writing anything, so a sub-sitting the merge drops never reaches the store.
    sets = [{a["atom_id"] for a in r["admissions"]} for r in subs]
    kept_ix, merged = _merge_overlaps(sets)
    kept = set(kept_ix)
    rep["merged"] = [{"label": subs[m["i"]]["seed_ref"], "into": subs[m["into"]]["seed_ref"],
                      "jaccard": m["jaccard"], "atoms": len(sets[m["i"]])} for m in merged]
    rep["kept"] = len(kept_ix)

    pset = set(patoms)
    covered: set = set()
    for i, rec in enumerate(subs):
        s = sets[i]
        # Tiered on admitted atoms (what the reader actually gets), not on region_atoms.
        tier = sb.tier_for_reading(rec["atoms"])
        keep = i in kept
        write = persist and keep and tier == "standalone"
        if write:
            sst.record_sitting(conn, rec)
            rep["persisted"] += 1
        if keep:
            covered |= s
        into = next((m for m in rep["merged"] if m["label"] == rec["seed_ref"]), None)
        rep["sub"].append({
            "label": rec["seed_ref"], "sitting_id": rec["sitting_id"], "atoms": rec["atoms"],
            "tokens": rec["tokens"], "region_atoms": rec["region_atoms"], "stop": rec["stop"],
            "tier": tier, "kept": keep, "persisted": write,
            "overlap_parent": round(len(s & pset) / len(s), 3) if s else 0.0,
            "new_atoms": len(s - pset),
            "merged_into": into["into"] if into else None,
            "jaccard": into["jaccard"] if into else None,
        })
    rep["parent_uncovered"] = len(pset - covered)
    return rep


def sweep_k(conn, sitting_id: str, *, ks, floor: float = sb.FLOOR_ZOOM) -> dict:
    """D2b: fracture the parent at every k in `ks` and compare. Persists nothing.

    The column that decides is `distinct` (post-merge part count), not `k` — asking for 12
    centroids doesn't mean 12 real parts. All k values are scored in one `_relevance()` scan
    (their centroids stacked into one matrix) so the sweep stays cheap. Measures regions, not
    admissions: no MMR ordering, no budget, no near-duplicate skipping.
    """
    parent = sst.get_sitting(conn, sitting_id)
    if parent is None:
        raise KeyError(f"no sitting {sitting_id!r}")
    patoms = [a["atom_id"] for a in parent["admissions"]]
    rep = {"parent_sitting_id": sitting_id, "parent_ref": parent["seed_ref"],
           "parent_atoms": len(patoms), "parent_floor": parent["floor"], "floor": floor,
           "chunks": 0, "rows": [], "reason": None}
    if not patoms:
        return {**rep, "reason": "parent sitting has no atoms"}
    V = _parent_chunks(conn, patoms)
    if V is None:
        return {**rep, "reason": "parent atoms have no embedded chunks"}
    rep["chunks"] = len(V)

    plans, blocks = [], []
    for want in dict.fromkeys(int(x) for x in ks):
        C, _ = _kmeans(V, _fit_k(want, len(V)))
        plans.append({"k": want, "k_used": len(C), "at": sum(len(b) for b in blocks)})
        blocks.append(C)
    scored = sv._relevance(conn, np.vstack(blocks), entry_modes=sv.REGION_VISIBLE)
    pset = set(patoms)
    # Everything below is SEED-INDEPENDENT and so is computed once for the whole sweep rather than
    # once per k: which atoms any candidate region could hold, their chunk spans, and what each
    # costs rendered in full. Only a LONG atom's projection depends on the centroid it is scored
    # against, so only those atoms need per-chunk vectors decoded, and only those are re-projected
    # per region.
    every = sorted({a for a, v in scored.items() if float(v.max()) >= floor})
    spans = sre._spans(conn, every)
    whole = sre.whole_tokens(spans)
    long_ids = {a for a, t in whole.items() if t > sre.LONG_ATOM_TOKENS}
    vecs = sv._atom_chunk_vectors(conn, sorted(long_ids))

    for p, C in zip(plans, blocks):
        lo, hi = p["at"], p["at"] + len(C)
        regions = [{a for a, v in scored.items() if v[j] >= floor} for j in range(lo, hi)]
        kept_ix, merged = _merge_overlaps(regions)
        sizes = sorted(len(regions[i]) for i in kept_ix)
        union: set = set().union(*[regions[i] for i in kept_ix]) if kept_ix else set()
        dropped = sum(1 for a in patoms
                      if a not in scored or float(scored[a][lo:hi].max()) < floor)
        # RENDERED tokens, per region against ITS OWN centroid — the number a build of that
        # sub-region would actually bill. A stored-size forecast would tell a human choosing k that
        # a paper-heavy fracture fits when the build will cut it.
        tokens = 0
        for i in kept_ix:
            ids = sorted(regions[i])
            tokens += sum(whole[a] for a in ids if a not in long_ids)
            cut = [a for a in ids if a in long_ids]
            if cut:
                tokens += sum(x["tokens"] for x in sre.projection(
                    conn, cut, vectors=vecs, seed_vector=C[i], floor=floor, spans=spans).values())
        rep["rows"].append({
            "k": p["k"], "k_used": p["k_used"], "distinct": len(kept_ix), "merged": len(merged),
            "med_size": int(np.median(sizes)) if sizes else 0, "max_size": max(sizes, default=0),
            "new_atoms": len(union - pset), "parent_dropped": dropped,
            "tokens": tokens,
        })
    return rep


def render_zoom(rep: dict) -> str:
    """The zoom report as the lines a human reads. See `docs/plans/2026-08-10-sitting-zoom.md`."""
    out = [f"parent: {rep['parent_ref']} — {rep['parent_atoms']} atoms at floor "
           f"{rep['parent_floor']} · {rep['chunks']} content chunks"]
    if rep["reason"]:
        return "\n".join(out + [f"no fracture: {rep['reason']}"])
    how = (f"derived: {rep['parent_atoms']} atoms / {ZOOM_TARGET_ATOMS} target"
           if rep["k_requested"] is None else f"requested {rep['k_requested']}")
    out += [f"k = {rep['k']} ({how})", ""]
    for s in rep["sub"]:
        # The note says WHY a row did not persist, and the three reasons are different facts. Keying
        # it on the `persisted` flag alone reported every row of a `--dry-run` as a sprout.
        if not s["kept"]:
            note = f"merged into {s['merged_into']} — not persisted"
        elif s["tier"] != "standalone":
            note = f"{s['tier'].upper()} — not persisted"
        else:
            note = "" if s["persisted"] else "not persisted (dry run)"
        out.append(f"  {s['label']:<14} {s['atoms']:>4}a  ~{s['tokens'] // 1000:>4}K tok  "
                   f"{s['stop']:<11} overlap w/ parent {s['overlap_parent']:.0%}  "
                   f"+{s['new_atoms']} new  {s['sitting_id']}  {note}".rstrip())
    for m in rep["merged"]:
        out.append(f"  dropped: {m['label']} vs {m['into']} J={m['jaccard']} "
                   f"→ merged into {m['into']}")
    out.append(f"  {rep['parent_dropped']} parent atoms fell below every sub-centroid "
               f"(fringe, still in the parent sitting)")
    if rep["parent_uncovered"] != rep["parent_dropped"]:
        out.append(f"  {rep['parent_uncovered']} parent atoms are in no kept sub-sitting "
                   f"(the line above, plus what the merge and the budget left out)")
    out.append(f"  {rep['persisted']} of {rep['k']} sub-sittings persisted")
    return "\n".join(out)


def render_sweep(rep: dict) -> str:
    """The D2b comparison table. `distinct` is the column that names the region's part count."""
    out = [f"parent: {rep['parent_ref']} — {rep['parent_atoms']} atoms at floor "
           f"{rep['parent_floor']} · {rep['chunks']} content chunks",
           f"fracturing at floor {rep['floor']} · persists NOTHING", ""]
    if rep["reason"]:
        return "\n".join(out + [f"no fracture: {rep['reason']}"])
    out.append(" k   distinct  merged   med size   max size   new atoms   parent dropped   Σtok")
    for r in rep["rows"]:
        out.append(f"{r['k']:>2}   {r['distinct']:>6}   {r['merged']:>5}   {r['med_size']:>7}   "
                   f"{r['max_size']:>8}   {r['new_atoms']:>8}   {r['parent_dropped']:>13}   "
                   f"{r['tokens'] // 1000:>4}K")
    out.append("")
    out.append(_plateau_note(rep["rows"]))
    return "\n".join(out)


def _plateau_note(rows: list) -> str:
    """Name the part count ONLY when `distinct` actually stopped growing — reading the largest
    `distinct` unconditionally would just report whatever k the sweep happened to stop at.
    """
    if not rows:
        return "no rows — nothing to read"
    for i, r in enumerate(rows[:-1]):
        if all(later["distinct"] <= r["distinct"] for later in rows[i + 1:]):
            return (f"distinct plateaus at {r['distinct']} from k={r['k']} → the region holds "
                    f"~{r['distinct']} separable sub-conversations; past that k only buys "
                    f"duplicates")
    last = rows[-1]
    return (f"NO PLATEAU — distinct is still growing at k={last['k']} ({last['distinct']} "
            f"distinct, {last['merged']} merged). This region has no natural part count at this "
            f"floor: raise k further, raise the floor, or pick k from the size columns instead of "
            f"from a plateau that is not there.")


