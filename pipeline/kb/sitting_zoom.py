"""
pipeline/kb/sitting_zoom.py — fracture a recorded sitting into its sub-conversations. $0, local.

Split out of `sitting_builder.py` 2026-08-16 (pure move, no behavior change) — see that module's
docstring for WHY re-seeding beats partitioning and what zoom costs. `zoom` re-runs the
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
