"""
pipeline/kb/sitting_vectors.py — the chunk-vector plumbing every sitting module streams through.

Zero dependency on any other `sitting_*` module — this is the leaf every one of them (and
`sitting_scheduler.py` externally) reads chunk vectors through, so it has to stay that way or the
import graph gets a cycle. `_relevance` lives here rather than in `sitting_builder.py` for the same
reason: it implements the "membership" formula `sitting_builder`'s docstring names (max-pooled
cosine against a seed), but `sitting_zoom.py`'s `zoom`/`sweep_k` and `sitting_scheduler.py` need the
identical scan, and none of those three should have to reach INTO `sitting_builder` for a function
that has no build-loop state of its own.

HUMAN_ATTESTED lives here too, re-exported from `schema.py` under this name because every membership
query in the sitting rail — builder, store, zoom, render — filters by it, and it is the shared
vocabulary the whole rail speaks, not a builder-specific concept.
"""
from __future__ import annotations

import numpy as np

from . import schema
from .embed import stored_dtype

# ── Which atoms may be assembled ────────────────────────────────────────────────
# Human-attested only — the anti-narrowing clause. Every mode here entered because a person acted
# (David saved it, a confirmed Oracle wrote it, an Oracle pointed at it). The invariant it protects:
# machine-discovered atoms must not feed the query generator, or the system narrows onto its own
# output invisibly. That rests ENTIRELY on `frontier` — written only by Frontier stage 3 — being
# outside this tuple. No other live mode is outside it; the retired `crawled` mode used to be, and
# the exclusion was justified by a v1 artifacts sweep that no longer exists (see
# docs/plans/2026-08-25-rename-github-crawled-to-oracle-footprint.md).
#
# Lives in `schema.py`, next to the `entry_mode` column it describes, and is re-exported here
# under this name because `retrieve.py` and the WHERE clauses below already read it as this name.
HUMAN_ATTESTED: tuple[str, ...] = schema.HUMAN_ATTESTED

# What a REGION may contain — the human-attested set plus Frontier's own finds. Deliberately NOT in
# `schema.py` and deliberately NOT folded into `HUMAN_ATTESTED`: those two are the same edit that
# `.guards.py`'s `human-attested-stays-human` rule exists to stop, because stage 1 generates its
# standing queries FROM `HUMAN_ATTESTED` and admitting stage 3's output there closes the loop.
#
# Reading a frontier atom is not the same act as generating queries from one. A sitting is a
# reading context: the union puts machine finds in front of a human, which is the whole point of
# having found them. What stays narrow is every decision the MACHINE makes on its own — which
# region to spend on (`sitting_scheduler`), what a seed is anchored to (`_vector_seed_atoms`), what
# the corpus' noise floor is (`calibrate_floor`), and what counts as read debt over David's own
# material (`sitting_store.coverage`). Each of those has a test naming it.
REGION_VISIBLE: tuple[str, ...] = HUMAN_ATTESTED + ("frontier",)

# Rows of vectors held in RAM during a streaming pass. 1024 x 4096 float32 = 16MB — deliberately
# small; never materialize a full chunk-vector matrix at once.
VEC_BATCH = 1024
SQL_VARS = 900              # under SQLite's default 999-parameter ceiling


# ── Vector plumbing ─────────────────────────────────────────────────────────────
def _dtype(conn) -> np.dtype:
    """The blob width `chunks.vector` is actually stored at.

    Read from `kb_meta`, never assumed — decoding a float16 blob as float32 (or vice versa)
    doesn't fail, it silently produces garbage cosines that still look plausible. Hardcoding the
    width here would reintroduce the failure `embed.assert_model` exists to prevent.
    """
    return np.dtype(stored_dtype(conn))


def _decode(blobs: list[bytes], dt: np.dtype) -> np.ndarray:
    """Blobs -> (n, dim) float32, L2-normalized rows."""
    mat = np.frombuffer(b"".join(blobs), dtype=dt).reshape(len(blobs), -1).astype(np.float32)
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    return mat


def _in_clause(n: int) -> str:
    return ",".join("?" * n)


def _human_clause(alias: str = "a", modes: tuple[str, ...] = HUMAN_ATTESTED) -> str:
    return f"{alias}.entry_mode IN ({_in_clause(len(modes))})"


def _relevance(conn, seeds: np.ndarray, *, restrict: set[str] | None = None,
               entry_modes: tuple[str, ...] = HUMAN_ATTESTED) -> dict:
    """`{atom_id: array of per-seed MAX cosine}` over every embedded chunk in `entry_modes`.

    ONE streaming pass, several seeds at a time — `sweep_k` stacks centroids across every k it
    tries, and `sitting_scheduler` stacks one anchor per region, so both would otherwise read the
    same blobs once per seed.

    Streams by design rather than materializing the full chunk-vector matrix: peak memory here is
    one batch regardless of corpus size.

    `entry_modes` DEFAULTS to the narrow set, so a caller widens on purpose and a new caller is
    narrow by accident — the same allow-list discipline `HUMAN_ATTESTED` itself states. What this
    scan admits is what `sitting_render.projection` then measures, so widening here cannot leave an
    atom admitted-but-unbilled: the two read the same list.
    """
    dt = _dtype(conn)
    seeds = np.atleast_2d(np.asarray(seeds, dtype=np.float32))
    seeds = seeds / (np.linalg.norm(seeds, axis=1, keepdims=True) + 1e-9)
    cur = conn.execute(
        f"SELECT c.atom_id, c.vector FROM chunks c JOIN atoms a ON a.atom_id = c.atom_id "
        f"WHERE c.vector IS NOT NULL AND {_human_clause(modes=entry_modes)}", entry_modes)
    out: dict[str, np.ndarray] = {}
    while True:
        rows = cur.fetchmany(VEC_BATCH)
        if not rows:
            return out
        keep = [r for r in rows if restrict is None or r["atom_id"] in restrict]
        if not keep:
            continue
        sims = _decode([r["vector"] for r in keep], dt) @ seeds.T      # (batch, n_seeds)
        for i, r in enumerate(keep):
            prev = out.get(r["atom_id"])
            out[r["atom_id"]] = sims[i] if prev is None else np.maximum(prev, sims[i])


def _atom_chunk_vectors(conn, atom_ids) -> dict:
    """`{atom_id: (n_chunks, dim) float32}` for a BOUNDED set of atoms.

    Only ever called with a seed set or a region, both small.

    Deliberately no length filter here — do not re-add a short-chunk cutoff. A character-count
    threshold can't separate machine-generated chunks from authored ones in this corpus (machine
    blocks run longer), so it drops real short posts while barely touching what it's meant to
    filter. See doc for the measurement behind this. Filtering machine-shaped chunks out of a seed
    must key on PROVENANCE (the `*Image:*` marker `vision.py` writes), never on length.
    """
    ids = list(dict.fromkeys(atom_ids))
    if not ids:
        return {}
    dt = _dtype(conn)
    every: dict[str, list] = {a: [] for a in ids}
    for i in range(0, len(ids), SQL_VARS):
        part = ids[i:i + SQL_VARS]
        rows = conn.execute(
            f"SELECT atom_id, vector FROM chunks "
            f"WHERE vector IS NOT NULL AND atom_id IN ({_in_clause(len(part))}) ORDER BY seq",
            part).fetchall()
        for r in rows:
            every[r["atom_id"]].append(r["vector"])
    return {a: _decode(v, dt) for a, v in every.items() if v}
