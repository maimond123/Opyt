"""Zoom — fracturing one region into sub-conversations, proven against planted geometry.

The claims here are the ones that would be invisible in production if they broke:

  • THE FRACTURE IS REPRODUCIBLE. A non-deterministic split makes every drop count in the report
    unfalsifiable, so nobody could ever check D1's flip condition.
  • OVER-SPLITTING SELF-CORRECTS. Derived-k is only safe because a homogeneous region collapses back
    to ~1 sub-region under the merge. That is asserted here, not assumed, because it is the entire
    argument for not running silhouette.
  • THE SWEEP WRITES NOTHING. It is a measurement, and a measurement that writes is a trap.
  • THE PLATEAU IS REAL. `distinct` naming the region's part count is the claim D2b rests on, so it
    is tested on a region whose true part count is planted and therefore known.
  • ZOOM IS FREE. No LLM call, no `frontier_queries` row. Positive control: the LLM entry point is
    stubbed to raise, and zoom must still complete.
"""
from __future__ import annotations

import hashlib
import sqlite3

import numpy as np
import pytest

from pipeline.kb import schema
from pipeline.kb import sitting_builder as sb
from pipeline.kb import sitting_render as sre
from pipeline.kb import sitting_store as sst
from pipeline.kb import sitting_vectors as sv
from pipeline.kb import sitting_zoom as sz


def _zoomed_from(conn, sitting_id: str) -> list:
    """The sub-sittings a zoom persisted under `sitting_id`, newest first. Was
    `sitting_zoom.zoomed_from`, deleted 2026-08-28 — these assertions were its only readers."""
    return [dict(r) for r in conn.execute(
        "SELECT sitting_id, seed_ref, floor, atoms, tokens, stop, built_at, read_at "
        "  FROM sittings WHERE parent_sitting_id = ? ORDER BY built_at DESC, seed_ref",
        (sitting_id,))]

# WIDE ON PURPOSE. At 8 dimensions the noise term below has only 6 to live in, so two random
# "unrelated" members land at cosine ~0.4 from each other by chance — which pushes a same-cluster
# pair past the builder's 0.95 near-duplicate ceiling and silently skips it. 24 dimensions puts
# random pairs near 0 and keeps the planted geometry the only thing the tests measure.
DIM = 24


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    from pipeline.kb.embed import ensure_kb_meta
    ensure_kb_meta(c, "fake", DIM, "local", "", storage_dtype="float32")
    yield c
    c.close()


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _axis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _atom(conn, atom_id, vecs, *, when="2026-08-01", who="x:user:1",
          entry_mode="user-saved", chars=800):
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode) "
                 "VALUES (?,?,?,?,?)", (atom_id, "x", who, when, entry_mode))
    pos = 0
    for seq, v in enumerate(vecs):
        text = f"{atom_id} chunk {seq} " + ("word " * max(1, chars // 5))
        conn.execute(
            "INSERT INTO chunks (atom_id, seq, char_start, char_end, text, vector) "
            "VALUES (?,?,?,?,?,?)",
            (atom_id, seq, pos, pos + len(text), text, np.asarray(v, dtype=np.float32).tobytes()))
        pos += len(text)
    conn.commit()


# A planted atom = SHARED on axis 0 (what the coarse parent is seeded on) + CLUSTER on its own axis
# (what makes it a knot) + NOISE in the leftover dimensions (what makes its neighbours DISTINCT).
#
# THE NOISE TERM IS LOAD-BEARING, and leaving it out is the first thing that went wrong here. Points
# an epsilon apart are near-duplicates at redundancy ~1.0, so the builder's 0.95 ceiling SKIPS them
# and a "12-atom cluster" arrives as one admitted atom. These weights put members at ~0.75 to each
# other (a cluster, not a repost), ~0.87 to their own centroid, and ~0.35 to a sibling centroid — so
# a 0.70 floor separates the clusters with room on both sides.
SHARED, CLUSTER, NOISE = 0.55, 0.669, 0.5


def _planted(conn, axes, per=12, *, rng_seed=0):
    """One cluster per axis in `axes`, `per` atoms each — a region whose true part count is KNOWN."""
    rng = np.random.default_rng(rng_seed)
    made = []
    for ax in axes:
        for i in range(per):
            g = rng.normal(size=DIM).astype(np.float32)
            g[0] = g[ax] = 0.0          # noise ORTHOGONAL to both dialled axes, so the cosines above
            g /= np.linalg.norm(g) + 1e-9   # are exact and the fixture stays readable
            v = np.zeros(DIM, dtype=np.float32)
            v[0], v[ax] = SHARED, CLUSTER
            aid = f"a:c{ax}-{i}"
            _atom(conn, aid, [_unit(v + NOISE * g)])
            made.append(aid)
    return made


def _long_atom(conn, atom_id="a:long", *, axis=1, chunks=5, chars=4000, rng_seed=7):
    """A planted atom big enough to be CUT — several chunks on the same cluster as `axis`, each
    with its own noise so a projection has something to choose between. Stands in for a full-text
    paper in a region of posts."""
    rng = np.random.default_rng(rng_seed)
    vecs = []
    for _ in range(chunks):
        g = rng.normal(size=DIM).astype(np.float32)
        g[0] = g[axis] = 0.0
        g /= np.linalg.norm(g) + 1e-9
        v = np.zeros(DIM, dtype=np.float32)
        v[0], v[axis] = SHARED, CLUSTER
        vecs.append(_unit(v + NOISE * g))
    _atom(conn, atom_id, vecs, chars=chars)
    return atom_id


def _parent(conn, floor=0.50, **kw):
    """The coarse region: a VECTOR seed on axis 0, so the parent's atoms are exactly the planted
    ones. Seeding from an atom would put an off-cluster atom into every count and blur the
    arithmetic these tests are checking."""
    seed = sb.resolve_seed(conn, vector=_axis(0), label="parent")
    return sb.build_sitting(conn, seed, floor=floor, **kw)


def _tables(conn) -> str:
    """A content hash of everything zoom could possibly write. Cheaper to compare than a file, and
    it ignores SQLite page churn, which is not what "persists nothing" is about."""
    rows = list(conn.execute("SELECT * FROM sittings ORDER BY sitting_id"))
    rows += list(conn.execute("SELECT * FROM sitting_atoms ORDER BY sitting_id, atom_id"))
    blob = "\n".join("|".join(str(v) for v in tuple(r)) for r in rows)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ── k-means ─────────────────────────────────────────────────────────────────────
def test_kmeans_is_deterministic():
    """Same matrix, same k, same seed → the same centroids, bit for bit. Everything the report
    claims about a fracture depends on being able to re-run it."""
    rng = np.random.default_rng(0)
    V = np.asarray([_unit(v) for v in rng.normal(size=(40, DIM))], dtype=np.float32)
    a, la = sz._kmeans(V, 4)
    b, lb = sz._kmeans(V, 4)
    assert np.array_equal(a, b) and np.array_equal(la, lb)


def test_kmeans_recovers_planted_clusters():
    """Three knots on three axes come back as three clusters, with every member in one of them."""
    pts = []
    for ax in (1, 2, 3):
        for i in range(8):
            v = np.zeros(DIM, dtype=np.float32)
            v[ax], v[4] = 1.0, 0.02 * i
            pts.append(_unit(v))
    C, labels = sz._kmeans(np.asarray(pts, dtype=np.float32), 3)
    assert len(C) == 3
    assert {len(labels[labels == j]) for j in range(3)} == {8}


def test_kmeans_returns_k_centroids_when_a_cluster_would_be_empty():
    """Ask for more clusters than there are distinct points. The empty-cluster re-seed is what keeps
    the return shape `k`, so no caller has to branch on a short result."""
    V = np.asarray([_axis(0)] * 3 + [_axis(1)] * 3, dtype=np.float32)
    C, labels = sz._kmeans(V, 5)
    assert len(C) == 5
    assert np.allclose(np.linalg.norm(C, axis=1), 1.0, atol=1e-4)
    assert set(labels.tolist()) <= set(range(5))


def test_kmeans_clamps_k_to_the_number_of_points():
    V = np.asarray([_axis(0), _axis(1)], dtype=np.float32)
    C, _ = sz._kmeans(V, 9)
    assert len(C) == 2


def test_derived_k_is_clamped_between_two_and_twelve():
    """D2's formula verbatim: `clamp(ceil(atoms / 60), 2, 12)`. 193 atoms -> 4 is the number the
    plan's own interaction walkthrough prints, so it is the one pinned here."""
    assert sz.derived_k(193) == 4
    assert sz.derived_k(120) == 2 and sz.derived_k(121) == 3      # the ceiling, not rounding
    assert sz.derived_k(1) == sz.ZOOM_K_MIN
    assert sz.derived_k(100_000) == sz.ZOOM_K_MAX


def test_k_is_reduced_when_the_parent_has_too_few_chunks(conn):
    """FAIL-SAFE: a thin parent fractures at a smaller k rather than erroring. Two chunks per
    cluster is the floor — below that a "cluster" is one point wearing a centroid."""
    _planted(conn, axes=(1, 2), per=2)
    p = _parent(conn)
    rep = sz.zoom(conn, p["sitting_id"], k=12, floor=0.70)
    assert rep["reason"] is None
    assert rep["k"] == rep["chunks"] // 2 < 12


# ── the fracture ────────────────────────────────────────────────────────────────
def test_the_fracture_is_reproducible(conn):
    """Same parent, same k, same seed → identical sub-region atom sets."""
    _planted(conn, axes=(1, 2, 3))
    p = _parent(conn)
    a = sz.zoom(conn, p["sitting_id"], k=3, floor=0.70, persist=False)
    b = sz.zoom(conn, p["sitting_id"], k=3, floor=0.70, persist=False)
    assert [s["label"] for s in a["sub"]] == [s["label"] for s in b["sub"]]
    assert [s["atoms"] for s in a["sub"]] == [s["atoms"] for s in b["sub"]]
    assert [s["region_atoms"] for s in a["sub"]] == [s["region_atoms"] for s in b["sub"]]


def test_a_two_topic_region_splits_into_two_near_disjoint_halves(conn):
    """The thing zoom is FOR. A parent holding two conversations comes back as two sub-sittings whose
    atom sets barely touch — and they survive the merge precisely because they are disjoint."""
    _planted(conn, axes=(1, 2))
    p = _parent(conn)
    rep = sz.zoom(conn, p["sitting_id"], k=2, floor=0.70)
    assert rep["kept"] == 2
    got = []
    for s in rep["sub"]:
        got.append({a["atom_id"] for a in
                    sst.get_sitting(conn, s["sitting_id"])["admissions"]})
    overlap = len(got[0] & got[1]) / len(got[0] | got[1])
    assert overlap < sz.ZOOM_MERGE_J, f"the two topics were not separated (J={overlap:.2f})"


def test_a_homogeneous_region_collapses_back_to_one(conn):
    """D2's SELF-CORRECTION, asserted rather than asserted-about. If a region is genuinely one
    conversation, its k centroids land on top of each other, their regions coincide, and the merge
    keeps one. That is the whole reason derived-k is allowed to over-ask."""
    _planted(conn, axes=(1,), per=14)
    p = _parent(conn)
    rep = sz.zoom(conn, p["sitting_id"], k=6, floor=0.70)
    assert rep["kept"] == 1
    assert len(rep["merged"]) == rep["k"] - 1
    assert all(m["jaccard"] >= sz.ZOOM_MERGE_J for m in rep["merged"])


def test_every_merge_is_logged_with_its_jaccard(conn):
    """D3 forbids a silent collapse: a dropped sub-region names what absorbed it and by how much."""
    _planted(conn, axes=(1,), per=14)
    p = _parent(conn)
    rep = sz.zoom(conn, p["sitting_id"], k=4, floor=0.70)
    assert rep["merged"]
    for m in rep["merged"]:
        assert m["into"] and m["label"] != m["into"] and 0.0 <= m["jaccard"] <= 1.0
    dropped = {s["label"] for s in rep["sub"] if not s["kept"]}
    assert dropped == {m["label"] for m in rep["merged"]}


def test_a_merged_sub_region_is_never_persisted(conn):
    """A duplicate row in the store is a PAID re-read: the reader's queue is every unread sitting."""
    _planted(conn, axes=(1,), per=14)
    p = _parent(conn)
    rep = sz.zoom(conn, p["sitting_id"], k=4, floor=0.70)
    for s in rep["sub"]:
        if not s["kept"]:
            assert sst.get_sitting(conn, s["sitting_id"]) is None


# ── D1's cost, reported ─────────────────────────────────────────────────────────
def test_fringe_parent_atoms_are_counted_not_hidden(conn):
    """THE PRICE OF RE-SEEDING. An atom that cleared the coarse floor from the parent can fail the
    finer one from every sub-centroid. It is not lost — it stays in the parent — but zoom is not
    coverage-preserving and the report has to say so."""
    _planted(conn, axes=(1, 2))
    # A fringe atom: admissible to the coarse parent, far from either cluster's centroid.
    fringe = np.zeros(DIM, dtype=np.float32)
    fringe[0], fringe[5] = 0.75, 0.66
    _atom(conn, "a:fringe", [_unit(fringe)])
    p = _parent(conn)
    assert "a:fringe" in {a["atom_id"] for a in p["admissions"]}
    rep = sz.zoom(conn, p["sitting_id"], k=2, floor=0.75)
    assert rep["parent_dropped"] == 1, "the fringe atom is the only one no sub-centroid reaches"
    assert rep["parent_uncovered"] >= rep["parent_dropped"]


def test_a_covered_parent_reports_no_drops(conn):
    """The counter is real, not a constant: a fracture that keeps everything says zero."""
    _planted(conn, axes=(1,), per=8)
    p = _parent(conn)
    rep = sz.zoom(conn, p["sitting_id"], k=2, floor=0.65)
    assert rep["parent_dropped"] == 0 and rep["parent_uncovered"] == 0


# ── D5: sub-sittings are ordinary sittings ──────────────────────────────────────
def test_sub_sittings_are_ordinary_sittings_with_a_parent_link(conn):
    """They persist as `seed_kind='vector'` rows labelled `<parent>/<i>`, so the ledger, the read
    queue, the daily cap and `--preview` all apply with no new code."""
    _planted(conn, axes=(1, 2))
    p = _parent(conn)
    rep = sz.zoom(conn, p["sitting_id"], k=2, floor=0.70)
    kids = _zoomed_from(conn, p["sitting_id"])
    assert len(kids) == rep["persisted"] == 2
    for row in kids:
        s = sst.get_sitting(conn, row["sitting_id"])
        assert s["seed_kind"] == "vector"
        assert s["seed_ref"].startswith(f"{p['seed_ref']}/")
        assert s["parent_sitting_id"] == p["sitting_id"]

    from pipeline.kb import sitting_reader as sr
    queued = {r["sitting_id"] for r in sr.unread_sittings(conn)}
    assert {row["sitting_id"] for row in kids} <= queued


def test_the_frontier_budget_applies_inside_a_fracture(conn, monkeypatch):
    """Zoom builds every sub-sitting through `sitting_builder.build_sitting`, so the per-region
    frontier budget (RULED 2026-08-27) caps a fracture child with no zoom-side copy of the rule to
    drift. A child is a FRESH region (`continues` is None), so its occupancy starts at zero and
    the cap binds inside the child build itself — this plants one more frontier atom than the
    budget allows in one cluster and reads the loser back out of the child's own skip record."""
    import json
    monkeypatch.setattr(sb, "FRONTIER_REGION_BUDGET", 1)
    _planted(conn, axes=(1, 2))
    rng = np.random.default_rng(11)
    for i in range(2):                       # cluster-1 members in every way but entry_mode
        g = rng.normal(size=DIM).astype(np.float32)
        g[0] = g[1] = 0.0
        g /= np.linalg.norm(g) + 1e-9
        v = np.zeros(DIM, dtype=np.float32)
        v[0], v[1] = SHARED, CLUSTER
        _atom(conn, f"f:{i}", [_unit(v + NOISE * g)], entry_mode="frontier")
    p = _parent(conn)
    sz.zoom(conn, p["sitting_id"], k=2, floor=0.70)

    kids = [sst.get_sitting(conn, r["sitting_id"]) for r in _zoomed_from(conn, p["sitting_id"])]
    hosts = [s for s in kids if {"f:0", "f:1"} & {a["atom_id"] for a in s["admissions"]}]
    assert hosts, "the fixture must land the frontier atoms inside a persisted fracture child"
    for s in hosts:
        got = {"f:0", "f:1"} & {a["atom_id"] for a in s["admissions"]}
        assert len(got) == 1, "a fracture child admitted more frontier atoms than the budget"
        dropped = {d["atom_id"] for d in json.loads(s["skipped"])
                   if d.get("reason") == "frontier_budget"}
        assert dropped == {"f:0", "f:1"} - got


def test_dormancy_is_scoped_per_sub_conversation(conn):
    """`generator_for` slugifies the label, so a sub-thread going quiet ages out its OWN queries and
    not its siblings'. That is the grain the per-region dormancy sweep needs."""
    from pipeline.kb import sitting_reader as sr
    assert sr.generator_for("mlx/3") == "sitting:mlx-3"
    assert sr.generator_for("mlx/3") != sr.generator_for("mlx/4")


def test_a_zoom_is_not_a_continuation(conn):
    """A sub-sitting is a fresh region at a finer floor, MEANT to overlap its siblings — not the next
    part of one region. Folding it into `continues` would put every sibling into its redundancy
    baseline and gut the material zoom exists to separate."""
    _planted(conn, axes=(1, 2))
    p = _parent(conn)
    rep = sz.zoom(conn, p["sitting_id"], k=2, floor=0.70)
    kid = next(s for s in rep["sub"] if s["persisted"])
    row = sst.get_sitting(conn, kid["sitting_id"])
    assert row["continues"] is None and row["prior_atoms"] == 0
    assert sre._part_index(conn, kid["sitting_id"]) == 1
    assert "continuation:" not in sre.render_sitting(conn, kid["sitting_id"])


def test_two_zooms_of_different_parents_in_one_second_do_not_collide(conn):
    """Both fractures label their first piece `<ref>/0`, at the same floor, in the same second. With
    only the label in the identity hash the second zoom would REPLACE the first."""
    import datetime as dt
    _planted(conn, axes=(1, 2))
    p1 = _parent(conn)
    p2 = _parent(conn, floor=0.45)            # same seed_ref, a different region
    assert p1["sitting_id"] != p2["sitting_id"]
    at = dt.datetime(2026, 8, 10, 12, 0, 0, tzinfo=dt.timezone.utc)
    a = sz.zoom(conn, p1["sitting_id"], k=2, floor=0.70, now=at)
    b = sz.zoom(conn, p2["sitting_id"], k=2, floor=0.70, now=at)
    assert not ({s["sitting_id"] for s in a["sub"]} & {s["sitting_id"] for s in b["sub"]})


def test_sprout_tier_sub_regions_are_reported_and_not_persisted(conn):
    """Below standalone there is not enough trajectory to narrate alone, so the sub-region is sprout
    mass — reported, and left for the sprouts digest instead of queued as its own paid read."""
    _planted(conn, axes=(1, 2), per=4)
    p = _parent(conn)
    rep = sz.zoom(conn, p["sitting_id"], k=2, floor=0.70)
    sprouts = [s for s in rep["sub"] if s["kept"] and s["tier"] != "standalone"]
    assert sprouts, "expected small sub-regions in this fixture"
    for s in sprouts:
        assert s["persisted"] is False
        assert sst.get_sitting(conn, s["sitting_id"]) is None


# ── zoom spends nothing ─────────────────────────────────────────────────────────
def test_zoom_makes_no_llm_call_and_writes_no_queries(conn, monkeypatch):
    """POSITIVE CONTROL. Zoom is pure geometry; if a model call ever crept in, this fails loudly
    rather than quietly billing a build that is documented as $0."""
    import pipeline.llm_client as llm

    def _boom(*a, **kw):
        raise AssertionError("zoom made an LLM call — it is pure geometry")

    monkeypatch.setattr(llm, "call", _boom)
    _planted(conn, axes=(1, 2))
    p = _parent(conn)
    rep = sz.zoom(conn, p["sitting_id"], k=2, floor=0.70)
    assert rep["persisted"] >= 1
    assert conn.execute("SELECT COUNT(*) FROM frontier_queries").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM frontier_reader_runs").fetchone()[0] == 0


def test_building_a_sub_sitting_is_not_reading_it(conn):
    """The ledger distinction survives the fracture: k free sub-sittings cover nothing until read."""
    _planted(conn, axes=(1, 2))
    p = _parent(conn)
    sz.zoom(conn, p["sitting_id"], k=2, floor=0.70)
    assert sst.coverage(conn)["read"] == 0


# ── the sweep ───────────────────────────────────────────────────────────────────
def test_the_sweep_persists_nothing(conn):
    """It is a MEASUREMENT. A measurement that writes is a trap: the sweep would queue up to
    2+3+4+6+8+12 sub-sittings for a question that was only ever "how many parts are there"."""
    _planted(conn, axes=(1, 2, 3))
    p = _parent(conn)
    before = _tables(conn)
    rep = sz.sweep_k(conn, p["sitting_id"], ks=[2, 3, 4, 6, 8], floor=0.70)
    assert rep["rows"]
    assert _tables(conn) == before


def test_the_sweep_scores_every_k_in_one_relevance_pass(conn, monkeypatch):
    """`_relevance` already takes several seed vectors, so 5 k values stack into one matrix. Without
    this the sweep is 23 sequential corpus scans and stops feeling free — and a measurement nobody
    runs decides nothing."""
    _planted(conn, axes=(1, 2, 3))
    p = _parent(conn)
    calls = []
    real = sv._relevance

    def counted(c, seeds, **kw):
        calls.append(seeds)
        return real(c, seeds, **kw)

    monkeypatch.setattr(sv, "_relevance", counted)
    sz.sweep_k(conn, p["sitting_id"], ks=[2, 3, 4, 6, 8], floor=0.70)
    assert len(calls) == 1, f"{len(calls)} corpus scans for 5 k values"
    assert len(calls[0]) == 2 + 3 + 4 + 6 + 8


def test_distinct_after_merge_plateaus_at_the_planted_part_count(conn):
    """THE CLAIM D2b RESTS ON, asserted on data whose true answer is known. Three planted clusters:
    raising k past 3 must keep producing 3 distinct sub-regions, because the extra centroids land
    inside clusters that are already covered and merge away."""
    _planted(conn, axes=(1, 2, 3), per=8)
    p = _parent(conn)
    rep = sz.sweep_k(conn, p["sitting_id"], ks=[2, 3, 4, 6, 8], floor=0.70)
    got = {r["k"]: r["distinct"] for r in rep["rows"]}
    assert got[2] < 3, "asking for too FEW pieces must under-count, or the plateau means nothing"
    assert got[3] == 3
    assert got[4] == got[6] == got[8] == 3, f"distinct did not plateau at 3: {got}"
    assert "plateaus at 3 from k=3" in sz.render_sweep(rep)


def test_a_sweep_that_never_plateaus_says_so_instead_of_naming_a_number(conn):
    """THE REPORTING TRAP, caught on the real corpus 2026-08-10: reading the largest `distinct` as
    the answer finds a plateau wherever the sweep happened to stop. On the real mlx region distinct
    grew 6 -> 37 across k = 6 -> 60 and never flattened, and an unconditional line would have
    announced "~37 sub-conversations" purely because 60 was the last column."""
    rising = {"rows": [{"k": 2, "distinct": 2, "merged": 0}, {"k": 4, "distinct": 4, "merged": 0},
                       {"k": 8, "distinct": 7, "merged": 1}]}
    assert "NO PLATEAU" in sz._plateau_note(rising["rows"])
    flat = [{"k": 2, "distinct": 2, "merged": 0}, {"k": 4, "distinct": 3, "merged": 1},
            {"k": 8, "distinct": 3, "merged": 5}]
    assert "plateaus at 3 from k=4" in sz._plateau_note(flat)


def test_the_sweep_reports_the_drop_count_rising_with_k(conn):
    """`parent dropped` is D1's cost becoming visible: a finer split leaves more fringe atoms outside
    every sub-centroid. It must be monotone in the sense that it can never DECREASE below what the
    coarsest split already dropped by chance — here, simply that the column is real and bounded."""
    _planted(conn, axes=(1, 2, 3))
    p = _parent(conn)
    rep = sz.sweep_k(conn, p["sitting_id"], ks=[2, 4], floor=0.75)
    for r in rep["rows"]:
        assert 0 <= r["parent_dropped"] <= rep["parent_atoms"]
        assert r["tokens"] >= 0 and r["max_size"] >= r["med_size"]


def test_the_sweep_reprojects_only_the_long_atoms(conn, monkeypatch):
    """THE HOIST THE SWEEP'S COST RESTS ON. Rendered size is seed-dependent only above
    `LONG_ATOM_TOKENS`; below it an atom renders in full whatever centroid scored it. So the sweep
    fetches chunk spans once, bills every short atom once from `whole_tokens`, and re-projects only
    the long atoms — turning (k values x regions x atoms) into (k values x regions x LONG atoms),
    and decoding chunk vectors for the long atoms alone.

    Asserted on WHICH ids reach `projection`, because the arithmetic is identical either way: a
    regression here costs nothing but time, so nothing else would ever catch it."""
    _planted(conn, axes=(1, 2))
    long_id = _long_atom(conn, axis=1)
    p = _parent(conn)
    seen, real = [], sre.projection

    def counted(c, ids, **kw):
        seen.append(list(ids))
        return real(c, ids, **kw)

    monkeypatch.setattr(sre, "projection", counted)
    rep = sz.sweep_k(conn, p["sitting_id"], ks=[2, 3], floor=0.70)
    assert rep["rows"] and seen, "the sweep must still project the long material"
    assert {a for call in seen for a in call} == {long_id}
    assert all(r["tokens"] > 0 for r in rep["rows"])


def test_the_sweep_deduplicates_repeated_k_values(conn):
    _planted(conn, axes=(1, 2))
    p = _parent(conn)
    rep = sz.sweep_k(conn, p["sitting_id"], ks=[3, 3, 3], floor=0.70)
    assert len(rep["rows"]) == 1


# ── degradation ─────────────────────────────────────────────────────────────────
def test_zooming_an_unembedded_parent_returns_a_reason_not_a_crash(conn):
    """FAIL-SAFE, same rule as the builder: a store whose embed pass has not run degrades to an
    empty fracture that says which failure this is."""
    _planted(conn, axes=(1, 2))
    p = _parent(conn)
    conn.execute("UPDATE chunks SET vector = NULL")
    conn.commit()
    rep = sz.zoom(conn, p["sitting_id"], floor=0.70)
    assert rep["reason"] and rep["sub"] == [] and rep["k"] == 0
    assert sz.sweep_k(conn, p["sitting_id"], ks=[2, 3])["reason"]


def test_zooming_a_missing_sitting_raises(conn):
    with pytest.raises(KeyError):
        sz.zoom(conn, "deadbeefdeadbeef")


def test_an_older_store_gains_the_parent_column_on_open(kb_home):
    """THE MIGRATION PATH a fresh-DB test cannot reach: on a store that already has a `sittings`
    table, `CREATE TABLE IF NOT EXISTS` is a no-op and only `_ensure_column` can add the column —
    and the index on it must be created after, or the whole schema fails to open."""
    db = kb_home / "old.db"
    raw = sqlite3.connect(str(db))
    raw.executescript(
        "CREATE TABLE sittings (sitting_id TEXT PRIMARY KEY, built_at TEXT NOT NULL, "
        " seed_kind TEXT NOT NULL, seed_ref TEXT, seed_atom_ids TEXT, floor REAL NOT NULL, "
        " calibrated_floor REAL, lam REAL NOT NULL, ceiling REAL NOT NULL, "
        " budget_tokens INTEGER NOT NULL, region_atoms INTEGER, region_tokens INTEGER, "
        " atoms INTEGER, tokens INTEGER, stop TEXT NOT NULL, "
        " skipped_dupes INTEGER NOT NULL DEFAULT 0, read_at TEXT, read_status TEXT);")
    raw.commit()
    raw.close()

    c = schema.connect(db)                                # must not raise
    cols = {r[1] for r in c.execute("PRAGMA table_info(sittings)")}
    assert "parent_sitting_id" in cols and "skipped" in cols
    assert "lam" not in cols                              # subtractive, 2026-08-25
    c.close()


# ── CLI ─────────────────────────────────────────────────────────────────────────
def test_cli_zoom_prints_the_report_and_persists(conn, capsys):
    _planted(conn, axes=(1, 2))
    p = _parent(conn)
    conn.commit()
    assert sb.main(["zoom", "--sitting", p["sitting_id"], "--k", "2", "--floor", "0.70"]) == 0
    out = capsys.readouterr().out
    assert "parent:" in out and "k = 2" in out
    assert "parent atoms fell below every sub-centroid" in out
    assert _zoomed_from(conn, p["sitting_id"])


def test_cli_sweep_prints_the_table_and_writes_nothing(conn, capsys):
    _planted(conn, axes=(1, 2, 3))
    p = _parent(conn)
    conn.commit()
    before = _tables(conn)
    assert sb.main(["zoom", "--sitting", p["sitting_id"], "--sweep", "2,3,4",
                    "--floor", "0.70"]) == 0
    out = capsys.readouterr().out
    assert "distinct" in out and "persists NOTHING" in out
    assert _tables(conn) == before


def test_cli_zoom_dry_run_records_nothing(conn, capsys):
    _planted(conn, axes=(1, 2))
    p = _parent(conn)
    conn.commit()
    before = _tables(conn)
    assert sb.main(["zoom", "--sitting", p["sitting_id"], "--k", "2", "--floor", "0.70",
                    "--dry-run"]) == 0
    assert _tables(conn) == before


def test_cli_zoom_on_a_missing_sitting_exits_one(conn, capsys):
    assert sb.main(["zoom", "--sitting", "deadbeefdeadbeef"]) == 1
    assert "not found" in capsys.readouterr().err
