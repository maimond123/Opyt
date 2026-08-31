"""The seeded context loop, proven offline against hand-placed vectors.

These lock MECHANICS, not taste. Whether a sitting reads well is a question the corpus answers;
what has to hold no matter the corpus is narrower and every item is a failure that would be
invisible in production:

  • THE ANCHOR HOLDS. Relevance is measured against the seed forever. If it ever re-anchored on the
    growing set, a sitting would walk A -> B -> C into a different topic and still look healthy.
  • THE FLOOR IS MEMBERSHIP. Nothing below it is admitted, at any redundancy, for any lambda.
  • NEAR-DUPLICATES ARE SKIPPED AND COUNTED. A silent drop reports the same shape as a full read.
  • BUILDING IS NOT READING. A built-but-unread sitting must not count as coverage, or the ledger
    launders free work into a claim that the corpus was read.
  • MEMBERSHIP IS AN ALLOW-LIST. A mode nobody listed stays out no matter how close it sits to the
    seed. Positive control for the discipline, so the fixture mode is deliberately FICTIONAL — a
    real one would read as documentation of a live mode's status.
  • THE UNION IS WELDED. Frontier atoms JOIN regions (RULED 2026-08-24) — and the membership scan
    and the token scan must widen with the SAME tuple, or an admitted atom bills zero and the
    budget silently stops binding.
  • THE BUDGET BILLS WHAT THE READER SEES. A long atom renders as head + floor-matched sections, so
    billing its stored size would cut a part at a boundary the render never had. Billing and
    rendering are ONE function; the anti-drift property is the test.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from pipeline.kb import schema
from pipeline.kb import sitting_builder as sb
from pipeline.kb import sitting_render as sre
from pipeline.kb import sitting_store as sst
from pipeline.kb import sitting_vectors as sv

DIM = 8

# A fixed build moment. `_sitting_id` folds `built_at` in at SECOND resolution, so two builds in one
# test would otherwise collide or not, depending on how fast the machine is.
_T0 = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    from pipeline.kb.embed import ensure_kb_meta
    ensure_kb_meta(c, "fake", DIM, "local", "", storage_dtype="float32")
    yield c
    c.close()


def _unit(*weights) -> np.ndarray:
    """A float32 unit vector in DIM dimensions from the leading weights given."""
    v = np.zeros(DIM, dtype=np.float32)
    v[:len(weights)] = weights
    return v / (np.linalg.norm(v) + 1e-9)


ANCHOR = _unit(1)


def _at_cos(c: float, axis: int = 1) -> np.ndarray:
    """A unit vector at EXACTLY cosine `c` from `ANCHOR`, spread on `axis`.

    Cosines are stated rather than eyeballed because two thresholds sit close together here: the
    floor (0.68) admits, and the ceiling (0.95) rejects as a near-duplicate. Hand-picked weights
    landed candidates on the wrong side of the ceiling and the tests read as loop bugs. Two vectors
    built on DIFFERENT axes at cosine `c` are `c**2` from each other, so distinct axes keep mutual
    redundancy predictable too.
    """
    v = np.zeros(DIM, dtype=np.float32)
    v[0], v[axis] = c, float(np.sqrt(max(0.0, 1.0 - c * c)))
    return v / (np.linalg.norm(v) + 1e-9)


def _atom(conn, atom_id, vecs, *, when="2026-08-01", who="x:user:1",
          entry_mode="user-saved", chars=800):
    """One atom with `vecs` chunk vectors. `chars` sets each chunk's length, which drives the token
    estimate and therefore where the budget binds."""
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode) "
                 "VALUES (?,?,?,?,?)", (atom_id, "x", who, when, entry_mode))
    pos = 0
    for seq, v in enumerate(vecs):
        text = f"{atom_id} chunk {seq} " + ("word " * max(1, chars // 5))
        conn.execute(
            "INSERT INTO chunks (atom_id, seq, char_start, char_end, text, vector) "
            "VALUES (?,?,?,?,?,?)",
            (atom_id, seq, pos, pos + len(text), text,
             np.asarray(v, dtype=np.float32).tobytes()))
        pos += len(text)
    conn.commit()


# ── membership ──────────────────────────────────────────────────────────────────
def test_floor_decides_membership(conn):
    """An atom below the floor is not in the region, however little it duplicates."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:near", [_at_cos(0.85)])             # clears the floor, under the ceiling
    _atom(conn, "a:far", [_at_cos(0.20, axis=2)])      # noise
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    rec = sb.build_sitting(conn, seed, floor=0.68)
    got = {a["atom_id"] for a in rec["admissions"]}
    assert "a:near" in got
    assert "a:far" not in got
    assert rec["region_atoms"] == 1                     # the region excludes the seed itself


def test_max_pooled_over_chunks_floats_a_long_atom(conn):
    """One strongly on-topic passage admits its atom. A mean over chunks would sink it."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:mixed", [_at_cos(0.05), _at_cos(0.05), _at_cos(0.88)])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert "a:mixed" in {a["atom_id"] for a in rec["admissions"]}


def test_an_unlisted_entry_mode_is_unreachable(conn):
    """POSITIVE CONTROL for the allow-list discipline: a mode nobody put in `REGION_VISIBLE` stays
    out even sitting right on top of the seed. That is what makes a newly-invented mode excluded by
    DEFAULT rather than by someone remembering to exclude it."""
    _atom(conn, "a:seed", [ANCHOR])
    # Fictional ON PURPOSE. Every real mode either is in the allow-list or is `frontier`, which
    # the union admits — so naming one here would test that mode's status, not the discipline.
    _atom(conn, "a:unlisted", [_at_cos(0.85)], entry_mode="not-a-real-mode")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert "a:unlisted" not in {a["atom_id"] for a in rec["admissions"]}
    assert rec["region_atoms"] == 0


# ── the union ───────────────────────────────────────────────────────────────────
def test_a_frontier_atom_joins_the_region_and_bills_its_tokens(conn):
    """THE UNION, AND ITS ONE SILENT-FAILURE MODE (RULED 2026-08-24).

    Widening membership without widening `atom_tokens` is the hazard: the atom is admitted, then
    `toks.get(a, 0)` bills it ZERO, so `used` never reaches the budget, `stop` reports `saturation`
    on a region that was really cut, and the over-budget document gets chopped by the reader's
    `MAX_INPUT_CHARS` — from the END, which is the newest material. Nothing raises anywhere along
    that path, which is why this asserts BOTH halves rather than membership alone.
    """
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:found", [_at_cos(0.85)], entry_mode="frontier")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)

    found = [a for a in rec["admissions"] if a["atom_id"] == "a:found"]
    assert found, "a frontier atom must reach the region — that is what finding it was for"
    assert found[0]["tokens"] > 0, "admitted but billed zero: the budget has stopped binding"
    assert rec["tokens"] >= found[0]["tokens"]


def test_the_seed_is_never_anchored_on_a_frontier_atom(conn, monkeypatch):
    """The union widened MEMBERSHIP, never the anchor. `_vector_seed_atoms` picks the atoms whose
    centroid the whole region is measured against forever — seeding on machine finds would let the
    frontier decide what a typed phrase means, which is the anti-narrowing loop one level up from
    where the guard watches for it."""
    _atom(conn, "a:human", [_at_cos(0.80)])
    _atom(conn, "a:found", [ANCHOR], entry_mode="frontier")

    class _E:
        model, dim, provider, query_instruction = "fake", DIM, "local", ""
        batch_size = 8
        def embed(self, texts, role=None):
            return [ANCHOR]
    seed = sb.resolve_seed(conn, query="anything", embedder=_E())
    assert seed["atom_ids"] == ["a:human"], "the closest atom was the frontier one and it lost"


# ── the frontier region budget (RULED 2026-08-27) ───────────────────────────────
def test_the_frontier_budget_caps_a_build_and_top_rel_wins_the_slots(conn, monkeypatch):
    """The cap runs on the POOL, top-rel first — not first-come. The dates are adversarial: the
    weakest frontier candidate is the OLDEST, so a cap applied in chronological admission order
    would keep it and drop the strongest. The losers are RECORDED with the reason, in the record
    and in the store — a budget drop must not look like a floor rejection, and must not inflate
    the near-duplicate count."""
    monkeypatch.setattr(sb, "FRONTIER_REGION_BUDGET", 2)
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "f:hi", [_at_cos(0.90, axis=1)], entry_mode="frontier", when="2026-03-01")
    _atom(conn, "f:mid", [_at_cos(0.85, axis=2)], entry_mode="frontier", when="2026-02-01")
    _atom(conn, "f:lo", [_at_cos(0.75, axis=3)], entry_mode="frontier", when="2026-01-01")
    _atom(conn, "a:human", [_at_cos(0.80, axis=4)])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)

    got = {a["atom_id"] for a in rec["admissions"]}
    assert {"f:hi", "f:mid", "a:human"} <= got
    assert "f:lo" not in got
    drops = [s for s in rec["skipped"] if s.get("reason") == "frontier_budget"]
    assert [d["atom_id"] for d in drops] == ["f:lo"]
    assert drops[0]["rel"] == pytest.approx(0.75, abs=1e-3)
    assert rec["skipped_dupes"] == 0, "a budget drop is not a near-duplicate"
    row = conn.execute("SELECT skipped FROM sittings WHERE sitting_id = ?",
                       (rec["sitting_id"],)).fetchone()
    assert [d["atom_id"] for d in json.loads(row["skipped"])
            if d.get("reason") == "frontier_budget"] == ["f:lo"]


def test_the_budget_spans_the_chain_not_each_part(conn, monkeypatch):
    """Occupancy is counted over PRIOR chain members, so part 2 gets budget − spent slots, not a
    fresh budget. A per-part cap was rejected as a rate limiter, not a cap (decision 4): frontier
    arrival is machine-paced, so a cap that resets every part fills every night forever."""
    monkeypatch.setattr(sb, "FRONTIER_REGION_BUDGET", 3)
    _atom(conn, "a:seed", [ANCHOR], when="2026-01-01")
    _atom(conn, "f:a", [_at_cos(0.90, axis=1)], entry_mode="frontier", when="2026-02-01")
    _atom(conn, "f:b", [_at_cos(0.88, axis=2)], entry_mode="frontier", when="2026-03-01")
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    p1 = sb.build_sitting(conn, seed, floor=0.68)
    assert {"f:a", "f:b"} <= {a["atom_id"] for a in p1["admissions"]}

    _atom(conn, "f:c", [_at_cos(0.92, axis=3)], entry_mode="frontier", when="2026-04-01")
    _atom(conn, "f:d", [_at_cos(0.85, axis=4)], entry_mode="frontier", when="2026-05-01")
    _atom(conn, "a:late", [_at_cos(0.80, axis=5)], when="2026-04-01")
    p2 = sb.build_sitting(conn, seed, floor=0.68, continues=p1["sitting_id"])
    got2 = {a["atom_id"] for a in p2["admissions"] if not a["is_seed"]}
    assert "a:late" in got2, "the budget must never touch a human-attested member"
    assert "f:c" in got2 and "f:d" not in got2      # 3 − 2 spent = 1 slot; top rel takes it
    assert [s["atom_id"] for s in p2["skipped"]
            if s.get("reason") == "frontier_budget"] == ["f:d"]


def test_promoting_a_prior_frontier_member_frees_its_slot(conn, monkeypatch):
    """Decision 5: promotion frees a slot as a CONSEQUENCE, not a mechanism — occupancy is derived
    from CURRENT `entry_mode` at build time, never decremented. The contrast pair is the same
    continuation twice: with the budget spent the newcomer is dropped; after the incumbent's
    promotion the identical build admits it, and no code in the builder knew promotion happened."""
    from pipeline.kb import ingest_common
    monkeypatch.setattr(sb, "FRONTIER_REGION_BUDGET", 1)
    _atom(conn, "a:seed", [ANCHOR], when="2026-01-01")
    _atom(conn, "f:first", [_at_cos(0.90, axis=1)], entry_mode="frontier", when="2026-02-01")
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    p1 = sb.build_sitting(conn, seed, floor=0.68)
    assert "f:first" in {a["atom_id"] for a in p1["admissions"]}

    _atom(conn, "f:second", [_at_cos(0.88, axis=2)], entry_mode="frontier", when="2026-03-01")
    p2 = sb.build_sitting(conn, seed, floor=0.68, continues=p1["sitting_id"])
    assert "f:second" not in {a["atom_id"] for a in p2["admissions"]}
    assert [s["atom_id"] for s in p2["skipped"]
            if s.get("reason") == "frontier_budget"] == ["f:second"]

    ingest_common.promote_atom(conn, "f:first", "user-saved")
    p3 = sb.build_sitting(conn, seed, floor=0.68, continues=p2["sitting_id"])
    assert "f:second" in {a["atom_id"] for a in p3["admissions"]}


def test_a_region_with_no_frontier_candidates_is_untouched_by_the_budget(conn, monkeypatch):
    """Even a ZERO budget touches nothing human-attested — the cap keys on `entry_mode`, never on
    scores, so a human-only region cannot lose a member to it."""
    monkeypatch.setattr(sb, "FRONTIER_REGION_BUDGET", 0)
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:1", [_at_cos(0.85, axis=1)])
    _atom(conn, "a:2", [_at_cos(0.80, axis=2)])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert {"a:1", "a:2"} <= {a["atom_id"] for a in rec["admissions"]}
    assert rec["skipped"] == []


# ── the anchor ──────────────────────────────────────────────────────────────────
def test_relevance_is_anchored_to_the_seed_not_the_growing_set(conn):
    """THE DRIFT GUARD. `bridge` is close to the seed; `drifted` is close to `bridge` but far from
    the seed. A loop that re-anchored on what it just admitted would pull `drifted` in on the next
    turn. Anchored on the seed, it can never qualify."""
    _atom(conn, "a:seed", [_unit(1, 0, 0)])
    _atom(conn, "a:bridge", [_unit(0.75, 0.66, 0)])          # ~0.75 to seed
    _atom(conn, "a:drifted", [_unit(0.10, 0.75, 0.65)])      # ~0.72 to bridge, ~0.10 to seed
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    got = {a["atom_id"] for a in rec["admissions"]}
    assert "a:bridge" in got
    assert "a:drifted" not in got, "relevance re-anchored on the admitted set — the loop drifts"


# ── ordering ────────────────────────────────────────────────────────────────────
def test_admission_order_is_chronological_not_by_novelty(conn):
    """MMR NO LONGER ORDERS ANYTHING (RULED 2026-08-24). It only ever ordered the cut when the
    budget bound, and chronology took that job — for a reason MMR could not answer: MMR is
    ANTI-CLUSTERING by design, so repurposed as a partitioner it maximally SEVERS threads. A
    rebuttal scores redundant against the claim it answers, which put originals in part 1 and
    responses in part 2 systematically.

    `a:dupe` is nearer the seed and would have been DEFERRED by novelty; it is older, so it is read
    first. That inversion is the assertion — under any novelty-ordered loop the two swap.

    The old form of this test passed `lam=0.5` and `lam=0.95` and asserted the order did not move.
    That assertion is now unstatable, which is the point: `lam` is gone, so nothing is left to
    prove inert.
    """
    _atom(conn, "a:seed", [ANCHOR], when="2026-01-01")
    _atom(conn, "a:dupe", [_at_cos(0.93)], when="2026-02-01")
    _atom(conn, "a:fresh", [_at_cos(0.75, axis=2)], when="2026-03-01")
    order = [a["atom_id"] for a in sb.build_sitting(
        conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)["admissions"]
        if not a["is_seed"]]
    assert order == ["a:dupe", "a:fresh"]


def test_near_duplicates_are_skipped_and_counted(conn):
    """Above the ceiling an atom is not deferred, it is dropped — and RECORDED. A skipped-silently
    reader looks identical to one that read everything."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:repost", [_at_cos(0.99)])           # above the ceiling
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68,
                           ceiling=0.95)
    assert rec["skipped_dupes"] == 1
    assert rec["skipped"][0]["atom_id"] == "a:repost"
    assert "a:repost" not in {a["atom_id"] for a in rec["admissions"]}
    # And the list SURVIVES the write, not only the count. The ruling that kept the near-duplicate
    # skip (2026-08-24) is revisitable only on an observed skip that is a response rather than a
    # crosspost — which is a one-query audit against this column and unanswerable without it.
    row = conn.execute("SELECT skipped FROM sittings WHERE sitting_id = ?",
                       (rec["sitting_id"],)).fetchone()
    assert json.loads(row["skipped"]) == [{"atom_id": "a:repost", "red": rec["skipped"][0]["red"]}]


# ── stop modes ──────────────────────────────────────────────────────────────────
def test_saturation_when_the_pool_runs_dry(conn):
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:1", [_at_cos(0.90)])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert rec["stop"] == "saturation"


def test_budget_stop_leaves_the_remainder_in_the_region(conn):
    """A budget stop is not a failure — the unadmitted remainder is the next sitting's seed, so the
    region size must stay visible next to what was actually taken."""
    _atom(conn, "a:seed", [ANCHOR], chars=400)
    # Distinct axes, so they are 0.81 from EACH OTHER — all admissible, none a near-duplicate.
    for i in range(6):
        _atom(conn, f"a:{i}", [_at_cos(0.90, axis=1 + i)], chars=4000)
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]),
                           floor=0.68, budget_tokens=2_000)
    assert rec["stop"] == "budget"
    assert rec["atoms"] < rec["region_atoms"] + 1
    assert rec["region_tokens"] > rec["tokens"]


def test_tokens_exclude_the_chunk_overlap(conn):
    """The estimate spans the snapshot, so a multi-chunk atom is not inflated by the 200 characters
    adjacent chunks deliberately share."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:long", [_at_cos(0.85), _at_cos(0.85)], chars=1000)
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    billed = {a["atom_id"]: a["tokens"] for a in rec["admissions"]}
    span = conn.execute("SELECT MAX(char_end) - MIN(char_start) FROM chunks WHERE atom_id='a:long'"
                        ).fetchone()[0]
    assert billed["a:long"] == span // 4


# ── the tiered render (RULED 2026-08-24) ────────────────────────────────────────
def _long_atom(conn, atom_id, *, matching: int, off_topic: int, chars=3000):
    """One long atom: chunk 0 is the head, then `matching` chunks on the seed axis and `off_topic`
    chunks well away from it. Stands in for a full-text paper — abstract, then sections, most of
    which the region has no interest in."""
    # The head is on-topic but NOT the anchor itself: an atom carrying a chunk identical to the
    # seed is a near-duplicate and the ceiling drops it before any of this is reached.
    vecs = ([_at_cos(0.75, axis=2)] + [_at_cos(0.90, axis=1)] * matching
            + [_unit(0, 0, 1)] * off_topic)
    _atom(conn, atom_id, vecs, chars=chars)


def _section(doc: str, atom_id: str) -> str:
    """The rendered block for one atom — header to the next header."""
    after = doc.split(f"({atom_id})", 1)[1]
    return after.split("\n### ", 1)[0]


def test_a_long_atom_is_billed_and_rendered_at_its_projection(conn):
    """THE ANTI-DRIFT WELD, asserted as one identity: the tokens the budget charged and the tokens
    the document actually contains are the same number, because one function produced both.

    Split them and the failure is invisible and permanent — the part is cut at a boundary the
    render never had, and a closed part is read, claims-extracted and lens-cached, never
    repartitioned.
    """
    _atom(conn, "a:seed", [ANCHOR], chars=400)
    _long_atom(conn, "a:paper", matching=1, off_topic=8)
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)

    billed = {a["atom_id"]: a["tokens"] for a in rec["admissions"]}["a:paper"]
    assert billed <= sre.LONG_ATOM_TOKENS, "a projection may not cost more than the bar that cut it"

    body = _section(sre.render_sitting(conn, rec["sitting_id"]), "a:paper")
    assert "not shown" in body, "the pointer to the full text is the whole recovery hatch"
    assert len(body) // 4 <= billed + 200, "the document is bigger than the budget was told"


def test_a_short_atom_is_untouched_by_the_projection(conn):
    """Under the bar, nothing changes — no pointer line, no missing sections. The projection is for
    the accident of PDF-mirror luck, not a general compression pass."""
    _atom(conn, "a:seed", [ANCHOR], chars=400)
    _atom(conn, "a:post", [_at_cos(0.85)], chars=800)
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    doc = sre.render_sitting(conn, rec["sitting_id"])
    assert "not shown" not in doc


def test_prefetched_spans_bill_exactly_what_a_fresh_fetch_bills(conn):
    """`projection` accepts the chunk spans a caller already fetched, because the span query is
    SEED-INDEPENDENT and `sitting_zoom.sweep_k` scores one atom set against every centroid of every
    k. Same input, same number: a `spans=` that shifted a single token would make the sweep's
    forecast disagree with the build it forecasts, and k would be chosen on a lie."""
    _atom(conn, "a:seed", [ANCHOR], chars=400)
    _long_atom(conn, "a:paper", matching=1, off_topic=8)
    ids = ["a:seed", "a:paper"]
    vecs = sv._atom_chunk_vectors(conn, ids)
    kw = {"vectors": vecs, "seed_vector": ANCHOR, "floor": 0.68}

    fresh = sre.projection(conn, ids, **kw)
    handed = sre.projection(conn, ids, spans=sre._spans(conn, ids), **kw)
    assert fresh == handed
    assert fresh["a:paper"]["seqs"] is not None, "the fixture must exercise the long-atom branch"


def test_a_short_atom_costs_the_same_against_any_seed(conn):
    """Why `sweep_k` may bill short atoms ONCE instead of re-projecting them per centroid: under
    `LONG_ATOM_TOKENS` an atom renders in full, so its cost cannot depend on what it was scored
    against. Break this and the sweep's fast path silently under- or over-bills every short atom in
    every candidate region."""
    _atom(conn, "a:seed", [ANCHOR], chars=400)
    _atom(conn, "a:post", [_at_cos(0.85)], chars=800)
    _long_atom(conn, "a:paper", matching=1, off_topic=8)
    ids = ["a:seed", "a:post", "a:paper"]
    vecs = sv._atom_chunk_vectors(conn, ids)
    whole = sre.whole_tokens(sre._spans(conn, ids))
    short = [a for a in ids if whole[a] <= sre.LONG_ATOM_TOKENS]
    assert short and whole["a:paper"] > sre.LONG_ATOM_TOKENS

    for seed in (ANCHOR, _unit(0, 0, 1)):
        proj = sre.projection(conn, ids, vectors=vecs, seed_vector=seed, floor=0.68)
        assert {a: proj[a]["tokens"] for a in short} == {a: whole[a] for a in short}


def test_membership_and_the_ceiling_still_see_the_whole_atom(conn):
    """The render is a PROJECTION, not a second copy. An atom that qualifies only through one
    buried section still qualifies — recall-generous membership is untouched, and cutting at
    membership is what the ruling explicitly refused."""
    _atom(conn, "a:seed", [ANCHOR], chars=400)
    # Head and every other section are far from the seed; ONE buried section is on topic.
    _atom(conn, "a:paper", [_unit(0, 0, 1)] * 4 + [_at_cos(0.90)] + [_unit(0, 0, 1)] * 4,
          chars=3000)
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert "a:paper" in {a["atom_id"] for a in rec["admissions"]}


def test_the_projection_shrinks_what_the_budget_charges(conn):
    """The number that makes part math work: the same corpus fits more atoms per part once the
    budget stops paying for methodology sections nobody in this region asked about."""
    _atom(conn, "a:seed", [ANCHOR], chars=400)
    for i in range(3):
        _long_atom(conn, f"a:paper{i}", matching=1, off_topic=8)
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)

    stored = conn.execute(
        "SELECT SUM(span) FROM (SELECT MAX(char_end) - MIN(char_start) AS span FROM chunks "
        " WHERE atom_id LIKE 'a:paper%' GROUP BY atom_id)").fetchone()[0] // 4
    assert rec["tokens"] < stored, "billed the stored size — part boundaries will be wrong"


# ── continuations ───────────────────────────────────────────────────────────────
def _budget_region(conn, n=6, *, cos=0.90, chars=4000):
    """A seed plus `n` distinct admissible atoms, each big enough that a small budget stops early.

    Dated one month apart, ascending, because the budget cut is chronological now: `a:0` is the
    oldest and belongs to part 1 by construction."""
    _atom(conn, "a:seed", [ANCHOR], chars=400, when="2026-01-01")
    for i in range(n):
        _atom(conn, f"a:{i}", [_at_cos(cos, axis=1 + i)], chars=chars,
              when=f"2026-{i + 2:02d}-01")


def _dates(conn, rec) -> list:
    """The `when_ts` of a part's non-seed atoms, in admission order."""
    ids = [a["atom_id"] for a in rec["admissions"] if not a["is_seed"]]
    got = {r["atom_id"]: r["when_ts"] for r in conn.execute(
        "SELECT atom_id, when_ts FROM atoms")}
    return [got[a] for a in ids]


def test_continuation_does_not_reread_a_prior_part(conn):
    """The half that already worked: an atom read in part 1 leaves part 2's pool."""
    _budget_region(conn)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    p1 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000)
    assert p1["stop"] == "budget"
    p2 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000,
                          continues=p1["sitting_id"])
    read1 = {a["atom_id"] for a in p1["admissions"] if not a["is_seed"]}
    read2 = {a["atom_id"] for a in p2["admissions"] if not a["is_seed"]}
    assert read1 and read2
    assert not (read1 & read2), "a continuation re-read an atom the previous part already covered"
    assert p2["prior_atoms"] == len(read1)


def test_continuation_carries_prior_atoms_into_the_redundancy_baseline(conn):
    """THE BUG THIS FEATURE EXISTS TO FIX, in its measured shape.

    `a:orig` and `a:copy` are the same essay under two ids — identical vectors, so they are at
    redundancy 1.00 to each other, well above the 0.95 ceiling. Part 1's budget admits exactly one of
    them. Part 2 must SKIP the other, which it can only do if part 1's atoms are in its redundancy
    baseline. Excluding prior atoms from the pool cannot catch this: the copy is a DIFFERENT
    atom_id, so pool exclusion never sees it.

    Without the fix the copy scores red = 0.88 (its cosine to the seed), sails under the ceiling, and
    is admitted — which is exactly what happened to Soren Larson's essay across the two RL parts.
    """
    _atom(conn, "a:seed", [ANCHOR], chars=400, when="2026-01-01")
    _atom(conn, "a:orig", [_at_cos(0.88, axis=1)], chars=4000, when="2026-02-01")
    _atom(conn, "a:copy", [_at_cos(0.88, axis=1)], chars=4000, when="2026-03-01")  # same vector
    _atom(conn, "a:other", [_at_cos(0.75, axis=2)], chars=4000, when="2026-04-01")
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    # Sized to admit ONE atom. Which one is decided by DATE now, not by an MMR tie-break on
    # atom_id: the original was published first, so part 1 takes it and part 2 meets the copy.
    p1 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=1_000)
    admitted1 = [a["atom_id"] for a in p1["admissions"] if not a["is_seed"]]
    assert admitted1 == ["a:orig"]

    p2 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=1_000, continues=p1["sitting_id"])
    assert "a:copy" in {s["atom_id"] for s in p2["skipped"]}, \
        "the cross-source duplicate was admitted again — redundancy is anchored on the seeds only"
    assert "a:copy" not in {a["atom_id"] for a in p2["admissions"]}
    assert p2["skipped_dupes"] >= 1


def test_a_part_is_a_contiguous_stretch_of_the_timeline(conn):
    """THE PARTITIONER (RULED 2026-08-24). Part 1 is the oldest stretch, part 2 picks up where it
    stopped, and neither interleaves with the other — which is the property MMR could not give at
    any lambda, because MMR spreads similar items and a rebuttal is similar to its target."""
    _budget_region(conn, n=6)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    p1 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000)
    p2 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000, continues=p1["sitting_id"])

    d1, d2 = _dates(conn, p1), _dates(conn, p2)
    assert d1 == sorted(d1) and d2 == sorted(d2), "a part is not in reading order"
    assert max(d1) < min(d2), "the parts interleave — this is not a partition of the timeline"
    # And the HEADER says the same thing. Seeds are re-admitted into every part, so counting them
    # stretches each part's reported range to the seed's own date and two exactly-contiguous parts
    # announce overlapping ranges. Measured on the live store 2026-08-25.
    assert sre.part_span(conn, p1["sitting_id"])[1] < sre.part_span(conn, p2["sitting_id"])[0]


def test_a_part_holding_only_its_seeds_does_not_claim_a_successor(conn):
    """`stop='budget'` means A NEXT PART WOULD READ SOMETHING. Seeds are re-admitted into every
    part, so a part that admitted nothing else read nothing new — and its successor would render
    the identical document, which the scheduler's remainder channel would pay for on every pass,
    forever. Reachable by hand with a budget under the seed mass."""
    _budget_region(conn, n=3)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    rec = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=10)
    assert [a["atom_id"] for a in rec["admissions"]] == ["a:seed"]
    assert rec["stop"] == "saturation"


def test_no_material_is_dropped_by_age_only_deferred(conn):
    """THE AMENDMENT-3 LINE. A part cut DEFERS; the deleted `bookmark_reader`'s recency window
    DELETED. Every atom the region holds appears in exactly one part, and the union is the whole
    region — a budget that discarded the tail would look identical from inside part 1."""
    _budget_region(conn, n=6)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    parts, prev = [], None
    for _ in range(6):
        rec = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000, continues=prev)
        parts.append(rec)
        prev = rec["sitting_id"]
        if rec["stop"] != "budget":
            break
    got = [a["atom_id"] for p in parts for a in p["admissions"] if not a["is_seed"]]
    assert sorted(got) == [f"a:{i}" for i in range(6)]
    assert len(got) == len(set(got)), "an atom was read in two parts"


def test_a_late_arriving_old_atom_folds_into_the_open_part(conn):
    """Membership is time-BLIND, and this is where that shows. An atom dated before part 1's whole
    range, saved after part 1 was closed, is not old news to be skipped and not a reason to
    repartition — it joins the open part, out of order but visibly dated. A date-bounded pool
    would silently never read it."""
    _budget_region(conn, n=6)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    p1 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000)
    _atom(conn, "a:ancient", [_at_cos(0.90, axis=7)], chars=4000, when="2020-01-01")
    p2 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000, continues=p1["sitting_id"])

    assert "a:ancient" in {a["atom_id"] for a in p2["admissions"]}
    d2 = _dates(conn, p2)
    assert d2[0].startswith("2020"), "it should sort to the front of the part it landed in"


def test_a_near_duplicate_is_still_skipped_during_a_chronological_cut(conn):
    """The ceiling test survived the partitioner swap. It is the one thing that still runs
    per-admission, and it is what keeps a reposted thread from spending a part's budget twice."""
    _atom(conn, "a:seed", [ANCHOR], chars=400, when="2026-01-01")
    _atom(conn, "a:orig", [_at_cos(0.88)], chars=1000, when="2026-02-01")
    _atom(conn, "a:repost", [_at_cos(0.88)], chars=1000, when="2026-03-01")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert [a["atom_id"] for a in rec["admissions"] if not a["is_seed"]] == ["a:orig"]
    assert rec["skipped"] == [{"atom_id": "a:repost", "red": 1.0}]


def test_a_chain_inherits_every_ancestor_not_just_the_parent(conn):
    """Part 3's baseline includes part 1. Walking the links is what makes this true without the
    caller accumulating an atom list it can silently under-report."""
    _budget_region(conn, n=6)     # DIM is 8, so axis 1..6 is the room there is for distinct atoms
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    p1 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000)
    p2 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000, continues=p1["sitting_id"])
    p3 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000, continues=p2["sitting_id"])
    seen = {a["atom_id"] for p in (p1, p2) for a in p["admissions"] if not a["is_seed"]}
    got3 = {a["atom_id"] for a in p3["admissions"] if not a["is_seed"]}
    assert not (seen & got3)
    assert p3["prior_atoms"] == len(seen)
    assert sb.chain_atom_ids(conn, p3["sitting_id"]) and \
        set(sb.chain_atom_ids(conn, p2["sitting_id"])) >= seen


def test_a_continuation_does_not_collide_onto_its_parent(conn):
    """Both parts are built inside the same second by any real continuation script, and `built_at` is
    second-resolution. Without the parent in the id hash, part 2 would REPLACE part 1."""
    _budget_region(conn)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    now = __import__("datetime").datetime(2026, 8, 10, 12, 0, 0,
                                          tzinfo=__import__("datetime").timezone.utc)
    p1 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000, now=now)
    p2 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000,
                          continues=p1["sitting_id"], now=now)
    assert p1["sitting_id"] != p2["sitting_id"]
    assert conn.execute("SELECT COUNT(*) FROM sittings").fetchone()[0] == 2


def test_an_unresolvable_parent_raises_rather_than_building_a_fresh_sitting(conn):
    """The degraded form IS the bug: a silent fallback re-reads everything and reports
    `skipped_dupes=0`, which is indistinguishable from a clean continuation."""
    _budget_region(conn)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    with pytest.raises(sb.ChainError):
        sb.build_sitting(conn, seed, floor=0.68, continues="deadbeefdeadbeef")


def test_the_seed_stays_in_every_part(conn):
    """A part is read standalone by a fresh agent, so it keeps the atoms the region is anchored to."""
    _budget_region(conn)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    p1 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000)
    p2 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000, continues=p1["sitting_id"])
    assert "a:seed" in {a["atom_id"] for a in p2["admissions"] if a["is_seed"]}


def test_remaining_mass_shrinks_with_each_part(conn):
    """`region_atoms` is what is STILL admissible, so it is the number that says whether another part
    is worth building. The full region stays recoverable via `prior_atoms`."""
    _budget_region(conn, n=6)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    p1 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000)
    p2 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000, continues=p1["sitting_id"])
    assert p2["region_atoms"] < p1["region_atoms"]
    assert p2["region_atoms"] + p2["prior_atoms"] == p1["region_atoms"]


def test_the_render_tells_a_part_two_reader_it_is_reading_part_two(conn):
    """A fresh agent handed part 2 with no marker would report a 24% slice as the whole region.
    The header states the STRETCH it covers and says where the rest went — earlier parts are in
    the claims table, never re-rendered as text."""
    _budget_region(conn)
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    p1 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000)
    p2 = sb.build_sitting(conn, seed, floor=0.68, budget_tokens=2_000, continues=p1["sitting_id"])
    md = sre.render_sitting(conn, p2["sitting_id"])
    assert "Part 2 of this region, covering " in md
    assert "appear as claims above" in md
    assert p1["sitting_id"] in md


def test_an_older_store_gains_the_chain_columns_on_open(kb_home):
    """THE MIGRATION PATH, which a fresh-DB test cannot reach.

    Every other test here builds an empty store, where `CREATE TABLE` supplies the new columns. A
    real store already has a `sittings` table, so `CREATE TABLE IF NOT EXISTS` is a no-op and only
    `_ensure_column` can add them. Creating the index on `continues` inside the DDL script rather
    than after that call made the whole schema fail to open on exactly those stores — the store did
    not degrade, it refused to connect.
    """
    import sqlite3
    from pipeline.kb import schema as sch

    db = kb_home / "old.db"
    raw = sqlite3.connect(str(db))
    raw.executescript(
        "CREATE TABLE sittings (sitting_id TEXT PRIMARY KEY, built_at TEXT NOT NULL, "
        " seed_kind TEXT NOT NULL, seed_ref TEXT, seed_atom_ids TEXT, floor REAL NOT NULL, "
        " calibrated_floor REAL, lam REAL NOT NULL, ceiling REAL NOT NULL, "
        " budget_tokens INTEGER NOT NULL, region_atoms INTEGER, region_tokens INTEGER, "
        " atoms INTEGER, tokens INTEGER, stop TEXT NOT NULL, "
        " skipped_dupes INTEGER NOT NULL DEFAULT 0, read_at TEXT, read_status TEXT);")
    raw.execute(
        "INSERT INTO sittings (sitting_id, built_at, seed_kind, seed_ref, seed_atom_ids, floor, "
        " lam, ceiling, budget_tokens, stop, read_at, read_status) "
        "VALUES ('old1','2026-07-01T00:00:00+00:00','query','mlx','[\"a:1\"]',0.67,0.7,0.95,"
        " 120000,'saturation','2026-07-02T00:00:00+00:00','ok')")
    raw.execute(
        "INSERT INTO sittings (sitting_id, built_at, seed_kind, seed_ref, seed_atom_ids, floor, "
        " lam, ceiling, budget_tokens, stop) "
        "VALUES ('old2','2026-07-03T00:00:00+00:00','query','mlx','[\"a:1\"]',0.67,0.7,0.95,"
        " 120000,'budget')")
    raw.commit()
    raw.close()

    c = sch.connect(db)                                   # must not raise
    cols = {r[1] for r in c.execute("PRAGMA table_info(sittings)")}
    assert {"continues", "prior_atoms", "region_key", "seed_vector"} <= cols
    # SUBTRACTIVE, 2026-08-25: the table is REBUILT without `lam`, not left carrying a dead dial.
    assert "lam" not in cols

    # THE BACKFILL IS EXACT, unlike `atoms.first_seen`'s: every input to the key is a column of the
    # row itself, so a key recomputed on migration is the same key a build would have written —
    # under the recipe that no longer takes `lam`.
    row = c.execute("SELECT region_key, seed_vector, read_at, read_status, stop FROM sittings "
                    "WHERE sitting_id='old1'").fetchone()
    assert row["region_key"] == sch.region_key("query", "mlx", 0.67, 0.95, 120000)
    # THE READ HISTORY SURVIVES THE REBUILD. Losing it would make the scheduler re-read and re-pay
    # for every region in the store on the next connect — the exact failure the whole table's
    # "append, never overwrite" rule exists to prevent.
    assert row["read_at"] == "2026-07-02T00:00:00+00:00" and row["read_status"] == "ok"
    assert row["stop"] == "saturation"
    # And the rebuild leaves the table INDEXED — dropping it dropped its indexes.
    names = {r[1] for r in c.execute("PRAGMA index_list(sittings)")}
    assert {"idx_sittings_read", "idx_sittings_continues", "idx_sittings_parent",
            "idx_sittings_region"} <= names
    # TWO PARTS OF ONE REGION STILL SHARE A KEY after the restamp. `region_key` is what the
    # scheduler's `MAX(read_at) GROUP BY region_key` groups on, so a restamp that split one region
    # into two would make it re-read and re-pay for the whole store.
    # And `seed_vector` is deliberately NOT backfilled — see `_backfill_region_keys`. A legacy row
    # heals on demand through `sitting_builder.ensure_seed_vector`, in the layer that owns vectors.
    assert row["seed_vector"] is None
    assert c.execute("SELECT COUNT(DISTINCT region_key) FROM sittings").fetchone()[0] == 1
    c.close()


def test_a_fresh_sitting_records_no_chain(conn):
    """Regression guard on the additive columns: part 1 is not a continuation of anything."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:near", [_at_cos(0.85)])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert rec["continues"] is None and rec["prior_atoms"] == 0
    row = sst.get_sitting(conn, rec["sitting_id"])
    assert row["continues"] is None and row["prior_atoms"] == 0
    assert "continuation:" not in sre.render_sitting(conn, rec["sitting_id"])


# ── seeds ───────────────────────────────────────────────────────────────────────
class _Embedder:
    """A phrase -> a vector THE TEST CHOSE. Offline, so the seed path is provable with no network.

    Its identity fields match what the `conn` fixture writes into `kb_meta`, because `resolve_seed`
    runs `assert_model` before it uses the vector. Pass different ones to prove that guard bites.
    """

    def __init__(self, vec, *, model="fake", provider="local", query_instruction="", dim=DIM):
        self._vec, self.model, self.provider, self.dim = vec, model, provider, dim
        self.query_instruction = query_instruction

    def embed(self, texts, role=None):
        assert role == "query", "a seed phrase must be embedded on the QUERY arm"
        return [np.asarray(self._vec, dtype=np.float32) for _ in texts]


def test_a_phrase_seeds_on_its_nearest_atoms_with_no_literal_match(conn):
    """⚠️ THE WHOLE REASON THE SEED LOOKUP MOVED OFF FTS. Nothing here is indexed for full-text
    search at all, which is the extreme form of the measured failure: `optimistic oracle` and
    `peer prediction` matched ZERO rows on the real corpus while the reader had emitted both as
    standing queries FROM that very region. The ideas were in the corpus; the words were not."""
    _atom(conn, "a:near", [_at_cos(0.75)])
    _atom(conn, "a:far", [_at_cos(0.10, axis=2)])
    seed = sb.resolve_seed(conn, query="a phrase nobody wrote down", embedder=_Embedder(ANCHOR))
    assert seed["kind"] == "query" and seed["atom_ids"] == ["a:near"]


def test_the_seed_is_the_centroid_of_chunks_not_the_phrase_vector(conn):
    """⚠️ THE PROPERTY THAT KEEPS `FLOOR_DEFAULT` VALID. The phrase vector only PICKS atoms — the
    anchor is the centroid of their CONTENT CHUNKS, so it stays in chunk space and every membership
    threshold keeps meaning what it was calibrated to mean."""
    _atom(conn, "a:near", [_at_cos(0.75)])
    seed = sb.resolve_seed(conn, query="anything", embedder=_Embedder(ANCHOR))
    assert float(seed["vector"] @ _at_cos(0.75)) > 0.999      # the atom's chunk...
    assert float(seed["vector"] @ ANCHOR) < 0.99              # ...not the phrase


def test_a_phrase_matching_nothing_close_enough_still_raises(conn):
    """"Nothing matched your phrase" and "this corner of the corpus is empty" are different facts and
    must not render the same. FTS returning zero rows used to enforce that for free; vector
    retrieval ALWAYS returns something, so `SEED_MATCH_FLOOR` is what restores the loud failure."""
    _atom(conn, "a:1", [ANCHOR])
    far = _at_cos(0.05, axis=3)                    # every atom scores ~0.05 against this phrase
    with pytest.raises(sb.SeedError):
        sb.resolve_seed(conn, query="zero knowledge proofs", embedder=_Embedder(far))


def test_a_weak_match_is_dropped_rather_than_padding_the_seed(conn):
    """The bar PRUNES as well as gates. Padding a seed up to `limit` with mediocre matches drags the
    centroid off the thing the caller named, and the region it then builds still looks plausible."""
    _atom(conn, "a:strong", [ANCHOR])
    for i in range(3):
        _atom(conn, f"a:weak{i}", [_at_cos(0.30, axis=2 + i)])   # under SEED_MATCH_FLOOR
    seed = sb.resolve_seed(conn, query="x", embedder=_Embedder(ANCHOR), limit=4)
    assert seed["atom_ids"] == ["a:strong"]


def test_a_query_seed_refuses_to_run_without_an_embedder(conn):
    """The metered call is a parameter, not a hidden construction, so a caller cannot embed by
    accident. The other two seed shapes stay entirely local and take none."""
    _atom(conn, "a:1", [ANCHOR])
    with pytest.raises(sb.SeedError):
        sb.resolve_seed(conn, query="mlx kernel")
    assert sb.resolve_seed(conn, atom_ids=["a:1"])["kind"] == "atoms"     # still free


def test_a_foreign_embedder_is_refused_before_its_vector_is_used(conn):
    """⚠️ POSITIVE CONTROL ON THE SUBSPACE GUARD — the single easiest way to get this path wrong.
    A seed centroid built from a query vector in one model's space, compared against chunks written
    in another's, produces cosines that mean NOTHING while the floor still looks calibrated. That is
    silent garbage, not a crash, so it has to fail here rather than downstream."""
    from pipeline.kb.embed import SubspaceError
    _atom(conn, "a:1", [ANCHOR])
    with pytest.raises(SubspaceError):
        sb.resolve_seed(conn, query="mlx kernel",
                        embedder=_Embedder(ANCHOR, model="some-other-model"))


def test_seed_uses_every_chunk_including_the_short_ones(conn):
    """A SHORT ATOM IS A WHOLE ATOM, and its one chunk is the only thing that can anchor it.

    This pins the deletion of a 200-char "content chunk" filter that used to run here. It was
    inert — `chunk.split_text` only emits a sub-overlap chunk when it is the atom's ONLY chunk, so
    the filter emptied the atom and the fallback handed the same chunks back — and its premise was
    backwards: in this corpus the long chunks are the machine-generated ones. The 18 reachable
    chunks it touched were all authored X one-liners like "give claude a wallet and it becomes
    unstoppable".
    """
    _atom(conn, "a:short", [ANCHOR], chars=150)        # one chunk, under the old 200-char bar
    seed = sb.resolve_seed(conn, atom_ids=["a:short"])
    assert float(seed["vector"] @ ANCHOR) > 0.999

    # And a short atom still joins a region grown from a different, normal-length seed. The seed
    # sits at 0.85 rather than on the anchor so `a:short` clears the floor without tripping the
    # near-duplicate ceiling, which would skip it for a reason that has nothing to do with length.
    _atom(conn, "a:seed", [_at_cos(0.85)])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert "a:short" in {a["atom_id"] for a in rec["admissions"]}


def test_vector_seed_needs_no_atoms(conn):
    """The path a density peak or a never-read centroid arrives on: a bare point, no seed atoms, so
    the first admission is pure relevance."""
    _atom(conn, "a:1", [ANCHOR])
    seed = sb.resolve_seed(conn, vector=ANCHOR, label="cluster-3")
    rec = sb.build_sitting(conn, seed, floor=0.68)
    assert rec["seed_kind"] == "vector" and rec["seed_atom_ids"] == []
    assert [a["atom_id"] for a in rec["admissions"]] == ["a:1"]


# ── region identity ─────────────────────────────────────────────────────────────
# TWO KEYS, TWO QUESTIONS. `sitting_id` says WHICH BUILD (it folds `built_at` in, so a rebuild
# appends rather than clobbering a read stamp). `region_key` says WHICH REGION the build is a read
# of, which is what `MAX(read_at) GROUP BY region_key` needs. Collapsing them either way breaks
# something silently, so both directions are pinned here.
def test_two_builds_of_one_region_share_a_region_key(conn):
    """Build the same seed twice and you get two events over one region. If the key moved with the
    build, last-read would be unfindable and the re-read trigger would fire forever on a region it
    had just read."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:near", [_at_cos(0.85)])
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx")
    one = sb.build_sitting(conn, seed, floor=0.68, now=_T0)
    two = sb.build_sitting(conn, seed, floor=0.68, now=_T0 + timedelta(days=30))

    assert one["sitting_id"] != two["sitting_id"]        # two events
    keys = {sst.get_sitting(conn, r["sitting_id"])["region_key"] for r in (one, two)}
    assert len(keys) == 1                                 # one region


def test_region_key_ignores_the_seed_set_churning(conn):
    """⚠️ D1, AND THE ONE PROPERTY MOST LIKELY TO BE "FIXED" BACK. The resolved seed atoms are
    deliberately OUT of the key, because any top-k membership churns as the corpus grows — ask for
    the best 4, save a better one next month, and one is displaced. With the ids in the key the same
    region would rename itself every time that happened."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:near", [_at_cos(0.85)])
    thin = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx"),
                            floor=0.68, now=_T0)
    grown = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed", "a:near"],
                                                   label="mlx"), floor=0.68, now=_T0)

    assert thin["seed_atom_ids"] != grown["seed_atom_ids"]      # the seed set really did move
    assert thin["sitting_id"] != grown["sitting_id"]            # so it IS a different build
    assert (sst.get_sitting(conn, thin["sitting_id"])["region_key"]
            == sst.get_sitting(conn, grown["sitting_id"])["region_key"])


def test_a_different_dial_is_a_different_region(conn):
    """The other direction. One phrase at two floors is one GENERATOR (queries refresh each other)
    and two REGIONS (read stamps must not). A key that ignored the dials would let a coarse read
    mark the fine region as read."""
    _atom(conn, "a:seed", [ANCHOR])
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx")
    coarse = sb.build_sitting(conn, seed, floor=0.68, now=_T0)
    fine = sb.build_sitting(conn, seed, floor=0.80, now=_T0)
    assert (sst.get_sitting(conn, coarse["sitting_id"])["region_key"]
            != sst.get_sitting(conn, fine["sitting_id"])["region_key"])


def test_a_vector_seed_round_trips_its_centroid(conn):
    """THE HOLE THE COLUMN WAS ADDED TO CLOSE. A `vector` seed has no `seed_atom_ids`, so before
    this its anchor existed nowhere — every fracture `zoom` produced was unreproducible, could not
    be continued, and could not have its new mass measured."""
    _atom(conn, "a:1", [ANCHOR])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, vector=_at_cos(0.9), label="cluster-3"),
                           floor=0.68)
    stored = sst.get_sitting(conn, rec["sitting_id"])["seed_vector"]
    assert stored is not None
    assert float(stored @ _at_cos(0.9)) > 0.999            # the anchor, not a re-derived one


def test_ensure_seed_vector_rebuilds_a_legacy_row_from_its_seed_atoms(conn):
    """The lazy repair path for a row written before the column existed. It rebuilds from the STORED
    seed atom ids — not by re-resolving the seed — so it reproduces the original centroid rather
    than whatever today's retrieval would pick."""
    _atom(conn, "a:seed", [ANCHOR])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    conn.execute("UPDATE sittings SET seed_vector = NULL")          # the pre-column shape
    conn.commit()
    assert sst.get_sitting(conn, rec["sitting_id"])["seed_vector"] is None

    v = sst.ensure_seed_vector(conn, rec["sitting_id"])
    assert float(v @ ANCHOR) > 0.999
    assert sst.get_sitting(conn, rec["sitting_id"])["seed_vector"] is not None   # persisted once


def test_ensure_seed_vector_refuses_a_legacy_vector_seed(conn):
    """It RAISES rather than inventing an anchor. A `vector` seed's centroid was never written down
    and its `seed_atom_ids` is empty by construction, so there is nothing to rebuild from — and
    degrading to a zero vector would report an EMPTY region (every cosine is 0.0) for a region that
    is merely un-migrated."""
    _atom(conn, "a:1", [ANCHOR])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, vector=ANCHOR, label="c0"), floor=0.68)
    conn.execute("UPDATE sittings SET seed_vector = NULL")
    conn.commit()
    with pytest.raises(sb.SeedError):
        sst.ensure_seed_vector(conn, rec["sitting_id"])


# ── calibration ─────────────────────────────────────────────────────────────────
def test_calibration_reports_the_noise_ceiling(conn):
    for i in range(30):
        v = np.zeros(DIM, dtype=np.float32)
        v[i % DIM] = 1.0
        _atom(conn, f"a:{i}", [v])
    cal = sb.calibrate_floor(conn, pairs=2_000)
    assert cal["p50"] is not None and cal["floor"] == round(cal["p99"] + 0.03, 2)


def _measurable_corpus(conn, n=220, rng_seed=3):
    """Enough DISTINCT chunks for the calibration to be enforceable — past
    `CALIBRATION_MIN_CHUNKS`. In `DIM` 8 dimensions random vectors sit far from orthogonal, so the
    measured ceiling lands well above `FLOOR_DEFAULT`, which is exactly the case the clamp is for."""
    rng = np.random.default_rng(rng_seed)
    for i in range(n):
        _atom(conn, f"a:noise-{i}", [_unit(*rng.normal(size=DIM))])


def test_a_floor_below_the_noise_ceiling_is_raised_to_it(conn):
    """ENFORCED, NOT WARNED (RULED 2026-08-25). Warn-only shipped an n=1 default: on a topically
    concentrated corpus the measured ceiling sits ABOVE `FLOOR_DEFAULT`, and the build logged a
    line nobody reads and then admitted the noise anyway. Below the ceiling an admission is not
    distinguishable from unrelated text, so the region is built at the ceiling instead."""
    _measurable_corpus(conn)
    _atom(conn, "a:seed", [ANCHOR])
    cal = sb.calibrate_floor(conn)
    assert cal["n_chunks"] >= sb.CALIBRATION_MIN_CHUNKS
    assert cal["floor"] > 0.5, "the fixture must actually sit above the requested floor"

    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.5)
    assert rec["floor"] == cal["floor"]


def test_the_clamp_only_raises(conn):
    """MAX(), never MIN(). A floor already above the ceiling is the caller being deliberately
    narrow, and there is no named caller for a deliberate below-noise pass. `calibrate_floor` still
    only measures — the `calibrate` CLI prints the raw numbers and changes nothing."""
    _measurable_corpus(conn)
    _atom(conn, "a:seed", [ANCHOR])
    assert sb.calibrate_floor(conn)["floor"] < 0.99
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.99)
    assert rec["floor"] == 0.99


def test_a_chained_part_keeps_its_frozen_floor_when_the_ceiling_rises(conn):
    """AT REGION CREATION ONLY. A region's floor is frozen into its `region_key`, and the measured
    ceiling drifts with the corpus. Re-clamping part N would change that key — orphaning the
    notebook chain and the read stamp, and making the scheduler re-read and re-pay for the whole
    region. The floor is a property of the region, not of the day it was continued on."""
    _atom(conn, "a:seed", [ANCHOR], when="2026-01-01")
    _atom(conn, "a:one", [_at_cos(0.80)], when="2026-02-01")
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    p1 = sb.build_sitting(conn, seed, floor=0.68)
    assert p1["floor"] == 0.68                            # tiny corpus: unclamped

    _measurable_corpus(conn)                              # the ceiling rises under the region
    assert sb.calibrate_floor(conn)["floor"] > 0.68
    p2 = sb.build_sitting(conn, seed, floor=p1["floor"], continues=p1["sitting_id"],
                          now=_T0 + timedelta(minutes=1))
    assert p2["floor"] == 0.68
    assert (sst.get_sitting(conn, p2["sitting_id"])["region_key"]
            == sst.get_sitting(conn, p1["sitting_id"])["region_key"])


def test_a_tiny_corpus_clamps_nothing(conn):
    """THE MIN-SAMPLE GUARD. `calibrate_floor` draws `CALIBRATION_PAIRS` pairs; under
    `CALIBRATION_MIN_CHUNKS` distinct chunks those draws are resampling a handful of values, so the
    p99 is an accident of which few atoms exist. Clamping on that number would push seeds into
    `SeedError` on a store's first day — the one day nobody can debug it."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:near", [_at_cos(0.85)])
    cal = sb.calibrate_floor(conn)
    assert cal["n_chunks"] < sb.CALIBRATION_MIN_CHUNKS
    assert cal["floor"] > 0.68, "otherwise this test passes for the wrong reason"

    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert rec["floor"] == 0.68


# ── the ledger ──────────────────────────────────────────────────────────────────
def test_building_is_not_reading(conn):
    """AMENDMENT 2's core distinction. A sitting on disk covers nothing until an agent reads it."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:near", [_at_cos(0.85)])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    cov = sst.coverage(conn)
    assert cov["read"] == 0 and cov["never_read"] == 2 and cov["built_unread"] == 2

    sst.mark_read(conn, rec["sitting_id"])
    cov = sst.coverage(conn)
    assert cov["read"] == 2 and cov["never_read"] == 0 and cov["built_unread"] == 0
    assert cov["pct_read"] == 100.0


def test_never_read_mass_includes_what_no_seed_reached(conn):
    """The structural blind spot: an atom below the floor from every seed ever chosen is never read
    AND, without this table, never reported unread."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:orphan", [_unit(0, 1)])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    sst.mark_read(conn, rec["sitting_id"])
    assert sst.unread_atom_ids(conn) == {"a:orphan"}
    assert sst.coverage(conn)["never_read"] == 1


def test_the_read_debt_ledger_stays_over_davids_own_material(conn):
    """The union widened MEMBERSHIP, not the ledger (RULED 2026-08-24).

    `coverage` answers "how much of what I saved have I actually read". Frontier's finds are
    unbounded and arrive nightly, so counting them makes the denominator outrun any read, and the
    sprouts digest — fed from `unread_atom_ids` — floods with machine pulls. The frontier atom below
    IS in the region and IS read; it is simply not debt.
    """
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:found", [_at_cos(0.85)], entry_mode="frontier")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert "a:found" in {a["atom_id"] for a in rec["admissions"]}
    sst.mark_read(conn, rec["sitting_id"])

    cov = sst.coverage(conn)
    assert cov["total"] == 1 and cov["read"] == 1        # the seed alone is the denominator
    assert sst.unread_atom_ids(conn) == set()


# ── per-lens read state (Job N) ─────────────────────────────────────────────────
def test_a_lens_read_stamp_is_independent_of_queries(conn):
    """`mark_lens_read` (Job N's `claims`) must not touch, or be touched by, `sittings.read_at` —
    the two lenses guard their own re-read independently (D20)."""
    _atom(conn, "a:seed", [ANCHOR])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    sid = rec["sitting_id"]
    assert sst.lens_read_state(conn, sid, "claims") is None

    sst.mark_lens_read(conn, sid, "claims")
    assert sst.lens_read_state(conn, sid, "claims")["read_status"] == "ok"
    assert sst.get_sitting(conn, sid)["read_at"] is None      # queries' own stamp: untouched

    sst.mark_read(conn, sid)                                   # queries reads it separately
    assert sst.get_sitting(conn, sid)["read_at"] is not None
    assert sst.lens_read_state(conn, sid, "claims")["read_status"] == "ok"  # unaffected either way


def test_a_claims_only_read_counts_toward_coverage(conn):
    """The Amendment 2 ledger measures whether a person got material out of a region, not which
    lens did it — a region read only for `claims` is not still 'unread' mass."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:near", [_at_cos(0.85)])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    sid = rec["sitting_id"]
    assert sst.coverage(conn)["read"] == 0

    sst.mark_lens_read(conn, sid, "claims")
    cov = sst.coverage(conn)
    assert cov["read"] == 2 and cov["never_read"] == 0 and cov["sittings_read"] == 1
    assert sst.unread_atom_ids(conn) == set()


def test_a_sitting_read_by_both_lenses_is_not_counted_twice(conn):
    """UNION, not UNION ALL — a sitting both lenses have read must still count as ONE read sitting."""
    _atom(conn, "a:seed", [ANCHOR])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    sid = rec["sitting_id"]
    sst.mark_read(conn, sid)
    sst.mark_lens_read(conn, sid, "claims")
    assert sst.coverage(conn)["sittings_read"] == 1


def test_record_and_get_claims_round_trip(conn):
    _atom(conn, "a:seed", [ANCHOR])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    sid = rec["sitting_id"]
    n = sst.record_claims(conn, sid, [
        {"claim": "x402 processed $1.6M/month", "falsified_by": "onchain data shows otherwise",
         "atom_ids": ["a:seed"]},
    ])
    assert n == 1
    claims = sst.get_claims(conn, sid)
    assert len(claims) == 1
    assert claims[0]["claim"] == "x402 processed $1.6M/month"
    assert claims[0]["atom_ids"] == ["a:seed"]


# ── sprouts digest (Job C, folded into Job L as a fifth lens) ───────────────────
def test_sprouts_digest_covers_only_what_no_read_sitting_covered(conn):
    """The material selector David ruled on 2026-08-16: `unread_atom_ids`, used as is. A read atom
    must not appear in the digest; an orphan no seed ever reached must."""
    _atom(conn, "a:seed", [ANCHOR])
    _atom(conn, "a:orphan", [_unit(0, 1)], when="2026-03-01")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    sst.mark_read(conn, rec["sitting_id"])
    dig = sre.render_sprouts_digest(conn)
    assert dig["atoms"] == 1 and not dig["truncated"]
    assert "a:orphan" in dig["document"] and "a:seed" not in dig["document"]


def test_sprouts_digest_is_empty_when_everything_is_read(conn):
    """The negative control — once every human-attested atom is covered, there is nothing left to
    digest, and that must render as an empty result rather than an empty-looking document."""
    _atom(conn, "a:seed", [ANCHOR])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    sst.mark_read(conn, rec["sitting_id"])
    assert sre.render_sprouts_digest(conn) == {"document": "", "atoms": 0, "truncated": False}


def test_sprouts_digest_is_chronological(conn):
    """Every other render in this module is chronological; this material has no seed to anchor an
    arc, but time order is still the one order it renders in."""
    _atom(conn, "a:mid", [ANCHOR], when="2026-04-01")
    _atom(conn, "a:early", [_unit(0, 1)], when="2026-01-01")
    _atom(conn, "a:late", [_unit(0, 0, 1)], when="2026-09-01")
    md = sre.render_sprouts_digest(conn)["document"]
    order = [ln.rsplit("(", 1)[1].rstrip(")") for ln in md.splitlines() if ln.startswith("### ")]
    assert order == ["a:early", "a:mid", "a:late"]


def test_sprouts_digest_names_itself_a_grab_bag_not_one_conversation(conn):
    """A fresh reader must not mistake unrelated orphans for a thread that failed to cohere — the
    header has to say so plainly, the same job the instruction text does at the lens layer."""
    _atom(conn, "a:orphan", [ANCHOR])
    md = sre.render_sprouts_digest(conn)["document"]
    assert "NOT one conversation" in md


def test_sprouts_digest_truncates_at_the_char_budget_and_says_so(conn, monkeypatch):
    """No silent caps — a truncated digest must say how much was left out, the same convention
    `sitting_reader.render_prompt` uses for an oversized sitting."""
    monkeypatch.setattr(sre, "SPROUTS_DIGEST_MAX_CHARS", 400)
    for i in range(5):
        _atom(conn, f"a:{i}", [_unit(0, 1 + i)], when=f"2026-0{1 + i}-01", chars=400)
    dig = sre.render_sprouts_digest(conn)
    assert dig["truncated"] is True
    assert dig["atoms"] == 5                    # the true count, even though not all are shown
    assert "TRUNCATED" in dig["document"]


# ⚠️ `test_densest_unread_points_at_the_biggest_untouched_mass` WAS HERE AND IS DELETED WITH THE
# FUNCTION (2026-08-16, D13). It asserted "the standing seed source" — that while unread mass
# remains there is always a densest part of it, so coverage could progress with no external signal.
# That capability was removed on purpose: under lazy creation a machine-proposed seed builds a
# region nobody asked for, and density ranks the most REPETITIVE mass rather than the most
# important. The ledger it fed (`coverage`, `unread_atom_ids`) is still tested directly above.


def test_rebuilding_a_sitting_keeps_its_read_stamp(conn):
    """A rebuild must never turn a sitting that was read back into unread mass."""
    _atom(conn, "a:seed", [ANCHOR])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    sst.mark_read(conn, rec["sitting_id"])
    sst.record_sitting(conn, rec)
    assert sst.get_sitting(conn, rec["sitting_id"])["read_at"] is not None


def test_tiering_keeps_small_regions_as_sprouts(conn):
    assert sb.tier_for_material(12) == "standalone"
    assert sb.tier_for_material(4) == "sprout"
    assert sb.tier_for_material(2) == "ledger"


def test_reading_tier_agrees_with_material_tier_today(conn):
    """The two questions share one bucket table, and that sharing is a coincidence rather than a
    definition — so it gets a test that will fail loudly the day someone moves one of them."""
    for n in (0, 2, 3, 9, 10, 12, 400):
        assert sb.tier_for_reading(n) == sb.tier_for_material(n)


def test_reading_tier_gates_on_admitted_not_region(conn):
    """The zoom persistence gate must never see material nobody will read. A 12-atom region skipped
    down to 7 is a sprout, not a standalone."""
    assert sb.tier_for_material(12) == "standalone"
    assert sb.tier_for_reading(7) == "sprout"


# ── render ──────────────────────────────────────────────────────────────────────
def test_render_is_chronological_not_admission_ordered(conn):
    """Time order is what produces "what moved". Score order hides a reversal inside a thread."""
    _atom(conn, "a:seed", [ANCHOR], when="2026-06-01")
    _atom(conn, "a:old", [_at_cos(0.80)], when="2026-01-01")
    _atom(conn, "a:new", [_at_cos(0.90, axis=2)], when="2026-08-01")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    md = sre.render_sitting(conn, rec["sitting_id"])
    # The section headings, not the raw ids — the seed is also named in the header block.
    order = [ln.rsplit("(", 1)[1].rstrip(")") for ln in md.splitlines() if ln.startswith("### ")]
    assert order == ["a:old", "a:seed", "a:new"]


def test_render_reports_author_concentration(conn):
    """A single-author sitting generates self-referential queries, so the reader has to be able to
    see that it is reading one person's log."""
    _atom(conn, "a:seed", [ANCHOR], who="x:taelin")
    _atom(conn, "a:2", [_at_cos(0.85)], who="x:taelin")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert "x:taelin wrote 100%" in sre.render_sitting(conn, rec["sitting_id"])


def test_render_removes_the_chunk_overlap(conn):
    _atom(conn, "a:seed", [ANCHOR])
    conn.execute("INSERT INTO atoms (atom_id, source_type, who_id, when_ts, entry_mode) "
                 "VALUES ('a:ov','x','x:u','2026-08-01','user-saved')")
    for seq, (cs, ce, txt) in enumerate([(0, 10, "ABCDEFGHIJ"), (8, 18, "IJKLMNOPQR")]):
        conn.execute("INSERT INTO chunks (atom_id, seq, char_start, char_end, text, vector) "
                     "VALUES (?,?,?,?,?,?)",
                     ("a:ov", seq, cs, ce, txt, _at_cos(0.85).tobytes()))
    conn.commit()
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    assert "ABCDEFGHIJKLMNOPQR" in sre.render_sitting(conn, rec["sitting_id"])


def test_artifacts_write_under_the_sandboxed_home(conn, kb_home):
    _atom(conn, "a:seed", [ANCHOR])
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"], label="mlx"),
                           floor=0.68)
    paths = sre.write_artifacts(conn, rec["sitting_id"])
    assert str(kb_home) in paths["markdown"]
    assert paths["markdown"].endswith(".md")


# ── degradation ─────────────────────────────────────────────────────────────────
def test_no_embedded_chunks_yields_an_empty_sitting_not_a_crash(conn):
    """FAIL-SAFE: a store whose embedding pass has not run yet degrades to an empty result."""
    _atom(conn, "a:seed", [ANCHOR])
    seed = sb.resolve_seed(conn, atom_ids=["a:seed"])
    conn.execute("UPDATE chunks SET vector = NULL")
    conn.commit()
    rec = sb.build_sitting(conn, seed, floor=0.68)
    assert rec["stop"] == "empty" and rec["atoms"] == 0


def test_undated_atoms_trail_the_chronology(conn):
    """An empty `when_ts` sorts before every real date, so an undated paper would OPEN a document
    the prompt calls chronological — the reader's first evidence for "what moved" would have no
    position in time. Measured on the real KISS1R sitting, where the seed paper carries no date."""
    _atom(conn, "a:seed", [ANCHOR], when="2026-06-01")
    _atom(conn, "a:dated", [_at_cos(0.85)], when="2026-01-01")
    _atom(conn, "a:undated", [_at_cos(0.90, axis=2)], when="")
    rec = sb.build_sitting(conn, sb.resolve_seed(conn, atom_ids=["a:seed"]), floor=0.68)
    md = sre.render_sitting(conn, rec["sitting_id"])
    order = [ln.rsplit("(", 1)[1].rstrip(")") for ln in md.splitlines() if ln.startswith("### ")]
    assert order == ["a:dated", "a:seed", "a:undated"]
