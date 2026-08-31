"""Frontier stage 1 — a standing query can be CLAIMED by more than one generator.

`sitting_reader.py` calls `frontier_queries.generator` the ownership key: it decides which rows a
re-seed of a region refreshes, and which rail's verdicts may move a row's counters. That intent is
right. The column could not carry it, because `upsert_queries` overwrote it on collision and the
last writer inherited every earlier region's rights.

It held while regions were far apart and one query had one region. `zoom` fractures a region into
siblings reading overlapping material, so one query now arrives from several generators as a matter
of course — four sub-reads of one region emitted 76 queries into 75 rows.

The failures fenced off here, all silent, and all worse under a design where survival IS the
verdict:

  • A VERDICT THROWN AWAY. `apply_verdicts` matched on the column, so once a sibling emitted the
    same text, the first region's explicit keep or drop landed as `unmatched` and was discarded.
  • A COUNTER SHARED. One region's `drop` slowed a query another had just kept, and one region's
    `keep` erased another's accumulated drops.
  • A QUERY THE REGION CANNOT SEE. `active_queries(generator=...)` filtered on the column, so the
    query left that region's standing list — and a query the reader is not shown cannot be
    verdicted, cannot be revived, and gets its wording re-invented on the next read, orphaning
    stage 2's watermark.
"""
from __future__ import annotations

import pytest

from pipeline.kb import frontier_queries as fq
from pipeline.kb import schema
from pipeline.kb import sitting_reader as sr

PARENT, SUB0, SUB2 = "sitting:mlx", "sitting:mlx-0", "sitting:mlx-2"


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _q(text: str) -> dict:
    return {"text": text, "rationale": "because", "target_sources": ["arxiv"], "atom_ids": []}


def _v(text: str, verdict: str = "keep") -> dict:
    return {"text": text, "verdict": verdict, "reason": "r", "atom_ids": ["x:1"]}


def _claims(conn, text: str) -> dict[str, int]:
    """{generator: miss_count} for one query."""
    qid = fq.query_id_for(fq.normalize(text))
    return {r[0]: r[1] for r in conn.execute(
        "SELECT generator, miss_count FROM frontier_query_generators WHERE query_id=?", (qid,))}


def _query(conn, text: str):
    return conn.execute("SELECT * FROM frontier_queries WHERE normalized=?",
                        (fq.normalize(text),)).fetchone()


# ── the many-to-many is recorded ────────────────────────────────────────────────
def test_three_generators_emitting_one_query_leave_three_claims(conn):
    for gen in (PARENT, SUB0, SUB2):
        fq.upsert_queries(conn, [_q("continuous batching MLX")], generator=gen)
    assert set(_claims(conn, "continuous batching MLX")) == {PARENT, SUB0, SUB2}
    assert len(list(conn.execute("SELECT 1 FROM frontier_queries"))) == 1   # still ONE query row


def test_the_generator_column_freezes_at_origin(conn):
    """It used to take the latest writer, which is what handed away the earlier region's rights."""
    fq.upsert_queries(conn, [_q("continuous batching MLX")], generator=PARENT)
    fq.upsert_queries(conn, [_q("continuous batching MLX")], generator=SUB2)
    row = _query(conn, "continuous batching MLX")
    assert row["generator"] == PARENT          # origin, like created_at — not ownership
    assert row["emit_count"] == 2              # and it is still one row being collided onto


# ── verdicts reach the right row ────────────────────────────────────────────────
def test_a_dispossessed_region_can_still_verdict_its_own_query(conn):
    """THE bug, in the form that matters on the verdict model: PARENT emitted it first, SUB2
    emitted it later. PARENT's verdict used to land as `unmatched` and be thrown away."""
    fq.upsert_queries(conn, [_q("shared thread")], generator=PARENT)
    fq.upsert_queries(conn, [_q("shared thread")], generator=SUB2)
    res = fq.apply_verdicts(conn, [_v("shared thread", "drop")], generator=PARENT)
    assert res == {"kept": 0, "dropped": 1, "unmatched": 0}
    assert _claims(conn, "shared thread")[PARENT] == 1


def test_a_generator_with_no_claim_is_still_unmatched(conn):
    """The scoping must not become a free-for-all: a rail that never asked for this query has no
    standing to move its counters."""
    fq.upsert_queries(conn, [_q("shared thread")], generator=PARENT)
    res = fq.apply_verdicts(conn, [_v("shared thread", "drop")], generator="bookmark-reader")
    assert res == {"kept": 0, "dropped": 0, "unmatched": 1}
    assert _claims(conn, "shared thread") == {PARENT: 0}


# ── the counter is per claim, and the query's speed is the MIN ──────────────────
def test_one_regions_drop_does_not_slow_a_query_another_region_keeps(conn):
    fq.upsert_queries(conn, [_q("shared thread")], generator=PARENT)
    fq.upsert_queries(conn, [_q("shared thread")], generator=SUB2)
    for _ in range(3):
        fq.apply_verdicts(conn, [_v("shared thread", "drop")], generator=SUB2)
    assert _claims(conn, "shared thread") == {PARENT: 0, SUB2: 3}
    # SUB2 wants it monthly; PARENT still wants it daily. MIN keeps it daily.
    assert _query(conn, "shared thread")["miss_count"] == 0


def test_a_query_only_slows_once_every_claimant_has_let_it(conn):
    fq.upsert_queries(conn, [_q("shared thread")], generator=PARENT)
    fq.upsert_queries(conn, [_q("shared thread")], generator=SUB2)
    for _ in range(3):
        fq.apply_verdicts(conn, [_v("shared thread", "drop")], generator=SUB2)
    assert _query(conn, "shared thread")["miss_count"] == 0
    for _ in range(2):
        fq.apply_verdicts(conn, [_v("shared thread", "drop")], generator=PARENT)
    assert _query(conn, "shared thread")["miss_count"] == 2      # MIN(3, 2)


def test_one_regions_keep_does_not_erase_another_regions_drops(conn):
    """The counter was shared on the query row, so this used to reset it to 0 for everyone."""
    fq.upsert_queries(conn, [_q("shared thread")], generator=PARENT)
    fq.upsert_queries(conn, [_q("shared thread")], generator=SUB2)
    for _ in range(2):
        fq.apply_verdicts(conn, [_v("shared thread", "drop")], generator=SUB2)
    fq.apply_verdicts(conn, [_v("shared thread", "keep")], generator=PARENT)
    assert _claims(conn, "shared thread") == {PARENT: 0, SUB2: 2}    # SUB2's drops survive


def test_a_keep_never_writes_status(conn):
    """Retirement stays a human act — an automatic path must not un-retire either."""
    fq.upsert_queries(conn, [_q("shared thread")], generator=PARENT)
    fq.retire_query(conn, "shared thread")
    fq.apply_verdicts(conn, [_v("shared thread", "keep")], generator=PARENT)
    assert _query(conn, "shared thread")["status"] == "retired"


# ── what the reader is shown ────────────────────────────────────────────────────
def test_the_scoped_read_shows_a_region_a_query_a_sibling_emitted_later(conn):
    fq.upsert_queries(conn, [_q("shared thread")], generator=PARENT)
    fq.upsert_queries(conn, [_q("shared thread")], generator=SUB2)
    assert [r["normalized"] for r in fq.active_queries(conn, generator=PARENT)] == ["shared thread"]
    assert [r["normalized"] for r in fq.active_queries(conn, generator=SUB2)] == ["shared thread"]


def test_the_scoped_read_excludes_a_region_that_never_asked(conn):
    fq.upsert_queries(conn, [_q("shared thread")], generator=PARENT)
    assert fq.active_queries(conn, generator=SUB2) == []
    assert len(fq.active_queries(conn)) == 1                       # unscoped: still executable


def test_a_slowed_query_stays_on_the_list_and_a_retired_one_does_not(conn):
    """The load-bearing rule of the decay design, now through the join."""
    fq.upsert_queries(conn, [_q("slowed")], generator=PARENT)
    fq.upsert_queries(conn, [_q("gone")], generator=PARENT)
    for _ in range(12):
        fq.apply_verdicts(conn, [_v("slowed", "drop")], generator=PARENT)
    fq.retire_query(conn, "gone")
    assert [r["normalized"] for r in fq.active_queries(conn, generator=PARENT)] == ["slowed"]


def test_the_scoped_read_returns_query_columns_not_claim_columns(conn):
    """`SELECT *` across the join would shadow the query's own miss_count and last_emitted_at."""
    fq.upsert_queries(conn, [_q("shared thread")], generator=PARENT)
    row = fq.active_queries(conn, generator=PARENT)[0]
    assert set(row.keys()) == {r[1] for r in conn.execute("PRAGMA table_info(frontier_queries)")}


# ── migrating a store written before claims existed ─────────────────────────────
def test_an_existing_query_row_is_backfilled_into_a_claim(kb_home, tmp_path):
    """Pre-existing rows had exactly one claimant, so the column IS that claim — counter included."""
    db = tmp_path / "old.db"
    c = schema.connect(db)
    c.execute("DELETE FROM frontier_query_generators")           # simulate the pre-claims store
    c.execute("""INSERT INTO frontier_queries
                   (query_id, text, normalized, generator, status, emit_count, miss_count,
                    created_at, last_emitted_at)
                 VALUES ('abc','Old Thread','old thread',?, 'active', 4, 2,
                         '2026-01-01T00:00:00+00:00','2026-06-01T00:00:00+00:00')""", (PARENT,))
    c.commit()
    c.close()

    c = schema.connect(db)                                       # DDL + backfill run on connect
    row = c.execute("SELECT * FROM frontier_query_generators WHERE query_id='abc'").fetchone()
    assert row["generator"] == PARENT and row["miss_count"] == 2
    assert row["first_emitted_at"] == "2026-01-01T00:00:00+00:00"
    c.close()


def test_the_backfill_is_idempotent_and_does_not_reset_a_claims_counter(kb_home, tmp_path):
    db = tmp_path / "old.db"
    c = schema.connect(db)
    fq.upsert_queries(c, [_q("shared thread")], generator=PARENT)
    for _ in range(3):
        fq.apply_verdicts(c, [_v("shared thread", "drop")], generator=PARENT)
    assert _claims(c, "shared thread")[PARENT] == 3
    c.close()

    c = schema.connect(db)                                       # re-run the DDL + backfill
    assert _claims(c, "shared thread") == {PARENT: 3}            # INSERT OR IGNORE touches nothing
    c.close()


# ── the machine-lane quota (K=3, RULED 2026-08-25) ──────────────────────────────
# The union puts crawler-found atoms in every region, so a read of a machine-heavy region could
# mint an unbounded standing watch-list off the system's own output. K bounds question-list
# OWNERSHIP, not money — watermarked pulls are near-free.

def _atom(conn, atom_id: str, entry_mode: str) -> str:
    schema.upsert_atom(conn, {"atom_id": atom_id, "source_type": "x", "entry_mode": entry_mode})
    return atom_id


def _mq(text: str, atom_ids) -> dict:
    return {"text": text, "rationale": "r", "target_sources": ["arxiv"], "atom_ids": list(atom_ids)}


@pytest.fixture()
def lanes(conn):
    """One machine-found atom and one human-attested one to cite."""
    return _atom(conn, "x:found", "frontier"), _atom(conn, "x:saved", "user-saved")


def _emit(conn, queries: list[dict], generator: str) -> tuple[list[dict], list[str]]:
    """The reader's real sequence: classify + clamp, THEN write. Lane reaches the row only through
    the clamp, so a test that seeds `upsert_queries` directly is testing a state the rail cannot
    produce."""
    kept, clamped = sr._clamp_machine_lane(conn, queries, generator=generator)
    fq.upsert_queries(conn, kept, generator=generator)
    return kept, clamped


def test_one_human_citation_makes_the_whole_query_human(conn, lanes):
    """Mixed counts as HUMAN (ruled). The human material is what motivated the question, and the
    quota exists to bound what the crawler can put on the list ON ITS OWN."""
    found, saved = lanes
    kept, clamped = sr._clamp_machine_lane(
        conn, [_mq("mixed", [found, saved]), _mq("pure", [found])], generator=PARENT)
    assert clamped == []
    assert [q["lane"] for q in kept] == [fq.LANE_HUMAN, fq.LANE_MACHINE]


def test_the_human_lane_is_never_clamped(conn, lanes):
    """No quota on David's own material — the region's question list is his to grow."""
    _found, saved = lanes
    kept, clamped = sr._clamp_machine_lane(
        conn, [_mq(f"q{i}", [saved]) for i in range(8)], generator=PARENT)
    assert len(kept) == 8 and clamped == []


def test_the_fourth_machine_query_is_dropped_in_emission_order(conn, lanes):
    """Emission order IS the priority order — the model puts what it thinks matters first, and
    there is no better signal available at this point."""
    found, _saved = lanes
    kept, clamped = sr._clamp_machine_lane(
        conn, [_mq(f"m{i}", [found]) for i in range(5)], generator=PARENT)
    assert [q["text"] for q in kept] == ["m0", "m1", "m2"]
    assert clamped == ["m3", "m4"]


def test_the_allowance_counts_claims_not_origin_rows(conn, lanes):
    """THE TRAP. `frontier_queries.generator` is ORIGIN, frozen at first insert. A query a SIBLING
    region said first but this region also claims is invisible to an origin-keyed count, so the
    region silently runs at double quota — the same defect this whole file exists for, arriving
    through a new door."""
    found, _saved = lanes
    _emit(conn, [_mq("held", [found])], SUB2)      # origin = the sibling
    _emit(conn, [_mq("held", [found])], PARENT)    # PARENT claims it too
    assert _query(conn, "held")["generator"] == SUB2

    kept, clamped = sr._clamp_machine_lane(
        conn, [_mq(f"m{i}", [found]) for i in range(4)], generator=PARENT)
    assert len(kept) == 2 and clamped == ["m2", "m3"], "the sibling-originated claim held a slot"


def test_re_emitting_a_query_already_held_costs_no_slot(conn, lanes):
    """Otherwise a region at quota clamps its OWN standing set on the very next read and thrashes
    it in and out forever — the query it just paid to confirm becomes the one it drops."""
    found, _saved = lanes
    _emit(conn, [_mq(f"m{i}", [found]) for i in range(3)], PARENT)
    assert len(fq.machine_lane_claims(conn, PARENT)) == 3

    kept, clamped = sr._clamp_machine_lane(
        conn, [_mq(f"m{i}", [found]) for i in range(3)], generator=PARENT)
    assert len(kept) == 3 and clamped == []


def test_a_retired_query_gives_its_slot_back(conn, lanes):
    """Retirement is the human's door out of the query set; a retired row that still held a slot
    would make the quota unrecoverable without a schema edit."""
    found, _saved = lanes
    _emit(conn, [_mq(f"m{i}", [found]) for i in range(3)], PARENT)
    conn.execute("UPDATE frontier_queries SET status='retired' WHERE normalized='m0'")
    conn.commit()

    kept, clamped = sr._clamp_machine_lane(conn, [_mq("fresh", [found])], generator=PARENT)
    assert len(kept) == 1 and clamped == []


def test_the_lane_is_one_way_sticky_to_human(conn, lanes):
    """Every other descriptive column is last-writer-wins. Left that way, this one flaps as two
    regions re-emit the same text from different material, the quota count goes nondeterministic,
    and a query oscillates in and out of the clamp forever."""
    found, saved = lanes
    _emit(conn, [_mq("thread", [saved])], PARENT)      # cited human material first
    assert _query(conn, "thread")["lane"] == fq.LANE_HUMAN
    _emit(conn, [_mq("thread", [found])], SUB2)        # a sibling re-emits it off machine material
    assert _query(conn, "thread")["lane"] == fq.LANE_HUMAN, "human is a floor, not a state"

    # The other direction moves, because a machine query CAN earn its way into the human lane.
    _emit(conn, [_mq("other", [found])], PARENT)
    assert _query(conn, "other")["lane"] == fq.LANE_MACHINE
    _emit(conn, [_mq("other", [saved])], SUB2)
    assert _query(conn, "other")["lane"] == fq.LANE_HUMAN


def test_an_older_store_gains_the_lane_column_and_its_rows_hold_no_slot(kb_home, tmp_path):
    """Migration + the reading of NULL. Every pre-existing row was emitted when regions held
    human-attested atoms only, so NULL is honestly human and must not consume the quota — a store
    that upgraded into an instantly-full quota would go quiet and never say why."""
    db = tmp_path / "old.db"
    c = schema.connect(db)
    c.execute("""INSERT INTO frontier_queries
                   (query_id, text, normalized, generator, status, emit_count, miss_count,
                    created_at, last_emitted_at, lane)
                 VALUES ('abc','Old','old',?, 'active', 1, 0,
                         '2026-01-01T00:00:00+00:00','2026-06-01T00:00:00+00:00', NULL)""",
              (PARENT,))
    c.execute("INSERT INTO frontier_query_generators (query_id, generator, first_emitted_at, "
              " last_emitted_at, miss_count) VALUES ('abc',?, '2026-01-01','2026-06-01',0)",
              (PARENT,))
    c.commit()

    assert "lane" in {r[1] for r in c.execute("PRAGMA table_info(frontier_queries)")}
    assert fq.machine_lane_claims(c, PARENT) == set()
    c.close()
