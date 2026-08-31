"""Frontier stage 1 — the generator registry, and the claim that cannot vote.

A query's stage-2 SPEED is `MIN(miss_count)` across its claims, chosen so a query two regions care
about runs at the pace of whichever still wants it most. That aggregate assumed every claimant can
change its mind. A claimant that cannot — one never shown the standing list again, so no occasion
ever arises at which its keep/drop verdict could arrive — inserts at 0 and stays at 0 forever.

`MIN` could not tell "this asker is maximally engaged" from "this asker is structurally incapable
of ever saying otherwise", so a single frozen claim pinned every query it touched to the fastest
tier permanently — silencing the bookmark reader's explicit drops on any query they shared. v1 read
silence as death and retired three live queries by arithmetic; this was the same mistake with the
sign flipped, reading permanent silence as maximal life.

⚠️ THE ORIGINAL EXAMPLE WAS `sitting:*`, AND IT IS NO LONGER ONE. The sitting reader was taught
verdicts on 2026-08-16 (D11) and registers itself votable, so no shipping generator declares
`votable=0` today. These tests therefore construct the non-votable claim directly rather than
borrowing a real rail's flag — which is the more honest shape anyway: what is fenced off here is
the AGGREGATE's behaviour, and it must hold for whichever generator is write-once next.

Fenced off here:

  • THE PIN. A votable claim at 15 and a frozen claim at 0 must resolve to 15, not 0.
  • THE UPSERT BACK DOOR, which is the one that would have shipped. `upsert_queries` used to write
    `miss_count = 0` onto the query row directly. That agreed with the MIN only while every claim
    voted. Once the MIN counts votable claims only, a NEW sitting emitting a query the bookmark
    reader had already dropped 15 times would reset the shared row to daily and silently undo the
    entire fix.
  • THE KILL SWITCH'S SUBTLETY. Retiring a dead region must not take down a thread the bookmark
    reader is still asking for.
"""
from __future__ import annotations

import pytest

from pipeline.kb import frontier_queries as fq
from pipeline.kb import schema

BOOKMARKS = "bookmark-reader"
REGION = "sitting:mlx"
OTHER_REGION = "sitting:kiss1r"
SHARED = "KV cache quantization MLX"


@pytest.fixture()
def conn(kb_home):
    c = schema.connect()
    yield c
    c.close()


def _q(text: str) -> dict:
    return {"text": text, "rationale": "because", "target_sources": ["arxiv"], "atom_ids": []}


def _speed(conn, text: str) -> int:
    row = conn.execute("SELECT miss_count FROM frontier_queries WHERE normalized=?",
                       (fq.normalize(text),)).fetchone()
    return row["miss_count"]


def _drop(conn, text: str, generator: str, times: int) -> None:
    for _ in range(times):
        fq.apply_verdicts(conn, [{"text": text, "verdict": "drop"}], generator=generator)


# ── registration ────────────────────────────────────────────────────────────────
def test_emitting_registers_the_channel(conn):
    """Nothing has to remember to register: the first emission does it."""
    fq.upsert_queries(conn, [_q(SHARED)], generator=REGION, votable=False, label="the mlx region")
    row = conn.execute("SELECT * FROM frontier_generators WHERE generator=?", (REGION,)).fetchone()
    assert row["votable"] == 0
    assert row["label"] == "the mlx region"
    assert row["status"] == "active"


def test_re_registration_never_un_retires(conn):
    """Retirement is a human decision; an automatic path must not overturn it by accident."""
    fq.upsert_queries(conn, [_q(SHARED)], generator=REGION, votable=False)
    fq.retire_generator(conn, REGION)
    fq.upsert_queries(conn, [_q("something else")], generator=REGION, votable=False)
    row = conn.execute("SELECT status FROM frontier_generators WHERE generator=?",
                       (REGION,)).fetchone()
    assert row["status"] == "retired"


# ── the pin ─────────────────────────────────────────────────────────────────────
def test_a_frozen_claim_does_not_win_the_min(conn):
    """THE BUG, in one test. Without the votable rule this asserts 0 and the query runs daily
    forever on the strength of a claimant that can never speak."""
    fq.upsert_queries(conn, [_q(SHARED)], generator=BOOKMARKS, votable=True)
    fq.upsert_queries(conn, [_q(SHARED)], generator=REGION, votable=False)
    _drop(conn, SHARED, BOOKMARKS, 15)
    assert _speed(conn, SHARED) == 15          # not MIN(15, 0)


def test_two_votable_claims_still_take_the_min(conn):
    """The original semantics survive where they were right: a query two LIVE askers want runs at
    the pace of whichever still wants it most."""
    fq.upsert_queries(conn, [_q(SHARED)], generator=BOOKMARKS, votable=True)
    fq.upsert_queries(conn, [_q(SHARED)], generator="second-reader", votable=True)
    _drop(conn, SHARED, BOOKMARKS, 12)
    assert _speed(conn, SHARED) == 0           # the second claim never dropped it


def test_only_frozen_claims_falls_back_to_daily(conn):
    """An empty votable aggregate is NO evidence about speed, and this rail's fail-safe direction
    runs toward pulling — one artifact search is the cost of being wrong."""
    fq.upsert_queries(conn, [_q(SHARED)], generator=REGION, votable=False)
    fq.upsert_queries(conn, [_q(SHARED)], generator=OTHER_REGION, votable=False)
    assert _speed(conn, SHARED) == 0


# ── the upsert back door ────────────────────────────────────────────────────────
def test_a_new_sitting_does_not_reset_an_accumulated_drop(conn):
    """THE REGRESSION THAT WOULD HAVE SHIPPED. `upsert_queries` wrote miss_count=0 onto the query
    row directly; the fix is only real if a later non-votable emission leaves the speed alone."""
    fq.upsert_queries(conn, [_q(SHARED)], generator=BOOKMARKS, votable=True)
    _drop(conn, SHARED, BOOKMARKS, 15)
    assert _speed(conn, SHARED) == 15

    fq.upsert_queries(conn, [_q(SHARED)], generator=OTHER_REGION, votable=False)
    assert _speed(conn, SHARED) == 15          # the sitting's arrival changed nothing


def test_a_votable_re_emission_still_resets(conn):
    """The other direction must keep working: re-emission by a channel that CAN vote is a keep."""
    fq.upsert_queries(conn, [_q(SHARED)], generator=BOOKMARKS, votable=True)
    _drop(conn, SHARED, BOOKMARKS, 15)
    fq.upsert_queries(conn, [_q(SHARED)], generator=BOOKMARKS, votable=True)
    assert _speed(conn, SHARED) == 0


# ── the kill switch ─────────────────────────────────────────────────────────────
def test_retiring_a_region_retires_only_what_it_alone_claimed(conn):
    solo = "interaction nets GPU compiler"
    fq.upsert_queries(conn, [_q(SHARED), _q(solo)], generator=REGION, votable=False)
    fq.upsert_queries(conn, [_q(SHARED)], generator=BOOKMARKS, votable=True)

    out = fq.retire_generator(conn, REGION)
    assert out["generator_retired"] is True
    assert out["queries_retired"] == 1

    statuses = {r["normalized"]: r["status"] for r in
                conn.execute("SELECT normalized, status FROM frontier_queries")}
    assert statuses[fq.normalize(solo)] == "retired"
    assert statuses[fq.normalize(SHARED)] == "active"    # the bookmark reader still asks


def test_a_retired_channel_stops_counting_as_a_live_claimant(conn):
    """Retire both claimants in turn and the shared query finally goes with the second."""
    fq.upsert_queries(conn, [_q(SHARED)], generator=REGION, votable=False)
    fq.upsert_queries(conn, [_q(SHARED)], generator=OTHER_REGION, votable=False)
    assert fq.retire_generator(conn, REGION)["queries_retired"] == 0
    assert fq.retire_generator(conn, OTHER_REGION)["queries_retired"] == 1


def test_retiring_an_unknown_channel_changes_nothing(conn):
    fq.upsert_queries(conn, [_q(SHARED)], generator=BOOKMARKS, votable=True)
    out = fq.retire_generator(conn, "sitting:never-existed")
    assert out == {"generator_retired": False, "queries_retired": 0}
    assert _speed(conn, SHARED) == 0


def test_generators_listing_reports_claims_per_channel(conn):
    fq.upsert_queries(conn, [_q(SHARED), _q("another thread")], generator=REGION, votable=False)
    fq.upsert_queries(conn, [_q(SHARED)], generator=BOOKMARKS, votable=True)
    by_name = {r["generator"]: r for r in fq.generators(conn)}
    assert by_name[REGION]["claims"] == 2
    assert by_name[BOOKMARKS]["claims"] == 1
    assert by_name[BOOKMARKS]["votable"] == 1


def test_the_reader_prompt_derives_its_source_list_and_never_restates_it():
    """`parse_response` silently drops any target_source outside `VALID_SOURCES` and tells the
    model nothing, so a prompt carrying its own copy of the list fails invisibly the moment the
    two disagree — which a second hand-maintained copy always does eventually. It sat at nine
    names in both places until openalex made it ten."""
    from pipeline.kb import reader_core as core
    from pipeline.kb import sitting_reader as sr

    assert "__VALID_SOURCES__" not in sr._SYSTEM, "the interpolation did not run"
    for name in core.VALID_SOURCES:
        assert name in sr._SYSTEM, f"{name} is routable but the reader is never told about it"
