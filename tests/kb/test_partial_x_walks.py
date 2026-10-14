"""A partial X walk claims exactly what it walked: the coverage writers' rule.

A walk cut short writes atoms, and the writers must not stamp the window that was asked for onto
a pull that never covered it. `covered_from` only ever widens (MIN), and None means "nothing to report", never "reset". These
pin the two writers that apply it: `coverage.record_coverage` and `oracle_refresh.save_pull`.

The X timeline scraper walks one page per unit and keeps its own frontier
(`tests/kb/background_scrapers/x_timeline/test_oracle_walks.py`).
"""
from __future__ import annotations


import pytest

from pipeline.kb import schema
from pipeline.oracles import coverage


@pytest.fixture()
def conn(kb_home, tmp_path):
    c = schema.connect(tmp_path / "opyt.db")
    yield c
    c.close()


# ── the store records what was reached, not what was asked ────────────────────
#
# Recording the requested window unconditionally is harmless only while a complete walk is the
# sole writer (asked == reached), and becomes a lie once a partial walk can write atoms.
# `record_pull` documents `covered_from` as "the oldest instant this pull reached", only ever
# widening (MIN), with None meaning "no lower bound to report".

def _pair(conn, cid="x:user:99", handle="carol"):
    from pipeline.kb import resolve
    from pipeline.oracles import registry
    schema.upsert_entity(conn, cid, name="Carol", profile={"handle": handle})
    resolve.resolve_entities(conn)
    registry.upsert_source(conn, registry.SourceRow(canonical_id=cid, source_type="x",
                                        source_key=handle, status="trusted"))
    return cid


def _stored(conn, cid):
    from pipeline.oracles import registry
    return next(r for r in registry.list_sources(conn, canonical_ids=[cid])).covered_from


def _record(conn, cid, rec, *, asked_days=183):
    from datetime import timedelta
    from pipeline.timeparse import utc_now
    coverage.record_coverage(conn, cid, [rec], x_since=utc_now() - timedelta(days=asked_days),
                             web_since=None)


def _x_rec(**kw):
    return {"url": "https://x.com/carol", "type": "x", "action": "ingested", **kw}


def test_a_measured_reach_beats_the_window_that_was_asked_for(conn):
    cid = _pair(conn)

    _record(conn, cid, _x_rec(covered_from="2026-03-01T00:00:00+00:00"))

    assert _stored(conn, cid).startswith("2026-03-01")


def test_a_partial_walk_records_no_frontier_at_all(conn):
    """Presence, not truthiness: `rec["covered_from"] or asked` would fall straight back to
    the requested window here, which is the claim a partial walk must refuse."""
    cid = _pair(conn)

    _record(conn, cid, _x_rec(covered_from=None))

    assert _stored(conn, cid) is None


def test_a_source_that_measures_nothing_still_falls_back_to_the_asked_window(conn):
    """Substack, blog and OpenAlex are bounded by a snapshot hash, not a meter, so asked and
    reached cannot diverge for them, so falling back to the asked window is right."""
    from pipeline.kb import resolve
    from pipeline.oracles import registry
    schema.upsert_entity(conn, "x:user:98", name="Dan", profile={"handle": "dan"})
    resolve.resolve_entities(conn)
    registry.upsert_source(conn, registry.SourceRow(canonical_id="x:user:98", source_type="x",
                                        source_key="dan", status="trusted"))

    _record(conn, "x:user:98",
            {"url": "https://x.com/dan", "type": "x", "action": "ingested"}, asked_days=10)

    assert _stored(conn, "x:user:98") is not None


def test_a_shallow_partial_never_raises_an_existing_deeper_floor(conn):
    """The MIN merge, which `record_pull` already implements — this is the caller proving it uses
    it. A background pass that only reached March must not undo a walk that reached January."""
    cid = _pair(conn)
    _record(conn, cid, _x_rec(covered_from="2026-01-01T00:00:00+00:00"))

    _record(conn, cid, _x_rec(covered_from="2026-06-01T00:00:00+00:00"))

    assert _stored(conn, cid).startswith("2026-01-01")


def test_a_partial_after_a_complete_walk_leaves_the_floor_where_it_was(conn):
    """The case that would otherwise erase coverage: a later partial reports None, and None means
    "nothing to report", never "reset"."""
    cid = _pair(conn)
    _record(conn, cid, _x_rec(covered_from="2026-01-01T00:00:00+00:00"))

    _record(conn, cid, _x_rec(covered_from=None))

    assert _stored(conn, cid).startswith("2026-01-01")


def test_save_pull_records_a_partial_walk_the_same_way(conn):
    """Two writers, one rule. A `summary.get(...) or since` shape would hide the same bug as
    `_record_coverage`: a partial walk reports `covered_from: None` deliberately, and `or` would
    send exactly that case down the fallback, stamping the window we asked for onto a pull that
    never covered it. `save_pull` is the refresh writer, for the web scraper."""
    from datetime import timedelta

    from pipeline.oracles import registry
    from pipeline.timeparse import utc_now

    cid = _pair(conn, cid="x:user:88", handle="dana")
    row = next(r for r in registry.list_sources(conn, canonical_ids=[cid]) if r.source_type == "x")
    summary = {"source": "x-footprint", "added": 4, "partial": True, "covered_from": None}

    out = coverage.save_pull(conn, row, summary, "ingested", None,
                        since=utc_now() - timedelta(days=183))

    assert out["status"] == "ingested" and out["new_atoms"] == 4
    assert _stored(conn, cid) is None                 # atoms landed; no frontier was claimed
