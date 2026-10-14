"""`pipeline/frontier/cadence.py` — when a pair is due and what window it asks for, one test per
row of each decision's table.

Plain values in, an answer out: no store, no clock, no adapter beyond an object with one attribute.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.frontier import cadence

_NOW = datetime(2026, 8, 10, 12, 0, 0, tzinfo=timezone.utc)


def _mark(last_pulled_at, tried_at=None):
    return {"last_pulled_at": last_pulled_at, "tried_at": tried_at}


def _ttl_hours(query_id, source):
    return cadence.TTL_HOURS * cadence.jitter(query_id, source)


# ── is_due ──────────────────────────────────────────────────────────────────────
def test_a_pair_with_no_watermark_is_due():
    assert cadence.is_due(None, "arxiv", query_id="q", now=_NOW) is True


@pytest.mark.parametrize("stamp", [None, "", "not a date"])
def test_a_watermark_with_no_readable_last_pull_is_due(stamp):
    assert cadence.is_due(_mark(stamp), "arxiv", query_id="q", now=_NOW) is True


def test_a_pair_pulled_at_least_its_ttl_ago_is_due():
    ttl = _ttl_hours("q", "github")
    last = _NOW - timedelta(hours=ttl)
    assert cadence.is_due(_mark(last.isoformat()), "github", query_id="q", now=_NOW) is True


def test_a_pair_tried_since_its_last_pull_waits_an_hour_from_the_try():
    """A failed request or a refused window: what the hourly pass gave it."""
    last, tried = _NOW - timedelta(days=3), _NOW - timedelta(minutes=30)
    mark = _mark(last.isoformat(), tried.isoformat())
    assert cadence.due_at(mark, "arxiv", query_id="q") == tried + cadence.RETRY_AFTER
    assert cadence.is_due(mark, "arxiv", query_id="q", now=_NOW) is False
    assert cadence.is_due(mark, "arxiv", query_id="q", now=tried + cadence.RETRY_AFTER) is True


def test_a_successful_pull_stamps_both_so_its_ttl_governs():
    last = _NOW - timedelta(hours=1)
    mark = _mark(last.isoformat(), last.isoformat())
    assert cadence.due_at(mark, "arxiv", query_id="q") == last + timedelta(
        hours=_ttl_hours("q", "arxiv"))


def test_a_pair_pulled_less_than_its_ttl_ago_is_not_due():
    ttl = _ttl_hours("q", "github")
    last = _NOW - timedelta(hours=ttl - 1)
    assert cadence.is_due(_mark(last.isoformat()), "github", query_id="q", now=_NOW) is False


# ── since_for ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mark", [None, _mark(None), _mark("garbage")])
def test_a_pair_with_no_readable_stamp_looks_back_the_first_pull_window(mark):
    assert cadence.since_for(mark, now=_NOW) == _NOW - timedelta(days=cadence.FIRST_PULL_DAYS)


def test_a_stamped_pair_resumes_an_overlap_behind_its_last_pull():
    last = _NOW - timedelta(days=2)
    assert (cadence.since_for(_mark(last.isoformat()), now=_NOW)
            == last - timedelta(hours=cadence.OVERLAP_HOURS))


# ── lookback_floor ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("adapter", [object(), type("A", (), {"min_lookback_days": 0})()])
def test_an_adapter_declaring_no_lookback_keeps_its_since(adapter):
    since = _NOW - timedelta(hours=30)
    assert cadence.lookback_floor(adapter, since, _NOW) == since


def test_a_declared_lookback_reaches_back_that_many_days_but_never_forward():
    wide = type("A", (), {"min_lookback_days": 30})()
    assert cadence.lookback_floor(wide, _NOW - timedelta(hours=30), _NOW) == _NOW - timedelta(days=30)
    older = _NOW - timedelta(days=45)
    assert cadence.lookback_floor(wide, older, _NOW) == older


# ── window_ok ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("days", [0, 14, cadence.MAX_WINDOW_DAYS])
def test_a_window_of_zero_to_sixty_days_is_ok(days):
    assert cadence.window_ok(_NOW - timedelta(days=days), _NOW) is True


@pytest.mark.parametrize("days", [-1, cadence.MAX_WINDOW_DAYS + 1])
def test_a_negative_or_over_sixty_day_window_is_refused(days):
    assert cadence.window_ok(_NOW - timedelta(days=days), _NOW) is False


# ── plan ────────────────────────────────────────────────────────────────────────
def _q(qid, *sources):
    return {"query_id": qid, "target_sources": json.dumps(list(sources))}


def _pairs(pairs):
    return [(p.query["query_id"], p.source) for p in pairs]


def test_a_pair_with_no_adapter_is_reported_and_never_due():
    p = cadence.plan([_q("q1", "biorxiv", "arxiv")], {}, {"arxiv"}, now=_NOW)
    assert _pairs(p.no_adapter) == [("q1", "biorxiv")]
    assert _pairs(p.due) == [("q1", "arxiv")]


def test_due_pairs_come_stalest_first_and_a_never_pulled_pair_first_of_all():
    marks = {("old", "arxiv"): _mark((_NOW - timedelta(days=30)).isoformat()),
             ("older", "arxiv"): _mark((_NOW - timedelta(days=40)).isoformat())}
    p = cadence.plan([_q("old", "arxiv"), _q("older", "arxiv"), _q("never", "arxiv")], marks,
                     {"arxiv"}, now=_NOW)
    assert _pairs(p.due) == [("never", "arxiv"), ("older", "arxiv"), ("old", "arxiv")]
    assert p.due[0].mark is None and p.due[1].mark is marks[("older", "arxiv")]


def test_a_pair_with_an_adapter_that_is_not_due_is_left_out():
    marks = {("fresh", "arxiv"): _mark((_NOW - timedelta(hours=1)).isoformat())}
    p = cadence.plan([_q("fresh", "arxiv")], marks, {"arxiv"}, now=_NOW)
    assert p.due == [] and p.no_adapter == []
