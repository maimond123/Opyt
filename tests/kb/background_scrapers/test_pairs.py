"""The pairs the scrapers walk (`background_scrapers.pairs`), and what a failure costs one."""
from __future__ import annotations

from datetime import timedelta

from pipeline.timeparse import parse_ts
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import Outcome
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles.sources import pair_rows

from ._support import NOW, pair, stored


def test_walked_pairs_are_a_confirmed_oracles_of_the_types_asked_for(conn):
    cid = pair(conn, "carol")
    registry.upsert_source(conn, registry.SourceRow(cid, "substack", "https://carol.substack.com"))
    pair(conn, "dana", status="paused")
    registry.upsert_source(conn, registry.SourceRow("x:user:gone", "x", "ghost"))       # no Oracle
    assert [r.source_key for r in pair_rows.walked(conn, ("x",))] == ["carol"]
    assert [r.source_type for r in pair_rows.walked(conn, ("x", "substack"), cid)] == ["substack",
                                                                                   "x"]


def test_a_pair_waiting_or_given_up_is_not_free(conn):
    pair(conn, "carol", retry_at=(NOW + timedelta(minutes=5)).isoformat())
    pair(conn, "dana", failures=retries.GIVE_UP_AFTER, last_status=retries.BLIP)
    pair(conn, "erin", failures=retries.GIVE_UP_AFTER - 1, last_status=retries.BLIP)
    rows = pair_rows.walked(conn, ("x",))
    assert [r.source_key for r in pair_rows.free(rows, NOW)] == ["erin"]
    assert [r.source_key for r in pair_rows.gave_up(rows)] == ["dana"]
    assert pair_rows.next_free(rows, NOW) == NOW + timedelta(minutes=5)


def _charge(conn, row, kind, detail):
    """Charge a failure on the pair as the loop does; returns its item, whose prior gives it back."""
    item = pair_rows.item(row, "x")
    retries.charge(conn, item, Outcome(kind, failure=kind, detail=detail),
                   retries.Outages(retries.OUTAGE_PAUSES_S), NOW)
    return item


def test_a_charge_is_recorded_on_the_row_and_its_undo_puts_it_back(conn, no_jitter):
    pair(conn, "carol", failures=2, last_status=retries.BLIP, last_error="before")
    item = _charge(conn, stored(conn), retries.BLIP, "HTTP 503")
    row = stored(conn)
    assert (row.failures, row.last_status, row.last_error) == (3, retries.BLIP, "HTTP 503")
    assert parse_ts(row.retry_at) == NOW + timedelta(minutes=5)
    pair_rows.PAIR_ROWS.write(conn, item.key, item.prior)
    row = stored(conn)
    assert (row.failures, row.retry_at, row.last_error) == (2, None, "before")


def test_the_tenth_failure_in_a_row_gives_up_and_clears_the_retry(conn):
    pair(conn, "carol", failures=retries.GIVE_UP_AFTER - 1, last_status=retries.ACCOUNT,
         retry_at=NOW.isoformat())
    _charge(conn, stored(conn), retries.ACCOUNT, "suspended")
    row = stored(conn)
    assert row.failures == retries.GIVE_UP_AFTER and row.retry_at is None
    assert pair_rows.free([row], NOW + timedelta(days=30)) == []
    assert roster.list_oracles(conn)                       # the Oracle itself is untouched
