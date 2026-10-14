"""Check-on-ask (`oracles.checks`): a check stamps the Oracle's X pairs and wakes the scrapers.
Where it stands is `progress`'s to say."""
from __future__ import annotations

from pipeline.oracles import registry
from pipeline.oracles import checks

from ._support import NOW, pair, stored


def test_a_check_stamps_the_pairs_and_wakes_the_scrapers(conn, monkeypatch):
    woken = []
    monkeypatch.setattr("pipeline.worker.job_queue.request_now",
                        lambda rail, **kw: woken.append((rail, kw)) or True)
    cid = pair(conn, last_pulled_at=NOW.isoformat())
    checks.request_check(conn, cid, now=NOW)
    assert woken and woken[0][0] == "scrapers"
    assert stored(conn).check_requested_at == NOW.isoformat()


def test_a_check_counts_only_pairs_the_scraper_walks(conn, monkeypatch):
    """A paused pair, or one whose Oracle is gone, is never walked: a check of it would wait
    forever, so it wakes nothing."""
    woken = []
    monkeypatch.setattr("pipeline.worker.job_queue.request_now",
                        lambda rail, **kw: woken.append(rail) or True)
    checks.request_check(conn, pair(conn, status="paused"), now=NOW)
    registry.upsert_source(conn, registry.SourceRow("x:user:gone", "x", "ghost"))
    checks.request_check(conn, "x:user:gone", now=NOW)
    assert woken == []


def test_a_check_on_a_store_no_scraper_has_run_on_stamps_without_raising(kb_home, monkeypatch):
    """Found by freshness against 5a: the scrapers' tables exist only once one has run, and a
    fresh install is exactly where a check must still land."""
    from pipeline.kb import schema
    monkeypatch.setattr("pipeline.worker.job_queue.request_now", lambda *a, **k: True)
    conn = schema.connect()
    pair(conn)
    checks.request_check(conn, stored(conn).canonical_id, now=NOW)
    assert stored(conn).check_requested_at == NOW.isoformat()
    conn.close()
