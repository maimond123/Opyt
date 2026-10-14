"""The recommendation pass on the web scraper (`background_scrapers.web.substack_recommendations`):
its weekly clock, a new Substack Oracle making it due sooner, and a lock leaving it due."""
from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest

from pipeline.curation import clock
from pipeline.curation import recommended as recs
from pipeline.worker.quota_loop import source

from .._support import NOW
from tests.kb.recommendation_support import run_pass
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles.sources.web import recommendations as web_recs


@pytest.fixture()
def substack(conn):
    roster.upsert_oracle(conn, "x:user:1", name="One")
    registry.upsert_source(conn, registry.SourceRow("x:user:1", "substack", "https://one.example.com",
                                        added_at="2026-09-01 00:00:00"))
    return conn


@pytest.fixture()
def passes(monkeypatch):
    ran = []
    monkeypatch.setattr(web_recs, "read_publication", lambda url: ran.append(url) or [])
    return ran


def _attempted(conn, when):
    clock.record_run(conn, recs.COLLECTOR, status="ok", now=when.isoformat())


def test_a_pass_is_due_weekly(substack):
    assert web_recs.FLOOR_HOURS == 7 * 24
    assert web_recs.due(substack, NOW)
    _attempted(substack, NOW)
    assert not web_recs.due(substack, NOW + timedelta(days=6))
    assert web_recs.due(substack, NOW + timedelta(days=7))
    assert web_recs.Recommendations().next_due(substack, NOW) == NOW + timedelta(days=7)


def test_a_substack_oracle_added_since_the_last_pass_makes_one_due(substack):
    _attempted(substack, NOW)
    roster.upsert_oracle(substack, "x:user:2", name="Two")
    registry.upsert_source(substack, registry.SourceRow("x:user:2", "substack", "https://two.example.com",
                                            added_at=(NOW + timedelta(hours=1)).strftime(
                                                "%Y-%m-%d %H:%M:%S")))
    assert web_recs.due(substack, NOW + timedelta(hours=2))


def test_a_lock_leaves_the_pass_due(substack, monkeypatch):
    def locked(url):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(web_recs, "read_publication", locked)
    with pytest.raises(sqlite3.OperationalError):
        run_pass(substack, now=NOW)
    assert clock.get_run(substack, recs.COLLECTOR) is None


def test_any_other_failure_counts_as_the_attempt(substack, monkeypatch):
    monkeypatch.setattr(web_recs, "read_publication",
                        lambda conn: (_ for _ in ()).throw(RuntimeError("HTTP 403")))
    assert run_pass(substack, now=NOW)["status"] == "partial"
    assert not web_recs.due(substack, NOW + timedelta(hours=1))


def test_the_source_is_scouting_and_waits_for_a_substack_oracle(conn, web_loop, passes):
    r = web_recs.Recommendations()
    assert r.klass == source.SCOUTING and r.floor == 0
    assert r.next_unit(conn, NOW) is None
    roster.upsert_oracle(conn, "x:user:1", name="One")
    registry.upsert_source(conn, registry.SourceRow("x:user:1", "substack", "https://one.example.com",
                                        added_at=(NOW - timedelta(hours=1)).isoformat()))
    assert run_pass(conn, now=NOW)["status"] == "ok"
    assert passes == ["https://one.example.com"] and r.next_unit(conn, NOW) is None


def test_restart_does_not_reread_publication_and_new_pass_keeps_corroboration(substack, passes):
    r = web_recs.Recommendations()
    ctx = source.Context("web")
    unit = r.next_unit(substack, NOW)
    assert r.record(substack, unit, r.fetch(substack, unit, ctx)).status == "started"
    unit = r.next_unit(substack, NOW)
    assert r.record(substack, unit, r.fetch(substack, unit, ctx)).status == "ok"
    assert web_recs.Recommendations().next_unit(substack, NOW).key == "recommendations:finish"
    assert run_pass(substack, NOW)["status"] == "ok"
    assert passes == ["https://one.example.com"]
