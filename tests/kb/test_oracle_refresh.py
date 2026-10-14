"""Oracle refresh: the freshness snapshot (`status_summary`, on the scrapers' failure records),
the deepen target, and the persistence rules for one pull (`save_pull`, and the foreground
writer's copy of the connectivity rule).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from pipeline.kb import schema
from pipeline.oracles import windows
from pipeline.platforms import connections
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import Retry
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles.pairs import seed_from_entities
from pipeline.oracles import coverage
from pipeline.oracles import freshness
from pipeline.oracles.sources import pair_rows
from pipeline.oracles.sources import scraper_tables

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)


# ── fixtures ────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def managed_x_session(monkeypatch):
    """Seeding registers the X pair only with a connected X session."""
    monkeypatch.setattr(connections, "_connected_x", lambda: True)


@pytest.fixture()
def store(kb_home):
    conn = schema.connect()
    registry.init_state_schema(conn)
    schema.upsert_entity(conn, "x:user:1", name="Will", identity_links=["https://willcb.com"], profile={"handle": "willccbb"})
    schema.upsert_entity(conn, "blog:willcb.com", name="Will", identity_links=["https://willcb.com"])
    schema.set_canonical_ids(conn, {"x:user:1": "x:user:1", "blog:willcb.com": "x:user:1"})
    roster.upsert_oracle(conn, "x:user:1", name="Will")
    seed_from_entities(conn)
    yield conn
    conn.close()


def _pair(conn, stype, key=None):
    return next(r for r in registry.list_sources(conn)
                if r.source_type == stype and (key is None or r.source_key == key))


def _stamp(conn, stype, hours_ago, key=None):
    row = _pair(conn, stype, key)
    when = (NOW - timedelta(hours=hours_ago)).isoformat()
    registry.record_pull(conn, row, last_status="ingested", cursor_ts=when, stamp=True, now=when)
    return _pair(conn, stype, key)


def _sources(out, stype):
    return [s for o in out["oracles"] for s in o["sources"] if s["source_type"] == stype]


# ── status ──────────────────────────────────────────────────────────────────────
def test_status_surfaces_a_frozen_oracle(store):
    _stamp(store, "x", 1)
    out = freshness.status_summary(store)
    assert out["tracked_pairs"] == 2
    [blog] = _sources(out, "blog")
    assert blog["never_refreshed"] is True and blog["stale"] is True
    # The effective ttl, so the report and the gate cannot disagree.
    assert abs(blog["ttl_hours"] - 336.0) <= 336.0 * windows.TTL_JITTER
    assert blog["ttl_hours"] != 336.0 or windows.TTL_JITTER == 0


def test_status_degrades_rather_than_raising(store, monkeypatch):
    monkeypatch.setattr(registry, "list_sources", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("x")))
    out = freshness.status_summary(store)
    assert out["tracked_pairs"] == 0 and "error" in out


def test_a_failing_pair_reports_the_scrapers_failure_record(store):
    """What a pair's failures cost is the scrapers' retry policy, recorded on its own row."""
    retry = (NOW + timedelta(hours=6)).isoformat()
    pair_rows.PAIR_ROWS.write(store, pair_rows.pair_key(_pair(store, "blog")),
                          Retry(retries.ACCOUNT, 2, retry, "403 Forbidden"))
    pair_rows.PAIR_ROWS.write(store, pair_rows.pair_key(_pair(store, "x")),
                          Retry(retries.BLIP, retries.GIVE_UP_AFTER, None, "timeout"))

    out = freshness.status_summary(store)

    [blog], [x] = _sources(out, "blog"), _sources(out, "x")
    assert (blog["failures"], blog["given_up"], blog["retry_at"], blog["last_error"]) == (
        2, False, retry, "403 Forbidden")
    assert x["given_up"] is True and x["retry_at"] is None
    assert "breaker_open" not in blog


def test_a_pair_with_no_failures_carries_no_failure_record(store):
    out = freshness.status_summary(store)
    assert all("given_up" not in s for o in out["oracles"] for s in o["sources"])


def test_a_found_source_whose_first_pull_keeps_failing_is_reported(store):
    """A found source has no pair until its first pull lands, so the pair rows are blind to it.
    One merely owed a first pull is not reported here: that is the archives milestone's."""
    scraper_tables.init_schema(store)
    store.executemany(
        "INSERT INTO found_sources (canonical_id, source_type, url, found_at, last_status, "
        "failures, retry_at, last_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [("x:user:1", "substack", "https://will.substack.com", NOW.isoformat(), "blip", 3,
          (NOW + timedelta(minutes=5)).isoformat(), "Cloudflare challenge"),
         ("x:user:1", "github", "https://github.com/willccbb", NOW.isoformat(), None, None,
          None, None)])
    store.commit()

    out = freshness.status_summary(store)

    [found] = _sources(out, "substack")
    assert found["source_key"] == "https://will.substack.com"
    assert found["never_refreshed"] is True and found["last_error"] == "Cloudflare challenge"
    assert _sources(out, "github") == []                  # owed, not failing
    assert out["tracked_pairs"] == 2                       # a found source is not a pair


def test_needs_attention_is_the_stale_fraction_alone(store):
    """No consent condition: Opyt never asks."""
    out = freshness.status_summary(store)
    assert out["stale_fraction"] == 1.0 and out["needs_attention"] is True
    assert "consented" not in out and "background scrapers" in out["note"]

    for stype in ("x", "blog"):                            # pulled just now
        registry.record_pull(store, _pair(store, stype), last_status="ingested", stamp=True)
    out = freshness.status_summary(store)
    assert out["needs_attention"] is False and "note" not in out


# ── the window the user actually asked for ────────

def test_target_for_prefers_the_deeper_of_asked_and_default():
    """`BACKFILL_TARGET_DAYS` is the floor for pairs nobody asked about — its own comment says
    a 2-year window belongs to a user who asks for it. `requested_from` is the durable record
    that someone did."""
    default = windows.deepen_target(NOW)
    row = registry.SourceRow("x:user:1", "x", "p1")

    assert windows.target_for(row, NOW) == default                     # nobody asked
    row.requested_from = (NOW - timedelta(days=365)).isoformat()
    assert windows.target_for(row, NOW) == NOW - timedelta(days=365)   # deeper ask honoured
    row.requested_from = (NOW - timedelta(days=30)).isoformat()
    assert windows.target_for(row, NOW) == default                     # shallower ask: floor holds


# ── the connectivity rule ──────────────────────────────────────

def _with_frontier(conn, *, days_back, pulled_hours_ago=100):
    """A pair with a frontier and a last pull. `record_pull` mutates the row it is handed, so
    every test below captures the values it compares against before the call under test."""
    row = _pair(conn, "x")
    pulled = (NOW - timedelta(hours=pulled_hours_ago)).isoformat()
    registry.record_pull(conn, row, last_status="ingested", stamp=True, now=pulled,
                   covered_from=(NOW - timedelta(days=days_back)).isoformat())
    return _pair(conn, "x")


def _cut_short(days_back):
    """What a top-started walk that stopped `days_back` down returns."""
    return {"source": "x-footprint", "added": 4, "partial": True, "started_at_top": True,
            "covered_from": (NOW - timedelta(days=days_back)).isoformat()}


def test_a_top_started_walk_cut_short_above_the_last_pull_claims_nothing(store):
    """The connectivity rule. Last pulled 60 days ago; a top-started walk stopped 10 days down.
    `[60d, 10d)` was never read — claiming 10d would leave the older frontier standing (MIN), and
    stamping would move the forward window past the gap for good."""
    row = _with_frontier(store, days_back=120, pulled_hours_ago=60 * 24)
    pulled, frontier = row.last_pulled_at, row.covered_from

    coverage.save_pull(store, row, _cut_short(10), "ingested", None, since=NOW - timedelta(days=61))

    after = _pair(store, "x")
    assert after.last_pulled_at == pulled                    # no stamp…
    assert after.covered_from == frontier                    # …and no claim
    assert after.last_status == "ingested"                   # the atoms still landed


def test_a_top_started_walk_cut_short_below_the_last_pull_is_connected(store):
    row = _with_frontier(store, days_back=120, pulled_hours_ago=5 * 24)
    pulled, frontier = row.last_pulled_at, row.covered_from

    coverage.save_pull(store, row, _cut_short(10), "ingested", None, since=NOW - timedelta(days=6))

    after = _pair(store, "x")
    assert after.last_pulled_at != pulled                    # stamped
    assert after.covered_from == frontier                    # MIN: the deeper frontier stands


def test_the_foreground_writer_applies_the_same_connectivity_rule(store):
    row = _with_frontier(store, days_back=120, pulled_hours_ago=60 * 24)
    pulled, frontier = row.last_pulled_at, row.covered_from
    rec = {"type": "x", "url": "https://x.com/willccbb", "action": "ingested", "partial": True,
           "covered_from": (NOW - timedelta(days=10)).isoformat()}

    coverage.record_coverage(store, "x:user:1", [rec], x_since=NOW - timedelta(days=183),
                             web_since=None)

    after = _pair(store, "x")
    assert after.last_pulled_at == pulled and after.covered_from == frontier


def test_the_foreground_writer_still_stamps_a_connected_partial_walk(store):
    row = _with_frontier(store, days_back=120, pulled_hours_ago=5 * 24)
    pulled = row.last_pulled_at
    rec = {"type": "x", "url": "https://x.com/willccbb", "action": "ingested", "partial": True,
           "covered_from": (NOW - timedelta(days=10)).isoformat()}

    coverage.record_coverage(store, "x:user:1", [rec], x_since=NOW - timedelta(days=183),
                             web_since=None)

    assert _pair(store, "x").last_pulled_at != pulled
