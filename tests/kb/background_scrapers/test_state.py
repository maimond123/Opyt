"""The background scrapers' stored state: the additive columns and tables, asks, scraper state,
and the seam to the write helper another workstream owns (`write_tx`)."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from pipeline.kb import writes
from pipeline.worker.quota_loop import loop_state
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles.asks import ask_for, ask_for_head, asks_by_head, list_asks, record_ask
from pipeline.oracles.found import list_found
from pipeline.oracles import windows
from pipeline.oracles.sources import scraper_tables

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    scraper_tables.init_schema(c)
    yield c
    c.close()


def _ago(days: int) -> datetime:
    return NOW - timedelta(days=days)


# ── additive schema ───────────────────────────────────────────────────────────────

def test_init_schema_is_idempotent(conn):
    scraper_tables.init_schema(conn)
    scraper_tables.init_schema(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"oracle_asks", "milestones_shown"} <= tables
    assert "scraper_state" not in tables, "a loop's state lives in loops.db"


# ── asks ──────────────────────────────────────────────────────────────────────────

def test_a_narrower_ask_never_takes_back_a_wider_one(conn):
    record_ask(conn, "a", x_from=_ago(365), now=NOW)
    record_ask(conn, "a", x_from=_ago(183), now=NOW + timedelta(hours=1))
    ask = ask_for(conn, "a")
    assert ask.x_from == "2025-10-03T12:00:00+00:00"
    assert ask.asked_at == "2026-10-03T13:00:00+00:00"          # every ask moves it

    record_ask(conn, "a", x_from=_ago(730), now=NOW)
    assert ask_for(conn, "a").x_from == "2024-10-03T12:00:00+00:00"


def test_an_ask_with_nothing_for_a_kind_leaves_that_kind_alone(conn):
    record_ask(conn, "a", x_from=_ago(365), web_from=windows.ALL_TIME, now=NOW)
    record_ask(conn, "a", scholar_from=_ago(730), now=NOW)
    ask = ask_for(conn, "a")
    assert ask.x_from == "2025-10-03T12:00:00+00:00"
    assert ask.web_from == windows.ALL_TIME
    assert ask.scholar_from == "2024-10-03T12:00:00+00:00"


def test_all_time_is_wider_than_any_date(conn):
    record_ask(conn, "a", web_from=windows.ALL_TIME, now=NOW)
    record_ask(conn, "a", web_from=_ago(365), now=NOW)
    assert ask_for(conn, "a").web_from == windows.ALL_TIME


def test_picks_keep_the_order_they_were_first_asked_in(conn):
    for cid in ("a", "b", "c"):
        record_ask(conn, cid, x_from=_ago(183), now=NOW)
    record_ask(conn, "a", x_from=_ago(365), now=NOW)
    assert [(a.canonical_id, a.position) for a in list_asks(conn)] == [
        ("a", 1), ("b", 2), ("c", 3)]


def test_rediscover_stays_set_until_discovery_clears_it(conn):
    record_ask(conn, "a", rediscover=True, now=NOW)
    record_ask(conn, "a", now=NOW)
    assert ask_for(conn, "a").rediscover is True


def test_an_unreadable_window_is_refused_not_dropped(conn):
    with pytest.raises(ValueError):
        record_ask(conn, "a", x_from="last spring", now=NOW)
    assert ask_for(conn, "a") is None


# ── scraper state ─────────────────────────────────────────────────────────────────

def test_since_survives_a_recheck_that_finds_the_same_state(conn):
    loop_state.set_state("q", "paused", cause="x_session", now=100.0)
    loop_state.set_state("q", "paused", cause="x_session", now=200.0)
    row = loop_state.read_state("q")
    assert (row["since"], row["beat_at"]) == (100.0, 200.0)

    loop_state.set_state("q", "running", now=300.0)
    assert loop_state.read_state("q")["since"] == 300.0


# ── the write seam ────────────────────────────────────────────────────────────────

def test_write_tx_commits_and_rolls_back(conn):
    with writes.write_tx(conn):
        conn.execute("INSERT INTO milestones_shown VALUES ('a', 'reach', 'now')")
    with pytest.raises(RuntimeError):
        with writes.write_tx(conn):
            conn.execute("INSERT INTO milestones_shown VALUES ('b', 'reach', 'now')")
            raise RuntimeError("boom")
    assert [r[0] for r in conn.execute("SELECT canonical_id FROM milestones_shown")] == ["a"]
    assert not conn.in_transaction


def test_write_tx_ends_a_transaction_left_open(conn):
    conn.execute("INSERT INTO milestones_shown VALUES ('leaked', 'reach', 'now')")
    assert conn.in_transaction
    with writes.write_tx(conn):
        pass
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM milestones_shown").fetchone()[0] == 1


# ── found sources, and an ask re-arming what gave up ──────────────────────────────

@pytest.fixture()
def store(kb_home):
    from pipeline.kb import schema
    c = schema.connect()
    registry.init_state_schema(c)
    scraper_tables.init_schema(c)
    yield c
    c.close()


def test_a_found_source_round_trips_to_the_shape_discovery_gave(store):
    store.execute("INSERT INTO found_sources (canonical_id, source_type, url, metadata, found_at) "
                  "VALUES ('x:user:1', 'blog', 'https://a.test', ?, '2026-10-03T12:00:00Z')",
                  ('{"trust": {"trusted": true}, "metadata": {"shape": "personal"}}',))
    store.commit()
    (found,) = list_found(store, "x:user:1")
    assert found.pulled_at is None
    assert found.source() == {"source_type": "blog", "url": "https://a.test",
                              "trust": {"trusted": True}, "metadata": {"shape": "personal"}}


def test_an_ask_re_arms_everything_of_that_oracle_and_nothing_else(store):
    for cid in ("x:user:1", "x:user:2"):
        roster.upsert_oracle(store, cid, name=cid)
        registry.upsert_source(store, registry.SourceRow(cid, "x", cid[-1]))
        store.execute("INSERT INTO found_sources (canonical_id, source_type, url, found_at, "
                      "failures, retry_at) VALUES (?, 'blog', 'https://a.test', 'now', 10, NULL)",
                      (cid,))
    record_ask(store, "x:user:2", web_from=_ago(30), now=NOW)
    store.execute("UPDATE oracle_sources SET failures=10, retry_at=NULL, last_error='gone'")
    store.execute("UPDATE found_sources SET failures=10")
    store.execute("UPDATE oracle_asks SET discovery_failures=10")
    store.commit()
    record_ask(store, "x:user:1", web_from=_ago(30), now=NOW)

    pairs = {r.canonical_id: r for r in registry.list_sources(store)}
    assert pairs["x:user:1"].failures == 0 and pairs["x:user:1"].last_error == "gone"
    assert pairs["x:user:2"].failures == 10
    found = {f.canonical_id: f.failures for f in list_found(store)}
    assert found == {"x:user:1": 0, "x:user:2": 10}
    asks = {a.canonical_id: a.discovery_failures for a in list_asks(store)}
    assert asks == {"x:user:1": 0, "x:user:2": 10}


def test_an_ask_is_found_under_its_oracles_current_head(store):
    from pipeline.kb import schema
    schema.upsert_entity(store, "substack:alice", name="Alice")
    schema.upsert_entity(store, "x:user:9", name="Alice")
    record_ask(store, "substack:alice", web_from=_ago(30), now=NOW)
    store.execute("UPDATE entities SET canonical_id='x:user:9'")
    store.commit()
    assert ask_for(store, "x:user:9") is None
    assert ask_for_head(store, "x:user:9").canonical_id == "substack:alice"
    assert set(asks_by_head(store)) == {"x:user:9"}


@pytest.mark.parametrize("beat_at, expected", [
    (None, False),                                       # never beat
    (1000.0 - 30, True),                                 # a beat ago
    (1000.0 - loop_state.STALE_BEAT_S, True),            # at the bound
    (1000.0 - loop_state.STALE_BEAT_S - 1, False),       # silent past it
])
def test_is_a_loop_still_beating(beat_at, expected):
    assert loop_state.is_beating(beat_at, 1000.0) is expected
