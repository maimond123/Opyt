"""The ask path (`oracles.admit.record`, `windows.parse`): a tool's request becomes owed work, and nothing starts."""
from __future__ import annotations

import threading
from datetime import timedelta

import pytest

from pipeline.kb import schema
from pipeline.platforms import connections
from pipeline.worker.job_queue import URGENT_PRIORITY
from pipeline.timeparse import parse_ts, utc_now
from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles.asks import ask_for
from pipeline.oracles import windows
from pipeline.oracles import admit

from ._support import pair, stored


@pytest.fixture()
def woken(monkeypatch):
    calls = []
    monkeypatch.setattr("pipeline.worker.job_queue.request_now",
                        lambda rail, **kw: calls.append((rail, kw.get("priority"))) or True)
    return calls


@pytest.fixture(autouse=True)
def x_connected(monkeypatch):
    monkeypatch.setattr(connections, "_connected_x", lambda: True)


def _venue(conn, cid="openalex:S123", name="ChemRxiv"):
    schema.upsert_entity(conn, cid, name=name)
    roster.upsert_oracle(conn, cid, name=name)
    return cid


# ── windows ───────────────────────────────────────────────────────────────────────

def test_the_defaults_are_six_months_of_x_and_everything_else():
    win = windows.parse(now=utc_now())
    assert abs((utc_now() - win.x_from) - timedelta(days=183)) < timedelta(minutes=1)
    assert (win.web_from, win.scholar_from, win.check) == (windows.ALL_TIME, windows.ALL_TIME, False)


def test_all_is_an_instant_never_none():
    """None already means "nothing asked of this kind"; `record_ask` would read "all" as that."""
    win = windows.parse("2yr", "all", "all", now=utc_now())
    assert win.web_from == windows.ALL_TIME and win.scholar_from == windows.ALL_TIME
    assert parse_ts(windows.parse(web_lookback="1yr", now=utc_now()).web_from) > utc_now() - timedelta(days=366)


def test_an_unknown_x_preset_refuses_before_anything_is_written():
    with pytest.raises(ValueError):
        windows.parse("1y", now=utc_now())


def test_a_top_up_asks_for_nothing_it_does_not_name():
    win = windows.parse("since_last", now=utc_now())
    assert (win.check, win.x_from, win.web_from, win.scholar_from) == (True, None, None, None)
    assert not win.asks
    assert windows.parse("since_last", web_lookback="all", now=utc_now()).web_from == windows.ALL_TIME


# ── record ────────────────────────────────────────────────────────────────────────

def test_an_ask_is_recorded_and_the_scrapers_woken_and_nothing_starts(conn, woken):
    cid = pair(conn)
    threads = threading.active_count()

    out = admit.record(conn, [cid], windows.parse("1yr", now=utc_now()))

    ask = ask_for(conn, cid)
    assert parse_ts(ask.x_from) < utc_now() - timedelta(days=364)
    assert ask.web_from == windows.ALL_TIME and ask.scholar_from == windows.ALL_TIME
    assert woken == [("scrapers", URGENT_PRIORITY)]
    assert out == {"needs_topics": [], "checked": [], "woken": True}
    assert threading.active_count() == threads


def test_an_ask_only_widens(conn, woken):
    cid = pair(conn)
    admit.record(conn, [cid], windows.parse("2yr", now=utc_now()))
    wide = ask_for(conn, cid).x_from
    admit.record(conn, [cid], windows.parse("6mo", now=utc_now()))
    assert ask_for(conn, cid).x_from == wide


def test_a_tool_can_be_the_first_writer_on_a_fresh_store(kb_home, woken):
    """Only a scraper thread created these tables until now; a tool writing first would have hit
    "no such table"."""
    conn = schema.connect()
    schema.upsert_entity(conn, "x:user:9", name="Nia", profile={"handle": "nia"})
    roster.upsert_oracle(conn, "x:user:9", name="Nia")

    admit.record(conn, ["x:user:9"], windows.parse(now=utc_now()))

    assert ask_for(conn, "x:user:9") is not None
    assert [r.source_key for r in registry.list_sources(conn)] == ["nia"]   # registered at once
    conn.close()


def test_topics_are_stored_where_archives_reads_them(conn, woken):
    schema.upsert_entity(conn, "openalex:A42", name="Ada")
    roster.upsert_oracle(conn, "openalex:A42", name="Ada")

    admit.record(conn, ["openalex:A42"], windows.parse(now=utc_now()), scholar_topics=["T1", "T2"])

    assert registry.topic_filter_for(conn, "openalex:A42", "A42") == "T1|T2"


def test_a_venue_without_topics_gets_no_paper_ask(conn, woken):
    """A venue publishes far more than one pull holds."""
    cid = _venue(conn)
    out = admit.record(conn, [cid], windows.parse(now=utc_now()))
    assert out["needs_topics"] == [cid]
    assert ask_for(conn, cid).scholar_from is None

    out = admit.record(conn, [cid], windows.parse(now=utc_now()), scholar_topics=["T9"])
    assert out["needs_topics"] == []
    assert ask_for(conn, cid).scholar_from == windows.ALL_TIME


def test_since_last_is_a_check_not_an_ask(conn, woken):
    cid = pair(conn, last_pulled_at=(utc_now() - timedelta(days=3)).isoformat())
    admit.record(conn, [cid], windows.parse(now=utc_now()))
    before = ask_for(conn, cid)

    out = admit.record(conn, [cid], windows.parse("since_last", now=utc_now()))

    assert out["checked"] == [cid]
    assert stored(conn).check_requested_at is not None
    assert ask_for(conn, cid).asked_at == before.asked_at     # nothing re-asked


def test_a_top_up_of_someone_never_pulled_is_their_first_ask(conn, woken):
    cid = pair(conn)
    admit.record(conn, [cid], windows.parse("since_last", now=utc_now()))
    ask = ask_for(conn, cid)
    assert ask is not None and ask.web_from == windows.ALL_TIME and ask.x_from is not None


def test_an_ask_lets_its_milestones_be_told_again(conn, woken):
    cid = pair(conn)
    admit.record(conn, [cid], windows.parse("6mo", now=utc_now()))
    conn.executemany("INSERT INTO milestones_shown VALUES (?, ?, 'x')",
                     [(cid, m) for m in ("newest_posts", "archives", "reach", "stuck")])
    conn.commit()

    admit.record(conn, [cid], windows.parse("6mo", now=utc_now()))           # no wider on X: reach stays told
    shown = {r[0] for r in conn.execute("SELECT milestone FROM milestones_shown")}
    assert shown == {"newest_posts", "reach"}

    admit.record(conn, [cid], windows.parse("2yr", now=utc_now()))
    assert {r[0] for r in conn.execute("SELECT milestone FROM milestones_shown")} == {
        "newest_posts"}

