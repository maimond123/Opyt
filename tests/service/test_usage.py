"""`usage_daily` — the shape of what a served read leaves behind.

The ruling (docs/plans/2026-08-27-what-opyt-collects.md): the intrusive part of a request log is
the JOIN — who read from whom and when, at full resolution — not the facts. One row per (day,
owner, reader, tool) keeps every metric the stats page reads and drops the trace. These tests pin
the aggregation itself, because a table that happens to hold one row per request today, and would
hold one per request under any traffic, is an event log wearing a different name.
"""
from __future__ import annotations

from service import store


def _rows():
    conn = store.connect()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM usage_daily ORDER BY tool")]
    finally:
        conn.close()


def test_same_day_reads_aggregate_to_one_row(svc):
    """Two searches, one row. This is the whole point of the shape: the second read increments a
    counter rather than appending a timestamped record of when it happened."""
    for _ in range(2):
        svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"},
                        headers=svc.reader_hdr)

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["owner"] == svc.owner
    assert rows[0]["reader"] == store.token_hash(svc.reader_token)
    assert rows[0]["tool"] == "search"
    assert rows[0]["n"] == 2
    assert store.usage_total(svc.owner) == 2


def test_each_tool_gets_its_own_row(svc):
    svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"}, headers=svc.reader_hdr)
    svc.client.post(f"/v1/kb/{svc.owner}/aggregate", json={}, headers=svc.reader_hdr)

    assert [r["tool"] for r in _rows()] == ["aggregate", "search"]
    assert store.usage_total(svc.owner) == 2
    assert store.usage_total(svc.owner, store.token_hash(svc.reader_token)) == 2


def test_zero_results_counted(svc):
    """The funnel number the stats page reads. A search that returned nothing still counts as a
    read; what it adds on top is the zero-result flag, and a search WITH hits must not."""
    svc.client.post(f"/v1/kb/{svc.owner}/search",
                    json={"query": "zzz-nothing-matches-this-zzz", "mode": "bm25"},
                    headers=svc.reader_hdr)
    svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent", "mode": "bm25"},
                    headers=svc.reader_hdr)

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["n"] == 2
    assert rows[0]["zero_results"] == 1


def test_a_non_search_tool_never_sets_zero_results(svc):
    svc.client.post(f"/v1/kb/{svc.owner}/aggregate", json={}, headers=svc.reader_hdr)
    assert _rows()[0]["zero_results"] == 0


def test_no_exchanges_table(svc):
    """The tombstone, asserted rather than trusted: `connect()` runs a DROP, so a dev database
    carrying the old table loses it on the next open."""
    conn = store.connect()
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "exchanges" not in names
    assert "usage_daily" in names


def test_the_day_column_is_a_day_not_a_timestamp(svc):
    """A per-request timestamp is the resolution the ruling refuses. `YYYY-MM-DD` is ten
    characters and cannot carry one, so the property is checked on the value, not on intent."""
    svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"}, headers=svc.reader_hdr)
    day = _rows()[0]["day"]
    assert len(day) == 10 and day.count("-") == 2
