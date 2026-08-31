"""What the service counts, and what it must never write down.

`usage_daily` is the meter a rate-limit policy will eventually read: which day, whose knowledge
base, which reader, which tool, how many. The query is not a column and must never become one
(R10). What the counting SHAPE guarantees — one row per day rather than one per request — is
pinned in `test_usage.py`; this file is about what is never written down at all.

This file also states the limit of that promise out loud, because promising more than the
architecture gives is the failure worth guarding against here. BM25 tokenizes the query STRING,
so on a keyword or hybrid read the text reaches this process. "Reader queries are blind" is a
RETENTION commitment — the text is not written down — not a property the design enforces. Only
the semantic arm is structurally blind, because a vector is all that crosses.
"""
from __future__ import annotations

from service import store

SECRET = "zzsecretphrasezz"


def test_a_served_read_is_counted(svc):
    before = store.usage_total(svc.owner)

    svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"}, headers=svc.reader_hdr)
    svc.client.post(f"/v1/kb/{svc.owner}/aggregate", json={}, headers=svc.reader_hdr)

    conn = store.connect()
    try:
        rows = conn.execute("SELECT owner, reader, tool FROM usage_daily").fetchall()
        assert store.usage_total(svc.owner) == before + 2
        assert {r["tool"] for r in rows} == {"search", "aggregate"}
        assert {r["owner"] for r in rows} == {svc.owner}
        assert {r["reader"] for r in rows} == {store.token_hash(svc.reader_token)}
    finally:
        conn.close()


def test_a_refused_read_spends_nobody_s_allowance(svc):
    """Counted AFTER the read runs, so the meter reflects reads that happened. A 401 or 403 is
    not a read — it returned nothing, and charging for it would let anyone exhaust a stranger's
    allowance by sending a wrong token."""
    before = store.usage_total(svc.owner)

    svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"})
    svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"}, headers=svc.owner_hdr)

    assert store.usage_total(svc.owner) == before


def test_the_query_text_is_not_in_the_audit_trail(svc, emb, capsys):
    """The retention commitment, checked against the whole file rather than against one column:
    a query that leaked into any column, index or free page of `service.db` would be findable
    here. The process output is checked too — the other place a body could be written down."""
    from tests.service.conftest import query_vector

    svc.client.post(f"/v1/kb/{svc.owner}/search",
                    json={"query": SECRET, "query_vector": query_vector(emb, SECRET)},
                    headers=svc.reader_hdr)
    svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": SECRET, "mode": "bm25"},
                    headers=svc.reader_hdr)

    assert SECRET.encode() not in (svc.home / "service.db").read_bytes()
    captured = capsys.readouterr()
    assert SECRET not in captured.out and SECRET not in captured.err


def test_the_usage_table_has_no_column_that_could_hold_a_query(svc):
    """Asserted on the SCHEMA, not on the rows. A row-level check passes on an empty table and
    on the day someone adds the column but has not filled it yet; this fails the moment the
    column exists. The exact set is the assertion — a new column has to come through here."""
    conn = store.connect()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(usage_daily)")}
    finally:
        conn.close()
    assert cols == {"day", "owner", "reader", "tool", "n", "zero_results"}
