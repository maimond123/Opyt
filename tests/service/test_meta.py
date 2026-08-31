"""`GET /v1/kb/{owner}/meta` — the embedding identity a remote reader has to match.

The reader embeds their own query on their own machine, so they need to know which model the
owner's export was built with. That fact lives in the export's `kb_meta` row and nowhere else.
This endpoint reads it from the served file on demand and stores no copy: `kb_meta` changes the
day the owner re-embeds and re-uploads, and a snapshot taken at redeem time would be a second
home for a fact the file owns.

Scoped like every other read: a reader token reaches exactly the knowledge base it was minted
against.
"""
from __future__ import annotations

import sqlite3

from pipeline.kb import peers
from pipeline.kb.embed import read_kb_meta


def test_meta_reports_the_exports_embedding_identity(svc):
    r = svc.client.get(f"/v1/kb/{svc.owner}/meta", headers=svc.reader_hdr)
    assert r.status_code == 200, r.text

    conn, _ = peers.open_peer(svc.owner)
    try:
        truth = read_kb_meta(conn)
    finally:
        conn.close()

    body = r.json()
    assert body["owner"] == svc.owner
    assert (body["model"], body["dim"], body["provider"], body["query_instruction"]) == (
        truth["model"], truth["dim"], truth["provider"], truth["query_instruction"])


def test_meta_follows_a_re_upload(svc, export_file):
    """Read on demand, never stored. The owner re-embeds at a different width and re-uploads;
    the very next meta call reports the new one, with no cache to invalidate on this side."""
    before = svc.client.get(f"/v1/kb/{svc.owner}/meta", headers=svc.reader_hdr).json()

    conn = sqlite3.connect(export_file)
    try:
        conn.execute("UPDATE kb_meta SET embed_model = 'other/model-v2' WHERE id = 1")
        conn.commit()
    finally:
        conn.close()
    up = svc.client.post(f"/v1/upload/{svc.owner}", content=export_file.read_bytes(),
                         headers=svc.owner_hdr)
    assert up.status_code == 200, up.text

    after = svc.client.get(f"/v1/kb/{svc.owner}/meta", headers=svc.reader_hdr).json()
    assert before["model"] != "other/model-v2"
    assert after["model"] == "other/model-v2"


def test_meta_needs_a_reader_token_for_that_kb(svc):
    from service import store

    assert svc.client.get(f"/v1/kb/{svc.owner}/meta").status_code == 401

    other = store.mint_token("somebody-else", "reader", label="not this kb")
    r = svc.client.get(f"/v1/kb/{svc.owner}/meta",
                       headers={"Authorization": f"Bearer {other}"})
    assert r.status_code == 403


def test_meta_on_an_unserved_kb_is_a_404(svc):
    from service import store

    token = store.mint_token("nobody", "reader", label="x")
    r = svc.client.get("/v1/kb/nobody/meta", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 404
