"""The public stats page — the roll-up, and nobody's name.

Ruled 2026-08-26: the metrics surface is a PUBLIC aggregate page, not an operator dashboard. One
page is both the transparency mechanism (a reader can check what the totals say against what
`TELEMETRY.md` claims is collected) and the record of whether anyone used this. That is why these
tests assert two things at once — that the numbers are real, and that no identity survives the
roll-up.

NARROWED 2026-09-01 by R5a, in exactly one place. Self-service publishing removed the human who
used to know how much each owner stored, so `/v1/stats` now carries one per-knowledge-base list:
routing key, bytes, and the date it was first published. That is not a retreat from the ruling —
R4 made the routing key an ASSIGNED address rather than a name anybody chose, and the list
carries no label, no reader and no traffic. The HTML page still shows totals only.
"""
from __future__ import annotations

import json

from service import store
from tests.service.conftest import LABEL

SENTINEL_OWNER = "zzz-owner-sentinel"
SENTINEL_LABEL = "zzz-reader-sentinel"


def test_stats_needs_no_token(svc):
    """Unauthenticated by design: a transparency page behind a credential is not one."""
    assert svc.client.get("/v1/stats").status_code == 200
    assert svc.client.get("/stats").status_code == 200


def test_stats_reflects_usage(svc):
    before = svc.client.get("/v1/stats").json()
    assert before["reads_total"] == 0
    assert before["kbs_published"] == 1          # the export uploaded by the fixture
    assert before["readers_total"] == 1
    assert before["codes_minted"] == 1 and before["codes_redeemed"] == 1
    assert before["zero_result_rate"] is None    # no searches yet: not zero, unknown

    svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"}, headers=svc.reader_hdr)
    svc.client.post(f"/v1/kb/{svc.owner}/search",
                    json={"query": "zzz-nothing-matches-this-zzz", "mode": "bm25"},
                    headers=svc.reader_hdr)
    svc.client.post(f"/v1/kb/{svc.owner}/aggregate", json={}, headers=svc.reader_hdr)

    after = svc.client.get("/v1/stats").json()
    assert after["reads_total"] == 3
    assert after["reads_30d"] == 3
    assert after["reads_by_tool"] == {"search": 2, "aggregate": 1}
    assert after["active_readers_30d"] == 1
    assert after["zero_result_rate"] == 0.5
    assert after["generated_at"]


def test_stats_leaks_no_identity(svc):
    """THE policy pin for this endpoint. Seed a distinctively named owner and reader, generate
    traffic, then assert that no LABEL and no token hash reaches either response body — checked
    against the whole body rather than key by key, so a key added later is covered.

    A label is the one string in this database that a person chose to describe a person: it is
    how an owner names their readers so they can revoke them, and how an owner names themselves.
    A routing key is not — see `test_the_page_shows_totals_only`, which pins the one place a key
    is published and why."""
    token = store.mint_token(SENTINEL_OWNER, "reader", label=SENTINEL_LABEL)
    store.record_usage(SENTINEL_OWNER, store.token_hash(token), "search")
    svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"}, headers=svc.reader_hdr)

    hashes = {store.token_hash(token), store.token_hash(svc.reader_token),
              store.token_hash(svc.owner_token)}

    js, page = svc.client.get("/v1/stats"), svc.client.get("/stats")
    assert js.status_code == 200 and page.status_code == 200
    for body in (json.dumps(js.json()), page.text):
        assert SENTINEL_LABEL not in body
        assert LABEL not in body
        assert not any(h in body for h in hashes)
    # A knowledge base nobody published stores nothing and appears nowhere: the list is written
    # by an upload, not by minting a token.
    assert SENTINEL_OWNER not in json.dumps(js.json())
    assert SENTINEL_OWNER not in page.text


def test_the_page_shows_totals_only_and_the_key_rides_the_json(svc):
    """The split R5a asks for. An operator has to be able to see WHICH knowledge base is eating
    the disk — that is the entire abuse response, since nothing is checked at admission — and a
    notifier reads JSON. The human-readable page keeps its promise that every number on it is a
    total."""
    js = svc.client.get("/v1/stats").json()
    row = next(r for r in js["stored_bytes_by_kb"] if r["owner"] == svc.owner)
    assert row["bytes"] == svc.export.stat().st_size
    assert row["owner_since"]
    assert js["stored_bytes_total"] == row["bytes"]
    assert "label" not in row

    assert svc.owner not in svc.client.get("/stats").text


def test_unpublishing_stops_counting_the_bytes_but_keeps_the_start_date(svc):
    """A stored-bytes number that only ever climbs is not a disk-usage number. The row survives
    because `first_published_at` is the fact that cannot be recovered — a directory listing never
    had it, and it is what pricing would be built on."""
    before = next(r for r in svc.client.get("/v1/stats").json()["stored_bytes_by_kb"]
                  if r["owner"] == svc.owner)

    svc.client.post("/v1/unpublish", headers=svc.owner_hdr)

    js = svc.client.get("/v1/stats").json()
    after = next(r for r in js["stored_bytes_by_kb"] if r["owner"] == svc.owner)
    assert after["bytes"] == 0
    assert js["stored_bytes_total"] == 0
    assert after["owner_since"] == before["owner_since"]


def test_a_second_upload_moves_the_last_date_and_not_the_first(svc):
    """`first_published_at` is written once. An upsert that refreshed it would erase the only
    copy of it on the second push, which is every push after the first."""
    before = next(r for r in svc.client.get("/v1/stats").json()["stored_bytes_by_kb"]
                  if r["owner"] == svc.owner)
    conn = store.connect()
    try:
        conn.execute("UPDATE owner_uploads SET first_published_at = '2020-01-01 00:00:00', "
                     "last_published_at = '2020-01-01 00:00:00' WHERE owner = ?", (svc.owner,))
        conn.commit()
    finally:
        conn.close()

    svc.client.post(f"/v1/upload/{svc.owner}", content=svc.export.read_bytes(),
                    headers=svc.owner_hdr)

    after = next(r for r in svc.client.get("/v1/stats").json()["stored_bytes_by_kb"]
                 if r["owner"] == svc.owner)
    assert after["owner_since"] == "2020-01-01 00:00:00"
    assert after["last_published_at"] > "2020-01-01 00:00:00"
    assert after["bytes"] == before["bytes"]


def test_the_html_page_is_self_contained(svc):
    """No external assets: the page must render with nothing fetched. A stats page that phones
    a CDN would hand a third party the visitor list this whole design refuses to keep."""
    page = svc.client.get("/stats")
    assert page.status_code == 200
    html = page.text
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html.lower()
