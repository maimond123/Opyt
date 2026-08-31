"""The public stats page — the roll-up, and nobody's name.

Ruled 2026-08-26: the metrics surface is a PUBLIC aggregate page, not an operator dashboard. One
page is both the transparency mechanism (a reader can check what the totals say against what
`TELEMETRY.md` claims is collected) and the record of whether anyone used this. That is why these
tests assert two things at once — that the numbers are real, and that no identity survives the
roll-up.
"""
from __future__ import annotations

import json

from service import store

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
    traffic, then assert that neither name nor any token hash reaches either response body —
    checked against the whole body rather than key by key, so a key added later is covered."""
    token = store.mint_token(SENTINEL_OWNER, "reader", label=SENTINEL_LABEL)
    store.record_usage(SENTINEL_OWNER, store.token_hash(token), "search")
    svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent"}, headers=svc.reader_hdr)

    hashes = {store.token_hash(token), store.token_hash(svc.reader_token),
              store.token_hash(svc.owner_token)}

    js, page = svc.client.get("/v1/stats"), svc.client.get("/stats")
    assert js.status_code == 200 and page.status_code == 200
    for body in (json.dumps(js.json()), page.text):
        assert SENTINEL_OWNER not in body
        assert SENTINEL_LABEL not in body
        assert svc.owner not in body
        assert not any(h in body for h in hashes)


def test_the_html_page_is_self_contained(svc):
    """No external assets: the page must render with nothing fetched. A stats page that phones
    a CDN would hand a third party the visitor list this whole design refuses to keep."""
    page = svc.client.get("/stats")
    assert page.status_code == 200
    html = page.text
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html.lower()
