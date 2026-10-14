"""Search and open over plain HTTP, read in pages.

`reading` pages for the connector already. These hold that the two HTTP routes pass an `offset`
through, so a plain-HTTP reader is not stopped at 25 results or at the first 60,000 characters,
and that a request with no `offset` is answered as it always was: an install's own reads of a
shared library send none.
"""
from __future__ import annotations

from pipeline.retrieval import reads
from service import db, reading, usage

QUERY = "agent framework autonomous tools"


def _search(svc, **body) -> dict:
    r = svc.client.post(f"/v1/kb/{svc.owner}/search", headers=svc.reader_hdr,
                        json={"query": QUERY, **body})
    assert r.status_code == 200, r.text
    return r.json()


def _open(svc, atom_id: str, **body) -> dict:
    r = svc.client.post(f"/v1/kb/{svc.owner}/open", headers=svc.reader_hdr,
                        json={"atom_id": atom_id, **body})
    assert r.status_code == 200, r.text
    return r.json()


def _reads(owner: str) -> int:
    conn = db.connect()
    try:
        return usage.total_reads(conn, owner)
    finally:
        conn.close()


# ── search ───────────────────────────────────────────────────────────────────────

def test_search_pages_join_into_the_ranking_one_deep_search_returns(svc, monkeypatch):
    monkeypatch.setattr(reading, "K_MAX", 2)
    whole = [h["atom_id"] for h in reads.run_kb_search(QUERY, k=50, kb=svc.owner)["hits"]]
    assert len(whole) > 4, "the fixture must match more than two pages"

    seen, offset, pages = [], 0, []
    while offset is not None:
        pages.append(_search(svc, k=2, offset=offset))
        seen += [h["atom_id"] for h in pages[-1]["hits"]]
        offset = pages[-1]["next_offset"]

    assert seen == whole and len(pages) > 2
    cut = next(n for n in pages[0]["notices"] if n["code"] == "results_truncated")
    assert "Pass offset=2" in cut["message"]


def test_a_search_with_no_offset_is_answered_as_it_always_was(svc, monkeypatch):
    """No `next_offset`, and the notice a reader that cannot page was already given. Sending
    `offset: null` is sending none."""
    monkeypatch.setattr(reading, "K_MAX", 2)
    want = reads.run_kb_search(QUERY, k=2, kb=svc.owner)

    for body in ({}, {"offset": None}):
        got = _search(svc, k=9, **body)
        assert "next_offset" not in got
        assert {**got, "notices": None} == {**want, "notices": None}
        cut = next(n for n in got["notices"] if n["code"] == "results_truncated")
        assert "at most 2 per search" in cut["message"] and "offset" not in cut["message"]

    paged = _search(svc, k=9, offset=0)
    assert paged["next_offset"] == 2 and paged["hits"] == want["hits"]


def test_a_search_offset_past_the_depth_says_so_and_is_not_counted(svc):
    before = _reads(svc.owner)
    for offset in (reading.SEARCH_DEPTH_MAX, -1):
        got = _search(svc, offset=offset)
        assert "outside" in got["error"] and "next_offset" in got["error"]
    assert _reads(svc.owner) == before


# ── open ─────────────────────────────────────────────────────────────────────────

def test_a_long_item_is_read_whole_in_pages(svc, monkeypatch):
    monkeypatch.setattr(reading, "OPEN_CHARS_MAX", 8)
    whole = reads.kb_open("substack:post1", kb=svc.owner)["raw"]
    assert len(whole) > 16, "the fixture item must be longer than two pages"

    first = _open(svc, "substack:post1")
    assert first["raw"] == whole[:8] and first["next_offset"] == 8
    text, page = first["raw"], first
    while page["next_offset"] is not None:
        page = _open(svc, "substack:post1", offset=page["next_offset"])
        assert page["raw_offset"] == len(text) and page["raw_chars"] == len(whole)
        text += page["raw"]
    assert text == whole and page["body_state"] == "partial"


def test_an_item_that_fits_one_page_is_answered_as_it_always_was(svc):
    """With no `offset`, and with `offset: 0`: the local answer, with no paging field on it."""
    want = reads.kb_open("github:root/agentkit", kb=svc.owner)
    assert _open(svc, "github:root/agentkit") == want
    assert _open(svc, "github:root/agentkit", offset=0) == want
    assert not {"next_offset", "raw_offset", "raw_chars"} & set(want)


def test_an_open_offset_outside_the_text_says_so_and_is_not_counted(svc):
    before = _reads(svc.owner)
    for offset in (10_000_000, -1):
        got = _open(svc, "github:root/agentkit", offset=offset)
        assert "outside this item's text" in got["error"] and "next_offset" in got["error"]
    assert _reads(svc.owner) == before
