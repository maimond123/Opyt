"""Throttling requests does not bound extraction — one call can return everything.

So the RESPONSE is what gets capped, and this file checks each surface separately, including the
one that needs no cap. That last case is the point of the file as much as the first two: a cap
that can never fire is a cap in name only, and the way to hold a bound you did not have to
enforce is to assert it here so a future edit has to decide about it.
"""
from __future__ import annotations

import pytest

from pipeline.retrieval import reads as kb_entry
from service import reading as service_reading


def test_k_is_clamped_server_side(svc, monkeypatch):
    """`k` is caller-controlled with no maximum in the tool layer, which is correct locally — it
    is your own store. Over the wire it is the size of one extraction, so the server decides."""
    monkeypatch.setattr(service_reading, "K_MAX", 2)
    got = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={"query": "agent", "k": 2000},
                          headers=svc.reader_hdr)
    assert got.status_code == 200
    assert len(got.json()["hits"]) == 2
    # The clamp is applied to the ARGUMENT, so the envelope is what the clamped query really
    # returned — including the `cutoff` that says something was left behind — except the "cut"
    # notice, whose "raise k" stops being true at the cap.
    want = kb_entry.run_kb_search("agent", k=2, kb=svc.owner)
    body = got.json()
    assert {**body, "notices": None} == {**want, "notices": None}
    cut = next(n for n in body["notices"] if n["code"] == "results_truncated")
    assert "at most 2 per search" in cut["message"] and "raise k" not in cut["message"]


def test_a_k_under_the_cap_is_untouched(svc):
    got = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={"query": "agent", "k": 3},
                          headers=svc.reader_hdr).json()
    assert len(got["hits"]) == 3


def test_an_oversized_snapshot_is_truncated_and_says_so(svc, monkeypatch):
    """`body_state` needs no new signalling: it exists precisely so a truncated snapshot is never
    quoted as a whole article, and every consumer already reads it. A byte cap that silently
    returned a prefix would make a stub indistinguishable from an article."""
    monkeypatch.setattr(service_reading, "OPEN_CHARS_MAX", 24)
    hit = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={"query": "agent framework"},
                          headers=svc.reader_hdr).json()["hits"][0]

    whole = kb_entry.kb_open(hit["atom_id"], kb=svc.owner)
    assert len(whole["raw"]) > 24, "the fixture body must exceed the cap"

    capped = svc.client.post(f"/v1/kb/{svc.owner}/open", json={"atom_id": hit["atom_id"]},
                             headers=svc.reader_hdr).json()
    assert len(capped["raw"]) == 24 and capped["next_offset"] == 24
    assert capped["raw_chars"] == len(whole["raw"])
    assert whole["raw"].startswith(capped["raw"])
    assert capped["body_state"] == "partial"


def test_a_snapshot_under_the_cap_keeps_its_own_body_state(svc):
    """The cap must not overwrite a state the store actually recorded. The fixture's atom B is
    stored `partial` and atom A `complete`; under the cap both come back as themselves."""
    for atom_id, expected in (("github:root/agentkit", "complete"),
                              ("github:stranger/agents", "partial")):
        got = svc.client.post(f"/v1/kb/{svc.owner}/open", json={"atom_id": atom_id},
                              headers=svc.reader_hdr).json()
        assert got["body_state"] == expected, atom_id


@pytest.mark.parametrize("key,bound", [("top_entities", 15),
                                       ("recent_descriptions", 12)])
def test_aggregates_lists_are_already_bounded(svc, key, bound):
    """The cap the service does NOT apply, asserted at the boundary rather than enforced twice.

    Every list `kb_aggregate` returns is `LIMIT`ed in its own SQL and its two dicts are keyed on
    closed enums, so there is no unbounded surface here — a truncation branch in the handler
    could never fire. This assertion is what keeps that true: raising one of those LIMITs breaks
    it, and whoever does has to decide about this endpoint instead of silently widening it."""
    got = svc.client.post(f"/v1/kb/{svc.owner}/aggregate", json={}, headers=svc.reader_hdr).json()
    assert len(got[key]) <= bound


def test_pages_of_a_text_join_back_into_the_whole_of_it(monkeypatch):
    """`page` on plain values: consecutive pages are the text, and only the last has no next."""
    monkeypatch.setattr(service_reading, "OPEN_CHARS_MAX", 16)
    text = "abcdefghij" * 7
    got, offset = [], 0
    while offset is not None:
        part = service_reading.page(text, offset)
        got.append(part["raw"])
        assert part["raw_chars"] == 70 and part["body_state"] == "partial"
        offset = part["next_offset"]
    assert "".join(got) == text and len(got) == 5


def test_a_text_that_fits_one_page_is_answered_unchanged():
    """No paging fields on an ordinary item: its answer is byte-for-byte the local one."""
    assert service_reading.page("short text", 0) == {}


def test_where_the_next_page_starts():
    """`next_offset` on plain values: after what this page showed, until nothing ranked is left or
    the depth is reached."""
    assert service_reading.next_offset(0, 25, 120) == 25
    assert service_reading.next_offset(25, 25, 50) is None          # the last page
    assert service_reading.next_offset(0, 10, 10) is None           # everything fit
    assert service_reading.next_offset(225, 25, 900) is None        # the depth
    assert service_reading.next_offset(0, 0, 0) is None
