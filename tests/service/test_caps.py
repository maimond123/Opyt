"""Throttling requests does not bound extraction — one call can return everything.

So the RESPONSE is what gets capped, and this file checks each surface separately, including the
one that needs no cap. That last case is the point of the file as much as the first two: a cap
that can never fire is a cap in name only, and the way to hold a bound you did not have to
enforce is to assert it here so a future edit has to decide about it.
"""
from __future__ import annotations

import pytest

from opyt_core import kb as kb_entry
from service import app as service_app
from tests.service.conftest import query_vector


def test_k_is_clamped_server_side(svc, emb, monkeypatch):
    """`k` is caller-controlled with no maximum in the tool layer, which is correct locally — it
    is your own store. Over the wire it is the size of one extraction, so the server decides."""
    monkeypatch.setattr(service_app, "K_MAX", 2)
    got = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={"query": "agent", "k": 2000,
                                "query_vector": query_vector(emb, "agent")},
                          headers=svc.reader_hdr)
    assert got.status_code == 200
    assert len(got.json()["hits"]) == 2
    # The clamp is applied to the ARGUMENT, so the envelope is what the clamped query really
    # returned — including the `cutoff` that says something was left behind.
    assert got.json() == kb_entry.run_kb_search("agent", k=2, kb=svc.owner,
                                                embedder=_reader_embedder(svc, emb, "agent"))


def test_a_k_under_the_cap_is_untouched(svc, emb):
    got = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={"query": "agent", "k": 3,
                                "query_vector": query_vector(emb, "agent")},
                          headers=svc.reader_hdr).json()
    assert len(got["hits"]) == 3


def test_an_oversized_snapshot_is_truncated_and_says_so(svc, emb, monkeypatch):
    """`body_state` needs no new signalling: it exists precisely so a truncated snapshot is never
    quoted as a whole article, and every consumer already reads it. A byte cap that silently
    returned a prefix would make a stub indistinguishable from an article."""
    monkeypatch.setattr(service_app, "OPEN_BYTES_MAX", 24)
    hit = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={"query": "agent framework",
                                "query_vector": query_vector(emb, "agent framework")},
                          headers=svc.reader_hdr).json()["hits"][0]

    whole = kb_entry.kb_open(hit["atom_id"], kb=svc.owner)
    assert len(whole["raw"].encode()) > 24, "the fixture body must exceed the cap"

    capped = svc.client.post(f"/v1/kb/{svc.owner}/open", json={"atom_id": hit["atom_id"]},
                             headers=svc.reader_hdr).json()
    assert len(capped["raw"].encode()) <= 24
    assert whole["raw"].startswith(capped["raw"])
    assert capped["body_state"] == "partial"


def test_a_snapshot_under_the_cap_keeps_its_own_body_state(svc, emb):
    """The cap must not overwrite a state the store actually recorded. The fixture's atom B is
    stored `partial` and atom A `complete`; under the cap both come back as themselves."""
    for atom_id, expected in (("github:root/agentkit", "complete"),
                              ("github:stranger/agents", "partial")):
        got = svc.client.post(f"/v1/kb/{svc.owner}/open", json={"atom_id": atom_id},
                              headers=svc.reader_hdr).json()
        assert got["body_state"] == expected, atom_id


@pytest.mark.parametrize("key,bound", [("top_topics", 15), ("top_entities", 15),
                                       ("recent_descriptions", 12)])
def test_aggregates_lists_are_already_bounded(svc, key, bound):
    """The cap the service does NOT apply, asserted at the boundary rather than enforced twice.

    Every list `kb_aggregate` returns is `LIMIT`ed in its own SQL and its two dicts are keyed on
    closed enums, so there is no unbounded surface here — a truncation branch in the handler
    could never fire. This assertion is what keeps that true: raising one of those LIMITs breaks
    it, and whoever does has to decide about this endpoint instead of silently widening it."""
    got = svc.client.post(f"/v1/kb/{svc.owner}/aggregate", json={}, headers=svc.reader_hdr).json()
    assert len(got[key]) <= bound


def _reader_embedder(svc, emb, query):
    from pipeline.kb import peers
    from pipeline.kb.embed import PrecomputedEmbedder
    conn, _ = peers.open_peer(svc.owner)
    try:
        return PrecomputedEmbedder(conn, query_vector(emb, query))
    finally:
        conn.close()
