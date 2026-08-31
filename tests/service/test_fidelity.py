"""The one assertion that pins the whole hop.

If a knowledge base read over HTTP returns anything other than what the same read returns
in-process against the same export, then the Phase-2 seam was drawn in the wrong place and the
service is a second implementation of retrieval rather than an adapter in front of the first.
So this file compares whole response bodies, not fields — a projection that quietly dropped a key
must fail here rather than somewhere downstream in a host model's reasoning.
"""
from __future__ import annotations

import json

import pytest

from opyt_core import kb as kb_entry
from pipeline.kb import peers
from pipeline.kb.embed import PrecomputedEmbedder
from tests.kb.test_export import _QUERIES
from tests.service.conftest import query_vector

# EVERY spec crosses, `entry_mode` included. It used to be excluded because `run_kb_search` had
# no parameter for it; it gained one on 2026-08-27, when the default scope became HUMAN_ATTESTED
# and `entry_mode` started deciding the SHAPE of the answer rather than just filtering it. A
# filter the adapter drops silently is a foreign search answering a different question than the
# identical local call, which is precisely what this file exists to catch.


def _local(owner, spec, vector=None):
    """The same call a reader makes against a peer sitting on their own disk."""
    embedder = None
    if vector is not None:
        conn, _label = peers.open_peer(owner)
        try:
            embedder = PrecomputedEmbedder(conn, vector)
        finally:
            conn.close()
    return kb_entry.run_kb_search(**spec, kb=owner, embedder=embedder)


@pytest.mark.parametrize("spec", _QUERIES, ids=lambda s: json.dumps(s, sort_keys=True))
def test_the_hop_changes_nothing_about_a_search(svc, emb, spec):
    """Every shape the read path has, over HTTP, against the in-process answer.

    The reader's own query vector is supplied, which is the served path: the semantic arm runs
    from floats that crossed the wire and the server embeds nothing."""
    vector = query_vector(emb, spec["query"])
    got = svc.client.post(f"/v1/kb/{svc.owner}/search", json={**spec, "query_vector": vector},
                          headers=svc.reader_hdr)
    assert got.status_code == 200, got.text
    assert got.json() == _local(svc.owner, spec, vector)


def test_the_hop_changes_nothing_when_the_reader_sends_no_vector(svc):
    """The other branch, and it must be the same function too. With no `query_vector` the server
    falls back to `embedder_for_store`, which cannot reach this store's model from this install
    and degrades to the keyword arm — carrying the `vector_arm_unavailable` notice that explains
    which half ran. That degradation is Phase 2's, not the service's, and the body proves it."""
    spec = {"query": "agent framework"}
    got = svc.client.post(f"/v1/kb/{svc.owner}/search", json=spec, headers=svc.reader_hdr)
    assert got.status_code == 200, got.text
    assert got.json() == _local(svc.owner, spec)
    assert any(n["code"] == "vector_arm_unavailable" for n in got.json()["notices"])


def test_every_hit_names_whose_knowledge_base_it_came_from(svc, emb):
    """Provenance travels on the CARD, not just the envelope — a host that quotes one hit into a
    document has to carry the attribution with it (X2). Inherited from Phase 2 unchanged, and
    asserted here because the hop is exactly where it would be easy to lose."""
    got = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={"query": "agent framework",
                                "query_vector": query_vector(emb, "agent framework")},
                          headers=svc.reader_hdr).json()
    assert got["hits"], "the fixture corpus must answer this query or the assertion is vacuous"
    assert {h["kb"] for h in got["hits"]} == {svc.owner}
    assert any(n["code"] == "foreign_kb" for n in got["notices"])


def test_the_hop_changes_nothing_about_open(svc, emb):
    hit = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={"query": "agent framework",
                                "query_vector": query_vector(emb, "agent framework")},
                          headers=svc.reader_hdr).json()["hits"][0]
    got = svc.client.post(f"/v1/kb/{svc.owner}/open", json={"atom_id": hit["atom_id"]},
                          headers=svc.reader_hdr)
    assert got.status_code == 200, got.text
    assert got.json() == kb_entry.kb_open(hit["atom_id"], kb=svc.owner)
    assert got.json()["raw"], "the snapshot body must survive the hop"


def test_the_hop_changes_nothing_about_aggregate(svc):
    for scope in (None, {"tags": ["ai-agents"]}, {"source_type": "github"},
                  {"date_from": "2025-01-01"}):
        got = svc.client.post(f"/v1/kb/{svc.owner}/aggregate", json={"scope": scope},
                              headers=svc.reader_hdr)
        assert got.status_code == 200, got.text
        assert got.json() == kb_entry.kb_aggregate(scope=scope, kb=svc.owner), scope


def test_the_reader_pays_for_embedding_and_the_service_never_does(svc, emb, monkeypatch):
    """I10, asserted at the seam where money would be spent.

    A `query_vector` search must not construct an embedder on the server AT ALL — not a cached
    one, not a cheap one. Both factories are replaced with something that raises, so any call
    fails the test loudly instead of quietly reaching for a key the server must not hold."""
    from pipeline.kb import embed

    def boom(*a, **kw):
        raise AssertionError("the service built an embedder for a reader-supplied query vector")

    monkeypatch.setattr(embed, "get_kb_embedder", boom)
    monkeypatch.setattr(embed, "embedder_for_store", boom)
    monkeypatch.setattr(kb_entry, "get_kb_embedder", boom)
    monkeypatch.setattr(kb_entry, "embedder_for_store", boom)

    got = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={"query": "agent framework", "mode": "semantic",
                                "query_vector": query_vector(emb, "agent framework")},
                          headers=svc.reader_hdr)
    assert got.status_code == 200, got.text
    assert got.json()["trace"]["ran"] == "semantic"
    assert got.json()["hits"], "the vector arm must actually have run on the sent vector"


def test_a_query_vector_of_the_wrong_width_is_refused_not_guessed(svc):
    """A subspace mismatch is the failure that does not announce itself: a wrong-width vector
    crashes several frames deep, and a right-width one from the wrong model returns confident
    garbage. The checkable half is checked at the boundary it arrived through, and the answer is
    the reader's to fix — never a silent drop to the keyword arm."""
    got = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={"query": "agent", "query_vector": [0.1] * 7},
                          headers=svc.reader_hdr)
    assert got.status_code == 400
    assert "dimensions" in got.json()["detail"]
