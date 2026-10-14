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

from pipeline.retrieval import reads as kb_entry
from tests.kb.test_export import _QUERIES

# EVERY spec crosses, `entry_mode` included. The default scope is HUMAN_ATTESTED, so `entry_mode`
# decides the SHAPE of the answer rather than just filtering it. A
# filter the adapter drops silently is a foreign search answering a different question than the
# identical local call, which is precisely what this file exists to catch.


def _local(owner, spec, as_kb=None):
    """The same call a reader makes against a peer sitting on their own disk."""
    return kb_entry.run_kb_search(**spec, kb=owner, as_kb=as_kb)


@pytest.mark.parametrize("spec", _QUERIES, ids=lambda s: json.dumps(s, sort_keys=True))
def test_the_hop_changes_nothing_about_a_search(svc, spec):
    """Every shape the read path has, over HTTP, against the in-process answer."""
    got = svc.client.post(f"/v1/kb/{svc.owner}/search", json=spec, headers=svc.reader_hdr)
    assert got.status_code == 200, got.text
    assert got.json() == _local(svc.owner, spec)


@pytest.mark.parametrize("spec", _QUERIES, ids=lambda s: json.dumps(s, sort_keys=True))
def test_as_kb_relabels_the_envelope_and_changes_nothing_else(svc, spec):
    """R4, held to the same standard as the hop itself: `as_kb` is LABELING, so an answer sent
    with one must equal the in-process answer computed with the same label — every hit, score,
    notice and trace field, not just the `kb` keys. A field that quietly took a different path
    when a label was present would be the split leaking into routing."""
    got = svc.client.post(f"/v1/kb/{svc.owner}/search", json={**spec, "as_kb": "leo"},
                          headers=svc.reader_hdr)
    assert got.status_code == 200, got.text
    assert got.json() == _local(svc.owner, spec, as_kb="leo")


def test_the_routing_key_still_selects_the_store_a_label_never_does(svc):
    """The other half of "labeling-only": `as_kb` must not be able to reach a store. A label
    naming a knowledge base that does not exist changes only what the answer is called."""
    body = {"query": "agent framework", "as_kb": "a-name-nobody-published"}
    got = svc.client.post(f"/v1/kb/{svc.owner}/search", json=body,
                          headers=svc.reader_hdr).json()
    assert got["hits"], "the store the URL named must still have answered"
    assert {h["kb"] for h in got["hits"]} == {"a-name-nobody-published"}


def test_an_absent_as_kb_falls_back_to_the_routing_key(svc):
    """A pre-R4 client sends no label. It gets today's behaviour rather than a null `kb` field,
    which is what lets the wire protocol change land on its own."""
    got = svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent framework"},
                          headers=svc.reader_hdr).json()
    assert {h["kb"] for h in got["hits"]} == {svc.owner}


def test_an_older_clients_mode_and_query_vector_are_ignored(svc):
    """A client from before whole-item search still sends `mode` and `query_vector`. They are
    dropped, not refused, so its search answers like any other."""
    spec = {"query": "agent framework"}
    got = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={**spec, "mode": "semantic", "query_vector": [0.1] * 7},
                          headers=svc.reader_hdr)
    assert got.status_code == 200, got.text
    assert got.json() == _local(svc.owner, spec)


def test_every_hit_names_whose_knowledge_base_it_came_from(svc):
    """Provenance travels on the CARD, not just the envelope — a host that quotes one hit into a
    document has to carry the attribution with it (X2). Inherited from Phase 2 unchanged, and
    asserted here because the hop is exactly where it would be easy to lose."""
    got = svc.client.post(f"/v1/kb/{svc.owner}/search",
                          json={"query": "agent framework"}, headers=svc.reader_hdr).json()
    assert got["hits"], "the fixture corpus must answer this query or the assertion is vacuous"
    assert {h["kb"] for h in got["hits"]} == {svc.owner}
    assert any(n["code"] == "foreign_kb" for n in got["notices"])


def test_the_hop_changes_nothing_about_open(svc):
    hit = svc.client.post(f"/v1/kb/{svc.owner}/search", json={"query": "agent framework"},
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
