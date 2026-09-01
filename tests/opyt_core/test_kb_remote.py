"""The HTTP branch of the `kb=` entry points, proven against the real service in-process.

The one theorem: `kb=` naming a SERVED knowledge base returns the same envelope as `kb=` naming
the same export on disk. Everything else here is the failure half — a revoked token, a dead
service, an unreachable embedding provider, a stale cached meta — each of which must come back
as an envelope with a sentence, never as an exception out of a tool call (P3).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import requests as real_requests

from opyt_core import kb as kb_entry
from opyt_core import kb_remote
from pipeline.kb import peers
from pipeline.kb.embed import PrecomputedEmbedder
from service import store
from tests.kb.test_export import _QUERIES
from tests.opyt_core.conftest import PEER
from tests.service.conftest import query_vector

# Every spec crosses, `entry_mode` included — see tests/service/test_fidelity.py for why it
# stopped being an exception.


def _local(owner, spec, vector=None):
    """The same read against the same export as a FILE peer — the equality's right-hand side.

    `as_kb=PEER` is what makes this an equality at all after R4. The routing key selects the
    store on both sides; the label is what the reader's install calls it, and the served side
    sends its own. So the theorem is now stronger than "the envelope crosses verbatim": the
    answer is identical AND already carries the reader's own name for the knowledge base."""
    embedder = None
    if vector is not None:
        conn, _label = peers.open_peer(owner)
        try:
            embedder = PrecomputedEmbedder(conn, vector)
        finally:
            conn.close()
    return kb_entry.run_kb_search(**spec, kb=owner, as_kb=PEER, embedder=embedder)


@pytest.mark.parametrize("spec", _QUERIES, ids=lambda s: json.dumps(s, sort_keys=True))
def test_remote_search_equals_local(remote, emb, monkeypatch, spec):
    """The envelope crosses verbatim: a served peer and a file peer are indistinguishable
    downstream, for every shape the read path has. `embedder_from_meta` is replaced with the
    fixture embedder (the one the corpus was built with) because the real one would need this
    install to reach the owner's provider — the transport is under test, not the provider axis."""
    monkeypatch.setattr(kb_remote, "embedder_from_meta", lambda meta, **kw: emb)
    got = kb_entry.run_kb_search(**spec, kb=PEER)
    vector = (query_vector(emb, spec["query"])
              if spec.get("mode", "hybrid") in ("hybrid", "semantic") else None)
    assert got == _local(remote.svc.owner, spec, vector)


def test_remote_open_and_aggregate_equal_local(remote, emb, monkeypatch):
    monkeypatch.setattr(kb_remote, "embedder_from_meta", lambda meta, **kw: emb)
    owner = remote.svc.owner
    hit = kb_entry.run_kb_search("agent framework", kb=PEER)["hits"][0]
    opened = kb_entry.kb_open(hit["atom_id"], kb=PEER)
    assert opened == kb_entry.kb_open(hit["atom_id"], kb=owner, as_kb=PEER)
    assert opened["raw"], "the snapshot body must survive the hop"
    for scope in (None, {"tags": ["ai-agents"]}, {"source_type": "github"}):
        assert (kb_entry.kb_aggregate(scope=scope, kb=PEER)
                == kb_entry.kb_aggregate(scope=scope, kb=owner, as_kb=PEER)), scope


def test_every_kb_field_carries_the_readers_own_name_not_the_routing_key(remote, emb,
                                                                        monkeypatch):
    """R4's whole payoff, at the surface a host model actually reads.

    The URL path segment is the routing key — after R5 an opaque `secrets.token_hex(6)` that
    nobody types. Without `as_kb` the envelope came back naming it: hit cards saying
    `kb: "a3f9c2e1"`, a trace saying the same, and a `foreign_kb` notice instructing the host to
    pass that string back to `open()` — which resolves nowhere on this install, because the
    reader registered the peer as `x`. Rewriting those strings afterwards would mean editing
    prose inside a notice; sending the name instead means nothing is ever rewritten."""
    monkeypatch.setattr(kb_remote, "embedder_from_meta", lambda meta, **kw: emb)
    owner = remote.svc.owner
    out = kb_entry.run_kb_search("agent framework", kb=PEER)

    assert out["hits"], "the fixture corpus must answer this or the assertion is vacuous"
    assert {h["kb"] for h in out["hits"]} == {PEER}
    assert out["trace"]["kb"] == PEER
    notice = next(n for n in out["notices"] if n["code"] == "foreign_kb")
    assert notice["kb"] == PEER
    assert f"kb='{PEER}'" in notice["message"]
    assert owner not in notice["message"]

    assert kb_entry.kb_open(out["hits"][0]["atom_id"], kb=PEER)["kb"] == PEER
    assert kb_entry.kb_aggregate(kb=PEER)["kb"] == PEER


def test_the_name_the_notice_hands_back_actually_opens_the_atom(remote, emb, monkeypatch):
    """The `foreign_kb` notice is meant to be repeated verbatim, so the string in it has to be a
    name this install resolves. Following the instruction is the test."""
    monkeypatch.setattr(kb_remote, "embedder_from_meta", lambda meta, **kw: emb)
    out = kb_entry.run_kb_search("agent framework", kb=PEER)
    hit = out["hits"][0]

    opened = kb_entry.kb_open(hit["atom_id"], kb=hit["kb"])
    assert opened.get("error") is None
    assert opened["atom_id"] == hit["atom_id"]


def test_a_reader_cannot_relabel_a_served_kb_as_their_own(remote, emb, monkeypatch):
    """`as_kb` arrives in an HTTP body, so it is untrusted. Every `kb_name == LOCAL_KB` test
    downstream means "this is my own store", and `kb_open`'s `raw_path` would hand back a path on
    the SERVER's filesystem. `_label_as` refuses that one name, so the label falls back to the
    routing key rather than turning a foreign read into a local one."""
    monkeypatch.setattr(kb_remote, "embedder_from_meta", lambda meta, **kw: emb)
    owner = remote.svc.owner
    hit = kb_entry.run_kb_search("agent framework", kb=PEER)["hits"][0]

    body = {"atom_id": hit["atom_id"], "as_kb": "me"}
    got = remote.svc.client.post(f"/v1/kb/{owner}/open", json=body,
                                 headers=remote.svc.reader_hdr).json()
    assert got["kb"] == owner
    assert got["raw_path"] is None


def test_revoked_token_is_a_notice_not_an_exception(remote):
    """Revocation is a server-side row delete, effective on the reader's next request — which
    must land as an empty envelope naming the fix (a new grant code), never as a raise."""
    store.revoke(remote.svc.owner, store.token_hash(remote.svc.reader_token))
    out = kb_entry.run_kb_search("agent framework", kb=PEER)
    assert out["hits"] == []
    assert any("grant code" in n["message"] for n in out["notices"])


def test_a_granted_but_unpublished_kb_reads_as_a_sentence_not_a_status(remote):
    """The 404's real caller, and the detached first publish is what makes it ordinary: `share`
    hands back an invite immediately and pushes in the background, so for a minute or two the
    link is live and the export is not. Without its own branch this is the generic "the service
    answered 404", which a host reads as a transient outage and retries — telling the reader Opyt
    is broken when nothing is. The status code must not surface, and the sentence must not tell
    them to ask the owner to publish, which the owner already did.

    (It is NOT reachable by unpublishing: that revokes the reader's token first, so the reader
    meets the 401 sentence instead. Which is right — the fix there really is a new grant code.)"""
    svc = remote.svc
    empty_hdr = {"Authorization": f"Bearer {store.mint_token('newcomer', 'owner')}"}
    code = svc.client.post("/v1/grant", json={"label": "eager"}, headers=empty_hdr).json()["code"]
    token = svc.client.post("/v1/redeem",
                            json={"code": code, "install_id": "i-3"}).json()["token"]
    peers.add("newcomer", "https://svc.test/v1/kb/newcomer", "Newcomer's KB", token=token)

    out = kb_entry.run_kb_search("agent framework", kb="newcomer")
    assert out["hits"] == []
    message = " ".join(n["message"] for n in out["notices"])
    assert "404" not in message
    assert "has not arrived on the service yet" in message
    assert "try again shortly" in message


def test_unpublishing_leaves_the_reader_with_the_revoked_token_sentence(remote):
    """Ordering check, from the reader's side: `unpublish` cuts tokens BEFORE deleting the file,
    so a reader never meets a live token against a missing export. They get the 401 sentence,
    which names the fix that actually works — the owner re-sharing."""
    remote.svc.client.post("/v1/unpublish", headers=remote.svc.owner_hdr)
    out = kb_entry.run_kb_search("agent framework", kb=PEER)
    assert out["hits"] == []
    assert any("grant code" in n["message"] for n in out["notices"])


def test_embed_failure_degrades_visibly(remote):
    """The export was embedded on a provider this install does not speak (`local`, the fixture
    corpus's), which is the real failure a reader hits — no synthetic breakage needed. The
    request must go out as bm25 and the envelope must carry the SAME degrade code the file-peer
    path emits, so hosts treat the two transports identically."""
    out = kb_entry.run_kb_search("agent framework", kb=PEER)
    assert out["trace"]["ran"] == "bm25"
    assert any(n["code"] == "vector_arm_unavailable" for n in out["notices"])
    assert out["hits"], "the keyword arm must actually have run"


def test_stale_meta_retries_once(remote, emb, monkeypatch):
    """The owner re-embedded at a new width and re-uploaded; this reader's cached meta is now a
    lie. The service 400s naming dimensions, the cache entry dies, meta is re-fetched, and the
    SAME search succeeds on the second attempt — invisible to the caller."""
    dims_seen = []

    def factory(meta, **kw):
        dims_seen.append(meta["dim"])
        if meta["dim"] == emb.dim:
            return emb
        return SimpleNamespace(
            embed=lambda texts, role="query": [np.zeros(meta["dim"], dtype=np.float32)])

    monkeypatch.setattr(kb_remote, "embedder_from_meta", factory)
    kb_remote._META_CACHE[remote.location] = {"model": "fake-bow", "dim": 7,
                                              "provider": "local", "query_instruction": ""}
    out = kb_entry.run_kb_search("agent framework", kb=PEER)
    assert dims_seen == [7, emb.dim]
    assert kb_remote._META_CACHE[remote.location]["dim"] == emb.dim
    assert out == _local(remote.svc.owner, {"query": "agent framework"},
                         query_vector(emb, "agent framework"))


def test_a_dead_service_is_an_empty_envelope_not_an_exception(remote, monkeypatch):
    """P3 at the transport: connection refused answers the way a missing file does, on all
    three entry points."""
    def refuse(*a, **kw):
        raise real_requests.ConnectionError("connection refused")

    monkeypatch.setattr(kb_remote, "requests", SimpleNamespace(post=refuse, get=refuse))
    out = kb_entry.run_kb_search("agent", kb=PEER)
    assert out["hits"] == [] and out["notices"]
    opened = kb_entry.kb_open("github:root/agentkit", kb=PEER)
    assert opened["error"]
    agg = kb_entry.kb_aggregate(kb=PEER)
    assert agg["total"] == 0 and agg["notices"]
