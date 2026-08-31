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
    """The same read against the same export as a FILE peer — the equality's right-hand side."""
    embedder = None
    if vector is not None:
        conn, _label = peers.open_peer(owner)
        try:
            embedder = PrecomputedEmbedder(conn, vector)
        finally:
            conn.close()
    return kb_entry.run_kb_search(**spec, kb=owner, embedder=embedder)


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
    assert opened == kb_entry.kb_open(hit["atom_id"], kb=owner)
    assert opened["raw"], "the snapshot body must survive the hop"
    for scope in (None, {"tags": ["ai-agents"]}, {"source_type": "github"}):
        assert (kb_entry.kb_aggregate(scope=scope, kb=PEER)
                == kb_entry.kb_aggregate(scope=scope, kb=owner)), scope


def test_revoked_token_is_a_notice_not_an_exception(remote):
    """Revocation is a server-side row delete, effective on the reader's next request — which
    must land as an empty envelope naming the fix (a new grant code), never as a raise."""
    store.revoke(remote.svc.owner, store.token_hash(remote.svc.reader_token))
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
