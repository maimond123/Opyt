"""The HTTP branch of the `kb=` entry points, proven against the real service in-process.

The one theorem: `kb=` naming a SERVED knowledge base returns the same envelope as `kb=` naming
the same export on disk. Everything else here is the failure half — a revoked token, a dead
service, an unpublished export — each of which must come back
as an envelope with a sentence, never as an exception out of a tool call (P3).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests as real_requests

from pipeline.retrieval import reads as kb_entry
from opyt_core import service_http
from pipeline.retrieval import peers
from service import tokens
from tests.kb.test_export import _QUERIES
from tests.opyt_core.conftest import PEER

# Every spec crosses, `entry_mode` included — see tests/service/test_fidelity.py for why it
# stopped being an exception.


def _local(owner, spec):
    """The same read against the same export as a FILE peer — the equality's right-hand side.

    `as_kb=PEER` is what makes this an equality at all after R4. The routing key selects the
    store on both sides; the label is what the reader's install calls it, and the served side
    sends its own. So the theorem is now stronger than "the envelope crosses verbatim": the
    answer is identical AND already carries the reader's own name for the knowledge base."""
    return kb_entry.run_kb_search(**spec, kb=owner, as_kb=PEER)


@pytest.mark.parametrize("spec", _QUERIES, ids=lambda s: json.dumps(s, sort_keys=True))
def test_remote_search_equals_local(remote, spec):
    """The envelope crosses verbatim: a served peer and a file peer are indistinguishable
    downstream, for every shape the read path has."""
    got = kb_entry.run_kb_search(**spec, kb=PEER)
    assert got == _local(remote.svc.owner, spec)


def test_remote_open_and_aggregate_equal_local(remote):
    owner = remote.svc.owner
    hit = kb_entry.run_kb_search("agent framework", kb=PEER)["hits"][0]
    opened = kb_entry.kb_open(hit["atom_id"], kb=PEER)
    assert opened == kb_entry.kb_open(hit["atom_id"], kb=owner, as_kb=PEER)
    assert opened["raw"], "the snapshot body must survive the hop"
    for scope in (None, {"tags": ["ai-agents"]}, {"source_type": "github"}):
        assert (kb_entry.kb_aggregate(scope=scope, kb=PEER)
                == kb_entry.kb_aggregate(scope=scope, kb=owner, as_kb=PEER)), scope


def test_every_kb_field_carries_the_readers_own_name_not_the_routing_key(remote):
    """R4's whole payoff, at the surface a host model actually reads.

    The URL path segment is the routing key — after R5 an opaque `secrets.token_hex(6)` that
    nobody types. Without `as_kb` the envelope came back naming it: hit cards saying
    `kb: "a3f9c2e1"`, a trace saying the same, and a `foreign_kb` notice instructing the host to
    pass that string back to `open()` — which resolves nowhere on this install, because the
    reader registered the peer as `x`. Rewriting those strings afterwards would mean editing
    prose inside a notice; sending the name instead means nothing is ever rewritten."""
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


def test_the_name_the_notice_hands_back_actually_opens_the_atom(remote):
    """The `foreign_kb` notice is meant to be repeated verbatim, so the string in it has to be a
    name this install resolves. Following the instruction is the test."""
    out = kb_entry.run_kb_search("agent framework", kb=PEER)
    hit = out["hits"][0]

    opened = kb_entry.kb_open(hit["atom_id"], kb=hit["kb"])
    assert opened.get("error") is None
    assert opened["atom_id"] == hit["atom_id"]


def test_a_reader_cannot_relabel_a_served_kb_as_their_own(remote):
    """`as_kb` arrives in an HTTP body, so it is untrusted. Every `kb_name == LOCAL_KB` test
    downstream means "this is my own store", and `kb_open` would read a paper's full text live on
    the SERVER. `_label_as` refuses that one name, so the label falls back to the
    routing key rather than turning a foreign read into a local one."""
    owner = remote.svc.owner
    hit = kb_entry.run_kb_search("agent framework", kb=PEER)["hits"][0]

    body = {"atom_id": hit["atom_id"], "as_kb": "me"}
    got = remote.svc.client.post(f"/v1/kb/{owner}/open", json=body,
                                 headers=remote.svc.reader_hdr).json()
    assert got["kb"] == owner


def test_revoked_token_is_a_notice_not_an_exception(remote):
    """Revocation is a server-side row delete, effective on the reader's next request — which
    must land as an empty envelope naming the fix (a new grant code), never as a raise."""
    tokens.revoke(remote.svc.owner, tokens.token_hash(remote.svc.reader_token))
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
    empty_hdr = {"Authorization": f"Bearer {tokens.mint_token('newcomer', 'owner')}"}
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


def test_a_dead_service_is_an_empty_envelope_not_an_exception(remote, monkeypatch):
    """P3 at the transport: connection refused answers the way a missing file does, on all
    three entry points."""
    def refuse(*a, **kw):
        raise real_requests.ConnectionError("connection refused")

    monkeypatch.setattr(service_http, "requests", SimpleNamespace(post=refuse, get=refuse))
    out = kb_entry.run_kb_search("agent", kb=PEER)
    assert out["hits"] == [] and out["notices"]
    opened = kb_entry.kb_open("github:root/agentkit", kb=PEER)
    assert opened["error"]
    agg = kb_entry.kb_aggregate(kb=PEER)
    assert agg["total"] == 0 and agg["notices"]
