"""R2 — the offer to share back, made once per peer and never repeated.

Sharing is one-directional (R2 rejected mutual-by-construction on a conversion argument:
requiring somebody to publish before they can read excludes anyone with an empty knowledge base,
which is most new installs). The second direction is an OFFER instead, made at the one moment it
is earned — the reader has just seen this work, on their own question.

Two properties, and the second is what makes the first worth anything: it fires on the first
read that returned something, and it never fires again. A latch in session state would re-offer
on every new session, which is a nag with extra steps, so the latch is on disk.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import mcp_server.atoms_tools as atoms_tools
from opyt_core import kb_remote
from pipeline.kb import peers
from tests.opyt_core.conftest import PEER


def _tools(mcp_stub):
    atoms_tools.register_atoms_tools(mcp_stub)
    return mcp_stub.got


@pytest.fixture()
def tools(remote, emb, monkeypatch):
    """The three atom tools, over the real served peer, with the session counters reset."""
    monkeypatch.setattr(kb_remote, "embedder_from_meta", lambda meta, **kw: emb)
    atoms_tools._reset_session()

    class _Fake:
        got: dict = {}

        def tool(self, *a, **kw):
            def deco(fn):
                self.got[fn.__name__] = fn
                return fn
            return deco

    stub = _Fake()
    stub.got = {}
    return SimpleNamespace(**_tools(stub))


def _offers(out) -> list[dict]:
    return [n for n in out["notices"] if n["code"] == "reciprocal_offer"]


def test_the_offer_rides_a_peers_first_answer(tools):
    out = tools.search("agent framework", kb=PEER)

    assert out["hits"], "the fixture corpus must answer this or the assertion is vacuous"
    offer = _offers(out)[0]
    assert offer["kb"] == PEER
    assert "share" in offer["message"]


def test_it_is_never_made_twice(tools):
    assert _offers(tools.search("agent framework", kb=PEER))
    assert _offers(tools.search("crypto rollup", kb=PEER)) == []
    assert _offers(tools.search("agent framework", kb=PEER)) == []


def test_it_survives_a_new_session(tools):
    """THE reason the latch is a file. Session-scoped, this re-offers every time the user opens
    a client — R2 says once, never repeated, and "once per session" is not once."""
    assert _offers(tools.search("agent framework", kb=PEER))

    atoms_tools._reset_session()
    assert _offers(tools.search("agent framework", kb=PEER)) == []


def test_a_search_of_your_own_store_never_offers(tools):
    """There is nobody to share back to. `kb=None` and `kb="me"` are the same store."""
    assert _offers(tools.search("agent framework")) == []
    assert _offers(tools.search("agent framework", kb="me")) == []


def test_an_empty_foreign_result_does_not_offer(tools):
    """Asking for a favour on the strength of a disappointment. The latch must also NOT trip, so
    the offer still gets made on the first read that actually works."""
    assert _offers(tools.search("zzz-nothing-matches-this-zzz", kb=PEER, mode="bm25")) == []
    assert _offers(tools.search("agent framework", kb=PEER))


def test_a_second_peer_gets_its_own_first_offer(tools, remote):
    """One latch per peer, not one per install: being asked about Alex's knowledge base says
    nothing about whether to ask about Sam's."""
    assert _offers(tools.search("agent framework", kb=PEER))
    peers.add("second", remote.location, "Another served KB", token=remote.svc.reader_token)

    offer = _offers(tools.search("agent framework", kb="second"))[0]
    assert offer["kb"] == "second"


def test_an_unwritable_marker_still_returns_results(tools, monkeypatch):
    """Fail-safe (P3): a notice is an addition to the answer and must never break it."""
    monkeypatch.setattr(atoms_tools.Path, "touch",
                        lambda self, *a, **kw: (_ for _ in ()).throw(OSError("read-only")))

    out = tools.search("agent framework", kb=PEER)
    assert out["hits"]


def test_the_frontier_notice_and_the_offer_never_both_ride(tools, monkeypatch):
    """Mutually exclusive by construction. Frontier's queue is the READER's own staged
    artifacts, so riding it on a foreign result would tell them their backlog grew because they
    looked at somebody else's knowledge base."""
    monkeypatch.setattr("mcp_server.frontier_tools.notice", lambda: {"staged": 7})

    foreign = tools.search("agent framework", kb=PEER)
    assert _offers(foreign) and "frontier" not in foreign

    own = tools.search("agent framework")
    assert _offers(own) == [] and own.get("frontier")
