"""Riders on a FOREIGN search, pinned through the real server and a real served peer.

The share-back offer rides a peer's first answer that found something, once per peer, latched on
disk; the `latest` count of the reader's own arrivals never rides a foreign answer. Pinned here,
beside the served-peer fixtures, through `tests/mcp_session.py`, like
`tests/kb/test_server_riders_pins.py`.
"""
from __future__ import annotations

import pytest

from pipeline.arrivals import notice as arrivals
from tests import mcp_session
from tests.kb.test_server_riders_pins import ARRIVED, LANDED, on_calls
from tests.opyt_core.conftest import PEER
from pipeline.oracles import told


@pytest.fixture()
def mcp(remote):
    return mcp_session.session()


def _codes(answer) -> list[str]:
    return [n.get("code") for n in answer["notices"]]


def test_a_peers_first_answer_carries_the_offer_last_and_no_latest_notice(mcp, monkeypatch):
    monkeypatch.setattr(arrivals, "notice", on_calls(ARRIVED))
    monkeypatch.setattr(told, "completion_notice", on_calls(LANDED, 1))
    out = mcp.call("search", query="agent framework", kb=PEER)

    assert out["hits"], "the fixture corpus must answer this or the assertion is vacuous"
    assert _codes(out)[-2:] == ["pull_finished", "reciprocal_offer"]
    assert out["notices"][-1]["kb"] == PEER
    assert "latest" not in out
    assert arrivals.notice.calls == [], "arrivals are the reader's own; never asked here"


def test_the_offer_is_made_once_per_peer_across_sessions(mcp):
    assert "reciprocal_offer" in _codes(mcp.call("search", query="agent framework", kb=PEER))
    assert "reciprocal_offer" not in _codes(mcp.call("search", query="agent framework", kb=PEER))
    later = mcp_session.session()                               # a new conversation
    assert "reciprocal_offer" not in _codes(later.call("search", query="agent framework", kb=PEER))


def test_a_peer_answer_with_no_hits_makes_no_offer(mcp):
    out = mcp.call("search", query="zzzz qqqq", kb=PEER)
    assert out["hits"] == []
    assert "reciprocal_offer" not in _codes(out)
