"""`oracle(action='screen')` says "still scoring" while the candidates' scores are being filled in,
so a thin, shifting roster is not read as the final ranking."""
from __future__ import annotations

import pytest

from pipeline.curation import collectors, still_scoring
from pipeline.platforms import connections


@pytest.fixture()
def platforms(monkeypatch):
    def set_(x=False, substack=False, pending=()):
        monkeypatch.setattr(connections, "_connected_x", lambda: x)
        monkeypatch.setattr(connections, "_connected_substack", lambda: substack)
        monkeypatch.setattr(collectors, "curation_pending",
                            lambda connected: sorted(set(pending) & connected))
    return set_


def test_a_connected_platform_never_read_is_still_scoring(kb_home, platforms):
    platforms(x=True, pending=["x"])
    scoring = still_scoring.still_scoring_notice()
    assert scoring["pending"] == ["x"] and "Still scoring" in scoring["message"]


def test_nothing_pending_says_nothing(kb_home, platforms):
    """A refresh of a platform already read changes counts, not who is a candidate: it is not
    "still scoring"."""
    platforms(x=True)
    assert still_scoring.still_scoring_notice() is None


def test_screen_carries_it(kb_home, platforms):
    from tests import mcp_session
    platforms(substack=True, pending=["substack"])
    screen = mcp_session.session().call("oracle", action="screen")
    assert screen["still_scoring"]["pending"] == ["substack"]
