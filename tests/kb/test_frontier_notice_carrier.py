"""Frontier stage 4 — the PUSH half, where the notice rides `search`.

The pull tool solves half the problem: it makes an autonomous discovery rail depend on the user
remembering it exists, which is the rail's own argument turned against it. The notice is the other
half, and everything it must do is about restraint:

  • ABSENT WHEN QUIET. `None` means the key is not in the envelope at all, so an unrelated
    conversation pays nothing for the frontier existing.
  • AT MOST ONCE PER SESSION. Twice is nagging, and every search would otherwise pay for a
    ranking pass over a table that only grows.
  • THE LATCH TRIPS ON AN EMIT, NOT ON A CHECK. Stage 2 runs detached and stages candidates
    mid-session; latching on a quiet look would blind the rest of the session to them.
  • IT NEVER BREAKS ITS CARRIER. A follower that can take down `search` is a worse bug than
    a frontier nobody sees.
"""
from __future__ import annotations

import json

import pytest

from mcp_server import atoms_tools
from pipeline.kb import frontier_surface as fs
from pipeline.kb import schema


@pytest.fixture(autouse=True)
def _fresh_session():
    atoms_tools._reset_session()
    yield
    atoms_tools._reset_session()


def _stage(conn, cid="arxiv:2501.00001"):
    conn.execute(
        """INSERT INTO frontier_candidates
             (candidate_id, source, title, url, published, summary, payload,
              status, first_seen_at, last_seen_at)
           VALUES (?,'arxiv',?,?,?,?,'{}','new',?,?)""",
        (cid, f"title {cid}", f"https://example/{cid}", "2026-08-11", "s" * 900,
         "2026-08-11", "2026-08-11"))
    conn.commit()
    return cid


def test_the_key_is_absent_when_the_frontier_is_quiet(kb_home):
    out = {"hits": []}
    atoms_tools._attach_frontier_notice(out)
    assert "frontier" not in out, "silence must be genuinely zero-footprint"


def test_the_notice_is_attached_once_and_not_again(kb_home):
    conn = schema.connect()
    _stage(conn)
    conn.close()

    first = {"hits": []}
    atoms_tools._attach_frontier_notice(first)
    assert first["frontier"]["unshown"] == 1

    second = {"hits": []}
    atoms_tools._attach_frontier_notice(second)
    assert "frontier" not in second, "twice in one session is nagging"


def test_a_quiet_check_does_not_spend_the_session_latch(kb_home):
    """Stage 2 is detached: candidates can land AFTER the first search of a session. If the latch
    tripped on the quiet look, everything staged mid-session would go unannounced until the user
    happened to start a new one."""
    early = {"hits": []}
    atoms_tools._attach_frontier_notice(early)
    assert "frontier" not in early

    conn = schema.connect()                     # stage 2 runs, mid-session
    _stage(conn)
    conn.close()

    later = {"hits": []}
    atoms_tools._attach_frontier_notice(later)
    assert later["frontier"]["unshown"] == 1, "the latch tripped on a check, not on an emit"


def test_a_broken_notice_costs_the_notice_and_nothing_else(kb_home, monkeypatch):
    from mcp_server import frontier_tools
    monkeypatch.setattr(frontier_tools, "notice",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    out = {"hits": [], "notices": []}
    atoms_tools._attach_frontier_notice(out)
    assert out == {"hits": [], "notices": []}


def test_the_notice_carries_a_count_and_a_pointer_never_the_digest(kb_home):
    conn = schema.connect()
    for i in range(5):
        _stage(conn, f"arxiv:{i}")
    conn.close()

    out = {}
    atoms_tools._attach_frontier_notice(out)
    n = out["frontier"]
    assert n["unshown"] == 5 and n["call"] == "frontier()"
    assert set(n["top"]) == {"candidate_id", "source", "title", "url", "published"}
    assert "candidates" not in n, "the ranked list costs nothing until someone asks for it"


def test_the_notice_does_not_mark_anything_shown(kb_home):
    """A push is an OFFER of attention, not a spend of it. Marking a candidate shown because a
    one-line count mentioned it would demote something nobody actually looked at."""
    conn = schema.connect()
    _stage(conn)
    conn.close()

    atoms_tools._attach_frontier_notice({})

    conn = schema.connect()
    assert conn.execute("SELECT COUNT(*) FROM frontier_candidate_events").fetchone()[0] == 0
    conn.close()


# ── The wiring itself ───────────────────────────────────────────────────────────
class _FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def test_search_really_carries_it(kb_home):
    """End-to-end through the registered tool, so the attachment cannot rot as a function nothing
    calls. v1 rode `search_papers_live`, which is not the tool anyone actually calls — this is.
    (`search_papers_live` was deleted outright 2026-08-16; this test is what the notice rides now.)

    `mode="bm25"` keeps the embedder (a paid call) out of it: what is under test is the envelope,
    not retrieval quality.
    """
    conn = schema.connect()
    _stage(conn)
    conn.close()

    mcp = _FakeMCP()
    atoms_tools.register_atoms_tools(mcp)
    out = mcp.tools["search"]("anything at all", mode="bm25")

    assert set(out) >= {"hits", "notices", "insights", "trace"}
    assert out["frontier"]["unshown"] == 1
    assert "frontier" not in mcp.tools["search"]("a second query", mode="bm25")
