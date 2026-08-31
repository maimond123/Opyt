"""The FastMCP `instructions` field — which as of 2026-08-13 carries NO prose.

W3 used to assert the routing bullets shipped: a DISCONFIRM checklist, a FRONTIER stage-4
model, a RETRIEVE/BROWSE/GROW/WRITE map. That blob was deleted (see
`docs/Future-Investigations/2026-08-13-the-deleted-instructions-blob.md` for the full text and
the four tests that pinned it). Those tests are gone with the text they pinned — a test that
asserts prose exists is exactly as stale as the prose, and keeping it would have blocked the
deletion rather than caught a bug.

What is tested now is the opposite property: `instructions` carries ONLY the frontier's session
notice, and it is None when the frontier is quiet. The blob is not coming back by accident.
"""
from __future__ import annotations

import asyncio


def test_instructions_is_none_when_the_frontier_is_quiet(monkeypatch, tmp_path):
    """The default is NO instructions at all.

    `instructions` is optional on `InitializeResult` (declared in `mcp.types`). Sending None is the
    honest encoding of "this server has nothing to say up front" — an empty string is a field
    the client must still handle. This also pins the blob's deletion: any prose reintroduced at
    the FastMCP call site fails here first.
    """
    from mcp_server import server

    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    assert server._setup_client_mcp().instructions is None


def test_the_session_notice_is_absent_when_the_frontier_is_quiet(monkeypatch, tmp_path):
    """`_frontier_session_notice` must return "" rather than a standing line. Text that always
    fires is text that stops being read, and it costs context for the whole session."""
    from mcp_server import server

    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    assert server._frontier_session_notice() == ""


def test_the_session_notice_never_breaks_the_server(monkeypatch):
    """It is built at startup. A DB hiccup here must cost the notice, not the server.

    This is the CLAUDE.md fail-safe invariant at the startup path. It matters more since the
    blob was deleted: the notice used to be appended to 1,400 tokens of prose that would have
    shipped regardless, and it is now the entire `instructions` value. So the failure mode moved
    from "the server starts with slightly less text" to "the server starts or it does not."
    """
    from mcp_server import frontier_tools, server

    monkeypatch.setattr(frontier_tools, "notice",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert server._frontier_session_notice() == ""

    mcp = server._setup_client_mcp()
    assert mcp.instructions is None
    names = [tool.name for tool in asyncio.run(mcp.list_tools())]
    assert "search" in names          # the surface still registered, under its new name
    assert "opyt_search" not in names  # and not under the old one


def test_instructions_carries_only_the_notice_when_the_frontier_speaks(monkeypatch):
    """When the frontier DOES have something, that is the whole of `instructions` — no preamble
    in front of it, and no leading blank lines left over from when it was appended to a blob."""
    from mcp_server import frontier_tools, server

    monkeypatch.setattr(frontier_tools, "notice",
                        lambda: {"unshown": 3, "total": 3,
                                 "top": {"title": "A Paper About Things"}})
    body = server._setup_client_mcp().instructions

    assert body is not None
    assert body.startswith("FRONTIER STATE AT SESSION START:")
    assert "3 staged artifacts" in body
    assert "A Paper About Things" in body
    # The deleted blob's landmarks must not reappear.
    for gone in ("• DISCONFIRM", "• RETRIEVE", "• RADAR", "vault", "open_opyt"):
        assert gone not in body
