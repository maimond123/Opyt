"""The FastMCP `instructions` field — one paragraph of prose since 2026-09-16, none before.

W3 used to assert the routing bullets shipped: a DISCONFIRM checklist, a FRONTIER stage-4
model, a RETRIEVE/BROWSE/GROW/WRITE map. That blob was deleted (see
`docs/Future-Investigations/2026-08-13-the-deleted-instructions-blob.md` for the full text and
the four tests that pinned it). Those tests are gone with the text they pinned — a test that
asserts prose exists is exactly as stale as the prose, and keeping it would have blocked the
deletion rather than caught a bug.

What is tested now is the opposite property: `instructions` carries ONE paragraph of standing
prose and the frontier's session notice, and nothing else. The blob is not coming back by accident.

AMENDED 2026-09-16. The "no prose at all" half of that ruling was overturned for exactly one
paragraph — `server._CITATION_RULE` — on a measured failure. The tool-description channel was
tried four times (a rule in `search`'s own docstring, then `cite`, then the link appended to
`description`) and four hosted phone answers came back as linkless digests anyway; asked why,
the same model produced every link instantly, saying "my formatting dropped them". A rule
attached to a TOOL gets skimmed; a directive in the session's setup is the channel that gets
obeyed. So the test below no longer asserts None — it asserts that `instructions` is that one
paragraph and NOTHING more, which is the property actually worth defending. The 1,400-token blob
is still gone and still must not return.

⚠️ This does knowingly spend what the old ruling protected: a standing line always fires, and
text that always fires is text that stops being read. It is one paragraph rather than a map of
the whole surface, and nothing DEPENDS on it — `search`'s description still carries the rule for
clients that drop `instructions`, and every hit still ships `cite` either way.
"""
from __future__ import annotations

import asyncio


def test_instructions_is_the_citation_rule_alone_when_the_frontier_is_quiet(monkeypatch, tmp_path):
    """The default is ONE paragraph — the standing citation rule — and not a word more.

    This is the blob's deletion still being pinned, just at its amended boundary: any SECOND
    paragraph reintroduced at the FastMCP call site fails here first. Equality, not `in`, is the
    whole point; a test that only checked the rule was present would let a map of the surface
    grow back underneath it one line at a time.
    """
    from mcp_server import server

    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    assert server._setup_client_mcp().instructions == server._CITATION_RULE


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
    assert mcp.instructions == server._CITATION_RULE   # the rule survives a dead frontier
    names = [tool.name for tool in asyncio.run(mcp.list_tools())]
    assert "search" in names          # the surface still registered, under its new name
    assert "forget" in names          # removal is reachable through the actual server
    assert "opyt_search" not in names  # and not under the old one


def test_instructions_carries_the_rule_then_the_notice_when_the_frontier_speaks(monkeypatch):
    """When the frontier DOES have something, `instructions` is the rule followed by the notice —
    in that order, with nothing else between them. The rule leads because it governs every answer
    in the session, while the notice is one session's state."""
    from mcp_server import frontier_tools, server

    monkeypatch.setattr(frontier_tools, "notice",
                        lambda: {"unshown": 3, "total": 3,
                                 "top": {"title": "A Paper About Things"}})
    body = server._setup_client_mcp().instructions

    assert body is not None
    assert body.startswith("CITING OPYT RESULTS:")
    assert "FRONTIER STATE AT SESSION START:" in body
    assert "3 staged artifacts" in body
    assert "A Paper About Things" in body
    # The deleted blob's landmarks must not reappear.
    for gone in ("• DISCONFIRM", "• RETRIEVE", "• RADAR", "vault", "open_opyt"):
        assert gone not in body


def test_optional_registration_diagnostics_leave_stdio_stdout_empty(monkeypatch, capsys):
    """A failed optional tool must not corrupt the JSON-RPC transport."""
    from mcp_server import atoms_tools, server

    monkeypatch.setattr(atoms_tools, "register_atoms_tools",
                        lambda mcp: (_ for _ in ()).throw(RuntimeError("unavailable")))

    server._setup_client_mcp()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err
