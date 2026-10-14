"""The FastMCP `instructions` field — one paragraph of prose.

`instructions` carries ONE paragraph of standing prose and nothing else; a long routing blob
is not coming back by accident. A test that asserts prose exists is as stale as the prose, so
the test pins the boundary, not the wording.

The one paragraph is `instructions.CITATION_RULE`, added on a measured failure. The tool-description channel was
tried four times (a rule in `search`'s own docstring, then `cite`, then the link appended to
`description`) and four hosted phone answers came back as linkless digests anyway; asked why,
the same model produced every link instantly, saying "my formatting dropped them". A rule
attached to a TOOL gets skimmed; a directive in the session's setup is the channel that gets
obeyed. So the test below asserts that `instructions` is that one
paragraph and NOTHING more, which is the property actually worth defending.

A standing line always fires, and
text that always fires is text that stops being read. It is one paragraph rather than a map of
the whole surface, and nothing DEPENDS on it — `search`'s description still carries the rule for
clients that drop `instructions`, and every hit still ships `cite` either way.
"""
from __future__ import annotations

import asyncio


def test_instructions_is_the_citation_rule_alone(monkeypatch, tmp_path):
    """ONE paragraph — the standing citation rule — and not a word more.

    Any SECOND
    paragraph reintroduced at the FastMCP call site fails here first. Equality, not `in`, is the
    whole point; a test that only checked the rule was present would let a map of the surface
    grow back underneath it one line at a time.
    """
    from mcp_server import build, instructions

    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    mcp = build.server()
    assert mcp.instructions == instructions.CITATION_RULE
    names = [tool.name for tool in asyncio.run(mcp.list_tools())]
    assert "search" in names          # the surface still registered, under its new name
    assert "forget" in names          # removal is reachable through the actual server
    assert "opyt_search" not in names  # and not under the old one


def test_optional_registration_diagnostics_leave_stdio_stdout_empty(monkeypatch, capsys):
    """A failed optional tool must not corrupt the JSON-RPC transport."""
    from mcp_server import atoms_tools, build

    monkeypatch.setattr(atoms_tools, "register_atoms_tools",
                        lambda mcp: (_ for _ in ()).throw(RuntimeError("unavailable")))

    build.server()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err
