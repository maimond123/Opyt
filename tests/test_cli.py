"""`opyt tools` and `opyt call`: the server's tools from a shell, for an agent with no MCP client.

Run against the real built server through `main`, so what is asserted is what the shell prints.
"""
from __future__ import annotations

import json

from mcp_server import cli


def run(capsys, *argv: str) -> tuple[int, dict | None, str]:
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, (json.loads(out) if out.strip() else None), err


def test_tools_lists_every_tool_with_its_description_and_whether_it_only_reads(capsys):
    code, out, _ = run(capsys, "tools")
    assert code == 0 and "CITING OPYT RESULTS" in out["instructions"]
    tools = {t["name"]: t for t in out["tools"]}
    assert tools["search"]["read_only"] is True and "query" in tools["search"]["parameters"]
    assert tools["forget"]["read_only"] is False
    assert all(t["description"] for t in tools.values())


def test_a_call_prints_the_answer_a_host_receives(capsys):
    code, out, _ = run(capsys, "call", "aggregate", "{}")
    assert code == 0 and "total" in out


def test_arguments_are_optional(capsys):
    code, out, _ = run(capsys, "call", "aggregate")
    assert code == 0 and "total" in out


def test_an_unknown_tool_is_refused_on_stderr(capsys):
    code, out, err = run(capsys, "call", "nope", "{}")
    assert code == 2 and out is None and "nope" in err


def test_arguments_that_are_not_a_json_object_are_a_usage_error(capsys):
    for bad in ("not json", "[1, 2]"):
        code, out, err = run(capsys, "call", "search", bad)
        assert code == 2 and out is None and "usage:" in err


def test_anything_else_prints_the_usage(capsys):
    for argv in ((), ("help",), ("tools", "extra")):
        code, out, err = run(capsys, *argv)
        assert code == 2 and out is None and "usage:" in err
