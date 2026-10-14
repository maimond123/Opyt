"""
mcp_server/cli.py — Opyt's tools from a shell, for an agent that has a terminal but no MCP client
(the agent on a Muse VM).

  opyt tools                     the server's instructions, then every tool: its name,
                                 description, parameters and whether it only reads
  opyt call <tool> ['<json>']    one call, its answer printed as JSON

A call goes through a real MCP client against the built server, in this process, so the answer is
exactly what an MCP host receives, riders included (`rider_hook.py`). Like `server.py`, it
bootstraps the home and starts no background work: `opyt-worker` collects.

Exit status: 0 for an answer (an answer can still say `"status": "error"`), 2 for a usage error or
a call the server refused.
"""
from __future__ import annotations

import asyncio
import json
import sys

from fastmcp import Client

from mcp_server import build
from opyt_core import bootstrap

USAGE = "usage: opyt tools | opyt call <tool> ['<json object of arguments>']"


def main(argv: list[str] | None = None) -> int:
    """Console entrypoint (`opyt`)."""
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["tools"] and len(args) == 1:
        request = None
    elif args[:1] == ["call"] and len(args) in (2, 3):
        try:
            arguments = json.loads(args[2]) if len(args) == 3 else {}
        except json.JSONDecodeError as e:
            return _usage(f"the arguments are not JSON ({e.msg})")
        if not isinstance(arguments, dict):
            return _usage("the arguments must be a JSON object")
        request = (args[1], arguments)
    else:
        return _usage()
    try:
        bootstrap.ensure_initialized()
    except Exception as e:
        build.diagnostic(f"[bootstrap] skipped: {e}")
    if request is None:
        _print(asyncio.run(_tools()))
        return 0
    answer, error = asyncio.run(_call(*request))
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    _print(answer)
    return 0


async def _tools() -> dict:
    async with Client(build.server()) as client:
        tools = await client.list_tools()
        return {"instructions": client.instructions,
                "tools": [{"name": t.name, "description": t.description,
                           "parameters": t.input_schema.get("properties", {}),
                           "read_only": bool(t.annotations and t.annotations.read_only_hint)}
                          for t in tools]}


async def _call(tool: str, arguments: dict) -> tuple[object, str | None]:
    """The answer, or the server's refusal (an unknown tool, a bad argument, a tool that raised)."""
    async with Client(build.server()) as client:
        result = await client.call_tool(tool, arguments, raise_on_error=False)
    text = result.content[0].text if result.content else ""
    if result.is_error:
        return None, f"opyt: {tool}: {text or 'the call failed'}"
    return (result.structured_content if result.structured_content is not None else text), None


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _usage(why: str | None = None) -> int:
    print(f"opyt: {why}\n{USAGE}" if why else USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
