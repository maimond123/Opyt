"""
mcp_server/rider_hook.py — the FastMCP middleware that runs the rider flow on every tool answer:
bind the call's arguments to the tool's own defaults, attach the riders, re-wrap the result.

It is the one place every `tools/call` passes through, so no tool module needs to know riders
exist. The re-wrapped result is built from the new answer the same way FastMCP builds one from a
tool's dict, so the wire text stays the answer serialized (pinned in
`tests/kb/test_server_riders_pins.py`).
"""
from __future__ import annotations

import sys
import time

import anyio
from fastmcp import FastMCP
from fastmcp.server.middleware import Middleware
from fastmcp.tools.base import ToolResult

from mcp_server import riders
from mcp_server.rider_rules import Call


def bind(parameters: dict, arguments: dict) -> dict:
    """The call's arguments with every omitted parameter at its declared default."""
    defaults = {name: spec["default"]
                for name, spec in (parameters.get("properties") or {}).items()
                if "default" in spec}
    return {**defaults, **arguments}


class RiderHook(Middleware):
    def __init__(self, mcp: FastMCP):
        self._mcp = mcp

    async def on_call_tool(self, context, call_next):
        started = time.monotonic()
        result = await call_next(context)
        if not isinstance(result, ToolResult) or not isinstance(result.structured_content, dict):
            return result
        answer = result.structured_content
        name = context.message.name
        tool = await self._mcp.get_tool(name)
        call = Call(name, bind(tool.parameters, context.message.arguments or {}))
        answered = time.monotonic()
        # The producers read SQLite, so the flow runs off the event loop, the way FastMCP runs a
        # sync tool body.
        attached = await anyio.to_thread.run_sync(riders.attach, call, answer)
        # One line per call, so a slow answer shows whether the tool or its riders spent the time.
        label = f"{name}/{call.args['action']}" if call.args.get("action") else name
        # stderr, never stdout: stdout is the stdio JSON-RPC transport.
        print(f"[tools] {label}: tool {answered - started:.1f}s, "
              f"riders {time.monotonic() - answered:.1f}s", file=sys.stderr, flush=True)
        if attached is answer:
            return result
        return ToolResult(structured_content=attached, meta=result.meta)
