"""One real MCP client session against the built server, called synchronously from a test.

Tool answers reach a host through `tools/call`, where the server's rider flow attaches notices
(`mcp_server/rider_hook.py`). A test that registers tools on a fake MCP and calls them directly
skips that path, so it cannot see a rider. This is the path a host sees: the real FastMCP server,
an in-process client, and every answer checked against its own wire text.

`session()` opens one; `tests/conftest.py` closes every open session after each test, so a test
can hold one across calls the way a host holds one across a conversation.
"""
from __future__ import annotations

import functools
import json
from contextlib import ExitStack

from anyio.from_thread import start_blocking_portal
from fastmcp import Client

from mcp_server import build


class McpSession:
    def __init__(self):
        mcp = build.server()          # built before anything is opened, so a failure leaks nothing
        self._stack = ExitStack()
        try:
            self._portal = self._stack.enter_context(start_blocking_portal())
            self._client = self._stack.enter_context(
                self._portal.wrap_async_context_manager(Client(mcp)))
            self._params = {tool.name: list(tool.parameters.get("properties") or {})
                            for tool in self._portal.call(mcp.list_tools)}
        except BaseException:
            self._stack.close()
            raise

    def call(self, tool: str, *args, **kwargs) -> dict:
        """The tool's answer as the host receives it. Positional arguments map onto the tool's
        parameters in order, like a direct call to the tool function."""
        kwargs.update(zip(self._params[tool], args, strict=False))
        result = self._portal.call(lambda: self._client.call_tool(tool, kwargs))
        answer = result.structured_content
        assert result.content[0].text == json.dumps(answer, separators=(",", ":"),
                                                    ensure_ascii=False)
        return answer

    def tools(self) -> dict:
        """`{name: callable}` for every registered tool, each calling through this session."""
        return {name: functools.partial(self.call, name) for name in self._params}

    @property
    def instructions(self) -> str | None:
        return self._client.instructions

    def close(self) -> None:
        self._stack.close()


_OPEN: list[McpSession] = []


def session() -> McpSession:
    opened = McpSession()
    _OPEN.append(opened)
    return opened


def close_all() -> None:
    while _OPEN:
        _OPEN.pop().close()
