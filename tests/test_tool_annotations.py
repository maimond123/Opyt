"""Every tool the server lists says what it does to the user's world, in MCP annotations.

A tool with none gets the spec's defaults, which treat it as a destructive, open-world write:
ChatGPT asks before every call, Claude's connector directory refuses the server, and a read-only
grant (Muse) has nothing to grant. Each tool declares its own at its decorator in
`mcp_server/*_tools.py`; this holds every tool to that, including ones added later.

Read through a real client, so what is asserted is what a host receives in `tools/list`.
"""
from __future__ import annotations

import asyncio

from fastmcp import Client

from mcp_server import build

DECLARED = {"title", "readOnlyHint", "destructiveHint", "openWorldHint"}


def _annotations() -> dict[str, dict]:
    """`{tool name: its annotations as the wire carries them}` for every tool the server lists."""
    async def listed():
        async with Client(build.server()) as client:
            return await client.list_tools()

    return {t.name: (t.annotations.model_dump(by_alias=True, exclude_none=True)
                     if t.annotations else {})
            for t in asyncio.run(listed())}


def test_every_tool_declares_its_annotations():
    missing = {name: sorted(DECLARED - set(ann)) for name, ann in _annotations().items()
               if not DECLARED <= set(ann)}
    assert not missing, missing


def test_only_the_reads_are_read_only_and_only_removal_is_destructive():
    """Pinned as sets, so a tool that changes sides, or a new tool on either side, is a decision
    made here rather than a default."""
    annotations = _annotations()
    assert {n for n, a in annotations.items() if a.get("readOnlyHint")} == {
        "search", "open", "aggregate", "latest", "look"}
    assert {n for n, a in annotations.items() if a.get("destructiveHint")} == {"forget", "unshare"}
