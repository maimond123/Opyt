"""
mcp_server/build.py — assemble the FastMCP server: its identity, the setup text, and every unit's
tools. This module defines no tools of its own; each unit's tool module registers its own.

`server.py` calls `server()` once per process, in both transports. Nothing here starts work:
the resident `opyt-worker` is the only thing that launches a rail (see `server.py`).
"""
from __future__ import annotations

import importlib
import sys

from fastmcp import FastMCP
from mcp.types import Icon

from mcp_server import instructions
from mcp_server.rider_hook import RiderHook

ICON = Icon(src="https://api.useopyt.com/icon.png", mimeType="image/png", sizes=["512x512"])

# Every unit's tool module and the function that registers it, in registration order.
#   atoms_tools    — search / open / aggregate, the LLM-free "trusted router" over the atom KB.
#   oracle_tools   — `oracle`, the "pick your people" screen, and `add_oracle`.
#   onboard_tools  — `onboard`, the setup steps in order.
#   arrivals_tools — `latest`, what arrived since the user last looked, and why.
#   hopper_tools   — `hopper`, the one manual "keep this" surface: one link to one atom.
#   forget_tools   — `forget`, remove atoms, an Oracle, or one of an Oracle's sources.
#   share_tools    — `share` / `accept` / `unshare`, knowledge-base sharing.
#   topic_tools    — `watch`: the topics the user follows, checked by the host.
#   look_tools     — `look`: what the watch sources have on a few named things, now.
#   page_tools     — `reader_view`: the sources behind an answer, on one private page.
TOOL_MODULES = (
    ("mcp_server.atoms_tools", "register_atoms_tools"),
    ("mcp_server.oracle_tools", "register_oracle_tools"),
    ("mcp_server.onboard_tools", "register_onboard_tools"),
    ("mcp_server.arrivals_tools", "register_arrivals_tools"),
    ("mcp_server.hopper_tools", "register_hopper_tools"),
    ("mcp_server.forget_tools", "register_forget_tools"),
    ("mcp_server.share_tools", "register_share_tools"),
    ("mcp_server.topic_tools", "register_topic_tools"),
    ("mcp_server.look_tools", "register_look_tools"),
    ("mcp_server.page_tools", "register_page_tools"),
)


def diagnostic(message: str) -> None:
    """Write startup diagnostics without corrupting the stdio JSON-RPC transport."""
    print(message, file=sys.stderr, flush=True)


def register_optional(mcp: FastMCP, module: str, register: str) -> None:
    """Register one unit's tools, or diagnose why not, so a stripped distribution (e.g. one
    without `pipeline.kb`) still starts with whatever surface it can offer."""
    try:
        getattr(importlib.import_module(module), register)(mcp)
    except Exception as e:
        diagnostic(f"[{module.rpartition('.')[2]}] registration skipped: {e}")


def server() -> FastMCP:
    """The server every session talks to, with the rider flow on every tool answer."""
    mcp = FastMCP("Opyt", instructions=instructions.CITATION_RULE, icons=[ICON])
    mcp.add_middleware(RiderHook(mcp))
    for module, register in TOOL_MODULES:
        register_optional(mcp, module, register)
    return mcp
