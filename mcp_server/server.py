#!/usr/bin/env python3
"""
OPYT Knowledge Base MCP Server

The process entry: bootstrap the home, build the server (`build.py`), serve stdio. It starts no
background work: the resident `opyt-worker` is the only thing that launches a rail; a tool that
needs one makes it due through `pipeline/worker/job_queue.request_now`.

Usage:
  python mcp_server/server.py
"""

import sys
from pathlib import Path

# Launched as `python mcp_server/server.py`, which puts mcp_server/ on sys.path but not the repo
# root; without this every tool module fails to import and the server starts with no tools.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server import build  # noqa: E402 — needs the repo root on sys.path, above
from opyt_core import bootstrap  # noqa: E402


def main() -> None:
    """Console entrypoint (`opyt-mcp`): serve stdio.

    It launches no rail: the process dies with its MCP client, so a rail it started would be
    killed mid-ingest or outlive its parent untracked."""
    try:
        bootstrap.ensure_initialized()
    except Exception as e:
        build.diagnostic(f"[bootstrap] skipped: {e}")

    build.server().run()


if __name__ == "__main__":
    main()
