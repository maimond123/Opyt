"""`opyt-mcp` startup, pinned: bootstrap, then stdio.

Nothing here serves: `FastMCP.run` is recorded instead of called.
"""
from __future__ import annotations

import sys

import pytest
from fastmcp import FastMCP

from mcp_server import server
from opyt_core import bootstrap


@pytest.fixture()
def started(monkeypatch, tmp_path):
    """Record what `main` starts instead of starting it."""
    monkeypatch.setenv("OPYT_HOME", str(tmp_path))
    record: list = []
    monkeypatch.setattr(bootstrap, "ensure_initialized", lambda: record.append(("bootstrap",)))
    monkeypatch.setattr(FastMCP, "run", lambda self, **kw: record.append(("mcp.run", kw)))
    return record


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["server.py", *argv])
    server.main()


def test_no_arguments_bootstraps_then_serves_stdio(started, monkeypatch):
    _main(monkeypatch)
    assert started == [("bootstrap",), ("mcp.run", {})]


def test_a_failed_bootstrap_is_diagnosed_on_stderr_and_the_server_still_starts(
        started, monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "ensure_initialized",
                        lambda: (_ for _ in ()).throw(RuntimeError("no template")))
    _main(monkeypatch)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[bootstrap] skipped: no template" in captured.err
    assert started == [("mcp.run", {})]
