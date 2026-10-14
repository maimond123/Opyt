"""
opyt_core/launcher.py

How opyt is launched with no shell around it: `uvx_command`, the absolute-`uvx` argv for any
published entry point (a desktop MCP client's config, through `install_client`), and
`running_distribution`, which copy of opyt is running and the worker launcher that matches it
(the LaunchAgent, through `install_worker`).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


def uvx_command(entry_point: str) -> list[str]:
    """The argv that launches one of opyt's published entry points, with no checkout involved.

    An absolute path to `uvx`: launchd starts both unattended launchers (a desktop MCP client, a
    LaunchAgent) with a minimal PATH and no shell profile, so `~/.local/bin` is not on it.

    `opyt@latest` is the only form that lets a published fix reach an installed user: uvx
    reuses the cached environment for a pinned or bare requirement. Offline, it runs the cached
    build. Do not reintroduce a pin here.
    """
    uvx = shutil.which("uvx") or str(Path.home() / ".local" / "bin" / "uvx")
    if not Path(uvx).exists():
        raise FileNotFoundError(
            "uvx not found. Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh")
    return [uvx, "--from", "opyt@latest", entry_point]


def running_distribution() -> dict:
    """Which copy of opyt is executing right now — `{"kind", "root", "launcher"}`.

    A checkout and the published wheel report the same `version('opyt')`, so the version cannot
    tell them apart. `source` when the running package sits in a tree with its own
    `pyproject.toml` (an editable install), and it launches its own console script; `published`
    otherwise, launched through `uvx_command`.
    """
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").exists():
        script = Path(sys.executable).with_name("opyt-worker")
        if script.exists():
            return {"kind": "source", "root": str(root), "launcher": [str(script)]}
        # Editable install with no console script: run the module through this interpreter rather
        # than a published build that is not this code.
        return {"kind": "source", "root": str(root),
                "launcher": [sys.executable, "-m", "pipeline.worker.supervisor"]}
    return {"kind": "published", "root": str(root), "launcher": uvx_command("opyt-worker")}
