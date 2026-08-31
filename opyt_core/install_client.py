"""
opyt_core/install_client.py

Registers the OPYT MCP server in a desktop MCP client's config (Cursor, Claude Desktop,
Windsurf) by merging into that client's JSON file, instead of requiring a manual hand-edit.

Each client stores MCP servers as `{"mcpServers": {"<name>": {...}}}` in a per-client JSON
file. install() parses the existing file (or starts fresh), inserts/replaces our `opyt` key
inside `mcpServers`, backs up the prior file, and writes valid JSON. Idempotent (re-running
is a no-op) and reversible (--uninstall).

The server command is `uvx --from opyt==<version> opyt-mcp`, with `uvx` resolved to an absolute
path at install time. That means a registered client needs no venv, no checkout, and no Python
of its own — `uv` supplies the interpreter and the package. Requires `uv` on PATH when you run
the installer:

    opyt-install-client --cursor
    opyt-install-client --all
    opyt-install-client --cursor --uninstall

This is a CLI command, not an MCP tool, since a first-run installer must live outside the
thing it installs.

NOTE (v1 scope): the CLIENTS paths below are macOS locations only; add Linux/Windows paths
when distribution expands. The merge logic itself is OS-independent.
"""
from __future__ import annotations

import argparse
import json
import shutil
from importlib.metadata import version
from pathlib import Path

# Key we register under inside each client's "mcpServers" object; capitalized so Claude Code's
# tool namespacing (mcp__<key>__<tool>) reads "Opyt".
SERVER_KEY = "Opyt"

# Older registrations used a lowercase key; install() drops these so a config never carries
# both "opyt" and "Opyt".
_LEGACY_KEYS = ["opyt"]

# client name -> its MCP config file (macOS). Same merge core for all; only the path differs.
CLIENTS: dict[str, Path] = {
    "cursor": Path.home() / ".cursor" / "mcp.json",
    "claude-desktop": Path.home() / "Library" / "Application Support" / "Claude"
    / "claude_desktop_config.json",
    "windsurf": Path.home() / ".codeium" / "windsurf" / "mcp_config.json",
}


def _server_spec() -> dict:
    """{command, args} block to register: an ABSOLUTE path to `uvx`, told to fetch this exact
    published version of opyt and run its `opyt-mcp` entry point.

    Absolute, not bare `uvx`: Claude Desktop spawns a server from a GUI app, which inherits a
    minimal PATH and never sources a shell profile — so the `~/.local/bin` that `uv`'s installer
    adds to your rc file does not exist as far as the spawned process is concerned.

    Version-pinned to the running distribution, so a config records what it was installed
    against and `uvx` resolves the same build every launch. Re-running this installer after an
    upgrade rewrites the pin (the spec differs, so `_is_current` returns False).

    Before 2026-08-29 this registered `sys.executable` plus an absolute path to this checkout's
    `mcp_server/server.py`, which welded every config to one folder that could never move.
    """
    uvx = shutil.which("uvx") or str(Path.home() / ".local" / "bin" / "uvx")
    if not Path(uvx).exists():
        raise FileNotFoundError(
            "uvx not found. Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh")
    return {"command": uvx, "args": ["--from", f"opyt=={version('opyt')}", "opyt-mcp"]}


def _load(path: Path) -> tuple[dict, bool]:
    """Return (config_dict, was_malformed). Missing file -> ({}, False). A present-but-
    unparseable file returns ({}, True) so the caller can refuse to overwrite it blindly."""
    if not path.exists():
        return {}, False
    try:
        data = json.loads(path.read_text() or "{}")
        return (data if isinstance(data, dict) else {}), False
    except json.JSONDecodeError:
        return {}, True


def _backup(path: Path) -> Path | None:
    """Copy the current file aside before rewriting it. Returns the backup path, or None if
    there was nothing to back up. Uses a stable suffix rather than timestamped copies."""
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".opyt-bak")
    shutil.copy2(path, bak)
    return bak


def _write(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def _is_current(cfg: dict, spec: dict) -> bool:
    servers = cfg.get("mcpServers", {})
    # Current only if our key matches and no legacy key lingers.
    return servers.get(SERVER_KEY) == spec and not any(k in servers for k in _LEGACY_KEYS)


def install(client: str, *, force: bool = False, dry_run: bool = False) -> dict:
    """Merge the opyt server into `client`'s config. Idempotent. Refuses to touch a malformed
    existing file unless force=True, which backs it up then overwrites with an opyt-only
    config, discarding unparseable prior content."""
    path = CLIENTS[client]
    spec = _server_spec()
    cfg, malformed = _load(path)

    if malformed and not force:
        bak = _backup(path)
        return {"client": client, "path": str(path), "status": "ERROR_MALFORMED",
                "backup": str(bak) if bak else None,
                "detail": "existing config is not valid JSON; can't merge without losing "
                          "unreadable content. Fix it, or re-run with --force to overwrite "
                          "with an opyt-only config (a backup was made)."}

    if _is_current(cfg, spec):
        return {"client": client, "path": str(path), "status": "ALREADY_CURRENT"}

    # Ensure mcpServers{} exists, drop any legacy key, then set ours.
    servers = cfg.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    migrated = [k for k in _LEGACY_KEYS if servers.pop(k, None) is not None]
    existed = SERVER_KEY in servers
    servers[SERVER_KEY] = spec
    cfg["mcpServers"] = servers

    if dry_run:
        return {"client": client, "path": str(path),
                "status": "DRY_RUN", "would_write": cfg}

    bak = _backup(path)
    _write(path, cfg)
    return {"client": client, "path": str(path),
            "status": "UPDATED" if existed else "INSTALLED",
            "migrated_legacy": migrated or None,
            "backup": str(bak) if bak else None}


def uninstall(client: str, *, dry_run: bool = False) -> dict:
    """Remove the opyt key from `client`'s config (leaving every other server intact)."""
    path = CLIENTS[client]
    cfg, malformed = _load(path)
    if malformed:
        return {"client": client, "path": str(path), "status": "ERROR_MALFORMED",
                "detail": "config is not valid JSON; not touching it."}
    servers = cfg.get("mcpServers", {})
    if not isinstance(servers, dict) or SERVER_KEY not in servers:
        return {"client": client, "path": str(path), "status": "NOT_PRESENT"}
    if dry_run:
        return {"client": client, "path": str(path), "status": "DRY_RUN_REMOVE"}
    bak = _backup(path)
    del servers[SERVER_KEY]
    cfg["mcpServers"] = servers
    _write(path, cfg)
    return {"client": client, "path": str(path), "status": "REMOVED",
            "backup": str(bak) if bak else None}


# This module writes only the opyt server entry in a client's mcp.json; never a second config
# file on the side.


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="opyt_core.install_client",
        description="Register (or remove) the OPYT MCP server in a desktop MCP client's config.")
    for name in CLIENTS:
        ap.add_argument(f"--{name}", action="store_true", help=f"target {name}")
    ap.add_argument("--all", action="store_true", help="target every known client")
    ap.add_argument("--uninstall", action="store_true", help="remove opyt instead of adding it")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a malformed config (backs it up first; loses unreadable content)")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    args = ap.parse_args(argv)

    targets = list(CLIENTS) if args.all else [n for n in CLIENTS if getattr(args, n.replace("-", "_"))]
    if not targets:
        ap.error("pick at least one client (e.g. --cursor) or --all")

    rc = 0
    for client in targets:
        try:
            if args.uninstall:
                res = uninstall(client, dry_run=args.dry_run)
            else:
                res = install(client, force=args.force, dry_run=args.dry_run)
        except FileNotFoundError as e:
            res = {"client": client, "status": "ERROR", "detail": str(e)}
        if str(res.get("status", "")).startswith("ERROR"):
            rc = 1
        # Compact, scannable one-line-per-client report.
        line = f"[{res['status']}] {client} -> {res.get('path', '')}"
        if res.get("backup"):
            line += f"  (backup: {res['backup']})"
        if res.get("detail"):
            line += f"\n    {res['detail']}"
        print(line)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
