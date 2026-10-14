"""
opyt_core/install_client.py

Registers the OPYT MCP server in a desktop MCP client's config (Cursor, Claude Desktop,
Windsurf) by merging into that client's JSON file, instead of requiring a manual hand-edit.

Each client stores MCP servers as `{"mcpServers": {"<name>": {...}}}` in a per-client JSON
file. install() parses the existing file (or starts fresh), inserts/replaces our `opyt` key
inside `mcpServers`, backs up the prior file, and writes valid JSON. Idempotent and reversible
(--uninstall, which also removes the background worker once no client still registers OPYT).

The server command is `uvx --from opyt@latest opyt-mcp`, so a registered client needs no venv,
checkout or Python of its own. Requires `uv` on PATH when you run the installer:

    opyt-install-client --cursor
    opyt-install-client --all
    opyt-install-client --cursor --uninstall

"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from opyt_core import install_worker
from opyt_core.launcher import uvx_command

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
    """The `{command, args}` block a client's mcp.json wants, around that launcher."""
    command, *args = uvx_command("opyt-mcp")
    return {"command": command, "args": args}


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


def merge_server(cfg: dict, spec: dict) -> tuple[dict, list[str], bool]:
    """`cfg` with our entry set under `mcpServers` and any legacy key dropped. Returns
    (config, legacy keys dropped, whether our key was already there)."""
    servers = cfg.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    migrated = [k for k in _LEGACY_KEYS if servers.pop(k, None) is not None]
    existed = SERVER_KEY in servers
    servers[SERVER_KEY] = spec
    cfg["mcpServers"] = servers
    return cfg, migrated, existed


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

    cfg, migrated, existed = merge_server(cfg, spec)
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


def _still_registered() -> tuple[list[str], list[str]]:
    """(clients that still register OPYT, clients whose config could not be read)."""
    kept, unreadable = [], []
    for client, path in CLIENTS.items():
        cfg, malformed = _load(path)
        if malformed:
            unreadable.append(client)
            continue
        servers = cfg.get("mcpServers", {})
        if isinstance(servers, dict) and any(k in servers for k in (SERVER_KEY, *_LEGACY_KEYS)):
            kept.append(client)
    return kept, unreadable


def remove_worker_if_unused(*, dry_run: bool = False) -> dict:
    """After an uninstall, remove the background worker too — unless OPYT is still in use.

    A removed OPYT must leave nothing relaunching. The worker stays while any known client still
    registers OPYT, or while a config cannot be read.
    """
    if sys.platform != "darwin":
        return {"status": "NOT_APPLICABLE"}
    kept, unreadable = _still_registered()
    if kept or unreadable:
        return {"status": "KEPT", "still_registered": kept, "unreadable": unreadable}
    return install_worker.uninstall(dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    args, targets = _parse(argv)
    rc = 0
    for client in targets:
        res = _run(client, args)
        if str(res.get("status", "")).startswith("ERROR"):
            rc = 1
        print(_client_line(client, res))
    if args.uninstall:
        line = _worker_line(remove_worker_if_unused(dry_run=args.dry_run))
        if line:
            print(line)
    return rc


def _parse(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
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
    return args, targets


def _run(client: str, args: argparse.Namespace) -> dict:
    try:
        if args.uninstall:
            return uninstall(client, dry_run=args.dry_run)
        return install(client, force=args.force, dry_run=args.dry_run)
    except FileNotFoundError as e:
        return {"client": client, "status": "ERROR", "detail": str(e)}


def _client_line(client: str, res: dict) -> str:
    """Compact, scannable one-line-per-client report."""
    line = f"[{res['status']}] {client} -> {res.get('path', '')}"
    if res.get("backup"):
        line += f"  (backup: {res['backup']})"
    if res.get("detail"):
        line += f"\n    {res['detail']}"
    return line


def _worker_line(res: dict) -> str | None:
    if res["status"] == "KEPT":
        why = ", ".join(res["still_registered"] + [f"{c} (unreadable config)"
                                                   for c in res["unreadable"]])
        return f"[KEPT] background worker — Opyt is still set up in {why}"
    if res["status"] == "NOT_APPLICABLE":
        return None
    return f"[{res['status']}] background worker -> {res.get('path', '')}"


if __name__ == "__main__":
    raise SystemExit(main())
