"""
opyt_core/bootstrap.py — first-run initialization so a FRESH machine works.

A stranger who installs OPYT has no ~/.opyt/settings.yaml and inherits the shipped template.
ensure_initialized() gives them a user-local config of their own, and never overwrites an
existing ~/.opyt/settings.yaml.

Everything here is idempotent and runs on EVERY session (plain stdio FastMCP, no daemon — one
server process per session, called from `mcp_server.server.main()` before serving). It is a
re-asserted preamble, not a one-time setup step.

`settings.yaml` writing is the only remaining step; the vault dir / empty-db touch and the
dev-checkout guard were removed 2026-08-13 as dead/broken. See
"""
from __future__ import annotations

import json

import yaml

from . import config


def ensure_config() -> str | None:
    """Write a user-local ~/.opyt/settings.yaml from the shipped template IFF this is a fresh
    install (no user config). Returns the written path, or None when one already exists.

    This is also the install receipt — the one artifact `ensure_initialized` leaves on disk at
    first run, and therefore the file to look at to answer "did my install work?"."""
    home = config.opyt_home()
    user_cfg = home / "settings.yaml"
    if user_cfg.exists():
        return None                      # respect an existing user config

    base = config.settings()             # the template (repo/packaged default)
    # a fresh user starts with no tracked people (don't inherit the author's list)
    cp = base.get("credible_people")
    if isinstance(cp, dict):
        cp["profiles"] = []

    home.mkdir(parents=True, exist_ok=True)
    user_cfg.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return str(user_cfg)


# Bootstrap does not create the database, deliberately. The store creates itself on the first
# write-capable `pipeline/kb/schema.py::connect()` (mkdirs + `init_kb_schema`); every read path
# opens write-capable so a never-ingested store returns empty instead of "no such table". This
# will need to change at the first real schema migration — see


def ensure_initialized() -> dict:
    """Idempotent per-session preamble. ONE step: write a user config if this is a fresh install.

    Runs on EVERY session, not once per machine — see the module docstring."""
    return {"config_written": ensure_config()}


if __name__ == "__main__":
    print(json.dumps(ensure_initialized(), indent=2))
