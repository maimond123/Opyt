"""
opyt_core/bootstrap.py — first-run initialization so a fresh machine works.

Everything here is idempotent and runs on every session, from `mcp_server.server.main()` before
serving.
"""
from __future__ import annotations

import yaml

from opyt_core import config, install_worker
from opyt_core.paths import opyt_home, owner_only


def ensure_config() -> str | None:
    """Write a user-local ~/.opyt/settings.yaml from the shipped template if none exists yet.
    Returns the written path, or None when one already exists.

    This is also the install receipt: the file that answers "did my install work?"."""
    home = opyt_home()
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
    owner_only(user_cfg)  # names what the user tracks
    return str(user_cfg)


# Bootstrap does not create the database: the store creates itself on the first write-capable
# `pipeline/kb/schema.py::connect()`.


def ensure_initialized() -> dict:
    """Idempotent per-session preamble: write a user config if this is a fresh install, and
    repair a sticky-disabled worker agent if the machine has one.

    The repair lives here because a session start is the only event guaranteed for a user who
    onboarded long ago. It queues nothing; it re-loads the process onboarding installed.
    `mcp_server.server.main` catches anything raised here."""
    return {"config_written": ensure_config(), "worker_repair": install_worker.repair()}

