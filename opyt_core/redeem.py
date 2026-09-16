"""
opyt_core/redeem.py — the install id, and nothing else.

THE FILE IS NAMED FOR A COMMAND THAT NO LONGER EXISTS. `opyt-redeem` was deleted on 2026-09-05:
`mcp_server/share_tools.accept` does the same job strictly better — it finds the code inside a
link, a fragment or a bare string. A bare code uses `config.service_url()`; a full link uses its
own host, with `useopyt.com` mapping to `config.DEFAULT_SERVICE_URL`. The command made the reader
type the URL as a positional argument. What is left here is the one function `accept` imports.

The module survives only to hold it. Do NOT read the name as evidence that a redeem path lives
here; `share_tools.accept` is the only one, and `pipeline/kb/peers.py` is where the row it writes
is explained.
"""
from __future__ import annotations

import uuid

from opyt_core.paths import opyt_path


def get_install_id() -> str:
    """A random id, minted once per install and sent with the redeem, so the service can count
    distinct installations (TELEMETRY.md: `tokens.install_id` — no account behind it, never
    linked to a person). Its one caller is `mcp_server/share_tools.accept`."""
    p = opyt_path("install_id")
    if p.exists():
        return p.read_text().strip()
    iid = uuid.uuid4().hex
    p.parent.mkdir(parents=True, exist_ok=True)   # accept may be the first opyt call ever made
    p.write_text(iid)
    return iid
