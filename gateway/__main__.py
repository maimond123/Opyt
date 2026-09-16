"""
gateway/__main__.py — `python -m gateway`.

One process, deliberately. There is no `--workers` knob and there must not be: each worker
would keep its own routing table and spawn its own child per user, so two workers would put
two `opyt-mcp` processes on one home. Every route here is I/O, so a second worker would buy
nothing anyway. Scale by making children cheaper, or by sharding users across boxes.

Bind to loopback and terminate TLS in front (nginx/caddy), the same shape as `service/`.
`OPYT_GATEWAY_BASE_URL` must be the PUBLIC https URL, because it is what the OAuth metadata
advertises and what Google redirects back to.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import uvicorn

from gateway.app import build_app

# Shared with `service/`, not copied. It configures error logs only here: access logging is off
# because hosted interaction capabilities live in request paths and must never reach a log.
LOG_CONFIG = Path(__file__).resolve().parent.parent / "service" / "log_config.json"


def main() -> None:
    if not LOG_CONFIG.exists():
        # Loud, not fail-safe. Falling back to uvicorn's default would silently start logging
        # every user's IP address, which is the one outcome this file exists to prevent.
        raise SystemExit(f"log config missing: {LOG_CONFIG}")

    uvicorn.run(
        build_app(),
        host=os.environ.get("OPYT_GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("OPYT_GATEWAY_PORT", 8080)),
        log_config=json.loads(LOG_CONFIG.read_text()),
        access_log=False,
    )


if __name__ == "__main__":
    main()
