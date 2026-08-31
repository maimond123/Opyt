"""The access log's format, pinned against the artifact that enforces it.

`uvicorn`'s default access formatter writes `%(client_addr)s` — every reader's IP against the
knowledge base they read. Nobody decided that, and the collection policy refuses it
(docs/plans/2026-08-27-what-opyt-collects.md, "What is never collected"). The config file is the
enforcement; this test is what stops the file from being edited back.

Asserted on the FILE, not on a running server: the deploy command names this file
(`--log-config service/log_config.json`), so the file is the thing that has to be right.
"""
from __future__ import annotations

import json
from pathlib import Path

LOG_CONFIG = Path(__file__).parents[2] / "service" / "log_config.json"


def test_access_log_format_has_no_client_addr():
    cfg = json.loads(LOG_CONFIG.read_text())
    fmt = cfg["formatters"]["access"]["fmt"]
    assert "client_addr" not in fmt
    assert "request_line" in fmt and "status_code" in fmt
