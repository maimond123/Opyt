"""The access log's format, pinned against the artifact that enforces it.

`uvicorn`'s default access formatter writes `%(client_addr)s` — every reader's IP against the
knowledge base they read. Nobody decided that, and the collection policy refuses it, per the collection policy. The config file is the
enforcement; this test is what stops the file from being edited back.

Asserted on the FILE, not on a running server: the deploy command names this file
(`--log-config service/log_config.json`), so the file is the thing that has to be right.
"""
from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path

import pytest

LOG_CONFIG = Path(__file__).parents[2] / "service" / "log_config.json"


def logged(name: str, msg: str, args: tuple) -> str:
    """The line the config writes for one uvicorn record, after the logger's filters."""
    logging.config.dictConfig(json.loads(LOG_CONFIG.read_text()))
    record = logging.LogRecord(name, logging.INFO, __file__, 0, msg, args, None)
    for f in logging.getLogger(name).filters:
        f.filter(record)
    handler = logging.getLogger("uvicorn.access").handlers[0]
    return handler.format(record) if name == "uvicorn.access" else record.getMessage()


def test_access_log_format_has_no_client_addr():
    cfg = json.loads(LOG_CONFIG.read_text())
    fmt = cfg["formatters"]["access"]["fmt"]
    assert "client_addr" not in fmt
    assert "request_line" in fmt and "status_code" in fmt


def test_websocket_lines_carry_no_address_and_no_session():
    """A WebSocket's accept line goes through uvicorn's error logger with the address first (the
    X sign-in page's stream); the config's filters blank it, and the session in the
    path."""
    line = logged("uvicorn.error", '%s - "WebSocket %s" [accepted]',
                  ("203.0.113.7:51234", "/x-signin/s/sess-secret/stream"))
    assert "203.0.113.7" not in line and "sess-secret" not in line
    assert '"WebSocket /x-signin/s/-/stream" [accepted]' in line


@pytest.mark.parametrize("path, kept", [
    ("/auth/callback?state=st-secret&code=4/code-secret&scope=openid", "/auth/callback"),
    ("/authorize?client_id=c&state=st-secret&code_challenge=x", "/authorize"),
    ("/x-signin/link-secret", "/x-signin/-"),
    ("/x-signin/link-secret/claim", "/x-signin/-/claim"),
    ("/x-signin/s/sess-secret/status", "/x-signin/s/-/status"),
    ("/v1/x-signin/collect", "/v1/x-signin/collect"),
    ("/v1/kb/a3f9c2e1b704/feed?since=2026-10-12&since_id=x", "/v1/kb/a3f9c2e1b704/feed"),
])
def test_access_lines_carry_no_credential_from_the_url(path, kept):
    """The connector's sign-in returns from Google with a one-time code in the query string, and
    the X sign-in's link and session are path segments; none of them reaches the log."""
    line = logged("uvicorn.access", '%s - "%s %s HTTP/%s" %d',
                  ("203.0.113.7:51234", "GET", path, "1.1", 200))
    assert f'"GET {kept} HTTP/1.1" 200' in line
    assert "secret" not in line and "?" not in line and "203.0.113.7" not in line
