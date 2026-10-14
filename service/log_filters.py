"""
service/log_filters.py — keep visitors' addresses and credentials out of the process log.

`log_config.json` drops the address from access lines. A WebSocket's lines take another path:
uvicorn writes "<address> - "WebSocket <path>" [accepted]" through its error logger, and the X
sign-in page's stream is a WebSocket. `NoClientAddr` blanks the address on those lines, as the
collection policy requires (`TELEMETRY.md`: addresses are never logged).

`NoTokensInPaths` keeps credentials that ride a request's URL out of both kinds of line: every
query string (the connector's sign-in returns from Google with its one-time code and state there)
and the X sign-in's link and session, each of which is a path segment that drives a live sign-in.
"""
from __future__ import annotations

import logging
import re

_SIGNIN_TOKEN = re.compile(r"^(/x-signin/(?:s/)?)[^/]+")


def log_path(path: str) -> str:
    """A request path as the log may hold it: no query string, no X sign-in link or session."""
    return _SIGNIN_TOKEN.sub(r"\1-", path.split("?", 1)[0])


def _websocket(record: logging.LogRecord) -> bool:
    return (isinstance(record.msg, str) and record.msg.startswith('%s - "WebSocket')
            and isinstance(record.args, tuple) and len(record.args) >= 2)


class NoClientAddr(logging.Filter):
    """Replace the address uvicorn puts first on a WebSocket line with "-"."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _websocket(record):
            record.args = ("-", *record.args[1:])
        return True


class NoTokensInPaths(logging.Filter):
    """Pass the path on an access line (address, method, path, version, status) or a WebSocket
    line (address, path, …) through `log_path`."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if record.name == "uvicorn.access" and isinstance(args, tuple) and len(args) == 5:
            record.args = (*args[:2], log_path(str(args[2])), *args[3:])
        elif _websocket(record):
            record.args = (args[0], log_path(str(args[1])), *args[2:])
        return True
