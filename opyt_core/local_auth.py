"""
opyt_core/local_auth.py
One-shot loopback HTTP capture for browser-driven credential acquisition.

Keys enter here over 127.0.0.1 and go straight to `set_key`, so a secret never has to pass
through chat. Nothing captured is ever returned to an MCP caller, logged, or echoed.

One route, `cb`: an OAuth redirect target where the credential arrives as a query param. The key
form for credentials with no sign-in flow is `opyt_core/key_page.py`, which reuses this listener
with its own handler; a form on the loopback is a credential-entry surface any local process can
reach, so it lives only from the `onboard(start=...)` call that opens it until its key is
collected, ten minutes at most.

Loopback-only, single-request, plain HTTP by design — no second route, no navigation, no TLS,
no assets. The response body is one self-contained `opyt_core.web_panel` string.
"""
from __future__ import annotations

import secrets
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from opyt_core import web_panel

_DEFAULT_TIMEOUT = 300.0


# Names no filesystem path: the user needs reassurance and the next step, not where the secret is.
_OK_PAGE = web_panel.render(
    "Opyt", ok=True,
    headline="Done — you can close this tab.",
    detail="Opyt stored the key on this machine and it never left it. Go back to your Claude "
           "conversation to carry on.")


class _Handler(BaseHTTPRequestHandler):
    capture: "Capture"          # injected per-instance below

    def log_message(self, *a):
        """Silence. stdout is the MCP JSON-RPC channel — a stray access-log line corrupts it."""

    def _authorized(self) -> bool:
        got = urlparse(self.path).path
        return secrets.compare_digest(got, self.capture.path)

    def _send(self, code: int, body: str):
        raw = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        # A bad nonce returns 404, not a shutdown — any local process can hit this port.
        if not self._authorized():
            return self._send(404, "not found")
        params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        self._send(200, _OK_PAGE)
        self.capture._deliver({"params": params})


class Capture:
    """A one-shot loopback listener. Use as a context manager, then `wait()`.

    Binds two sockets on the same port — `127.0.0.1` and `::1` — because macOS resolves
    `localhost` to `::1`, and a single dual-stack socket would require the `::` wildcard,
    which would put a credential capture on the LAN.
    """

    handler: type[_Handler] = _Handler      # what answers the one route
    route = "cb"

    def __init__(self, *, timeout: float = _DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.nonce = secrets.token_urlsafe(32)
        self.path = f"/{self.route}/{self.nonce}"
        self.port = 0
        self._servers: list[HTTPServer] = []
        self._threads: list[threading.Thread] = []
        self._done = threading.Event()
        self._payload: dict | None = None

    @property
    def url(self) -> str:
        """Always spelled `localhost`, never an IP literal — OAuth providers and users both
        expect that, and both sockets answer to it."""
        return f"http://localhost:{self.port}{self.path}"

    def _deliver(self, payload: dict):
        if not self._done.is_set():
            self._payload = payload
            self._done.set()

    def _bind(self, family, host, port) -> HTTPServer:
        cls = type("_S", (HTTPServer,), {"address_family": family})
        handler = type("_H", (self.handler,), {"capture": self})
        return cls((host, port), handler)

    def __enter__(self) -> "Capture":
        # Take an ephemeral port on v4, then claim the same port on v6; retry since another
        # process can grab the v6 side of that port in the gap.
        last: Exception | None = None
        for _ in range(8):
            v4 = self._bind(socket.AF_INET, "127.0.0.1", 0)
            port = v4.server_address[1]
            try:
                v6 = self._bind(socket.AF_INET6, "::1", port)
            except OSError as e:
                v4.server_close()
                last = e
                continue
            self.port, self._servers = port, [v4, v6]
            break
        else:
            raise OSError(f"could not bind a loopback port on both stacks: {last}")

        for s in self._servers:
            t = threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.1},
                                 daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def wait(self) -> dict | None:
        """Block until a valid request lands, or the timeout expires. None means timeout; the
        caller degrades to a printed URL."""
        return self._payload if self._done.wait(self.timeout) else None

    def __exit__(self, *exc):
        for s in self._servers:
            s.shutdown()
            s.server_close()
        for t in self._threads:
            t.join(timeout=2)
        return False


# ── The start of a browser-driven acquisition ──────────────────────────────────────────────────

def open_browser(url: str) -> bool:
    """Best effort. False is a normal outcome (headless box, remote shell); callers then hand
    the user the URL."""
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False
