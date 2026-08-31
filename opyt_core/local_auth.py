"""
opyt_core/local_auth.py
One-shot loopback HTTP capture for browser-driven credential acquisition.

Keys enter here over 127.0.0.1 and go straight to `set_key`, so a secret never has to pass
through chat. Nothing captured is ever returned to an MCP caller, logged, or echoed.

Two routes, one server:
  • `cb`    — an OAuth redirect target; the credential arrives as a query param.
  • `paste` — a form page whose POST body carries a key that has no OAuth flow.

Loopback-only, single-request, plain HTTP by design — do not grow this into a UI or add TLS.
"""
from __future__ import annotations

import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

_DEFAULT_TIMEOUT = 300.0
_MAX_BODY = 8192          # a key is ~64 bytes; this is a hostile-input cap, not a real limit

_OK_PAGE = ("<!doctype html><meta charset=utf-8><title>OPYT</title>"
            "<body style='font:16px system-ui;padding:3rem'>"
            "<h2>Done — you can close this tab.</h2>"
            "<p>OPYT stored the value locally in <code>~/.opyt/.env</code>. "
            "It never left this machine.</p>")

_FORM_PAGE = ("<!doctype html><meta charset=utf-8><title>OPYT — paste your key</title>"
              "<body style='font:16px system-ui;padding:3rem;max-width:34rem'>"
              "<h2>Paste your {label} key</h2>"
              "<p>This page is served by OPYT on your own machine "
              "(<code>127.0.0.1</code>). The value is written to "
              "<code>~/.opyt/.env</code> and never leaves this computer.</p>"
              "<form method=post><input name=value type=password style='width:100%;"
              "padding:.6rem;font:inherit' autofocus autocomplete=off>"
              "<button style='margin-top:1rem;padding:.6rem 1.2rem;font:inherit'>"
              "Save</button></form>")


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
        if self.capture.route == "paste" and not params:
            return self._send(200, _FORM_PAGE.format(label=self.capture.label))
        self._send(200, _OK_PAGE)
        self.capture._deliver({"params": params, "form": {}})

    def do_POST(self):
        if not self._authorized():
            return self._send(404, "not found")
        n = min(int(self.headers.get("Content-Length") or 0), _MAX_BODY)
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(n).decode("utf-8", "replace")).items()}
        self._send(200, _OK_PAGE)
        self.capture._deliver({"params": {}, "form": form})


class Capture:
    """A one-shot loopback listener. Use as a context manager, then `wait()`.

    Binds TWO sockets on the SAME port — `127.0.0.1` and `::1` — because macOS resolves
    `localhost` to `::1`, and a single dual-stack socket would require the `::` wildcard,
    exposing the form to the LAN.
    """

    def __init__(self, *, route: str, timeout: float = _DEFAULT_TIMEOUT, label: str = ""):
        if route not in ("cb", "paste"):
            raise ValueError(f"route must be 'cb' or 'paste', got {route!r}")
        self.route, self.timeout, self.label = route, timeout, label
        self.nonce = secrets.token_urlsafe(32)
        self.path = f"/{route}/{self.nonce}"
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
        handler = type("_H", (_Handler,), {"capture": self})
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
        """Block until a valid request lands, or the timeout expires. None means timeout —
        Fail-safe: the caller degrades to a printed URL, it never dead-ends."""
        return self._payload if self._done.wait(self.timeout) else None

    def __exit__(self, *exc):
        for s in self._servers:
            s.shutdown()
            s.server_close()
        for t in self._threads:
            t.join(timeout=2)
        return False
