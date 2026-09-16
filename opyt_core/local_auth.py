"""
opyt_core/local_auth.py
One-shot loopback HTTP capture for browser-driven credential acquisition.

Keys enter here over 127.0.0.1 and go straight to `set_key`, so a secret never has to pass
through chat. Nothing captured is ever returned to an MCP caller, logged, or echoed.

ONE route: `cb`, an OAuth redirect target, where the credential arrives as a query param.
A second `paste` route served a form for keys with no OAuth flow. Its caller was deleted on
2026-09-05 under the `retired-key-paste-module` guard, and this half outlived it by four days
on the strength of one test — its only caller. No remaining credential needs a form path (that
guard's message says why GitHub and Semantic Scholar are not it), so `do_POST`, the form page
and the `route`/`label` parameters went on 2026-09-09. The rule, kept where the story is not:
a form on the loopback is a credential-entry surface any local process can reach, so it needs
a live caller to exist at all.

Loopback-only, single-request, plain HTTP by design — no second route, no navigation, no TLS,
and nothing that fetches a stylesheet or an asset, because a single-request listener cannot
serve one. The RESPONSE BODY is not covered by that rule: it renders `opyt_core.web_panel`,
which is one self-contained string. That was settled on 2026-09-09, when the bare `<h2>` this
used to return turned out to be the first Opyt-served page any local user ever sees.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from opyt_core import web_panel

_DEFAULT_TIMEOUT = 300.0


# NO FILESYSTEM PATH. This page named the keys file until 2026-09-09 — first as a hardcoded
# `~/.opyt/.env`, which was wrong under `$OPYT_HOME`, then briefly as the resolved path, which
# was right and still not wanted. Somebody who has just authorised a third party needs the
# reassurance and the next step, not the location of a dotfile holding a secret. Resolving the
# path was patching the premise; deleting it is the fix.
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

    Binds TWO sockets on the SAME port — `127.0.0.1` and `::1` — because macOS resolves
    `localhost` to `::1`, and a single dual-stack socket would require the `::` wildcard,
    which would put a credential capture on the LAN.
    """

    def __init__(self, *, timeout: float = _DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.nonce = secrets.token_urlsafe(32)
        self.path = f"/cb/{self.nonce}"
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


# ── The other half of a browser-driven acquisition, shared by every flow that runs one ────────
# `Capture` is the loopback END of the round trip; these two are its START. They live here
# rather than in one flow's module because there are now TWO flows — `openrouter_oauth`
# (the user's own OpenRouter account) and `trial` (a key the gateway mints for them) — and a
# second private copy of either is the shape drift starts in. Neither touches a credential:
# one derives a public challenge, the other opens a URL.

def pkce_pair() -> tuple[str, str]:
    """A PKCE (verifier, S256 challenge). The VERIFIER never leaves the process that made it.

    RFC 7636 S256: only the challenge travels through the browser, so an authorization code
    stolen in transit is useless without the verifier that never went anywhere.
    """
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def open_browser(url: str) -> bool:
    """Best effort. False is a normal outcome, not an error: a headless box, a locked-down
    desktop or a remote shell all land here, and every caller degrades to handing the user
    the URL instead of dead-ending on it."""
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False
