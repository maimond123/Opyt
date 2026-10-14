"""
opyt_core/key_page.py
A one-time key page on the loopback, for a credential with no sign-in flow (OpenAlex, Semantic
Scholar): the user pastes the key into a page on their own machine, so it never passes through the
chat.

  GET  http://localhost:<port>/key/<nonce>   → one form, one field
  POST http://localhost:<port>/key/<nonce>   → the key, held in-process for `take`

Two calls, like GitHub's device sign-in: `begin` opens the page and returns its address for the
host to give the user as a link (nothing opens a browser, so the page never appears unannounced),
and `take`, on the next call, collects the key the page took. The page lives in this process
between the two, answers only on its nonce path, takes one key, and closes itself after
`LIFETIME`. `take` hands the value to its in-process caller, which checks it before storing;
nothing here stores, logs or echoes it, and no MCP caller ever sees it.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from html import escape as html_escape
from urllib.parse import parse_qs

from opyt_core import local_auth, web_panel

_MAX_BODY = 4096          # a key is under 200 characters; anything bigger is not one
LIFETIME = 600.0          # seconds a page waits for its key before it closes itself


def _form_html(name: str, signup_url: str) -> str:
    """The page: one password field, the link to get a key, nothing else."""
    form = (
        "<form method=post style=\"margin-top:1.4rem;display:grid;gap:.7rem\">"
        f"<input name=key type=password autocomplete=off required maxlength=512 "
        f"aria-label=\"{html_escape(name)} key\" placeholder=\"{html_escape(name)} key\" "
        "style=\"font:inherit;padding:.6rem;border:1px solid var(--line);border-radius:6px\">"
        "<button style=\"font:inherit;padding:.6rem;border:0;border-radius:6px;"
        "background:var(--text);color:#fff;cursor:pointer\">Save</button></form>"
        f"<p style=\"margin-top:1rem\"><a href=\"{html_escape(signup_url)}\" target=_blank "
        "rel=noopener>Get a key</a></p>")
    return web_panel.render(
        "Opyt", ok=True, headline=f"Add your {name} key",
        detail="It is stored on this machine only and never sent to the chat.", body=form)


_RECEIVED_PAGE = web_panel.render(
    "Opyt", ok=True,
    headline="Received — you can close this tab.",
    detail="Go back to your Claude conversation: Opyt checks the key there before keeping it.")


class _FormHandler(local_auth._Handler):
    """Serves the form on GET and takes the key on POST, both on the nonce path only."""
    capture: "KeyCapture"

    def do_GET(self):
        if not self._authorized():
            return self._send(404, "not found")
        self._send(200, self.capture.form)

    def do_POST(self):
        if not self._authorized():
            return self._send(404, "not found")
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= _MAX_BODY:
            return self._send(400, "bad request")
        fields = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
        key = (fields.get("key") or [""])[0].strip()
        if not key:
            return self._send(400, "bad request")
        self._send(200, _RECEIVED_PAGE)
        self.capture._deliver({"key": key})


class KeyCapture(local_auth.Capture):
    """`local_auth.Capture` answering one key form instead of an OAuth redirect."""
    handler = _FormHandler
    route = "key"

    def __init__(self, *, name: str, signup_url: str, timeout: float):
        super().__init__(timeout=timeout)
        self.form = _form_html(name, signup_url)


@dataclass
class _Open:
    capture: KeyCapture
    closer: threading.Timer
    expires_at: float


# The page waiting for a key, at most one.
_open: _Open | None = None


def begin(*, name: str, signup_url: str) -> str:
    """Open a fresh page, closing any earlier one, and return its address."""
    global _open
    close()
    capture = KeyCapture(name=name, signup_url=signup_url, timeout=0)
    capture.__enter__()
    closer = threading.Timer(LIFETIME, capture.__exit__, args=(None, None, None))
    closer.daemon = True
    closer.start()
    _open = _Open(capture, closer, time.monotonic() + LIFETIME)
    return capture.url


def waiting() -> str | None:
    """The open page's address while it still waits for its key, else None."""
    if _open is None or _open.expires_at <= time.monotonic() or _open.capture.wait():
        return None
    return _open.capture.url


def take() -> str | None:
    """The key the page took, closing the page; None when no key has arrived."""
    got = _open.capture.wait() if _open is not None else None
    if got is None:
        return None
    close()
    return got["key"]


def close() -> None:
    """Close the open page, if there is one."""
    global _open
    if _open is not None:
        _open.closer.cancel()
        _open.capture.__exit__(None, None, None)
        _open = None
