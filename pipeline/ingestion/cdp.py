"""
pipeline/ingestion/cdp.py

Drive a Chromium browser OPYT launched, over the Chrome DevTools Protocol, using nothing
but the standard library.

Why this exists: reading a Chromium cookie store from Python means asking macOS for the
browser's `<Browser> Safe Storage` keychain key, and a foreign process asking for it is
exactly what raises the native Keychain dialog. A browser OPYT launches decrypts its own
store as itself — it is on that keychain item's ACL — so cookies come back in plaintext over
this protocol with no dialog at all. See
docs/plans/2026-08-30-cdp-cookie-transplant-and-guided-login.md for the measured basis.

Stdlib-only on purpose. A WebSocket client is ~60 lines of socket + struct, and the
alternative (playwright, websockets, websocket-client) lands on the dependency budget
pyproject.toml:46-58 defends — including the `cryptography<49` pin that keeps `uvx --from
opyt` compiler-free on x86_64 Macs.

macOS-focused, like `browser_cookies`: `app_path` is an .app binary path and the Safe Storage
reasoning above is Keychain-specific.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# Chrome writes the port it actually bound here once it is ready to accept CDP. Reading it
# (rather than passing a fixed port) is what lets two OPYT processes run at once without
# colliding — every DEBUGGED launch passes --remote-debugging-port=0. `launch` below is the
# exception: a sign-in window carries no debugger at all.
_PORT_FILE = "DevToolsActivePort"

_LAUNCH_TIMEOUT = 30.0      # subprocess: can hang forever, so it gets a timeout
_CALL_TIMEOUT = 30.0        # network-shaped (loopback socket), same reasoning


class CDPError(RuntimeError):
    """A controlled-browser session could not be established or a call was refused."""


# ── browser process ──────────────────────────────────────────────────────────────

# Chrome writes ~130 MB of ML models and component payloads into EVERY profile it is handed, and
# hosted that is per USER, for the same bytes every time. Measured 2026-09-12 on the box against a
# throwaway profile that had visited ONE sign-in page: 46 MB `optimization_guide_model_store`,
# 38 MB `component_crx_cache`, 23 MB `WasmTtsEngine`, 7.6 MB `OnDeviceHeadSuggestModel` -- against
# 8 MB of actual profile. The same run with these flags leaves 5.5 MB total, a 96% cut, and it is
# the difference between roughly 4,000 and 30,000 users fitting on one disk.
#
# Cookies are the one part of a profile OPYT ever reads, and they are untouched: the two profiles
# were compared directly and both held the same 15 cookies for the same hosts.
#
# ⚠️ `--disable-background-networking` also stops variations and Safe Browsing list fetches.
# Neither is load-bearing for signing in, but it is the first flag to drop if a browser launched
# here ever starts behaving oddly.
LEAN_PROFILE_FLAGS = (
    "--disable-component-update",
    "--disable-background-networking",
    "--disable-features=OptimizationGuideModelDownloading,OptimizationHints,"
    "OptimizationHintsFetching,OptimizationTargetPrediction,TextSafetyClassifier,"
    "WasmTtsComponentUpdaterEnabled",
)


def chrome_argv(app_path: Path | str, user_data_dir: Path | str) -> list[str]:
    """The opening argv every Chrome OPYT launches shares: its profile, and no bloat in it.

    A builder rather than a constant because the profile is half of it, and a builder rather
    than three copies because a launch site that forgets `LEAN_PROFILE_FLAGS` costs 130 MB per
    user and nothing visible -- the failure is silent, which is exactly the kind that survives.
    Callers append their own flags and put the URL last.
    """
    return [str(app_path), f"--user-data-dir={user_data_dir}",
            "--no-first-run", "--no-default-browser-check", *LEAN_PROFILE_FLAGS]


def _launch(app_path: Path, user_data_dir: Path, *, headless: bool,
            url: str | None) -> subprocess.Popen:
    """Start `app_path` on `user_data_dir` with CDP enabled on an OS-assigned port."""
    # Chrome leaves this marker behind when its process is terminated during a gateway restart.
    # It is metadata for the previous browser, not a lock: reading it before Chrome writes a
    # fresh one connects to a dead port and makes the next owned session look unavailable.
    (user_data_dir / _PORT_FILE).unlink(missing_ok=True)
    argv = chrome_argv(app_path, user_data_dir) + ["--remote-debugging-port=0"]
    if headless:
        argv.append("--headless=new")
    if url:
        argv.append(url)
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_for_endpoint(user_data_dir: Path, proc: subprocess.Popen,
                       timeout: float = _LAUNCH_TIMEOUT) -> str:
    """Block until the browser publishes its DevTools endpoint, and return the ws:// URL.

    The file's two lines are the bound port and the browser-target path. Polling it is the
    only supported way to learn the port when launching with `--remote-debugging-port=0`."""
    path = user_data_dir / _PORT_FILE
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise CDPError(f"browser exited (code {proc.returncode}) before opening DevTools")
        try:
            port, target = path.read_text().splitlines()[:2]
            return f"ws://127.0.0.1:{int(port)}{target}"
        except (FileNotFoundError, ValueError, IndexError):
            time.sleep(0.1)
    raise CDPError(f"browser did not publish {_PORT_FILE} within {timeout:.0f}s")


# ── WebSocket (RFC 6455 client subset: text frames, masked out, unmasked in) ──────

class _Socket:
    def __init__(self, url: str, timeout: float = _CALL_TIMEOUT):
        u = urlparse(url)
        self._sock = socket.create_connection((u.hostname, u.port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self._sock.sendall(
            f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
            f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise CDPError("DevTools closed the connection during the handshake")
            buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise CDPError(f"DevTools refused the upgrade: {head.splitlines()[0]!r}")
        self._buf = rest
        self._next_id = 0
        self._events: list[dict] = []

    def _need(self, n: int) -> None:
        while len(self._buf) < n:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise CDPError("DevTools closed the connection")
            self._buf += chunk

    def _send_frame(self, payload: bytes, opcode: int = 0x1) -> None:
        mask = os.urandom(4)
        n = len(payload)
        if n < 126:
            head = struct.pack("!BB", 0x80 | opcode, 0x80 | n)
        elif n < 65536:
            head = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, n)
        else:
            head = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, n)
        self._sock.sendall(head + mask
                           + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def _recv_message(self) -> bytes:
        """One complete message, reassembling continuation frames and answering pings."""
        out = b""
        while True:
            self._need(2)
            b0, b1 = self._buf[0], self._buf[1]
            fin, opcode, n, off = b0 & 0x80, b0 & 0x0F, b1 & 0x7F, 2
            if n == 126:
                self._need(4); n = struct.unpack("!H", self._buf[2:4])[0]; off = 4
            elif n == 127:
                self._need(10); n = struct.unpack("!Q", self._buf[2:10])[0]; off = 10
            self._need(off + n)
            payload, self._buf = self._buf[off:off + n], self._buf[off + n:]
            if opcode == 0x8:
                raise CDPError("DevTools sent a close frame")
            if opcode == 0x9:            # ping — answer it and keep waiting
                self._send_frame(payload, opcode=0xA)
                continue
            if opcode == 0xA:            # pong — not ours to interpret
                continue
            out += payload
            if fin:
                return out

    def call(self, method: str, params: dict | None = None, *,
             session_id: str | None = None) -> dict:
        """Issue one CDP command and return its `result`, skipping interleaved events."""
        self._next_id += 1
        call_id = self._next_id
        message = {"id": call_id, "method": method, "params": params or {}}
        if session_id is not None:
            message["sessionId"] = session_id
        self._send_frame(json.dumps(message).encode())
        while True:
            msg = json.loads(self._recv_message())
            if msg.get("id") != call_id:
                if "method" in msg:
                    self._events.append(msg)
                continue                 # a protocol event, or an earlier call's reply
            if "error" in msg:
                raise CDPError(f"{method} failed: {json.dumps(msg['error'])}")
            return msg.get("result", {})

    def wait_for_event(self, method: str, *, session_id: str | None = None,
                       timeout: float = _CALL_TIMEOUT) -> dict | None:
        """Return the next matching CDP event, or None when the wait expires.

        Screencast frames are asynchronous CDP events. Keeping them here, beside command
        replies, lets a hosted login use the same small protocol client as the local reader
        without teaching either caller to parse WebSocket frames.
        """
        deadline = time.monotonic() + timeout
        while True:
            for i, event in enumerate(self._events):
                if event.get("method") == method and (
                        session_id is None or event.get("sessionId") == session_id):
                    return self._events.pop(i)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            old_timeout = self._sock.gettimeout()
            self._sock.settimeout(remaining)
            try:
                event = json.loads(self._recv_message())
            except socket.timeout:
                return None
            finally:
                self._sock.settimeout(old_timeout)
            if event.get("method") == method and (
                    session_id is None or event.get("sessionId") == session_id):
                return event
            if "method" in event:
                self._events.append(event)

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


# ── public surface ───────────────────────────────────────────────────────────────

@contextmanager
def controlled_browser(app_path: Path, *, user_data_dir: Path | None = None,
                       headless: bool = True, url: str | None = None):
    """A browser OPYT owns, yielding a `.call(method, params)` CDP session.

    `user_data_dir=None` means an ephemeral profile in a temp dir, removed on exit. Pass a
    directory when the CALLER owns that directory's lifetime — the transplant read builds a
    temp profile, seeds it with a copied cookie DB, and deletes it itself, so this must not
    delete it out from under the seeding step.

    The browser is always terminated on exit, including on error; a leaked headless Chrome
    holding a decrypted cookie store is the failure worth preventing."""
    ephemeral = user_data_dir is None
    data_dir = Path(tempfile.mkdtemp(prefix="opyt-cdp-")) if ephemeral else user_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    proc = _launch(app_path, data_dir, headless=headless, url=url)
    sock = None
    try:
        sock = _Socket(_wait_for_endpoint(data_dir, proc))
        yield sock
    finally:
        if sock is not None:
            sock.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            # `kill` ends the process, but only `wait` reaps it. This context manager owns
            # the browser's whole lifecycle, so leaving collection to its parent would strand
            # a zombie for every forced teardown.
            proc.wait()
        if ephemeral:
            shutil.rmtree(data_dir, ignore_errors=True)


def launch(app_path: Path, user_data_dir: Path, *, url: str | None = None) -> subprocess.Popen:
    """Start a browser OPYT owns HEADED and hand back the running process, teardown NOT
    included — this one deliberately leaks the window.

    The counterpart to `controlled_browser`, which owns the browser for the length of a read.
    A window a human has to use must outlive the call that opened it: the caller (guided
    login) returns while the user is still typing, and the user closes the window themselves.
    Terminating it here would kill a login mid-flow.

    `user_data_dir` is always persistent — the session the user creates in that window exists
    nowhere else, so an ephemeral profile would discard the thing it was opened for.

    No `--remote-debugging-port`, deliberately. Google refuses to sign in to a debugged
    browser ("This browser or app may not be secure"), and the port is the signal it reads:
    headed Chrome WITH the port reports `navigator.webdriver: true`, measured 2026-09-07
    (docs/plans/2026-09-07-hosted-x-google-sso-block-context.md). The hosted login desktop
    already obeys this boundary; this is the same rule at the local window. Nothing is lost —
    no caller ever attached to this window, and the session is read back later by the
    transplant, which relaunches separately WITH CDP."""
    user_data_dir.mkdir(parents=True, exist_ok=True)
    argv = chrome_argv(app_path, user_data_dir)
    if url:
        argv.append(url)
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_cookies(session: _Socket) -> list[dict]:
    """Every cookie the browser holds, decrypted, as CDP cookie dicts.

    `Storage.getCookies` is browser-scoped; `Network.getAllCookies` is page-scoped and errors
    on the browser target. Nothing needs to be navigated to for this to work."""
    return session.call("Storage.getCookies")["cookies"]


@dataclass(frozen=True)
class Page:
    """One attached browser page, limited to its own CDP target session."""

    _browser: _Socket
    session_id: str

    def call(self, method: str, params: dict | None = None) -> dict:
        return self._browser.call(method, params, session_id=self.session_id)

    def wait_for_event(self, method: str, *, timeout: float = _CALL_TIMEOUT) -> dict | None:
        return self._browser.wait_for_event(method, session_id=self.session_id, timeout=timeout)


def new_page(browser: _Socket, url: str) -> Page:
    """Create and attach to one Chrome page through the browser-level CDP endpoint."""
    target_id = browser.call("Target.createTarget", {"url": url})["targetId"]
    attached = browser.call("Target.attachToTarget", {"targetId": target_id, "flatten": True})
    return Page(browser, attached["sessionId"])
