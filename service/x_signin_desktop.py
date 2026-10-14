"""
service/x_signin_desktop.py — one X sign-in desktop on the sharing service.

A laptop-sized private screen (`pipeline/hosted_signin/display.py`), a Chrome on a throwaway
profile showing X's sign-in page in a narrow window in the screen's top-left corner, with no tabs
or address bar, and an x11vnc the visitor's page watches through; the page shows only that
window. When X shows its home page the watcher ends the stream, waits for
Chrome to write the session to disk, closes Chrome, and reads the session with a second, headless
Chrome (`cdp`); the profile is deleted when the desktop closes. No debugger is attached while the
visitor signs in.

The shape was measured on the service box: X refused a phone-shaped screen at the
username step, and accepted a laptop-sized one both with a full-screen window and with this
narrow one, in which X draws its simple full-width sign-in form that a phone can show whole. The wait for the write is there because two measurements
of a SIGTERM disagree: on the gateway box it left the cookie store unwritten, and on
the service box (a newer Chrome) it left the session in place four times out of four. Chrome
writes cookies on a ~30 s timer, so the wait is at most that, while X's timeline loads (~660 MB
at its peak, against ~465 MB for the sign-in alone).
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

from pipeline.hosted_signin import display
from pipeline.ingestion.utils import log
from pipeline.platforms.browser import cdp, profile
from pipeline.platforms.browser.profile import HostedBrowserError

LOGIN_URL = "https://x.com/login"
# The screen X sees, at density 1: laptop-sized, because X refuses a phone-shaped one. Chrome's
# window is narrower and sits at the top-left corner; the visitor's page shows only the window.
WIDTH, HEIGHT = 1600, 900
WINDOW_WIDTH, WINDOW_HEIGHT = 500, 900
# What the install's X collectors need: the cookies X collection needs.
SESSION_COOKIES = ("auth_token", "ct0", "twid")

_WATCH_SECONDS = 0.25
# The cookie store is the signal for an X whose home page is not titled in English. Chrome writes
# it on a ~30 s timer, so it is the slow path and is read less often.
_STORE_CHECK_SECONDS = 2.0
# Chromium's cookie commit interval, with margin: past this, nothing was pending.
_WRITE_WAIT_SECONDS = 35.0

# `on_done`'s outcomes.
CONNECTED, FAILED, EXPIRED = "connected", "failed", "expired"


def x_page_shown(window_tree: str) -> bool:
    """Whether `xwininfo -root -tree` lists a window titled for any X page ("Log in to X / X",
    "Home / X"): X has drawn something, so the visitor no longer looks at a blank browser."""
    return ' / X"' in window_tree


def home_shown(window_tree: str) -> bool:
    """Whether `xwininfo -root -tree` lists a window titled for X's home page: "Home / X", or
    "(3) Home / X" with unread notifications."""
    return 'Home / X"' in window_tree


def session_from(cookies: list[dict]) -> dict | None:
    """X's session from a CDP cookie list, or None when this profile never signed in."""
    found = {c["name"]: c["value"] for c in cookies
             if c.get("domain", "").lstrip(".") == "x.com" and c.get("name") in SESSION_COOKIES}
    return found if found.get("auth_token") and found.get("ct0") else None


class XSigninDesktop:
    """One sign-in's screen, browser, VNC server and watcher.

    `on_done(outcome, session)` is called once, from the watcher or the deadline: `CONNECTED` with
    the session, or `FAILED` / `EXPIRED` with None. The desktop has closed by then.
    """

    def __init__(self, deadline: float, on_done: Callable[[str, dict | None], None]) -> None:
        self.deadline = deadline                      # monotonic
        self._on_done = on_done
        self._lock = threading.RLock()
        self._closed = False
        self._finished = False
        self._profile = Path(tempfile.mkdtemp(prefix="opyt-x-signin-"))
        self._chrome_path: Path | None = None
        self._display: str | None = None
        self._xvfb: subprocess.Popen | None = None
        self._chrome: subprocess.Popen | None = None
        self._vnc: subprocess.Popen | None = None
        self._vnc_port: int | None = None
        self._timer: threading.Timer | None = None
        self._ready = False
        self._signed_in = False

    @property
    def vnc_port(self) -> int | None:
        with self._lock:
            return None if self._closed else self._vnc_port

    @property
    def ready(self) -> bool:
        """X has drawn its page. Until then the browser is a blank window, which the visitor's
        page covers with a loading message."""
        return self._ready

    @property
    def signed_in(self) -> bool:
        """X showed its home page, and the desktop is finishing: the stream has ended and the
        session is being written down and read."""
        return self._signed_in

    def start(self) -> None:
        """Open the screen, the browser and the VNC server, then start watching. Raises
        `HostedBrowserError` with nothing left running."""
        with self._lock:
            try:
                # Both refusals before anything starts: the watcher cannot see without xwininfo.
                self._chrome_path = profile.chrome_binary()
                display.binary("xwininfo")
                self._display, self._xvfb = display.start_xvfb(WIDTH, HEIGHT)
                self._chrome = self._launch_chrome()
                port = display.free_loopback_port()
                self._vnc = subprocess.Popen(
                    display.vnc_argv(self._display, port, threads=False),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                display.wait_for_vnc(self._vnc, port)
                display.require_alive(self._chrome)
                self._vnc_port = port
            except Exception:
                self.close()
                raise
            self._timer = threading.Timer(max(0.0, self.deadline - time.monotonic()),
                                          self._finish, (EXPIRED, None))
            self._timer.daemon = True
            self._timer.start()
            threading.Thread(target=self._watch, name="x-signin-watch", daemon=True).start()

    def extend(self, deadline: float) -> None:
        """Run until `deadline` instead: a desktop started before its page opened gets the full
        time from when it did."""
        with self._lock:
            if self._closed or self._finished:
                return
            self.deadline = deadline
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(max(0.0, deadline - time.monotonic()),
                                          self._finish, (EXPIRED, None))
            self._timer.daemon = True
            self._timer.start()

    def _launch_chrome(self) -> subprocess.Popen:
        """The visitor's browser: X's sign-in page, no tabs or address bar, in the narrow window
        at the screen's top-left corner. Its HOME is the profile, so Chrome's own files go where
        the profile goes."""
        return subprocess.Popen(
            cdp.chrome_argv(self._chrome_path, self._profile)
            + [f"--window-size={WINDOW_WIDTH},{WINDOW_HEIGHT}", "--force-device-scale-factor=1",
               "--window-position=0,0", f"--app={LOGIN_URL}"],
            env={**os.environ, "DISPLAY": self._display, "HOME": str(self._profile)},
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def restart_browser(self) -> None:
        """Put the browser back on X's sign-in page. The screen has no window manager, so a
        popup ("Continue with Google") cannot otherwise be closed."""
        with self._lock:
            if self._closed or self._chrome is None:
                raise HostedBrowserError("that sign-in is not running")
            display.reap(self._chrome)
            self._ready = False
            self._chrome = self._launch_chrome()
            display.require_alive(self._chrome)

    def _watch(self) -> None:
        """Until the desktop closes: finish on X's home page or a session in the store, and fail
        if the browser dies on its own."""
        checked_store = time.monotonic()
        while True:
            time.sleep(_WATCH_SECONDS)
            with self._lock:
                if self._closed:
                    return
                chrome, screen = self._chrome, self._display
            if chrome.poll() is not None:
                with self._lock:
                    replaced = self._chrome is not chrome
                if replaced:          # `restart_browser` swapped it while this looked
                    continue
                log("[x-signin] the sign-in browser exited on its own")
                self._finish(FAILED, None)
                return
            written = False
            tree = _window_tree(screen)
            if not self._ready and x_page_shown(tree):
                self._ready = True
            signed_in = home_shown(tree)
            if not signed_in and time.monotonic() - checked_store >= _STORE_CHECK_SECONDS:
                checked_store = time.monotonic()
                signed_in = written = _store_has_session(self._profile)
            if signed_in:
                self._end_stream()
                if not written:
                    self._await_write()
                self._finish(*self._read_session(chrome))
                return

    def _end_stream(self) -> None:
        """Stop showing the screen: the visitor's page says it is finishing, and nobody clicks
        around a timeline the desktop is about to close."""
        with self._lock:
            self._signed_in = True
            vnc, self._vnc_port = self._vnc, None
        if vnc is not None:
            display.reap(vnc)

    def _await_write(self) -> None:
        """Hold Chrome open until it has written the session to the profile, or the commit
        interval has passed."""
        began = time.monotonic()
        while time.monotonic() - began < _WRITE_WAIT_SECONDS and not self._closed:
            if _store_has_session(self._profile):
                log(f"[x-signin] Chrome wrote the session {time.monotonic() - began:.0f} s "
                    "after X's home page")
                return
            time.sleep(_STORE_CHECK_SECONDS / 2)
        log("[x-signin] Chrome had not written the session when the wait ran out; closing anyway")

    def _read_session(self, chrome: subprocess.Popen) -> tuple[str, dict | None]:
        """Close the visitor's browser, then read the session with a headless Chrome on the same
        profile."""
        display.reap(chrome)
        try:
            with cdp.controlled_browser(self._chrome_path, user_data_dir=self._profile,
                                        headless=True) as browser:
                session = session_from(cdp.get_cookies(browser))
        except Exception as e:
            log(f"[x-signin] the session could not be read: {type(e).__name__}: {e}")
            return FAILED, None
        if session is None:
            log("[x-signin] X showed its home page but the profile holds no session")
            return FAILED, None
        return CONNECTED, session

    def _finish(self, outcome: str, session: dict | None) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
        self.close()
        self._on_done(outcome, session)

    def close(self) -> None:
        """End every process and delete the profile. Safe to call more than once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
            procs = (self._vnc, self._chrome, self._xvfb)
            self._vnc_port = None
        # VNC first: the page loses the stream before anything beneath it closes.
        for proc in procs:
            if proc is not None:
                display.reap(proc)
        shutil.rmtree(self._profile, ignore_errors=True)


def _window_tree(screen: str) -> str:
    """Every window's title on `screen`, as `xwininfo` lists them; empty if it could not ask."""
    try:
        return subprocess.run([display.binary("xwininfo"), "-root", "-tree"],
                              env={**os.environ, "DISPLAY": screen}, capture_output=True,
                              text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError, HostedBrowserError):
        return ""


def _store_has_session(profile_dir: Path) -> bool:
    """Whether Chrome has written an X `auth_token` row to the profile's cookie store. Reads row
    presence only; the value is encrypted and is not touched here."""
    for rel in (("Default", "Network", "Cookies"), ("Default", "Cookies")):
        path = profile_dir.joinpath(*rel)
        if not path.exists():
            continue
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
            try:
                row = conn.execute("SELECT 1 FROM cookies WHERE host_key = '.x.com' "
                                   "AND name = 'auth_token'").fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            continue            # caught mid-write; the next look reads it
        if row is not None:
            return True
    return False
