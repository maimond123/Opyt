"""
pipeline/ingestion/guided_login.py

Create a browser session for a user who has none, by opening a browser OPYT owns and
letting them log in there.

A different verb from `browser_cookies`, which READS a session that already exists. This one
CREATES one, and it blocks on a human, so it never sits in that module's backend registry —
it is selected one layer up, by onboarding, when the scan came back empty.

The profile is persistent (`browser_cookies.opyt_session_root()`), because a session created
here exists nowhere else: an ephemeral profile would mean logging in on every run. Reading it
back is deliberately NOT this module's job. The profile root is an ordinary Chromium
user-data-dir, so `browser_cookies` picks it up as another backend (`<browser>@opyt`) and
transplants it like any other — one read path, not two, and one answer to "is there a
session" for both `read_cookies` and onboarding's scan.

macOS-focused, like `browser_cookies` and `cdp`.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.ingestion import browser_cookies as bc, cdp
from pipeline.ingestion.utils import SyncAuthError


def target_backend(browser: str | None = None) -> bc.BrowserBackend | None:
    """The browser a guided login should open: the requested one, else the highest-priority
    installed browser OPYT can launch itself. None when this machine has none.

    Only launchable (Chromium) backends qualify — OPYT can open Chrome for you, but it
    cannot drive Safari, and a login it cannot later read back is not a login."""
    launchable = [b for b in bc.installed_backends()
                  if b.app_path() is not None and not b.key.endswith(bc.OPYT_KEY_SUFFIX)]
    if browser:
        return next((b for b in launchable if b.key == browser), None)
    return launchable[0] if launchable else None


def session_dir(backend: bc.BrowserBackend) -> Path:
    """Where this browser's OPYT-owned profile lives. One directory per browser, named by
    the browser key, which is what `browser_cookies.opyt_session_backends()` reads back."""
    return bc.opyt_session_root() / backend.key


def start(login_url: str, *, browser: str | None = None) -> bc.BrowserBackend:
    """Open a browser window on OPYT's own profile, at `login_url`, and return immediately.

    Deliberately does not wait for the login. The window has to outlive this call — the user
    is going to spend a minute in it, and an MCP tool call that blocks on a human is a tool
    call that times out. The caller tells the user to come back; the next scan finds the
    session, because the profile is a normal Chromium user-data-dir that `browser_cookies`
    already enumerates.

    Nor does it close the window: the user closes it. A read of that profile copies the
    cookie DB, so a still-open window never blocks the session from being read.

    Raises SyncAuthError when no browser on this machine can be opened — the same typed
    failure every other "no session" path raises, so callers keep one except branch."""
    backend = target_backend(browser)
    if backend is None:
        raise SyncAuthError(
            "Opyt could not find a browser it can open for you to log in — it needs Chrome, "
            "Brave, Edge, Vivaldi or Opera installed. Log in using any browser you have, "
            "then re-run.")
    cdp.launch(backend.app_path(), session_dir(backend), url=login_url)
    return backend
