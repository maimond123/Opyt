"""
pipeline/ingestion/guided_login.py

Create OPYT's persistent login session for a source by opening a browser OPYT owns and letting
the user log in there. The URL is the only thing that differs per source — X and any other site
that needs a logged-in session take the same path through here.

A different verb from `browser_cookies`, which READS a session that already exists. This one
CREATES one, and it blocks on a human, so it never sits in that module's backend registry —
it is selected one layer up by the explicit `onboard(source=...)` action.

The profile is persistent (`browser_cookies.opyt_session_root()`), because a session created
here exists nowhere else: an ephemeral profile would mean logging in on every run. Reading it
back is deliberately NOT this module's job. The profile root is an ordinary Chromium
user-data-dir, so `browser_cookies.read_opyt_cookies` can transplant it without ever scanning
the user's normal browser profiles.

macOS-focused, like `browser_cookies` and `cdp`.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.ingestion import browser_cookies as bc, cdp
from pipeline.ingestion.utils import SyncAuthError


def target_backend() -> bc.BrowserBackend | None:
    """The highest-priority launchable Chromium browser, or None when none is installed.

    Only launchable (Chromium) backends qualify — OPYT can open Chrome for you, but it
    cannot create an OPYT-managed profile in Safari, and an arbitrary browser login cannot
    satisfy X setup."""
    launchable = [b for b in bc.installed_backends()
                  if b.app_path() is not None and not b.key.endswith(bc.OPYT_KEY_SUFFIX)]
    return launchable[0] if launchable else None


def session_dir(backend: bc.BrowserBackend) -> Path:
    """Where this browser's OPYT-owned profile lives. One directory per browser, named by
    the browser key, which is what `browser_cookies.opyt_session_backends()` reads back."""
    return bc.opyt_session_root() / backend.key


def start(login_url: str) -> bc.BrowserBackend:
    """Open a browser window on OPYT's own profile, at `login_url`, and return immediately.

    Deliberately does not wait for the login. The window has to outlive this call — the user
    is going to spend a minute in it, and an MCP tool call that blocks on a human is a tool
    call that times out. The caller tells the user to come back; the next scan finds the
    session, because the profile is a normal Chromium user-data-dir that the managed reader
    enumerates.

    Nor does it close the window: the user closes it. A read of that profile copies the
    cookie DB, so a still-open window never blocks the session from being read.

    Raises SyncAuthError when no browser on this machine can be opened — the same typed
    failure the managed setup path raises."""
    backend = target_backend()
    if backend is None:
        raise SyncAuthError(
            "Opyt could not find launchable Chromium to create its managed login profile — it "
            "needs Chrome, Brave, Edge, Vivaldi or Opera installed. Logging into an existing "
            "browser cannot connect a source to OPYT; install one, then call onboard(source=...) "
            "again.")
    cdp.launch(backend.app_path(), session_dir(backend), url=login_url)
    return backend
