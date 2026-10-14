"""Create OPYT's persistent login session for a source by opening a browser OPYT owns and letting
the user log in there; only the URL differs per source.

`cookies.py` reads sessions; this creates one and blocks on a human, so it is selected by the
explicit `onboard(source=...)` action, not the cookie backend registry. The profile is a
persistent ordinary Chromium user-data-dir, which `cookies.read_opyt_cookies` reads back.
macOS-focused, like `cookies` and `cdp`.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.platforms.browser import cdp, cookies as bc
from pipeline.ingestion.utils import SyncAuthError


def choose_launch_target(facts: list[tuple[bc.BrowserBackend, bool]]) -> bc.BrowserBackend | None:
    """The first ordinary browser in `facts` (installed backends in priority order, each paired
    with whether its executable exists) that OPYT can launch, or None.

    Only Chromium backends qualify (OPYT cannot create a managed profile in Safari). OPYT's own
    profiles are never a target: they are what a login creates."""
    return next((b for b, launchable in facts
                 if launchable and not b.key.endswith(bc.OPYT_KEY_SUFFIX)), None)


def target_backend() -> bc.BrowserBackend | None:
    """The highest-priority launchable Chromium browser, or None when none is installed."""
    return choose_launch_target([(b, b.app_path() is not None) for b in bc.installed_backends()])


def session_dir(backend: bc.BrowserBackend) -> Path:
    """Where this browser's OPYT-owned profile lives, named by the browser key that
    `cookies.opyt_session_backends()` reads back."""
    return bc.opyt_session_root() / backend.key


def start(login_url: str) -> bc.BrowserBackend:
    """Open a browser window on OPYT's own profile, at `login_url`, and return immediately.

    It does not wait for the login (an MCP call that blocks on a human times out) and does not
    close the window; the next scan finds the session. Raises SyncAuthError when no browser on
    this machine can be opened."""
    backend = target_backend()
    if backend is None:
        raise SyncAuthError(
            "Opyt could not find launchable Chromium to create its managed login profile — it "
            "needs Chrome, Brave, Edge, Vivaldi or Opera installed. Logging into an existing "
            "browser cannot connect a source to OPYT; install one, then call onboard(source=...) "
            "again.")
    cdp.launch(backend.app_path(), session_dir(backend), url=login_url)
    return backend
