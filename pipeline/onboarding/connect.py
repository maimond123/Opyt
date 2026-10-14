"""
pipeline/onboarding/connect.py — open OPYT's own sign-in for a login root (a local window, or the
sign-in Opyt's service runs), and hold while a just-made session does not answer yet.

A local sign-in leaves a marker in the home naming its root (`signin_opened`), so the next call
knows a login is expected and waits for it (`hold_for_signin`). Removed once the session answers,
or when the user answers the sources question another way.
"""
from __future__ import annotations

import sys
import time

from opyt_core.paths import opyt_path
from pipeline.ingestion.utils import SyncAuthError
from pipeline.onboarding import roots, service_signin
from pipeline.platforms import connections
from pipeline.platforms.browser import login


def route(platform: str, source: str) -> str:
    """Which sign-in `source` gets: Opyt's service for X off macOS, where the install is an
    agent's computer with no window a person can see; otherwise a window on this machine."""
    if platform != "darwin" and source == "x":
        return "service"
    return "local"


def start(source: str) -> dict:
    """Open OPYT's own login profile for `source`, or mint a service sign-in for it.

    Never scans the user's browsers: reading back a profile OPYT created reads OPYT's own session.
    """
    if route(sys.platform, source) == "service":
        return service_signin.connect()

    try:
        backend = login.start(roots.LOGIN_URLS[source])
    except SyncAuthError as e:
        return {"status": "needs_login", "phase": "sources", "source": source,
                "message": str(e)}
    _set_opened(source)
    return {"status": "awaiting_login", "phase": "sources", "source": source,
            "message": (f"{backend.label} just opened at {roots.LOGIN_URLS[source]}, on a profile "
                        f"OPYT owns — the user's own windows and logins are untouched. Tell the "
                        f"user to log in there, then call `onboard` again.")}


# Chrome keeps new cookies in memory and writes them to disk in batches, up to 30 s after a login
# sets them (measured: 30.4 s), and OPYT reads the disk. A call right after the user
# says "done" therefore waits this long for the session before saying it is not there.
_FLUSH_WAIT_S = 32.0
_POLL_S = 2.0

_OPENED_MARKER = "signin_opened"


def _opened() -> str | None:
    """The root whose local sign-in window was opened and has not produced a session yet."""
    try:
        word = opyt_path(_OPENED_MARKER).read_text().strip()
    except OSError:
        return None
    return word if word in roots.LOGIN_URLS else None


def _set_opened(source: str | None) -> None:
    """Best-effort: a home that cannot be written skips the wait and reads the disk once."""
    path = opyt_path(_OPENED_MARKER)
    try:
        if source is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
    except OSError:
        pass


def abandon_signin() -> None:
    """The user answered the sources question another way: no login is expected any more."""
    _set_opened(None)


def _answers(source: str) -> bool:
    """Whether `source`'s session is on disk and usable. Substack's cookie is there before the
    login (anonymous visitors get one), so it is asked an authenticated request."""
    try:
        if source == "x":
            return connections._connected_x()
        return (connections._connected_substack()
                and connections.managed_substack_session_ready(attempts=1, delay=0.0))
    except Exception:
        # Fail-safe: a check that cannot run must not strand a user who did sign in.
        return True


def hold_for_signin(*, clock=time.monotonic, sleep=time.sleep) -> dict | None:
    """The answer that holds while the sign-in this home opened has produced no session, or None
    once it has, or when no sign-in was opened. Returns the moment the session answers, and
    waits at most `_FLUSH_WAIT_S` for it."""
    source = _opened()
    if source is None:
        return None
    deadline = clock() + _FLUSH_WAIT_S
    while not (ready := _answers(source)) and clock() < deadline:
        sleep(_POLL_S)
    if not ready:
        return _signin_not_seen(source)
    _set_opened(None)
    return None


def _signin_not_seen(source: str) -> dict:
    """The wait ran out with no session. Says every way forward, because OPYT cannot see whether
    the window is still open."""
    return {
        "status": "awaiting_login", "phase": "sources", "source": source,
        "message": (
            f"The {source.title()} window isn't signed in yet. Tell the user to finish signing "
            f"in inside Opyt's window, then call `onboard` again. If that window is gone, "
            f"`onboard(source='{source}')` opens it again, and any other source word starts "
            f"with that one instead. Nothing was pulled, so nothing is half-done."),
    }


def hold_if_not_ready(state: dict) -> dict | None:
    """The answer that holds the curation walk while a connected session does not answer yet,
    or None when every connected session is ready."""
    not_ready = _connected_but_not_ready(state)
    return _login_not_ready(not_ready) if not_ready else None


def _connected_but_not_ready(state: dict) -> list[str]:
    """Connected login sources whose session is present but does not answer an authenticated
    request yet, so the caller can hold instead of building the curation walk on a session that
    will 401.

    Cookie presence is not validity: Substack hands anonymous visitors a `substack.sid`, and
    Chrome flushes cookies lazily. Substack only; X's `auth_token` has no anonymous form.
    """
    if not state["sources"].get("substack"):
        return []
    try:
        return [] if connections.managed_substack_session_ready() else ["substack"]
    except Exception:
        # Fail-safe: an unexpected verifier error must not strand a user who did sign in. Let the
        # flow proceed; the collector records a real skip rather than a lie if the session is dead.
        return []


def _login_not_ready(sources: list[str]) -> dict:
    """The window is open and OPYT can see it, but the sign-in has not registered yet. Ask the
    user to finish in that window and come back — nothing was pulled, so nothing is half-done."""
    names = _phrase([s.title() for s in sources])
    return {
        "status": "awaiting_login", "phase": "sources", "source": sources[0],
        "message": (
            f"The {names} window isn't signed in yet — I checked, and the session doesn't answer "
            f"an authenticated request. Finish signing in inside Opyt's window; it can take a few "
            f"seconds to register after you log in. Then tell me you're done and I'll check again. "
            f"Nothing was pulled, so nothing is half-done."),
    }


def _phrase(parts: list[str]) -> str:
    return " and ".join(parts) if len(parts) < 3 else ", ".join(parts[:-1]) + " and " + parts[-1]
