"""
service/x_signin.py — X sign-ins this service runs for installs with no browser of their own (an
agent's computer), from a minted link to a collected session.

An install mints one (`mint`), keeps the collect token, and shows its user the page link. Loading
the page spends nothing, because link scanners and previews fetch it first (one did, before
the user's browser): the page's script claims the sign-in (`claim`), which binds it to
that browser tab, then starts the desktop (`start`) and polls `status`. When
the desktop sees X's home page it hands the session here, and the install's next `collect` takes
it, once. Nothing is written to disk: the session waits in this process's memory for at most
`RESULT_TTL`, and the Chrome profile it came from is deleted when the desktop closes.

One desktop at a time, because the box holds one (a sign-in peaks at ~465 MB); a page behind it retries `start` until the desktop is free. When the slot is free at
`mint`, the desktop starts then, so X's page (seconds to load on this box's one core) is usually
drawn by the time the user taps the link; a warm desktop nobody opens closes after
`WARM_SECONDS` and frees the slot, and its sign-in can still start one when its page opens. The state is in memory,
so this needs the one uvicorn worker `DEPLOY.md` runs. Every state change happens in this module,
under `_lock`:

    minted ─page claims─▶ opened ─start─▶ running ─home page─▶ connected ─collect─▶ (gone)
                     (`status` says loading until X's page is drawn, and finishing while the
                      session is written down and read)
       │                    │               └─browser died / no session─▶ failed
       └──── ten minutes ───┴─▶ expired          └─ten minutes─▶ expired
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass

from pipeline.ingestion.utils import log
from service import x_signin_desktop
from service.x_signin_desktop import CONNECTED, EXPIRED, FAILED, XSigninDesktop

# How long a minted link may wait to be opened and started, a desktop may run, and a session may
# wait for its install; and how long a finished sign-in still answers `collect` with its outcome.
LINK_TTL = DESKTOP_TTL = RESULT_TTL = ENDED_TTL = 600.0
# How long a desktop started at `mint` waits for its page to open.
WARM_SECONDS = 180.0

MINTED, OPENED, RUNNING, FINISHING, LOADING = (
    "minted", "opened", "running", "finishing", "loading")
_ENDED = (FAILED, EXPIRED)


@dataclass
class _Signin:
    nonce: str                    # the page link; any fetch of it is harmless
    token: str                    # what the install collects with; never in a URL
    expires_at: float             # monotonic: when this state ends
    state: str = MINTED
    session: str | None = None    # the claiming tab's token; it alone drives the sign-in
    cookies: dict | None = None   # only while CONNECTED


_lock = threading.Lock()
_signins: dict[str, _Signin] = {}          # by collect token
_running: _Signin | None = None
_desktop: XSigninDesktop | None = None


# ── decisions ────────────────────────────────────────────────────────────────────────────────

def start_decision(state: str | None, holds_desktop: bool, desktop_busy: bool) -> str:
    """What a page's `start` gets: `attach` to the desktop warmed for it at mint, `start` one,
    wait while another sign-in holds it (`busy`), or the state it already reached."""
    if state == OPENED:
        if holds_desktop:
            return "attach"
        return "busy" if desktop_busy else "start"
    if state in (RUNNING, CONNECTED, FAILED):
        return state
    return EXPIRED                                  # unknown, or the link ran out first


def claim_decision(state: str | None, same_tab: bool) -> str:
    """What a page's `claim` gets: `claim` the sign-in for this tab, the same tab's `claimed`
    again (a reload), `taken` when another tab holds it, or the state it ended in."""
    if state == MINTED:
        return "claim"
    if state in (OPENED, RUNNING):
        return "claimed" if same_tab else "taken"
    if state in (CONNECTED, FAILED):
        return state
    return EXPIRED


def collect_reply(state: str | None) -> str:
    """What an install's `collect` gets for a sign-in in `state`."""
    if state in (MINTED, OPENED, RUNNING):
        return "waiting"
    if state in (CONNECTED, FAILED):
        return state
    return EXPIRED                                  # unknown, collected already, or ran out


# ── the flow ────────────────────────────────────────────────────────────────────────────────

def mint(now: float) -> dict:
    """A new sign-in: the collect token for the install, and the page path for its user. Warms
    a desktop for it when the slot is free."""
    signin = _Signin(nonce=secrets.token_urlsafe(24), token=secrets.token_urlsafe(32),
                     expires_at=now + LINK_TTL)
    with _lock:
        _sweep(now)
        _signins[signin.token] = signin
        desktop = _claim(signin, now + WARM_SECONDS) if _desktop is None else None
    if desktop is not None:
        _spawn(_start_desktop, signin, desktop)
    return {"signin": signin.token, "path": f"/x-signin/{signin.nonce}",
            "expires_in": int(LINK_TTL)}


def link_open(nonce: str, now: float) -> bool:
    """Whether the link still leads to a sign-in (one not ended or expired). Changes nothing."""
    with _lock:
        _sweep(now)
        signin = _find(nonce=nonce)
        return signin is not None and signin.state not in _ENDED


def claim(nonce: str, session: str | None, now: float) -> dict:
    """Bind the sign-in to the first tab whose script asks: `{status: "claimed", session}` with
    that tab's token, given again to the same tab (`session` from its storage) after a reload;
    `taken` for any other tab; or the state it ended in."""
    with _lock:
        _sweep(now)
        signin = _find(nonce=nonce)
        same_tab = (signin is not None and signin.session is not None and session is not None
                    and secrets.compare_digest(signin.session, session))
        decision = claim_decision(signin.state if signin else None, same_tab)
        if decision == "claim":
            signin.state, signin.session = OPENED, secrets.token_urlsafe(32)
            decision = "claimed"
        if decision == "claimed":
            return {"status": "claimed", "session": signin.session}
        return {"status": decision}


def start(session: str, now: float) -> dict:
    """Start this page's desktop if it is free. Starting takes seconds, so the slot is claimed
    under the lock and the desktop started outside it."""
    with _lock:
        _sweep(now)
        signin = _find(session=session)
        holds = signin is not None and signin is _running and _desktop is not None
        decision = start_decision(signin.state if signin else None, holds_desktop=holds,
                                  desktop_busy=_desktop is not None and not holds)
        if decision == "attach":
            signin.state, signin.expires_at = RUNNING, now + DESKTOP_TTL
            _desktop.extend(now + DESKTOP_TTL)
            return _started()
        if decision != "start":
            return _started() if decision == RUNNING else {"status": decision}
        desktop = _claim(signin, now + DESKTOP_TTL)
        signin.state, signin.expires_at = RUNNING, now + DESKTOP_TTL
    if not _start_desktop(signin, desktop):
        return {"status": FAILED}
    return _started()


def _claim(signin: _Signin, deadline: float) -> XSigninDesktop:
    """Give the slot to `signin`, under the lock: its desktop, not yet started."""
    global _running, _desktop
    desktop = XSigninDesktop(deadline, lambda outcome, cookies:
                             _desktop_done(signin, outcome, cookies))
    _running, _desktop = signin, desktop
    return desktop


def _start_desktop(signin: _Signin, desktop: XSigninDesktop) -> bool:
    """Start a claimed desktop outside the lock (it takes seconds). False when it would not."""
    try:
        desktop.start()
    except Exception as e:
        log(f"[x-signin] the desktop did not start: {type(e).__name__}: {e}")
        _desktop_done(signin, FAILED, None)
        return False
    return True


def _spawn(target, *args) -> None:
    """Run `target` on its own thread: a warm start must not hold up the install's `mint`."""
    threading.Thread(target=target, args=args, name="x-signin-warm", daemon=True).start()


def _started() -> dict:
    """What the page draws: the whole screen's size, and the window at its top-left corner that
    is the only part shown."""
    return {"status": RUNNING, "width": x_signin_desktop.WIDTH,
            "height": x_signin_desktop.HEIGHT,
            "window": {"width": x_signin_desktop.WINDOW_WIDTH,
                       "height": x_signin_desktop.WINDOW_HEIGHT},
            "expires_in": int(DESKTOP_TTL)}


def restart(session: str) -> bool:
    """Put this page's browser back on X's sign-in page. False when it is not running."""
    desktop = _desktop_of(session)
    if desktop is None:
        return False
    try:
        desktop.restart_browser()
    except Exception as e:
        log(f"[x-signin] the browser did not restart: {type(e).__name__}: {e}")
        return False
    return True


def vnc_port(session: str) -> int | None:
    """Where this page's screen is served on loopback, while its desktop runs."""
    desktop = _desktop_of(session)
    return desktop.vnc_port if desktop is not None else None


def status(session: str, now: float) -> str:
    """Where this page's sign-in stands: opened, loading (X's page not drawn yet), running,
    finishing (X showed its home page and the session is being read), connected, failed or
    expired."""
    with _lock:
        _sweep(now)
        signin = _find(session=session)
        if signin is None or signin.state == MINTED:
            return EXPIRED
        if signin is _running and _desktop is not None and signin.state == RUNNING:
            if _desktop.signed_in:
                return FINISHING
            if not _desktop.ready:
                return LOADING
        return signin.state


def collect(token: str, now: float) -> dict:
    """The install's answer: `waiting`, `connected` with the session (given once, then
    forgotten), `failed` or `expired`."""
    with _lock:
        _sweep(now)
        signin = _signins.get(token)
        reply = collect_reply(signin.state if signin else None)
        if reply == CONNECTED:
            del _signins[token]
            return {"status": CONNECTED, "session": signin.cookies}
        if reply in _ENDED:
            _signins.pop(token, None)
        return {"status": reply}


# ── state, under the lock ───────────────────────────────────────────────────────────────────

def _desktop_done(signin: _Signin, outcome: str, cookies: dict | None) -> None:
    """The desktop closed: free the slot and record what it ended with. A desktop warmed at mint
    that ends before its page started it (nobody opened the link in time, or it would not start)
    only frees the slot: the sign-in can still start one when its page opens."""
    global _running, _desktop
    now = time.monotonic()
    with _lock:
        if _running is signin:
            _running = _desktop = None
        if signin.state in (MINTED, OPENED) and outcome != CONNECTED:
            log(f"[x-signin] a desktop warmed for a sign-in closed before its page started it: "
                f"{outcome}")
            return
        signin.state = outcome
        signin.cookies = cookies if outcome == CONNECTED else None
        signin.expires_at = now + (RESULT_TTL if outcome == CONNECTED else ENDED_TTL)
    log(f"[x-signin] a sign-in ended: {outcome}")


def _desktop_of(session: str) -> XSigninDesktop | None:
    with _lock:
        signin = _find(session=session)
        return _desktop if signin is not None and signin is _running else None


def _find(*, nonce: str | None = None, session: str | None = None) -> _Signin | None:
    for signin in _signins.values():
        if (nonce is not None and secrets.compare_digest(signin.nonce, nonce)) or (
                session is not None and signin.session is not None
                and secrets.compare_digest(signin.session, session)):
            return signin
    return None


def _sweep(now: float) -> None:
    """Expire what ran out, and forget what ended long enough ago. A running desktop keeps its
    own deadline."""
    for token, signin in list(_signins.items()):
        if now < signin.expires_at or signin.state == RUNNING:
            continue
        if signin.state in _ENDED:
            del _signins[token]
        else:                                       # minted, opened, or a session never collected
            signin.state, signin.cookies = EXPIRED, None
            signin.expires_at = now + ENDED_TTL
