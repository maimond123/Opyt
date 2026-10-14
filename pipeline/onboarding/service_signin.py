"""
pipeline/onboarding/service_signin.py — connect X on an install with no browser a person can see
(an agent's computer, such as Muse's), through the sign-in Opyt's service runs
(`service/x_signin.py`).

Two calls, like the Google and GitHub sign-ins. The first mints a sign-in and answers with its
link for the user. Each call after that waits for the user, up to `_WAIT_SECONDS`, and once they
are in saves the session to `.env` as `OPYT_COOKIES_X`, where the X collectors read it
(`cookies.env_session`); worker jobs reload `.env` before they run. The collect token waits in
the home between calls, because each call can be a fresh process (`opyt call onboard`).

The user never sees or copies a cookie. Every failure is an answer, never a raise.
"""
from __future__ import annotations

import json
import os
import time

from opyt_core import keys
from opyt_core.paths import opyt_path, owner_only
from pipeline.sharing import account, client

_PENDING = "x-signin.json"
SESSION_ENV = "OPYT_COOKIES_X"
# Muse's network holds a first request to a new host for 20-85 s (measured), so an X
# request through it needs the longer timeout `x/local.py` reads from here.
TIMEOUT_ENV, TIMEOUT_SECONDS = "OPYT_X_TIMEOUT_S", "180"
_WAIT_SECONDS = 40.0
_POLL_SECONDS = 3.0


def connect() -> dict:
    """One step: a link for the user (`awaiting_login`), still waiting, or `connected`."""
    base = account.service_url()
    pending = _pending()
    try:
        if pending is not None:
            reply = _wait(base, pending["signin"])
            if reply["status"] == "connected":
                _save(reply["session"])
                _forget()
                return _connected()
            if reply["status"] == "waiting":
                return _awaiting(pending["link"], first=False)
            _forget()                 # failed or expired: a fresh link below
        minted = client.x_signin(base)
    except client.Failed:
        return _unreachable()
    link = base + minted["path"]
    _hold({"signin": minted["signin"], "link": link})
    return _awaiting(link, first=True)


def _wait(base: str, signin: str) -> dict:
    """The service's answer, asking again every few seconds while it is `waiting`."""
    deadline = time.monotonic() + _WAIT_SECONDS
    while True:
        reply = client.x_signin_collect(base, signin)
        if reply.get("status") != "waiting" or time.monotonic() >= deadline:
            return reply
        time.sleep(_POLL_SECONDS)


def _save(session: dict) -> None:
    """The session as a `Cookie:` header in `.env`, and in this process."""
    header = "; ".join(f"{name}={value}" for name, value in session.items())
    keys.set_key(SESSION_ENV, header)
    os.environ[SESSION_ENV] = header
    if not os.environ.get(TIMEOUT_ENV):
        keys.set_key(TIMEOUT_ENV, TIMEOUT_SECONDS)
        os.environ[TIMEOUT_ENV] = TIMEOUT_SECONDS


def _pending() -> dict | None:
    try:
        return json.loads(opyt_path(_PENDING).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _hold(pending: dict) -> None:
    path = opyt_path(_PENDING)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pending), encoding="utf-8")
    owner_only(path)


def _forget() -> None:
    opyt_path(_PENDING).unlink(missing_ok=True)


# ── answers ──────────────────────────────────────────────────────────────────────────────────

def _awaiting(link: str, *, first: bool) -> dict:
    lead = ("A sign-in link is ready in `login_url`. Present it to the user as a link whose text "
            "reads **Connect X to Opyt**; do not print the raw URL. It opens X's own sign-in in a "
            "private browser on Opyt's server, and they sign in as they would anywhere."
            if first else
            "They have not finished signing in yet. If they lost the link, present `login_url` "
            "again as **Connect X to Opyt**.")
    return {"status": "awaiting_login", "phase": "sources", "source": "x", "login_url": link,
            "message": lead + " Then call `onboard` again right away: it waits for them, and "
                              "their page tells them when they are done."}


def _connected() -> dict:
    return {"status": "connected", "phase": "sources", "source": "x",
            "message": "X is connected. Call `onboard` again to carry on."}


def _unreachable() -> dict:
    return {"status": "needs_login", "phase": "sources", "source": "x",
            "message": "Opyt's sign-in service did not answer. Call `onboard` again in a minute."}
