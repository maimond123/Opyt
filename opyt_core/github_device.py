"""
opyt_core/github_device.py
GitHub's device sign-in — a GitHub token with zero paste.

  POST github.com/login/device/code {client_id}       → a short user code and a device code
  the user enters the code at github.com/login/device and approves
  POST github.com/login/oauth/access_token {client_id, device_code, grant_type}
                                                       → an access token → keys.set_key

Two calls and no thread: `begin` asks GitHub for a code and returns it for the host to show (the
user must read it before approving, so it cannot wait inside the call that produced it), and
`finish` asks GitHub once whether the user approved. The device code lives in this process only
and expires on GitHub's clock.

Device flow needs only the app's public client id, never a client secret, and no scopes: OPYT
reads public repositories, which need none.

Nothing here returns a token to a caller: `finish` returns a status word, and the token goes
straight to `keys.set_key`.
"""
from __future__ import annotations

import time

import requests

from opyt_core import keys, local_auth
from opyt_core.credentials_registry import SERVICES

# The public client id of Opyt's GitHub OAuth app (device flow enabled, user tokens that do not
# expire: nothing here keeps a refresh token). Empty would report the step as unavailable.
CLIENT_ID = "Ov23liExaV6kLOA2PA74"

DEVICE_CODE_URL = "https://github.com/login/device/code"
TOKEN_URL = "https://github.com/login/oauth/access_token"
_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
_TIMEOUT = 15.0
ENV = SERVICES["github"]

# The one sign-in in flight: device code, user code, where to enter it, and when it expires.
_pending: dict | None = None


def pending() -> dict | None:
    """The sign-in in flight, or None when there is none or it has expired."""
    global _pending
    if _pending is not None and _pending["expires_at"] <= time.monotonic():
        _pending = None
    return _pending


def begin() -> dict:
    """Ask GitHub for a code, open the page to enter it, and return what the user must see."""
    global _pending
    if not CLIENT_ID:
        return {"status": "unavailable"}
    try:
        r = requests.post(DEVICE_CODE_URL, data={"client_id": CLIENT_ID, "scope": ""},
                          headers={"Accept": "application/json"}, timeout=_TIMEOUT)
        body = r.json()
        _pending = {"device_code": body["device_code"], "user_code": body["user_code"],
                    "url": body["verification_uri"],
                    "expires_at": time.monotonic() + float(body["expires_in"])}
    except (requests.RequestException, ValueError, KeyError) as e:
        return {"status": "failed", "message": f"GitHub did not start the sign-in: "
                                               f"{type(e).__name__}"}
    opened = local_auth.open_browser(_pending["url"])
    return {"status": "waiting", "user_code": _pending["user_code"],
            "open_this_url": _pending["url"], "browser_opened": opened}


def finish() -> dict:
    """Ask GitHub once whether the sign-in in flight was approved. `stored`, `waiting` (not
    approved yet), or `failed` (denied or expired: the next `begin` starts over)."""
    global _pending
    now = pending()
    if now is None:
        return {"status": "failed", "message": "no GitHub sign-in is in flight"}
    try:
        r = requests.post(TOKEN_URL, data={"client_id": CLIENT_ID,
                                           "device_code": now["device_code"],
                                           "grant_type": _GRANT},
                          headers={"Accept": "application/json"}, timeout=_TIMEOUT)
        body = r.json()
    except (requests.RequestException, ValueError) as e:
        # Transport trouble says nothing about the approval: keep the code and ask again later.
        return {"status": "waiting", "user_code": now["user_code"], "open_this_url": now["url"],
                "message": f"GitHub did not answer ({type(e).__name__})"}
    token = body.get("access_token")
    if token:
        keys.set_key(ENV, token)
        _pending = None
        return {"status": "stored"}
    if body.get("error") in ("authorization_pending", "slow_down"):
        return {"status": "waiting", "user_code": now["user_code"], "open_this_url": now["url"]}
    _pending = None
    return {"status": "failed", "message": body.get("error_description") or body.get("error")
            or "GitHub refused the sign-in"}
