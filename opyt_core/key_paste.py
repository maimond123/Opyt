"""
opyt_core/key_paste.py
Browser-form capture for keys that have NO OAuth flow (twitterapi.io, and GitHub if offered).

The value never touches chat. It is typed into a page served on 127.0.0.1, POSTed to this
process, VALIDATED, and only then written by `keys.set_key`. `acquire()` returns a status word;
the value never appears in the return payload, in a log, or in an exception message.

Validate BEFORE writing: `pipeline/credentials.py` does real per-service checks (a live GET for
GitHub, a real fetch for S2, a format check for twitterapi — which is deliberately NOT a network
call, because twitterapi is pay-per-use).
"""
from __future__ import annotations

import time
import webbrowser

from opyt_core import keys, local_auth
from pipeline import credentials

# Pause between opening the signup URL and the paste form so the browser keeps focus on the
# signup tab (opening both in the same tick gives focus to whichever opened second).
SIGNUP_FOCUS_DELAY_S = 0.4


def _open_browser(url: str) -> bool:
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


def _default_sleep(seconds: float) -> None:
    time.sleep(seconds)


def acquire(service: str, env_name: str, *, signup_url: str, label: str = "",
            timeout: float = 600.0) -> dict:
    """Open the signup page, then a local form, and store one validated key.

    The timeout is LONG (10 min) on purpose: this step can include creating an account.
    Fail-safe: a timeout returns `waiting` plus the signup URL, never a dead end.

    A `SIGNUP_FOCUS_DELAY_S` pause sits between the two `webbrowser.open()` calls — see the module
    docstring above it. No `signup_url` means nothing to keep focus over, so no delay either.
    """
    label = label or service
    if signup_url:
        _open_browser(signup_url)
        _default_sleep(SIGNUP_FOCUS_DELAY_S)
    with local_auth.Capture(route="paste", timeout=timeout, label=label) as cap:
        opened = _open_browser(cap.url)
        got = cap.wait()
        if got is None:
            return {"status": "waiting", "browser_opened": opened,
                    "get_your_key_at": signup_url, "paste_it_at": cap.url,
                    "message": (f"Waiting for your {label} key. Get one at the signup URL, "
                                f"paste it into the local page, then call `onboard` again. "
                                f"⚠️ Do NOT paste it into this chat — the local page keeps it "
                                f"off the transcript.")}
        value = (got["form"].get("value") or "").strip()

    if not value:
        return {"status": "invalid", "message": f"the {label} form was submitted empty."}
    ok, why = credentials.validate_credential(service, value)
    if not ok:
        # Not written. A bad key on disk is worse than no key: every later failure looks
        # like an outage instead of a setup mistake.
        return {"status": "invalid", "message": f"that {label} key did not validate: {why}"}
    keys.set_key(env_name, value)
    return {"status": "stored", "env": env_name,
            "message": f"{label} key validated and stored in ~/.opyt/.env (chmod 600)."}
