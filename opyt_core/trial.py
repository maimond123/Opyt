"""
opyt_core/trial.py
The starter allowance — an OpenRouter key OPYT mints FOR the user, so setup needs no account.

Same destination as `openrouter_oauth`, and deliberately the same SHAPE: a key that reaches
`keys.set_key` and nothing else. What differs is whose account is behind it. The OAuth flow ends
with the user's OWN OpenRouter account; this one ends with a key the OPYT gateway minted against
ITS balance under a hard spend cap and an expiry, which is what lets a first-time user reach a
working install without creating an account anywhere.

It is a DETOUR AROUND the OAuth step, never a replacement for it. Every trial key ends — the cap
is spent or the expiry passes — and that end hands straight back to `openrouter_oauth.acquire()`.
So nothing here invents a second credential shape: a trial key IS an OpenRouter key, in the env
var the registry names, read by the one backend. Code downstream of `keys.set_key` cannot tell
the two apart, and must not be taught to.

WHY A MARKER FILE, AND WHY IT IS NOT ONBOARDING STATE. An exhausted trial key and an unfunded
user key are the SAME 402 from OpenRouter, and they need opposite advice: "add credit to your
account" is a dead end for somebody who has no account, and "authorize OPYT" is noise for
somebody who already did. The key value carries no provenance, so the single bit — did OPYT mint
this — is written beside it. `retired-onboarding-state-file` bans a PHASE file and this is not
one: it describes a CREDENTIAL, it is written by the call that writes that credential and
deleted by the call that replaces it, and no phase is ever derived from it. `onboard_state`
does not read it; `readiness` does, to choose which of two sentences a user is shown.

MARKER FIRST, THEN THE KEY. A crash between the two writes must not leave a key that looks like
the user's own — that is the one state where we would tell them to fund an account they never
made. The other order fails safe: a marker with no key reads as `missing` with the trial already
spent, which sends them down the OAuth path they would have taken anyway.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

from opyt_core import keys, local_auth, openrouter_oauth
from opyt_core.paths import opyt_home

# The deployed gateway. Overridable because the distributable invariant forbids assuming ONE
# operator's box — a self-hoster points this at their own gateway and the flow is unchanged.
DEFAULT_GATEWAY_URL = "https://mcp.useopyt.com"
GATEWAY_ENV = "OPYT_TRIAL_GATEWAY_URL"

# The child-side half of the hosted channel, named under the OPYT_HOSTED_INTERACTION_* convention
# that `retired-hosted-login-channel-names` pins. A hosted child never opens a browser for this:
# it is already talking to a gateway that already knows who it is.
HOSTED_TRIAL_URL_ENV = "OPYT_HOSTED_INTERACTION_TRIAL_URL"

_MARKER_NAME = "trial_key.json"
_TIMEOUT = 30.0


class TrialError(RuntimeError):
    pass


# ── provenance ────────────────────────────────────────────────────────────────────────────────

def marker_path() -> Path:
    """Resolved at CALL time, like `keys.env_path` and for the same reason: `$OPYT_HOME` moves
    the home, and a second spelling of that path is the drift `paths.py` exists to prevent."""
    return opyt_home() / _MARKER_NAME


def read_marker() -> dict | None:
    """The trial record, or None. Unreadable counts as absent — a corrupt marker must not be
    the thing that stops a user setting up (fail-safe), and the worst it costs is one wrong
    sentence at the moment the key dies."""
    p = marker_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def is_trial() -> bool:
    """Whether the key this install holds was minted by OPYT rather than by the user."""
    return read_marker() is not None


def record(data: dict) -> None:
    """Write the provenance of a key OPYT just minted. NO SECRET GOES IN HERE — the hash is
    OpenRouter's own public identifier for the key, which is what a support conversation or a
    revocation needs and what a stolen marker cannot spend."""
    p = marker_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "hash": data.get("hash", ""),
        "limit": data.get("limit"),
        "expires_at": data.get("expires_at", ""),
        "issued_at": int(time.time()),
    }, indent=2) + "\n", encoding="utf-8")


def clear() -> None:
    """Forget the trial. Called by whoever stores a key the USER owns — miss this and their own
    funded account inherits the trial's copy the first time it runs dry."""
    try:
        marker_path().unlink()
    except OSError:
        pass


# ── acquisition ───────────────────────────────────────────────────────────────────────────────

def gateway_base() -> str:
    return (os.environ.get(GATEWAY_ENV) or DEFAULT_GATEWAY_URL).rstrip("/")


def hosted_enabled() -> bool:
    """Whether the gateway wired this child for minting. Separate from
    `openrouter_oauth.hosted_enabled`: an operator can run a gateway that routes OAuth callbacks
    and holds no management key at all, and that gateway must not advertise a trial it cannot
    mint."""
    return bool(os.environ.get(HOSTED_TRIAL_URL_ENV))


def available() -> bool:
    """Whether a trial is worth ATTEMPTING. Not a promise — the gateway decides, and a refusal
    degrades to the OAuth flow rather than to an error.

    A marker already on disk ends it here: this home has had its allowance, and re-asking would
    spend a round trip to be told so.
    """
    if is_trial():
        return False
    return hosted_enabled() or bool(gateway_base())


def _post_json(url: str, payload: dict | None, headers: dict[str, str]) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            return json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = json.loads(e.read().decode() or "{}").get("error", "")
        except Exception:
            pass
        raise TrialError(body or f"the gateway refused ({e.code})") from None
    except (OSError, urllib.error.URLError, ValueError) as e:
        raise TrialError(f"could not reach the gateway: {type(e).__name__}") from None


def _store(body: dict) -> dict:
    key = body.get("key")
    if not key:
        raise TrialError("the gateway returned no key")
    record(body)                          # marker FIRST — see the module docstring
    keys.set_key(openrouter_oauth.env_name(), key)
    return {"status": "stored",
            "message": "A starter allowance is set up. Nothing to sign up for."}


def _acquire_hosted() -> dict:
    """One call on the channel the gateway already authenticated. No browser, no user turn —
    the connector cannot exist without a Google login, so the gateway already knows who this is
    and there is nothing left to ask."""
    return _store(_post_json(
        os.environ[HOSTED_TRIAL_URL_ENV], None,
        {"X-Opyt-Hosted-Interaction-Key": os.environ.get("OPYT_HOSTED_INTERACTION_KEY", "")}))


def _offered_by_gateway() -> bool:
    """Ask the gateway whether it can mint, BEFORE sending anyone to it.

    Without this the only way to discover a gateway that holds no management key is to open a
    browser tab at it and let the user find the dead end — and then sit in the loopback capture
    for the full five minutes waiting for a code that is never coming. The preflight turns that
    into a silent fall-through to the OpenRouter approval, which is the path they would have
    taken anyway.

    Unreachable counts as "not offered" for the same reason: a tab pointed at a gateway that is
    down is worse for the user than no tab at all.
    """
    try:
        request = urllib.request.Request(f"{gateway_base()}/healthz", method="GET")
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return bool(json.loads(response.read().decode() or "{}").get("trial"))
    except Exception:
        return False


def _acquire_local(timeout: float) -> dict:
    """PKCE against the gateway, shaped exactly like the OpenRouter approval it replaces.

    THE KEY NEVER TRAVELS IN A URL. The browser carries a one-time code; the key comes back on
    this process's own HTTPS call, proved by a verifier that never left memory. The obvious
    shortcut — redirect to the loopback with the key in a query parameter — writes a live
    credential into browser history and into the referrer of anything that page touches.
    """
    if not _offered_by_gateway():
        raise TrialError("no starter allowance is offered by this gateway")
    verifier, challenge = local_auth.pkce_pair()
    with local_auth.Capture(timeout=timeout) as cap:
        url = f"{gateway_base()}/trial/start?" + urlencode(
            {"callback_url": cap.url, "code_challenge": challenge,
             "code_challenge_method": "S256"})
        opened = local_auth.open_browser(url)
        got = cap.wait()
        if got is None:
            return {"status": "waiting", "browser_opened": opened, "open_this_url": url,
                    # HOW to give it, not just to give it -- `open_this_url` stays a bare URL
                    # for a client that cannot render markdown. Same sentence as the hosted
                    # sign-in link in `mcp_server/onboard_tools.py`.
                    "message": ("Nobody has signed in yet. The link is in `open_this_url` — "
                                "present it to the user as a link whose text reads **Sign in to "
                                "Opyt**, not as a raw URL; only they can complete it. Nothing "
                                "was stored. After they finish, call `onboard` again.")}
        params = got["params"]
    if params.get("error"):
        # The gateway refuses in ONE direction — it declines to mint — and every reason it does
        # so has the same remedy, so the caller is told to fall through rather than to retry.
        return {"status": "unavailable", "reason": params["error"],
                "message": "No starter allowance is available for this account."}
    code = params.get("code")
    if not code:
        return {"status": "failed",
                "message": "The sign-in came back without a code. Call `onboard` again."}
    return _store(_post_json(f"{gateway_base()}/trial/exchange",
                             {"code": code, "code_verifier": verifier}, {}))


def acquire(*, timeout: float = 300.0) -> dict:
    """Claim the starter allowance. Returns a status word, NEVER a credential.

    Every failure here is soft on purpose. A gateway that is down, out of budget, or simply not
    configured must leave the user exactly where they would have been without a trial — in front
    of the OpenRouter approval — because the trial is a shortcut and a shortcut that fails is
    not an outage.
    """
    try:
        return _acquire_hosted() if hosted_enabled() else _acquire_local(timeout)
    except TrialError as e:
        return {"status": "unavailable", "reason": str(e),
                "message": f"No starter allowance is available ({e})."}
