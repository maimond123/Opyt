"""
opyt_core/openrouter_oauth.py
OpenRouter OAuth PKCE — the ONE key acquisition that needs zero paste.

  verifier → open https://openrouter.ai/auth?callback_url=http://localhost:<port>/cb/<nonce>
                  &code_challenge=<S256>&code_challenge_method=S256
  user clicks Approve
  loopback catches ?code= → POST /api/v1/auth/keys {code, code_verifier,
                            code_challenge_method} → keys.set_key(...)

No client registration or client secret. Local OPYT catches the callback on loopback. A hosted
child keeps the same PKCE verifier in memory and asks the gateway to route the one-time callback
back to that child; the gateway never receives the verifier or the resulting key.

Nothing here ever returns a key value to a caller. `acquire()` returns a status word. The
value goes straight to `keys.set_key` (chmod 600) and is never logged or echoed.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode

from opyt_core import keys, local_auth

AUTH_BASE = "https://openrouter.ai/auth"
KEYS_ENDPOINT = "https://openrouter.ai/api/v1/auth/keys"

SERVICE = "openrouter"
_HOSTED_TTL_SECONDS = 600.0
# Env var name is derived from the credential registry, never hardcoded here.


def env_name() -> str:
    """The environment variable this flow writes, straight from the registry."""
    from opyt_core.credentials_registry import by_service
    cred = by_service(SERVICE)
    if cred is None:                      # a registry edit must not crash setup
        raise OAuthError(f"no {SERVICE} row in the credential registry")
    return cred.env


class OAuthError(RuntimeError):
    pass


def hosted_enabled() -> bool:
    """Whether the gateway configured this process to receive hosted OAuth callbacks."""
    return os.environ.get("OPYT_HOSTED_OPENROUTER") == "1"


# PKCE and the browser open moved to `local_auth` on 2026-09-11, when `trial` became the
# second flow that needs both. They are re-exported under their old private names so every
# call site and test here keeps working; the implementations have ONE home.
_pkce_pair = local_auth.pkce_pair
_open_browser = local_auth.open_browser


def _store_user_key(key: str) -> str:
    """Store a key the USER owns, and retire any trial marker beside it. Returns the env name.

    BOTH homes land here, which is the whole point: the hosted flow stores inside
    `_HostedApproval.complete` and the local one at the end of `acquire`, so a clear placed in
    either caller would leave the other one wrong. `trial.clear` is what stops a funded personal
    account inheriting the starter allowance's copy the first time it runs dry.

    Imported lazily because `trial` imports THIS module for `env_name`; at module scope the two
    would deadlock the cycle.
    """
    from opyt_core import trial
    env = env_name()
    keys.set_key(env, key)
    trial.clear()
    return env


def _auth_url(callback: str, challenge: str) -> str:
    return f"{AUTH_BASE}?" + urlencode({"callback_url": callback,
                                        "code_challenge": challenge,
                                        "code_challenge_method": "S256"})


def _post_json(url: str, json: dict, timeout: float) -> dict:
    import requests
    r = requests.post(url, json=json, timeout=timeout)
    try:
        return r.json()
    except ValueError:
        raise OAuthError(f"HTTP {r.status_code}: {(r.text or '')[:200]}") from None


def _exchange(code: str, verifier: str, *, timeout: float = 30.0) -> str:
    body = _post_json(KEYS_ENDPOINT,
                      {"code": code, "code_verifier": verifier,
                       "code_challenge_method": "S256"}, timeout)
    key = body.get("key")
    if not key:
        raise OAuthError(f"no key in the exchange response: {str(body)[:200]}")
    return key


class _HostedApproval:
    """One hosted child's pending PKCE approvals. The verifier never leaves this process."""

    def __init__(self) -> None:
        self._pending: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def begin(self) -> str:
        register_url = os.environ.get("OPYT_HOSTED_INTERACTION_REGISTER_URL")
        interaction_key = os.environ.get("OPYT_HOSTED_INTERACTION_KEY")
        public_base = os.environ.get("OPYT_HOSTED_INTERACTION_URL")
        if not register_url or not interaction_key or not public_base:
            raise OAuthError("hosted OpenRouter approval is not configured by the gateway")

        nonce = secrets.token_urlsafe(32)
        verifier, challenge = _pkce_pair()
        with self._lock:
            # One OpenRouter approval at a time mirrors the gateway's one interaction route;
            # re-entry supersedes a stale link instead of retaining a queue of verifiers.
            self._pending.clear()
            self._pending[nonce] = (verifier, time.monotonic() + _HOSTED_TTL_SECONDS)

        request = urllib.request.Request(
            register_url,
            data=json.dumps({"kind": "openrouter", "nonce": nonce}).encode(),
            headers={"Content-Type": "application/json",
                     "X-Opyt-Hosted-Interaction-Key": interaction_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5):
                pass
        except (OSError, urllib.error.URLError):
            with self._lock:
                self._pending.pop(nonce, None)
            raise OAuthError("could not create the hosted OpenRouter approval link") from None

        callback = f"{public_base.rstrip('/')}/login/openrouter/{nonce}"
        return _auth_url(callback, challenge)

    def complete(self, nonce: str, code: str) -> bool:
        with self._lock:
            pending = self._pending.pop(nonce, None)
        if (pending is None or pending[1] <= time.monotonic() or not code
                or len(code) > 4096):
            return False
        try:
            key = _exchange(code, pending[0])
            _store_user_key(key)
        except Exception:
            return False
        return True


_hosted_approval = _HostedApproval()


def acquire(*, timeout: float = 300.0) -> dict:
    """Run the flow. Returns a status word, never a credential.

    No browser, or a timeout, degrades to `waiting` plus the URL to open by hand; re-calling
    `onboard` re-runs this instead of asking for a paste in chat.
    """
    if hosted_enabled():
        try:
            url = _hosted_approval.begin()
        except OAuthError as e:
            return {"status": "failed", "message": str(e)}
        return {
            "status": "waiting",
            "open_this_url": url,
            # HOW to give it, not just to give it -- `open_this_url` stays a bare URL for a
            # client that cannot render markdown, and the instruction is what one that can
            # obeys. Matches the hosted sign-in link in `mcp_server/onboard_tools.py`.
            "message": ("An approval link is ready in `open_this_url`. Present it to the user "
                        "as a link whose text reads **Approve OpenRouter for Opyt** — do not "
                        "print the raw URL. Only the user can approve it. It expires in ten "
                        "minutes. After they approve, call `onboard` again."),
        }

    verifier, challenge = _pkce_pair()
    with local_auth.Capture(timeout=timeout) as cap:
        url = _auth_url(cap.url, challenge)
        opened = _open_browser(url)
        got = cap.wait()
        if got is None:
            return {"status": "waiting", "browser_opened": opened, "open_this_url": url,
                    "message": ("Nobody has approved OpenRouter yet. The link is in "
                                "`open_this_url` — present it to the user as a link whose text "
                                "reads **Approve OpenRouter for Opyt**, not as a raw URL; only "
                                "they can approve it. Nothing was paid or stored. After they "
                                "approve, call `onboard` again.")}
        code = got["params"].get("code")
        if not code:
            return {"status": "failed",
                    "message": "OpenRouter redirected back without a code. Call `onboard` "
                               "again to restart the approval."}
    try:
        key = _exchange(code, verifier)
    except Exception as e:
        return {"status": "failed", "message": f"key exchange failed: {type(e).__name__}: {e}"}
    env = _store_user_key(key)
    # The PATH is resolved, never spelled: `$OPYT_HOME` relocates it, and this sentence is
    # shown to the user as a claim about where their key went. The local callback page carried
    # the same hardcoded literal until 2026-09-09 and told a sandboxed user the wrong file.
    return {"status": "stored", "env": env,
            "message": f"OpenRouter key stored in {keys.env_path()} (chmod 600)."}


def hosted_child_app(mcp_app):
    """Add the loopback-only OpenRouter callback receiver to one hosted child."""
    from contextlib import asynccontextmanager

    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    interaction_key = os.environ.get("OPYT_HOSTED_INTERACTION_KEY", "")

    def authorized(value: str | None) -> bool:
        return bool(interaction_key and value
                    and secrets.compare_digest(interaction_key, value))

    async def callback(request: Request):
        if not authorized(request.headers.get("X-Opyt-Hosted-Interaction-Key")):
            return JSONResponse({"error": "not_found"}, status_code=404)
        try:
            payload = await request.json()
            code = payload["code"]
        except (KeyError, TypeError, ValueError):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        if not isinstance(code, str) or len(code) > 4096:
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        stored = await asyncio.to_thread(_hosted_approval.complete,
                                         request.path_params["nonce"], code)
        return JSONResponse({"status": "stored" if stored else "unavailable"},
                            status_code=200 if stored else 400)

    @asynccontextmanager
    async def lifespan(app):
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    return Starlette(routes=[
        Route("/_hosted-openrouter/callback/{nonce}", callback, methods=["POST"]),
        Mount("/", app=mcp_app),
    ], lifespan=lifespan)
