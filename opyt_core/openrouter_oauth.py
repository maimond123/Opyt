"""
opyt_core/openrouter_oauth.py
OpenRouter OAuth PKCE — the ONE key acquisition that needs zero paste.

  verifier → open https://openrouter.ai/auth?callback_url=http://localhost:<port>/cb/<nonce>
                  &code_challenge=<S256>&code_challenge_method=S256
  user clicks Approve
  loopback catches ?code= → POST /api/v1/auth/keys {code, code_verifier,
                            code_challenge_method} → keys.set_key(...)

No client registration, no client secret, localhost callback on any port.

Nothing here ever returns a key value to a caller. `acquire()` returns a status word. The
value goes straight to `keys.set_key` (chmod 600) and is never logged or echoed.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import webbrowser
from urllib.parse import urlencode

from opyt_core import keys, local_auth

AUTH_BASE = "https://openrouter.ai/auth"
KEYS_ENDPOINT = "https://openrouter.ai/api/v1/auth/keys"

SERVICE = "openrouter"
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


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


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


def _open_browser(url: str) -> bool:
    try:
        return bool(webbrowser.open(url))
    except Exception:
        return False


def acquire(*, timeout: float = 300.0) -> dict:
    """Run the flow. Returns a status word, never a credential.

    No browser, or a timeout, degrades to `waiting` plus the URL to open by hand; re-calling
    `onboard` re-runs this instead of asking for a paste in chat.
    """
    verifier, challenge = _pkce_pair()
    with local_auth.Capture(route="cb", timeout=timeout) as cap:
        url = _auth_url(cap.url, challenge)
        opened = _open_browser(url)
        got = cap.wait()
        if got is None:
            return {"status": "waiting", "browser_opened": opened, "open_this_url": url,
                    "message": ("Waiting on OpenRouter approval. Open the URL above, click "
                                "Approve, then call `onboard` again. Nothing was paid or "
                                "stored.")}
        code = got["params"].get("code")
        if not code:
            return {"status": "failed",
                    "message": "OpenRouter redirected back without a code. Call `onboard` "
                               "again to restart the approval."}
    try:
        key = _exchange(code, verifier)
    except Exception as e:
        return {"status": "failed", "message": f"key exchange failed: {type(e).__name__}: {e}"}
    env = env_name()
    keys.set_key(env, key)
    return {"status": "stored", "env": env,
            "message": "OpenRouter key stored in ~/.opyt/.env (chmod 600)."}
