"""
service/google_device.py — Google's device sign-in, which ties a builder's library to a Google
account.

  POST oauth2.googleapis.com/device/code {client_id, scope=openid}  → a short user code and a
                                                                      device code
  the user enters the code at google.com/device and approves
  POST oauth2.googleapis.com/token {client_id, client_secret, device_code, grant_type}
                                                                    → an id_token naming the account

Only `openid` is asked for: `sub` is the one claim kept. The client is Google's "TVs and Limited
Input devices" type, and its id and secret are this service's environment, so the secret never
ships in the open-source client.

The id_token comes straight from Google's token endpoint over TLS, so its signature is not
re-checked (OIDC Core 3.1.3.7); its audience and issuer are.
"""
from __future__ import annotations

import base64
import json
import os
from typing import NamedTuple

import requests

CLIENT_ID_ENV = "OPYT_SERVICE_GOOGLE_DEVICE_CLIENT_ID"
CLIENT_SECRET_ENV = "OPYT_SERVICE_GOOGLE_DEVICE_CLIENT_SECRET"

DEVICE_CODE_URL = "https://oauth2.googleapis.com/device/code"
TOKEN_URL = "https://oauth2.googleapis.com/token"
_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
_ISSUERS = ("accounts.google.com", "https://accounts.google.com")
_TIMEOUT = 15.0


class Unavailable(RuntimeError):
    """No device client is configured here, or Google did not answer. Says nothing about whether
    the user approved."""


class Answer(NamedTuple):
    """What Google's token endpoint said: `approved` (with `sub`), `waiting`, or `failed`."""
    status: str
    sub: str | None = None
    message: str | None = None


def _client() -> tuple[str, str]:
    cid, secret = os.environ.get(CLIENT_ID_ENV), os.environ.get(CLIENT_SECRET_ENV)
    if not (cid and secret):
        raise Unavailable("Google sign-in is not configured on this service")
    return cid, secret


def request_code() -> dict:
    """A new sign-in: `{device_code, user_code, verification_url, expires_in}`."""
    cid, _secret = _client()
    try:
        r = requests.post(DEVICE_CODE_URL, data={"client_id": cid, "scope": "openid"},
                          timeout=_TIMEOUT)
        body = r.json()
        return {"device_code": body["device_code"], "user_code": body["user_code"],
                "verification_url": body["verification_url"],
                "expires_in": float(body["expires_in"])}
    except (requests.RequestException, ValueError, KeyError) as e:
        raise Unavailable(f"Google did not start the sign-in ({type(e).__name__})") from e


def poll(device_code: str) -> Answer:
    """Ask Google once whether the sign-in behind `device_code` was approved."""
    cid, secret = _client()
    try:
        r = requests.post(TOKEN_URL, data={"client_id": cid, "client_secret": secret,
                                           "device_code": device_code, "grant_type": _GRANT},
                          timeout=_TIMEOUT)
        body = r.json()
    except (requests.RequestException, ValueError) as e:
        raise Unavailable(f"Google did not answer ({type(e).__name__})") from e
    return read_answer(body, cid)


def read_answer(body: dict, client_id: str) -> Answer:
    """Google's token answer, read. Plain values in and out:

      • an id_token issued by Google to this client → `approved`, with its `sub`
      • `authorization_pending` / `slow_down`        → `waiting`
      • anything else (denied, expired, an id_token
        for another client or issuer)                → `failed`
    """
    if body.get("error") in ("authorization_pending", "slow_down"):
        return Answer("waiting")
    claims = _claims(body.get("id_token"))
    if claims is not None and claims.get("sub") and claims.get("aud") == client_id \
            and claims.get("iss") in _ISSUERS:
        return Answer("approved", sub=str(claims["sub"]))
    return Answer("failed", message=body.get("error_description") or body.get("error")
                  or "Google did not confirm the account")


def _claims(id_token: object) -> dict | None:
    """The payload of a JWT, or None for anything that is not one."""
    if not isinstance(id_token, str) or id_token.count(".") != 2:
        return None
    payload = id_token.split(".")[1]
    try:
        return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except ValueError:      # bad base64 and bad JSON both
        return None
