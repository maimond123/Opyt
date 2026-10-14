"""
pipeline/sharing/account.py — this install's standing with a sharing service: its address, its
owner token, its install id.

The address is config (`settings.yaml`'s `service_url`), the token a credential in `~/.opyt/.env`
(`OPYT_SERVICE_TOKEN`). The two always travel together, as `Account`.

Credentials and keys are reached through their modules, so a test that patches
`pipeline.credentials.get_credential` or `opyt_core.keys.set_key` patches what this calls.
"""
from __future__ import annotations

import os
import uuid
from typing import NamedTuple

from opyt_core import config, keys
from opyt_core.paths import opyt_path, owner_only
from pipeline import credentials

# Where an owner publishes when they have not said otherwise; `settings.yaml`'s `service_url` wins.
DEFAULT_SERVICE_URL = "https://api.useopyt.com"


class Account(NamedTuple):
    """The service this install publishes to, and its owner token there (None: never shared)."""
    url: str
    token: str | None


def service_url() -> str:
    """The service this install publishes its export to — `settings.yaml`'s `service_url`, or
    `DEFAULT_SERVICE_URL` (also when the config is unreadable). A reader uses it too: `accept`
    redeems a bare pasted code here."""
    try:
        val = config.settings().get("service_url")
    except Exception:
        return DEFAULT_SERVICE_URL
    if isinstance(val, str) and val.strip():
        return val.strip()
    return DEFAULT_SERVICE_URL


def load() -> Account:
    """This install's account: the configured address, trailing slash dropped, and its token."""
    return Account(service_url().rstrip("/"), credentials.get_credential("opyt_service"))


def save_token(account: Account, token: str) -> Account:
    """Keep a newly registered owner token in `.env`, and return the account that holds it."""
    keys.set_key("OPYT_SERVICE_TOKEN", token)
    return account._replace(token=token)


# A token registered for a call that publishes nothing (`client.held_token`).
_PENDING_TOKEN = "service_token_pending"


def pending_token() -> str | None:
    """An owner token registered for a Google connection the user has not finished, or for a page
    of sources. Kept out of `.env` until the connection finishes, so no publishing pass uploads a
    library whose connection was abandoned or denied, or that only ever made pages."""
    try:
        return opyt_path(_PENDING_TOKEN).read_text().strip() or None
    except OSError:
        return None


def hold_token(token: str) -> None:
    """Keep a newly registered owner token aside until a Google connection finishes."""
    p = opyt_path(_PENDING_TOKEN)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(token)
    owner_only(p)


def confirm_token(account: Account, token: str) -> Account:
    """The connection finished: the held token becomes this install's owner token."""
    acct = save_token(account, token)
    opyt_path(_PENDING_TOKEN).unlink(missing_ok=True)
    return acct


def drop_token() -> None:
    """The service holds nothing for this install any more: forget its owner token, and one held
    for a Google connection, so nothing publishes until `share` or a Google connection registers
    anew. `.env` keeps an empty `OPYT_SERVICE_TOKEN=`, which every reader takes as none."""
    keys.set_key("OPYT_SERVICE_TOKEN", "")
    # `opyt_core` copied `.env` into the environment at import, and a credential read falls back
    # to that copy when `.env` has no value.
    os.environ.pop("OPYT_SERVICE_TOKEN", None)
    opyt_path(_PENDING_TOKEN).unlink(missing_ok=True)


def install_id() -> str:
    """A random id, minted once per install and sent with the redeem, so the service can count
    distinct installations (TELEMETRY.md: `tokens.install_id` — no account behind it, never
    linked to a person). Its one caller is `accept`."""
    p = opyt_path("install_id")
    if p.exists():
        return p.read_text().strip()
    iid = uuid.uuid4().hex
    p.parent.mkdir(parents=True, exist_ok=True)   # accept may be the first opyt call ever made
    p.write_text(iid)
    return iid
