"""
pipeline/sharing/invite.py — the invite format: the code's alphabet, the link, and reading both back
out of whatever a person pasted.

One home for both sides of the wire. The service mints codes from `CODE_ALPHABET` and `CODE_LEN`
(`service/grants.mint_grant`), and `accept` finds them with `_CODE`, so the two can never drift
apart. A change here is a change to a contract between versions: an install and the hosted service
can run different ones.
"""
from __future__ import annotations

import re
import string
from typing import NamedTuple
from urllib.parse import urlparse

from pipeline.sharing import account

# Letters and digits only (no `-`/`_`): `_CODE` finds the code by that alphabet, so any other
# character would make an invite silently unfindable. 43 characters is ~256 bits.
CODE_ALPHABET = string.ascii_letters + string.digits
CODE_LEN = 43

# Finds a code in a whole invite, a fragment or a bare paste; bounded on both sides so it cannot
# take a nearly-right code out of a longer run.
_CODE = re.compile(rf"(?<![A-Za-z0-9])([A-Za-z0-9]{{{CODE_LEN}}})(?![A-Za-z0-9])")

# The public invite host. A link on it is redeemed at the hosted service; any other host in a
# pasted link is a self-hosted service and redeems at its own origin.
INVITE_HOST = "useopyt.com"
_INVITE_BASE = f"https://{INVITE_HOST}/invite"


class Found(NamedTuple):
    """A code read out of a paste, and the service that issued it."""
    code: str
    url: str


def link(code: str) -> str:
    """The link an owner sends: the public invite page, with the code in the fragment."""
    return f"{_INVITE_BASE}#{code}"


def parse(pasted: str, configured_url: str) -> Found | None:
    """The code inside whatever the user pasted, and the service to redeem it at. None if the paste
    holds no code.

    A full invite chooses its issuer: `useopyt.com` is the public invite host, while every other
    explicit host is a self-hosted service. A bare code has no issuer and uses `configured_url`,
    this install's configured service."""
    m = _CODE.search(pasted or "")
    if not m:
        return None
    parsed = urlparse(pasted.strip())
    if parsed.hostname and parsed.scheme in {"http", "https"}:
        url = (account.DEFAULT_SERVICE_URL if parsed.hostname.lower() == INVITE_HOST
               else f"{parsed.scheme}://{parsed.netloc}")
    else:
        url = configured_url
    return Found(m.group(1), url.rstrip("/"))
