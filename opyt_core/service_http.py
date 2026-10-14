"""
opyt_core/service_http.py — one HTTP exchange with an OPYT sharing service.

Every client of the service talks to it the same way: a bearer token in the header, a timeout on
the network call, a 2xx or a refusal, a JSON body, and the service's own error sentence when it
refuses. This is the one place that is written. What each caller says about a failure stays with
the caller, because a reader's search and an owner's upload fail into different sentences; the
three failure types below carry what they need to say it.

Two callers sit on this: `pipeline/sharing/client.py`, the owner's and redeemer's routes, and
`pipeline/retrieval/served.py`, a reader's reads of a served knowledge base.
"""
from __future__ import annotations

import requests
from requests import RequestException

TIMEOUT = 30   # seconds, on the network call — the one place a timeout belongs


class Unreachable(Exception):
    """The request got no answer. The message is the transport's own cause."""


class Unreadable(ValueError):
    """A 2xx whose body is not JSON — most often a proxy's page standing in for the service. A
    `ValueError`, as the JSON decoder's own error is, so a caller that caught that still does."""


class Refused(Exception):
    """The service answered with something other than a 2xx."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


def error_detail(r) -> str:
    """The service's own error sentence when there is one. A reverse proxy answering for a dead
    upstream sends HTML, not the service's JSON — hence the fallback to raw text."""
    try:
        return str(r.json().get("detail"))
    except Exception:
        return (r.text or "")[:200]


def request(method: str, url: str, *, token: str | None = None, json: dict | None = None,
            params: dict | None = None, data=None, timeout: float = TIMEOUT):
    """One exchange: the parsed JSON body of a 2xx. Raises `Unreachable`, `Refused` or
    `Unreadable`, so a caller that promises never to raise has exactly three things to catch.

    `data` is a raw body — a file object streams, so an export is never held in memory."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    kw: dict = {"headers": headers, "timeout": timeout}
    if params is not None:
        kw["params"] = params
    if json is not None:
        kw["json"] = json
    if data is not None:
        kw["data"] = data
    try:
        r = requests.get(url, **kw) if method == "get" else requests.post(url, **kw)
    except RequestException as e:
        raise Unreachable(str(e)) from e
    if not 200 <= r.status_code < 300:
        raise Refused(r.status_code, error_detail(r))
    try:
        return r.json()
    except ValueError as e:
        raise Unreadable(f"{url} answered with something that is not the service's JSON") from e
