"""Substack's HTTP reads: as the signed-in subscriber (managed session cookies), or anonymously.

The policy every Substack request shares lives here once: the Chrome TLS fingerprint, the user
agent, the session cookie header, which hosts honor that cookie, and the retry for Substack's
intermittent Cloudflare 403. Callers choose authenticated or anonymous; a read that does not need
the session must not spend it.
"""

import re
import time

from pipeline.host_access import request as host_request
from pipeline.platforms.browser.cookies import build_cookie_header
from pipeline.platforms.substack.contracts import SubstackPublicReadError

USER_AGENT = "Mozilla/5.0 (compatible; OPYT/1.0)"

_URL_HOST_RE = re.compile(r"https?://([^/:]+)", re.I)

_SESSION_COOKIE_HOST = re.compile(r"^(?:[^./]+\.)?substack\.com$", re.I)


def session_cookie_applies(base: str) -> bool:
    """Whether this publication's host honors the `substack.sid` session cookie.

    `build_cookie_header` applies no domain scoping, so the session goes wherever it is pointed;
    only `substack.com` and its subdomains accept it. The local saved source's
    `authenticated_body` composes this with whether a session was sent.
    """
    m = _URL_HOST_RE.match((base or "").strip())
    return bool(m and _SESSION_COOKIE_HOST.match(m.group(1)))


def authed_get_json(url: str, params: dict, cookies: dict):
    """GET as the logged-in subscriber: Chrome TLS fingerprint + the session Cookie
    header. Used for full paid-post bodies and the subscription list."""
    # Optional transport: import-only callers do not require it.
    from curl_cffi import requests as cffi_requests
    resp = host_request(cffi_requests.get,
        url,
        params=params,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "Cookie": build_cookie_header(cookies),
        },
        impersonate="chrome120",
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def authed_get_json_retry(url: str, params: dict, cookies: dict, *, referer: str,
                           retries: int = 4, backoff: float = 2.0):
    """Authed GET with retry for Substack's intermittent Cloudflare 403. Raises the last error if
    every attempt fails — the caller decides the fail-safe.

    `referer` is the page a browser would have been on when it made this request, and it is a
    parameter because the two callers are on different hosts: the saved list is read from
    `substack.com/saved`, a post body from that publication's own site. Sending the reader's
    referer to a publication host would be a cross-site referer no real browser produces.
    """
    # Optional transport: import-only callers do not require it.
    from curl_cffi import requests as cffi_requests
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = host_request(cffi_requests.get,
                url, params=params,
                headers={
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                    "Referer": referer,
                    "Cookie": build_cookie_header(cookies),
                },
                impersonate="chrome120", timeout=30,
            )
            if resp.status_code == 200 and resp.text.strip():
                return resp.json()
            last = RuntimeError(f"HTTP {resp.status_code}")  # Cloudflare 403 / empty body
        except Exception as e:
            last = e
        time.sleep(backoff * (attempt + 1))
    raise last or RuntimeError("authed fetch failed after retries")


def public_get_json(url: str, params: dict, *, referer: str | None = None):
    """GET a public Substack JSON route with no cookie at all, so the read does not spend the
    session's share of the Cloudflare 403 window."""
    # Optional transport: import-only callers do not require it.
    from curl_cffi import requests as cffi_requests
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = referer
    resp = host_request(cffi_requests.get, url, params=params, headers=headers,
                             impersonate="chrome120", timeout=30)
    if resp.status_code != 200 or not resp.text.strip():
        raise SubstackPublicReadError(f"HTTP {resp.status_code} from {url}")
    return resp.json()
