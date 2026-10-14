"""Run X reads through OPYT's managed local cookie session: requests and query-ID discovery."""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.ingestion.utils import SyncAuthError, log
from pipeline.platforms.browser import cookies as _bc
from pipeline.platforms.x.contracts import FALLBACK_BEARER, GRAPHQL_ROOT, XRequest, XResponse
from pipeline.platforms.x.rate import XRateLimited

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Seconds to wait on a request that carries the session. An agent's computer whose outbound
# gatekeeper holds credentialed requests for a minute or more sets `OPYT_X_TIMEOUT_S` (180 waited
# out Muse's hold).
_SESSION_TIMEOUT_S = float(os.environ.get("OPYT_X_TIMEOUT_S") or 30)

# `(?:""\+)?` because X has shipped the map both ways: `.u=e=>""+(({…` and `.u=e=>(({…`. Both
# forms are pinned in the tests.
_RUNTIME_MAP_RE = re.compile(
    r'\.u=(?P<key>\w+)=>(?:""\+)?\(\(\{(?P<names>.*?)\}\)\[(?P=key)\]\|\|(?P=key)\)'
    r'\+"\."\+\(\{(?P<hashes>.*?)\}\)\[(?P=key)\]\+"a\.js"')


def _runtime_chunks(html: str, base: str) -> dict[str, str]:
    """Map every lazily loaded chunk name to its URL; return empty when the map is absent."""
    runtime = _RUNTIME_MAP_RE.search(html)
    if not runtime:
        return {}
    hashes = dict(re.findall(r'(\d+):"([0-9a-f]+)"', runtime.group("hashes")))
    return {name: f"{base}{name}.{hashes[cid]}a.js"
            for cid, name in re.findall(r'(\d+):"([^"]+)"', runtime.group("names"))
            if cid in hashes}


def _runtime_chunk_urls(html: str, op: str, base: str) -> set[str]:
    """Select operation-named chunks from X's inline Webpack name/hash maps."""
    op_key = op.lower()
    return {url for name, url in _runtime_chunks(html, base).items() if op_key in name.lower()}


def discover_query_id(op: str, page_url: str, cookies: dict) -> str | None:
    """Fetch an authenticated X page and scan its entry and runtime bundles for an operation id.

    Operation-named chunks are scanned first; the remaining chunks are scanned in parallel as
    a last resort because some operation ids live in unnamed shared chunks.
    """
    try:
        # Optional transport: import-only callers do not require it.
        from curl_cffi import requests as cffi
    except ImportError:
        return None

    pat = re.compile(r'queryId:"([^"]+)",operationName:"' + re.escape(op) + '"')
    base = "https://abs.twimg.com/responsive-web/client-web/"
    cookie_header = _bc.build_cookie_header(cookies)

    def _get(url: str) -> str:
        return cffi.get(url, headers={"User-Agent": _UA}, impersonate="chrome120",
                        timeout=20).text

    try:
        html = cffi.get(page_url,
                        headers={"User-Agent": _UA, "Cookie": cookie_header},
                        impersonate="chrome120", timeout=_SESSION_TIMEOUT_S).text
    except Exception as e:
        log(f"[x-graphql] {op} queryId discovery: page fetch failed: {e}")
        return None

    entry = dict.fromkeys(re.findall(
        r'https://abs\.twimg\.com/responsive-web/client-web[^"\']+?\.js', html))
    chunks: set[str] = set(entry)
    chunks.update(_runtime_chunk_urls(html, op, base))
    for url in list(entry)[:6]:
        try:
            js = _get(url)
        except Exception:
            continue
        match = pat.search(js)
        if match:
            log(f"[x-graphql] discovered {op} queryId in {url.split('/')[-1]}")
            return match.group(1)
        for fragment in re.findall(
                r'"((?:bundle|api|endpoints|[\w.\-]+)\.[0-9a-f]{6,}a?\.js)"', js):
            chunks.add(base + fragment.split("/")[-1])

    op_key = op.lower()
    ordered = sorted(chunks, key=lambda url: (op_key not in url.lower(), url))
    scanned = 0
    for url in ordered[:150]:
        try:
            js = _get(url)
        except Exception:
            continue
        scanned += 1
        match = pat.search(js)
        if match:
            log(f"[x-graphql] discovered {op} queryId in {url.split('/')[-1]} "
                f"(after {scanned} chunks)")
            return match.group(1)

    rest = [url for url in _runtime_chunks(html, base).values() if url not in set(ordered[:150])]

    def _scan(url: str) -> str | None:
        try:
            match = pat.search(_get(url))
        except Exception:
            return None
        return match.group(1) if match else None

    if rest:
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {pool.submit(_scan, url): url for url in rest}
            for future in as_completed(futures):
                query_id = future.result()
                if query_id:
                    for other in futures:
                        other.cancel()
                    log(f"[x-graphql] discovered {op} queryId in "
                        f"{futures[future].split('/')[-1]} (full chunk scan)")
                    return query_id
    log(f"[x-graphql] {op} queryId not found after scanning {scanned + len(rest)} chunk(s)")
    return None


def read_x_cookies() -> dict:
    """Cookie dict from OPYT's one managed X session; never returns an empty dict."""
    cookies = _bc.read_opyt_cookies(["x.com", "twitter.com"], "auth_token", source="X")
    if not cookies.get("ct0"):
        log("[x-graphql] warning: ct0 (csrf) cookie missing — request may 403")
    if not cookies.get("twid"):
        log("[x-graphql] warning: twid cookie missing — the account walks can't name the viewer "
            "and will pause")
    return cookies


def auth_headers(cookies: dict, referer: str) -> dict:
    """The header set every GraphQL read needs. `referer` is per-surface (x.com/i/lists,
    x.com/i/bookmarks, …); X is lax about it but we send the honest one."""
    bearer = os.getenv("X_WEB_BEARER") or FALLBACK_BEARER
    return {
        "authorization": f"Bearer {bearer}",
        "x-csrf-token": cookies.get("ct0", ""),       # must equal the ct0 cookie
        "x-twitter-active-user": "yes",
        "x-twitter-auth-type": "OAuth2Session",
        "x-twitter-client-language": "en",
        "content-type": "application/json",
        "User-Agent": _UA,
        "Referer": referer,
        "Cookie": _bc.build_cookie_header(cookies),
    }


def has_session() -> bool:
    """Whether an OPYT-managed local browser profile is logged into X: a row-presence check of the
    profiles OPYT created, which never decrypts (so no macOS Keychain dialog)."""
    candidates, _ = _bc.list_opyt_logged_in(["x.com", "twitter.com"], "auth_token")
    return bool(candidates)


def rate_from_headers(headers) -> tuple[int, float] | None:
    """`(remaining, reset)` off a response's headers, or None.

    Fail-safe: a response without the headers (a proxy stripped them, a shape change) yields
    nothing rather than a zero — an absent meter must not look like a spent one.
    """
    try:
        return int(headers.get("x-rate-limit-remaining")), float(headers.get("x-rate-limit-reset"))
    except (TypeError, ValueError):
        return None


class LocalXTransport:
    """The cookie/header transport for local installs."""

    def __init__(self, cookies: dict, headers: dict) -> None:
        self.cookies = cookies
        self.headers = headers

    def request(self, request: XRequest) -> XResponse:
        # Optional transport: import-only callers do not require it.
        from curl_cffi import requests as cffi

        url = f"{GRAPHQL_ROOT}/{request.query_id}/{request.op}"
        if request.method == "POST":
            # The web client's own body: the same three values a GET sends, unencoded.
            body = {"variables": request.variables, "features": request.features,
                    "queryId": request.query_id}
            if request.field_toggles is not None:
                body["fieldToggles"] = request.field_toggles
            resp = cffi.post(url, json=body, headers=self.headers, impersonate="chrome120",
                             timeout=_SESSION_TIMEOUT_S)
        else:
            params = {
                "variables": json.dumps(request.variables, separators=(",", ":")),
                "features": json.dumps(request.features, separators=(",", ":")),
            }
            if request.field_toggles is not None:
                params["fieldToggles"] = json.dumps(request.field_toggles,
                                                    separators=(",", ":"))
            resp = cffi.get(url, params=params, headers=self.headers, impersonate="chrome120",
                            timeout=_SESSION_TIMEOUT_S)
        data = error = None
        try:
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:                       # re-raised by the client, after metering
            error = e
        return XResponse(resp.status_code, data, rate_from_headers(resp.headers), error,
                         resp.text)

    def discover_query_id(self, op: str, page_url: str) -> str | None:
        return discover_query_id(op, page_url, self.cookies)

    def viewer_id(self) -> str | None:
        """The logged-in user's own numeric id (rest_id), decoded from the `twid` cookie
        (`u=<id>`). This is the identity we match list ownership against — handles rename,
        ids don't. None if twid is absent (caller should fail safe, not guess)."""
        twid = (self.cookies.get("twid") or "").replace("u%3D", "").replace("u=", "")
        return twid or None

    def refusal(self, action: str, op: str, response: XResponse | None) -> Exception:
        if action == "missing_query_id":
            return RuntimeError(
                f"Could not resolve the {op} GraphQL queryId from the JS bundles. This is a "
                f"one-time developer seed (X-wide, not per-user): grab it from one real "
                f"{op} request URL (DevTools → Network) and pass it as default_seed / the env "
                f"override.")
        assert response is not None
        if action == "rate_limited":
            return XRateLimited(
                f"{op} rate-limited (429) by x.com — this operation's 15-minute window is spent. "
                f"(If this fired with budget left on the meter it means a loop failed to "
                f"terminate — that is a bug, not just throttling.)",
                op=op, reset_at=response.rate_limit[1] if response.rate_limit else None)
        if action == "unauthenticated":
            return SyncAuthError(
                f"x.com rejected the session ({response.status}) — cookie expired or "
                f"ct0/csrf mismatch. Re-log into x.com in Chrome. Body: {response.text[:300]}")
        if action == "drift":
            return RuntimeError(
                f"{op} GraphQL {response.status} — queryId or `features` drifted "
                f"(cache cleared; next run re-discovers). Body: {response.text[:500]}")
        if action == "graphql_errors":
            return RuntimeError(f"{op} GraphQL returned errors: {(response.data or {})['errors']}")
        return response.error or RuntimeError(f"{op} GraphQL {response.status}")
