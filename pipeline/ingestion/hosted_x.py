"""The X half of the hosted browser boundary.

Everything site-agnostic — the profile, the shared Chrome, the interactive login desktop, the
typed request result — lives in ``hosted_browser``. This module owns only what is true of X:
its URLs, its GraphQL root, its public web bearer, its csrf header, its queryId discovery, and
the one request that proves a profile is signed in.

Python never opens the profile's cookie database and never asks CDP for cookies. The local
cookie transport remains in ``x_graphql_core`` for local installs only.
"""
from __future__ import annotations

import json
import os
import re

from pipeline.ingestion.hosted_browser import (ChromeRequestResult, ChromeRequestRunner,
                                               ChromeRequestStatus, HostedBrowserError,
                                               profile_dir, shared_chrome)
from pipeline.ingestion.utils import SyncAuthError, log

_LOGIN_URL = "https://x.com/login"
_HOME_URL = "https://x.com/home"
_GRAPHQL_ROOT = "https://x.com/i/api/graphql"
# The viewer's own identity, and so the login-completion check. GraphQL rather than a REST
# `/1.1/` route because measured 2026-09-07 against a live session: `/i/api/graphql/*` answers
# with only the bearer and the ct0 csrf header, while the REST host additionally demands an
# `x-client-transaction-id` the client computes per request — a value Opyt cannot produce.
_VIEWER_OP = "Viewer"
# Two alphabets, deliberately not one pattern. Both are interpolated into a GraphQL URL path,
# so both are validated at that boundary — but an operation NAME is an identifier, while a
# queryId is base64url and routinely contains `-`. Measured live 2026-09-07:
# `UserByScreenName` is `Gb-d6r0vxPOADdG62OEBpQ` and `Bookmarks` is `iblrFnKr6PZUR-dWpfXG6g`.
# One pattern for both silently discarded every id with a dash, which is roughly half of them.
_OP_RE = re.compile(r"^[A-Za-z0-9_]+$")
_QID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_BUNDLE_ROOT = "https://abs.twimg.com/responsive-web/client-web/"
# X builds a lazily-loaded chunk's URL at call time from its inline Webpack `p.u` map, so that
# URL appears nowhere in the page as a literal. A scan that reads literals therefore sees only
# the entry bundles the document links, and an operation that lives in a route chunk is
# unreachable to it — not slow, unreachable.
#
# Measured live 2026-09-07 across all 11 operations Opyt resolves: eight sit in the three entry
# bundles (`vendor`, `en`, `main`) and never need this map. `Bookmarks` is in none of them; its
# id is found only through the map, in `shared~bundle.BookmarkFolders~bundle.Bookmarks.<hash>`.
# So this is a capability, not a speed-up. `906e51fa` shipped it as a latency fix and `a287880f`
# reverted it for saving no time — both sampled `UserByScreenName` and `UserTweets`, which live
# in `main`, so neither measurement could show the difference. Timing evidence is not a reason
# to remove it; a measurement showing `Bookmarks` in an entry bundle would be.
#
# Mirrors `x_graphql_core._runtime_chunk_urls` in JavaScript instead of reusing it, because the
# hosted scan runs inside Chrome and may not hand page contents back to this process. Change one
# and change the other. JS spells a named group `(?<k>)` / `\k<k>` where Python spells it
# `(?P<k>)` / `(?P=k)`, and `[\s\S]` where the Python relies on the map being one line.
_RUNTIME_CHUNK_JS_RE = (
    r'\.u=(?<key>\w+)=>""\+\(\(\{(?<names>[\s\S]*?)\}\)\[\k<key>\]\|\|\k<key>\)'
    r'\+"\."\+\(\{(?<hashes>[\s\S]*?)\}\)\[\k<key>\]\+"a\.js"'
)


def login_url() -> str:
    """Where a hosted X sign-in desktop opens."""
    return _LOGIN_URL


def _public_web_bearer() -> str:
    """The web client's public identifier, never a value from a user's profile or environment."""
    # X's web client requires this non-secret identifier on GraphQL requests. Deliberately do
    # not honor X_WEB_BEARER here: a hosted process must not acquire an operator-provided token.
    from pipeline.ingestion.x_graphql_core import FALLBACK_BEARER
    return FALLBACK_BEARER


def _headers_js() -> str:
    """X's request headers as page source, because the csrf value only exists in the page."""
    return f"""{{
                "authorization": "Bearer {_public_web_bearer()}",
                "x-csrf-token": decodeURIComponent(
                    (document.cookie.match(/(?:^|; )ct0=([^;]*)/) || [])[1] || ""),
                "x-twitter-active-user": "yes",
                "x-twitter-auth-type": "OAuth2Session",
                "x-twitter-client-language": "en",
                "content-type": "application/json",
            }}"""


def graphql(runner: ChromeRequestRunner, op: str, query_id: str, variables: dict, features: dict,
            *, field_toggles: dict | None = None) -> ChromeRequestResult:
    """Run one known GraphQL operation through `runner` and return the sanitized typed result."""
    if not _OP_RE.fullmatch(op) or not _QID_RE.fullmatch(query_id):
        return ChromeRequestResult(ChromeRequestStatus.REJECTED)
    params = {
        "variables": json.dumps(variables, separators=(",", ":")),
        "features": json.dumps(features, separators=(",", ":")),
    }
    if field_toggles is not None:
        params["fieldToggles"] = json.dumps(field_toggles, separators=(",", ":"))
    from urllib.parse import urlencode
    url = f"{_GRAPHQL_ROOT}/{query_id}/{op}?{urlencode(params)}"
    return runner.fetch_json(_HOME_URL, url, _headers_js())


def discover_query_id(runner: ChromeRequestRunner, op: str, page_url: str) -> str | None:
    """Ask Chrome to discover one operation id without returning page or bundle contents.

    The entry bundles the page links are searched first, then the operation's own chunks
    rebuilt from the Webpack map (see `_RUNTIME_CHUNK_JS_RE`). That order means an unparsable
    map costs nothing for the eight operations the entry bundles already carry.
    """
    if not _OP_RE.fullmatch(op) or not page_url.startswith("https://x.com/"):
        return None
    pattern = json.dumps(r'queryId:"([^\"]+)",operationName:"' + op + r'"')
    script = rf"""(async () => {{
        const pattern = new RegExp({pattern});
        const runtimePattern = new RegExp({json.dumps(_RUNTIME_CHUNK_JS_RE)});
        const opKey = {json.dumps(op.lower())};
        const find = (text) => (text || "").match(pattern)?.[1] || null;
        let page = "";
        try {{ page = await (await fetch({json.dumps(page_url)}, {{credentials: "include"}})).text(); }}
        catch (_ignored) {{ return {{qid: null}}; }}
        let found = find(page);
        if (found) return {{qid: found}};
        const sources = [...new Set([...page.matchAll(/https:\/\/abs\.twimg\.com\/responsive-web\/client-web[^"']+?\.js/g)].map(m => m[0]))];
        const runtime = page.match(runtimePattern);
        if (runtime) {{
            for (const [, id, name] of runtime.groups.names.matchAll(/(\d+):"([^"]+)"/g)) {{
                if (!name.toLowerCase().includes(opKey)) continue;
                const hash = runtime.groups.hashes.match(
                    new RegExp("(?:^|,)" + id + ':"([0-9a-f]+)"'));
                if (hash) sources.push({json.dumps(_BUNDLE_ROOT)} + name + "." + hash[1] + "a.js");
            }}
        }}
        for (const source of [...new Set(sources)].slice(0, 150)) {{
            try {{ found = find(await (await fetch(source)).text()); }} catch (_ignored) {{ continue; }}
            if (found) return {{qid: found}};
        }}
        return {{qid: null}};
    }})()"""
    raw = runner.evaluate(_HOME_URL, script)
    qid = raw.get("qid") if raw else None
    return qid if isinstance(qid, str) and _QID_RE.fullmatch(qid) else None


def resolve_query_id(runner: ChromeRequestRunner, op: str, page_url: str) -> str | None:
    """The cached operation id, else one discovered through this runner's own Chrome.

    Takes a live runner rather than opening one, because a caller already inside
    `shared_chrome()` cannot reach for it again: `_profile_lock` is not reentrant, so that
    second acquire would block until it times out. `validate()` is exactly such a caller.
    """
    from pipeline.ingestion import x_graphql_core as core
    cached = core.qid_cache_get(op)
    if cached:
        return cached
    discovered = discover_query_id(runner, op, page_url)
    if discovered:
        core.qid_cache_put(op, discovered)
    return discovered


def validate(runner: ChromeRequestRunner) -> ChromeRequestResult:
    """The only login-completion check: an authenticated request inside Chrome.

    A logged-out profile makes this `UNAUTHENTICATED` with no id (measured), so callers
    that gate on a truthy id cannot mistake a signed-out browser for a connection.
    """
    query_id = resolve_query_id(runner, _VIEWER_OP, _HOME_URL)
    if query_id is None:
        return ChromeRequestResult(ChromeRequestStatus.UNAVAILABLE)
    result = graphql(runner, _VIEWER_OP, query_id, {}, {})
    if result.status is not ChromeRequestStatus.OK:
        return result
    viewer = ((result.data or {}).get("data") or {}).get("viewer") or {}
    user = (viewer.get("user_results") or {}).get("result") or {}
    # The caller needs only the stable viewer id, not the rest of the viewer response.
    user_id = str(user.get("rest_id") or "")
    return ChromeRequestResult(ChromeRequestStatus.OK, {"id": user_id} if user_id else {})


class HostedXSession:
    """The hosted half of ``x_graphql_core``'s transport seam; it exposes no credentials."""

    def __init__(self, referer: str) -> None:
        self.referer = referer

    def resolve_query_id(self, op: str, *, default_seed: str = "",
                         env_var: str | None = None, page_url: str = _HOME_URL) -> str:
        from pipeline.ingestion import x_graphql_core as core
        if env_var and os.getenv(env_var):
            return os.environ[env_var]
        if default_seed:
            return default_seed
        # Checked before the lock on purpose: `shared_chrome()` launches a browser, and a cache
        # hit must not pay for one.
        cached = core.qid_cache_get(op)
        if cached:
            return cached
        with shared_chrome() as chrome:
            qid = resolve_query_id(chrome, op, page_url)
        if qid:
            return qid
        raise RuntimeError(f"Could not resolve the {op} GraphQL queryId through hosted Chrome")

    def graphql_get(self, op: str, query_id: str, variables: dict, features: dict, *,
                    field_toggles: dict | None = None, tolerate_errors: bool = False) -> dict:
        from pipeline.ingestion import x_graphql_core as core
        core._refuse_if_spent(op)
        with shared_chrome() as chrome:
            result = graphql(chrome, op, query_id, variables, features,
                             field_toggles=field_toggles)
        if result.rate_limit is not None:
            core._record_rate_values(op, result.rate_limit.remaining, result.rate_limit.reset_at)
        if result.status is ChromeRequestStatus.OK:
            data = result.data or {}
            if data.get("errors"):
                if not tolerate_errors:
                    raise RuntimeError(f"{op} GraphQL returned errors")
                log(f"[x-graphql] {op}: {len(data['errors'])} partial field error(s) tolerated")
            return data
        if result.status is ChromeRequestStatus.RATE_LIMITED:
            reset = result.rate_limit.reset_at if result.rate_limit else None
            raise core.XRateLimited(f"{op} rate-limited by x.com", op=op, reset_at=reset)
        if result.status is ChromeRequestStatus.UNAUTHENTICATED:
            raise SyncAuthError("x.com rejected the hosted Chrome session — reconnect X")
        if result.status is ChromeRequestStatus.REJECTED:
            core.qid_cache_clear(op)
            raise RuntimeError(f"{op} GraphQL request was rejected; queryId cache cleared")
        raise RuntimeError(f"{op} hosted Chrome request was unavailable")

    def viewer_id(self) -> str | None:
        if not profile_dir().exists():
            return None
        with shared_chrome() as chrome:
            result = validate(chrome)
        if result.status is not ChromeRequestStatus.OK:
            return None
        value = (result.data or {}).get("id")
        return str(value) if value else None


def has_connection() -> bool:
    """A hosted X connection exists only when Chrome validates the persistent profile."""
    try:
        return HostedXSession(_HOME_URL).viewer_id() is not None
    except HostedBrowserError:
        return False
