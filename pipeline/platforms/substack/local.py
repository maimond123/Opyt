"""Read the user's own Substack account through OPYT's managed local browser session.

Every reader here reads the session's cookies on this machine (`read_substack_cookies`).
"""

import json
import re
import time

from pipeline.host_access import request as host_request
from pipeline.ingestion.utils import SyncAuthError, log
from pipeline.platforms.browser.cookies import (
    build_cookie_header,
    list_opyt_logged_in,
    read_opyt_cookies,
)
from pipeline.platforms.substack import account as substack_account
from pipeline.platforms.substack import http as substack_http
from pipeline.platforms.substack import publication as substack_publication
from pipeline.platforms.substack.contracts import SubstackListingError

# Substack's session cookie (set on .substack.com when you're logged in). Presence
# of this is what distinguishes an authenticated subscriber request from the public
# archive read.
_SUBSTACK_AUTH_COOKIE = "substack.sid"


# The user's own "Saved posts" list — post-level curation (the X-bookmarks analog). Subject
# to an intermittent Cloudflare 403 that a retry clears (hence `http.authed_get_json_retry`).
_SAVED_ENDPOINT = "https://substack.com/api/v1/reader/saved"

# The user's own subscription list — which publications they subscribe to, and whether they pay.
# A different graph from `subscriber-lists?lists=following` (people). Keyed on the session, so it
# takes no user id. `page_v2` (not `all/v2` or `top/v2`) is the route that joins each subscription
# to its publication object.
_SUBS_ENDPOINT = "https://substack.com/api/v1/subscriptions/page_v2"


def read_substack_cookies() -> dict:
    """Read the OPYT-managed Substack session — the profile the user logged into during
    onboarding — and never the user's everyday browser, as X's `read_opyt_cookies` does.

    Raises SyncAuthError when no managed session exists (or several do); the caller falls back
    to the public archive.
    """
    return read_opyt_cookies("substack.com", _SUBSTACK_AUTH_COOKIE, source="substack")


def has_session() -> bool:
    """Whether an OPYT-owned local browser profile is logged into Substack. Scans only profiles
    OPYT created and decrypts nothing — see `connections._connected_substack`."""
    candidates, _ = list_opyt_logged_in(["substack.com"], _SUBSTACK_AUTH_COOKIE)
    return bool(candidates)


def session_ready(*, attempts: int, delay: float) -> bool:
    """Whether the managed local session answers an authenticated request, retried to ride out
    the seconds after a login — see `connections.managed_substack_session_ready` for the three
    outcomes and why a refusal counts as ready."""
    for i in range(attempts):
        try:
            if own_user_id(read_substack_cookies()):
                return True
            # None: a clean signed-out answer. Stable, but the cookie may still be settling right
            # after login, so spend the remaining attempts before concluding not-signed-in.
        except SyncAuthError:
            pass  # presence not settled yet — the flush / write-lock race; retry rides it out.
        except SubstackListingError:
            return True  # refused, not signed-out — do not blame the user's session for it.
        if i < attempts - 1:
            time.sleep(delay)
    return False


def _preloads(html: str):
    """The reader page's preloads JSON, in either shape the page has used.

    It used to inline them — `window._preloads = JSON.parse("{...}")` — and now reads them from
    a tag: `JSON.parse(document.querySelector('script#preloads-json[type="application/json"]')
    .textContent)`. None when neither shape is there.
    """
    tag = re.search(r'<script[^>]*id="preloads-json"[^>]*>(.*?)</script>', html, re.S)
    if tag:
        try:
            return json.loads(tag.group(1).strip())
        except Exception:
            pass
    marker = html.find("_preloads")
    call = html.find("JSON.parse(", marker) if marker >= 0 else -1
    quote = html.find('"', call) if call >= 0 else -1
    if quote < 0:
        return None
    try:
        return json.loads(json.JSONDecoder().raw_decode(html, quote)[0])
    except Exception:
        return None


def own_user_id(cookies: dict) -> int | None:
    """The logged-in user's own numeric id — needed to key the subscriber-lists call.

    Substack embeds it in the reader page's `window._preloads` JSON (there's no clean
    'me' JSON endpoint that survives the account API's bot protection). One HTML GET.

    Three outcomes, and the first two must stay distinct:

      raises `SubstackListingError`  the request was refused (a non-200, a transport failure), or
                                     the preloads are missing or in an unknown shape.
      returns None                   the page's preloads carry no signed-in user: signed out.
      returns an int                 the id.
    """
    # Optional transport: import-only callers do not require it.
    from curl_cffi import requests as cffi_requests
    try:
        resp = host_request(cffi_requests.get,
            "https://substack.com/inbox",
            headers={"User-Agent": substack_http.USER_AGENT, "Cookie": build_cookie_header(cookies)},
            timeout=30,
        )
    except Exception as e:
        raise SubstackListingError(f"reader page unreachable: {e}") from e
    if resp.status_code != 200:
        raise SubstackListingError(f"reader page refused: HTTP {resp.status_code}")
    html = resp.text
    data = _preloads(html)
    if data is None:
        raise SubstackListingError("reader page carries no _preloads JSON — the shape changed")
    return substack_account.preload_user(data)


def fetch_follows(cookies: dict, user_id: int | None = None) -> list[dict]:
    """The people the user follows on Substack, over the local cookie transport.

    Hits `/api/v1/user/{id}/subscriber-lists?lists=following`; `user_id` defaults to the
    logged-in user. Not the subscription list: `lists=following` is the Notes social graph
    (people), while subscriptions are read from `page_v2`.

    Returns [{name, url}]; an empty list means only that the account follows nobody. A refused
    read raises `SubstackListingError`."""
    user_id = user_id or own_user_id(cookies)
    if not user_id:
        return []
    url = f"https://substack.com/api/v1/user/{user_id}/subscriber-lists"
    try:
        data = substack_http.authed_get_json(url, {"lists": "following"}, cookies)
    except Exception as e:
        raise SubstackListingError(f"follow-list fetch failed: {e}") from e
    return substack_account.parse_subscriber_lists(data)


class LocalFollows:
    """The established cookie transport, behind the `FollowSource` call shape."""

    def follows(self, user_id: int | None = None) -> list[dict]:
        return fetch_follows(read_substack_cookies(), user_id)


class LocalSubscriptionList:
    """The cookie transport for one page of the subscription list.

    One request per page and no account-id lookup: `page_v2` is keyed on the session. Cookies
    are read on first use, matching `LocalSaved`.
    """

    def __init__(self) -> None:
        self._cookies: dict | None = None

    @property
    def cookies(self) -> dict:
        if self._cookies is None:
            self._cookies = read_substack_cookies()
        return self._cookies

    def page(self, cursor: str | None) -> dict | None:
        params = {"cursor": cursor} if cursor else {}
        try:
            return substack_http.authed_get_json_retry(_SUBS_ENDPOINT, params, self.cookies,
                                          referer="https://substack.com/inbox")
        except Exception as e:
            log(f"  [warn] subscription page fetch failed: {e}")
            return None


class LocalSaved:
    """The user's own browser session, used for both the saved list and each post's body.

    Cookies are read on first use, not at construction: `saved_source()` is called before the
    caller knows it will fetch anything, and reading the jar has a cost the phase probe's
    docstring spells out.
    """

    def authenticated_body(self, base: str) -> bool:
        """Did a body fetched from `base` carry a session the host accepted?

        A per-publication question: only `*.substack.com` honors the session, so a custom-domain
        fetch is anonymous and a paid post comes back as a teaser.
        """
        return substack_http.session_cookie_applies(base)

    def __init__(self) -> None:
        self._cookies: dict | None = None

    @property
    def cookies(self) -> dict:
        if self._cookies is None:
            self._cookies = read_substack_cookies()
        return self._cookies

    def page(self, cursor: str | None) -> dict | None:
        params = {"filter": "all"}
        if cursor:
            params["cursor"] = cursor
        try:
            return substack_http.authed_get_json_retry(_SAVED_ENDPOINT, params, self.cookies,
                                          referer="https://substack.com/saved")
        except Exception as e:
            log(f"  [warn] saved-posts page fetch failed: {e}")
            return None

    def full_post(self, base: str, slug: str) -> dict | None:
        return substack_publication.fetch_full_post(base, slug, self.cookies)
