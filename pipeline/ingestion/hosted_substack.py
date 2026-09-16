"""The Substack half of the hosted browser boundary.

Simpler than X: no GraphQL, no operation-id discovery, no bearer token. Every request below is a
plain authenticated GET that Chrome makes from a signed-in substack.com page, and only the JSON
body comes back. Nothing here reads a cookie, and a hosted child must not call
``sources.substack.read_substack_cookies`` — that reader returns cookie VALUES to Python, which
is the transport the 2026-09-06 session-token boundary forbids, and ``browser_cookies`` has no
first-class Linux Chrome backend to run it with anyway.

Payload SHAPES belong to ``sources.substack``; this module returns the raw JSON body so the
hosted and local transports parse a payload — or walk a cursor — exactly once, in one place. That
holds for all three account reads: the follow list, the subscription list, and the saved list.
"""
from __future__ import annotations

import json
from urllib.parse import quote

from pipeline.ingestion.hosted_browser import (ChromeRequestRunner, ChromeRequestStatus,
                                               HostedBrowserError, profile_dir,
                                               shared_chrome)

_LOGIN_URL = "https://substack.com/sign-in"
# The reader's own page, and the origin every request below runs from. It is also where the
# account id is found: Substack embeds it in this page's `_preloads` JSON, and there is no clean
# `me` endpoint that survives the account API's bot protection (measured for the local reader,
# `sources.substack.own_user_id`, and mirrored rather than replaced here).
_HOME_URL = "https://substack.com/inbox"
_SUBSCRIBER_LISTS = "https://substack.com/api/v1/user/{user_id}/subscriber-lists?lists=following"
# The reader's own SUBSCRIPTION list — publications, not people. Keyed on the session, so unlike
# `_SUBSCRIBER_LISTS` it needs no account id and costs one request instead of two. One page per
# call; the cursor walk lives in `sources.substack.fetch_subscription_list`, for the reason the
# saved list states.
_SUBSCRIPTIONS = "https://substack.com/api/v1/subscriptions/page_v2"
# The reader's own saved list. `filter=all` then filtered to posts client-side; `filter=post`
# 400s. One page per call — the cursor walk itself lives in `sources.substack.fetch_saved_posts`,
# because pagination is payload shape and the local transport would otherwise grow a second copy.
_SAVED = "https://substack.com/api/v1/reader/saved?filter=all"
_JSON_HEADERS_JS = '{"Accept": "application/json"}'


def login_url() -> str:
    """Where a hosted Substack sign-in desktop opens."""
    return _LOGIN_URL


def _own_user_id(runner: ChromeRequestRunner) -> int | None:
    """The signed-in account's numeric id, extracted inside the page.

    Mirrors `sources.substack.own_user_id` step for step: fetch the reader page, find the
    `_preloads` JSON literal, decode it twice (a JSON string that holds JSON), and take the
    user id. Only that ONE number leaves the renderer — never the page.
    """
    script = f"""(async () => {{
        let html = "";
        try {{ html = await (await fetch({json.dumps(_HOME_URL)},
                                         {{credentials: "include"}})).text(); }}
        catch (_ignored) {{ return {{id: null}}; }}
        const marker = html.indexOf("_preloads");
        if (marker < 0) return {{id: null}};
        const call = html.indexOf("JSON.parse(", marker);
        if (call < 0) return {{id: null}};
        const start = html.indexOf('"', call);
        if (start < 0) return {{id: null}};
        let end = start + 1;
        while (end < html.length && html[end] !== '"') {{
            end += html[end] === "\\\\" ? 2 : 1;
        }}
        if (end >= html.length) return {{id: null}};
        let preloads = null;
        try {{ preloads = JSON.parse(JSON.parse(html.slice(start, end + 1))); }}
        catch (_ignored) {{ return {{id: null}}; }}
        const id = (preloads && preloads.user) ? preloads.user.id : null;
        return {{id: typeof id === "number" ? id : null}};
    }})()"""
    raw = runner.evaluate(_HOME_URL, script)
    value = raw.get("id") if raw else None
    return value if isinstance(value, int) else None


def own_user_id() -> int | None:
    """The signed-in Substack account's id, or None when the profile holds no live session."""
    if not profile_dir().exists():
        return None
    with shared_chrome() as chrome:
        return _own_user_id(chrome)


def subscriber_lists(user_id: int) -> dict | None:
    """The raw `subscriber-lists` JSON body for one account, or None when the read failed.

    Cloudflare guards this endpoint and it 403s under bursty automation, so it is called once
    per sync and never in a loop. A refusal returns None rather than raising: the caller's
    fail-safe is an empty discovery pass, not a dead rail.
    """
    with shared_chrome() as chrome:
        result = chrome.fetch_json(
            _HOME_URL, _SUBSCRIBER_LISTS.format(user_id=user_id), _JSON_HEADERS_JS)
    return result.data if result.status is ChromeRequestStatus.OK else None


def saved_page(cursor: str | None = None) -> dict | None:
    """One page of the signed-in reader's saved posts, or None when the read failed.

    Same Cloudflare exposure as `subscriber_lists`, so a refusal returns None and
    `sources.substack.fetch_saved_posts` stops with what it has rather than claiming a complete
    list. The cursor is a value Substack minted, so it is percent-encoded before it goes into the
    query string.
    """
    url = _SAVED if not cursor else f"{_SAVED}&cursor={quote(cursor, safe='')}"
    with shared_chrome() as chrome:
        result = chrome.fetch_json(_HOME_URL, url, _JSON_HEADERS_JS)
    return result.data if result.status is ChromeRequestStatus.OK else None


def subscription_page(cursor: str | None = None) -> dict | None:
    """One page of the signed-in reader's subscription list, or None when the read failed.

    Simpler than `subscriber_lists`: no `own_user_id` call, because the route is keyed on the
    session. Same Cloudflare exposure, so a refusal returns None and the caller decides — page 1
    becomes a raise, a later page a LOUD partial list.

    The cursor is a value Substack minted, so it is percent-encoded before it goes into the query
    string. No response has ever carried one (see `fetch_subscription_list`), so this branch is
    unmeasured and mirrors `saved_page`, the one cursor walk that is.
    """
    url = _SUBSCRIPTIONS if not cursor else f"{_SUBSCRIPTIONS}?cursor={quote(cursor, safe='')}"
    with shared_chrome() as chrome:
        result = chrome.fetch_json(_HOME_URL, url, _JSON_HEADERS_JS)
    return result.data if result.status is ChromeRequestStatus.OK else None


def has_connection() -> bool:
    """A hosted Substack connection exists only when Chrome resolves the account id.

    That id is what every later read is keyed on, so resolving it is both the cheapest proof of
    a live session and the thing the connection is FOR. A signed-out profile is served the
    logged-out reader page, which carries no `user` in its preloads.
    """
    try:
        return own_user_id() is not None
    except HostedBrowserError:
        return False
