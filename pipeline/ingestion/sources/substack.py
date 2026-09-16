"""
pipeline/ingestion/sources/substack.py
Substack cookie auth, archive/post/subscription/saved-post fetching, and post→markdown rendering.

RSS feeds only return 15-20 posts. The ``/api/v1/archive`` endpoint returns ALL posts with full
``body_html``, paginated — which is what ``_fetch_all_posts`` walks.

Layer 1 only — see ``pipeline/ingestion/sources/__init__.py``. The Layer-2 note-writing walks
that used to sit above this (``ingest_substack.sync_substack`` / ``sync_saved_posts``) were
DELETED 2026-08-14 with the ``raw/`` rail. This module was deliberately NOT deleted with them:
``pipeline/kb/ingest_curation.py`` reaches it to land ATOMS. That layer split is exactly what
made the Layer-2 deletion safe — see the ``retired-sync-tool`` guard.

Three things about the USER'S OWN ACCOUNT need a session, and each has ONE entry point that
picks the transport: ``follow_source()`` for the people they follow, ``subscription_list_source()``
for the publications they subscribe to, ``saved_source()`` for the posts they saved. A collector
calls those and never a cookie reader, so a hosted home cannot reach the plaintext cookie path by
accident.

Follows and subscriptions are DIFFERENT GRAPHS and are read from different endpoints. Measured on
one real account 2026-09-08: 36 subscriptions, 21 follows, overlapping by ONE. Anything that
treats them as one list is wrong by construction.

One read here is about someone ELSE'S curation and needs no session: `fetch_recommendations`,
the publications a publication recommends. It picks no transport because there is nothing to
pick — it goes out anonymously through `_public_get_json`, and must keep doing so. Borrowing the
session for a read that does not need it spends the one resource the account reads cannot do
without.
"""

import re
import time
from datetime import datetime
from typing import NamedTuple

import html2text
import requests

from pipeline.ingestion.browser_cookies import build_cookie_header, read_opyt_cookies
from pipeline.ingestion.utils import log, SyncAuthError

FETCH_DELAY = 2  # seconds between API pages

# Archive page size. Substack does not return what you ask for — pages come back short of the
# requested limit but are still contiguous, so a short page is not the end of the archive. Advance
# the offset by what ARRIVED, and treat only an EMPTY page as the end. See doc for the measurement.
_ARCHIVE_PAGE_SIZE = 50
_ARCHIVE_RETRIES = 3
# Runaway backstop for the offset loop (mirrors _SAVED_MAX_PAGES) — at 50/page this is 25k
# posts, far past any real publication.
_ARCHIVE_MAX_PAGES = 500

# Substack's session cookie (set on .substack.com when you're logged in). Presence
# of this is what distinguishes an authenticated subscriber request from the public
# archive read.
_SUBSTACK_AUTH_COOKIE = "substack.sid"

_UA = "Mozilla/5.0 (compatible; OPYT/1.0)"

_URL_HOST_RE = re.compile(r"https?://([^/:]+)", re.I)

# The user's OWN "Saved posts" list — post-level curation (the true X-bookmarks analog). Subject
# to an intermittent Cloudflare 403 that a retry clears (hence _authed_get_json_retry).
_SAVED_ENDPOINT = "https://substack.com/api/v1/reader/saved"
# Runaway backstop for the cursor loop — 200 pages far exceeds any real saved list.
_SAVED_MAX_PAGES = 200

# The user's OWN subscription list — which PUBLICATIONS they subscribe to, and whether they pay.
# A different graph from `subscriber-lists?lists=following` (people), and the one that makes a
# newsletter reader with no Notes follows visible at all. Keyed on the SESSION: it takes no user
# id, so it costs one request where the follow read costs two.
#
# Two sibling routes exist and are the wrong ones. `/subscriptions/all/v2` drives the
# "search all subscriptions" modal; `/subscriptions/top/v2?layout=` is the sidebar's top-N grid.
# `page_v2` is the one that joins each subscription to its publication object, so a single
# request yields both the membership state and the {name, url} the entity keys on.
_SUBS_ENDPOINT = "https://substack.com/api/v1/subscriptions/page_v2"
# Runaway backstop for a cursor loop that has never been observed to run twice — see
# `fetch_subscription_list` on why the walk exists anyway. At the measured 36 rows/page this is
# 3,600 subscriptions.
_SUBS_MAX_PAGES = 100

# One publication's outbound recommendations — the only Substack read here that is about someone
# ELSE'S curation rather than the user's. PUBLIC: it answered 200 with no cookie at all (measured
# 2026-09-08), which is what lets it run on the Oracle rail with no session and no consent
# question of its own.
#
# Keyed on the publication's NUMERIC id and nothing else — a subdomain string returns
# `400 Invalid value` — so `fetch_publication_id` resolves one first.
_RECOMMENDATIONS_ENDPOINT = "https://substack.com/api/v1/recommendations/from"
# The route's measured MAXIMUM, not a page size: `limit=100` is rejected as `Invalid value` and
# there is no cursor, so this is the most one request can ever return. See `fetch_recommendations`
# on why a response of exactly this length is logged as possibly truncated.
_RECOMMENDATIONS_LIMIT = 99


# ── Cookie auth ────────────────────────────────────────────────────────────────

def read_substack_cookies(profile: str | None = None) -> dict:
    """Read the OPYT-MANAGED Substack session — the profile the user logged into during
    onboarding — and never the user's everyday browser. Mirrors X's `read_opyt_cookies`: OPYT
    reads only the session it created. Two things follow, and both were bugs before 2026-09-13:
    OPYT no longer silently reaches into the user's own Chrome cookie jar (the transplant made
    that possible without a Keychain prompt, so "we can't" was never the guardrail — this is),
    and a stale everyday-browser session can no longer SHADOW the live managed one. The measured
    failure: a dead `substack.sid` in normal Chrome won the generic reader's presence-ranked scan
    and 401'd, while the managed profile the user had just signed into saw all 36 subscriptions.

    Raises SyncAuthError when no managed session exists (or several do) — the caller falls back
    to the public archive rather than crashing. See
    docs/plans/2026-09-13-substack-managed-only-session.md; that plan's follow-up removes the
    now-dead `profile` argument end to end, which is why it is still accepted but selects nothing.

    LOCAL ONLY, and it refuses in a hosted child rather than degrading. This transport returns
    cookie VALUES to Python, which the 2026-09-06 session-token boundary forbids on a hosted
    home; `hosted_substack` asks Chrome to make the request instead. The same refusal guards
    `x_graphql_core.read_x_cookies`.
    """
    from pipeline.ingestion import hosted_browser
    if hosted_browser.enabled():
        raise RuntimeError(
            "hosted Substack must use the Chrome request runner, never read_substack_cookies")
    return read_opyt_cookies("substack.com", _SUBSTACK_AUTH_COOKIE, source="substack")


def has_managed_substack_session() -> bool:
    """Whether an OPYT-OWNED browser profile is logged into Substack.

    NOT `read_substack_cookies`, and it must never become it — they ask different questions.
    That reader asks "may I read the connected session for this fetch"; this asks "HAS the user
    connected Substack to OPYT". This one runs inside `onboard_state.derive` BEFORE any consent
    exists, so it scans only profiles OPYT created and decrypts nothing. Reading the user's jar to
    answer the second question is the 2026-08-20 cold-start finding that put a consent gate on
    `curation_catchup` in the first place.

    Since 2026-09-13 the two questions share an answer on every home: `read_substack_cookies` also
    reads the managed profile now, so a False here means the collector cannot run, exactly as for
    X. The redundant-offer cost of the old generic reader — a user logged into normal Chrome being
    asked to connect anyway — is gone with the reader that made it, not accepted.

    A hosted child answers this from its own Chrome profile instead. It has no local cookie jar
    to scan, and its profile is the only session container it has — so the question "has the
    user connected Substack to OPYT" is exactly "does that profile answer an authenticated
    request", which is what `hosted_substack.has_connection` asks.
    """
    from pipeline.ingestion import hosted_browser
    if hosted_browser.enabled():
        from pipeline.ingestion import hosted_substack
        return hosted_substack.has_connection()
    from pipeline.ingestion.browser_cookies import list_opyt_logged_in
    candidates, _ = list_opyt_logged_in(["substack.com"], _SUBSTACK_AUTH_COOKIE)
    return bool(candidates)


def managed_substack_session_ready(*, attempts: int = 4, delay: float = 1.0) -> bool:
    """Does the OPYT-managed Substack session answer an AUTHENTICATED request right now — with a
    short retry to ride out the seconds after a login? This is the VALIDITY counterpart to
    `has_managed_substack_session`'s PRESENCE, and onboarding's "done" gate turns on it.

    Why presence is not enough, measured across 2026-09-12/13: Substack hands an anonymous visitor
    a `substack.sid` the moment the sign-in page loads, so a row's existence proves nothing; and
    Chrome flushes cookies to disk lazily, so right after a real login the transplant read can miss
    the session or hit a write-locked DB and momentarily report "not connected". Both made
    onboarding act on a session that was present-but-not-usable and report "you follow nobody"
    where the truth was "give it a second". The retry absorbs the flush; `own_user_id` is the
    authority on live-ness.

    Returns True = SAFE TO PROCEED, meaning either the session authenticated (an int user id) OR
    the read was REFUSED in a way that is not the user's sign-in state (a Cloudflare/transport
    error). Translating a transient refusal into "you are not signed in" is the exact 2026-09-08
    `own_user_id` trap, and it must not be re-lived here. Returns False only when the session gives
    a clean signed-out / still-absent answer after every attempt — the one case where telling the
    user to finish signing in is correct.
    """
    from pipeline.ingestion import hosted_browser
    if hosted_browser.enabled():
        from pipeline.ingestion import hosted_substack
        return hosted_substack.has_connection()
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


def readable() -> bool:
    """Should a Substack collector be ATTEMPTED at all right now? The one home for that rule,
    because two rails ask it (`curation_catchup`, `substack_saved_catchup`) and they must not
    disagree.

    Gates on the OPYT-managed session on EVERY home. Until 2026-09-13 this was asymmetric — local
    returned True unconditionally because the reader could fall back to the user's own browser —
    but `read_substack_cookies` now reads only the managed profile, so "is Substack connected" is
    one question with one answer everywhere, exactly as for X (`has_managed_x_session`). Ungated,
    a home with no managed session would attempt the collector every pass and record `found=0`,
    which reads as "you follow nobody" where the truth is "Substack is not connected".
    """
    return has_managed_substack_session()


_SESSION_COOKIE_HOST = re.compile(r"^(?:[^./]+\.)?substack\.com$", re.I)


def _session_cookie_applies(base: str) -> bool:
    """Will the `substack.sid` session cookie be HONORED by this publication's host?

    `build_cookie_header` writes the `Cookie:` header by hand, so unlike a browser's cookie jar it
    applies no domain scoping — the session goes out to whatever host it is pointed at. Only
    `substack.com` and its subdomains accept it. Measured 2026-09-08: a custom-domain publication
    answered `confirmedLogin: False` with the session cookie present in the request.

    So "we sent a session" and "we made an authenticated request" are two different facts, and
    this is the second one. `_LocalSaved.authenticated_body` composes them; nothing else should
    have to know that the split exists.
    """
    m = _URL_HOST_RE.match((base or "").strip())
    return bool(m and _SESSION_COOKIE_HOST.match(m.group(1)))


def _authed_get_json(url: str, params: dict, cookies: dict):
    """GET as the logged-in subscriber: Chrome TLS fingerprint + the session Cookie
    header. Used for full paid-post bodies and the subscription list."""
    from curl_cffi import requests as cffi_requests
    resp = cffi_requests.get(
        url,
        params=params,
        headers={
            "Accept": "application/json",
            "User-Agent": _UA,
            "Cookie": build_cookie_header(cookies),
        },
        impersonate="chrome120",
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _authed_get_json_retry(url: str, params: dict, cookies: dict, *, referer: str,
                           retries: int = 4, backoff: float = 2.0):
    """Authed GET with retry for Substack's intermittent Cloudflare 403. Raises the last error if
    every attempt fails — the caller decides the fail-safe.

    `referer` is the page a browser would have been on when it made this request, and it is a
    parameter because the two callers are on different hosts: the saved list is read from
    `substack.com/saved`, a post body from that publication's own site. Sending the reader's
    referer to a publication host would be a cross-site referer no real browser produces.
    """
    from curl_cffi import requests as cffi_requests
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = cffi_requests.get(
                url, params=params,
                headers={
                    "Accept": "application/json",
                    "User-Agent": _UA,
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


# html2text converter
_h2t = html2text.HTML2Text()
_h2t.ignore_links      = False
_h2t.ignore_images     = False
_h2t.body_width        = 0
_h2t.unicode_snob      = True
_h2t.ignore_emphasis   = False


# ── API ──────────────────────────────────────────────────────────────────────

class SubstackListingError(RuntimeError):
    """The archive LISTING could not be completed — the walk was stopped part-way.

    Deliberately NOT `SubstackFetchError`, because the caller's fail-safe is different. A failed
    per-post fetch means one post is unknown: skip it, keep going. A failed LISTING means the
    run's inventory of what exists is a prefix, so continuing would ingest part of an archive and
    report it as the whole one. See doc for the incident this replaced."""


def _older_than(post: dict, since: datetime) -> bool:
    """Does this archive record definitely predate `since`? Only a PARSED date can say so.

    Fail-safe by construction: a missing, empty, or unparseable `post_date` — and a naive timestamp
    that will not compare against an aware `since` — returns False, so the record is RETAINED. An
    absent date is not evidence a post is old, and the same policy already governs the pagination
    stop below."""
    raw = post.get("post_date") or ""
    if not raw:
        return False
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")) < since
    except (ValueError, TypeError):
        return False


def _fetch_all_posts(
    substack_url: str, since: datetime | None = None, cookies: dict | None = None,
) -> list[dict]:
    """Paginate through Substack's archive API to get ALL posts.

    Unauthenticated (cookies=None) this reads the PUBLIC archive — paywalled posts
    come back preview-only. With `cookies` the request carries the subscriber session,
    so the list is fetched AS the user (full bodies come from _fetch_full_post, since
    even the authed archive list can return preview body_html for paid posts).

    `since` filters as well as bounds: records that definitely predate it are dropped from the
    result (`_older_than`), while the raw page still decides when to stop paginating. Without the
    filter the boundary page — the one straddling the date — returned its older half too, and the
    CLI's "on/after YYYY-MM-DD" was false for every one of them.

    Runs the whole walk on ONE `requests.Session` so Cloudflare's `__cf_bm` bot-management cookie
    persists across pages — applying rate limits per session rather than treating every page as a
    fresh first-contact client from the same IP.

    RAISES `SubstackListingError` when a page cannot be fetched, or comes back in a shape that is
    not a list of post records, after `_ARCHIVE_RETRIES` — see that class for why a partial list
    must never be returned as a complete one."""
    base = substack_url.rstrip("/")
    all_posts = []
    offset = 0
    headers = {"User-Agent": _UA}
    if cookies:
        headers["Cookie"] = build_cookie_header(cookies)

    session = requests.Session()
    try:
        for _page in range(_ARCHIVE_MAX_PAGES):
            log(f"  Fetching archive offset={offset} ...")
            batch = None
            last: Exception | None = None
            for attempt in range(_ARCHIVE_RETRIES):
                try:
                    resp = session.get(
                        f"{base}/api/v1/archive",
                        params={"sort": "new", "offset": offset, "limit": _ARCHIVE_PAGE_SIZE},
                        headers=headers,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    # The response is untrusted until it has the shape the loop indexes. A 200
                    # carrying a JSON OBJECT (Cloudflare, a rate-limit body, an endpoint change)
                    # used to reach `batch[-1]` and escape as `KeyError(-1)` — straight through
                    # the caller's `SubstackListingError` fail-safe, which is the only thing that
                    # means "write nothing, mark nothing". Raising INSIDE the retry loop puts a
                    # wrong-shaped page on the same footing as a failed one: retried, then a
                    # listing error — and a `ValueError`, not the escape type, so
                    # the only `SubstackListingError` that leaves this function is the loop's own.
                    if not isinstance(payload, list) or not all(
                            isinstance(post, dict) for post in payload):
                        raise ValueError(f"archive page was not a list of posts "
                                         f"(got {type(payload).__name__})")
                    batch = payload
                    break
                except Exception as e:
                    last = e
                    if attempt + 1 < _ARCHIVE_RETRIES:
                        # Intermittent Cloudflare 403 that a retry clears; escalating backoff
                        # covers both transient failure and throttling.
                        log(f"    [warn] archive page failed (attempt {attempt + 1}): {e}")
                        time.sleep(FETCH_DELAY * (attempt + 1))
            if batch is None:
                raise SubstackListingError(
                    f"archive listing stopped at offset={offset} after {_ARCHIVE_RETRIES} "
                    f"attempts ({len(all_posts)} posts listed so far): {last}"
                ) from last

            if not batch:            # the ONLY end-of-archive signal — a short page is not one
                break

            # `since` means on/after, so the BOUNDARY page — the one that straddles the date —
            # must be filtered, not just used to stop. Extending first and testing after handed
            # every older post on that page to the caller, which then fetched and ingested them.
            all_posts.extend(post for post in batch
                             if since is None or not _older_than(post, since))
            offset += len(batch)     # advance by what ARRIVED (unfiltered): the API under-fills

            # Stop once the page's OLDEST record predates `since`. Still read off the raw page:
            # this decides when to stop paginating, which is a different question from which
            # records to keep.
            if since and batch:
                last_date_str = batch[-1].get("post_date", "")
                if last_date_str:
                    try:
                        last_date = datetime.fromisoformat(last_date_str.replace("Z", "+00:00"))
                        if last_date < since:
                            break
                    except (ValueError, TypeError):
                        pass

            time.sleep(FETCH_DELAY)
        else:
            raise SubstackListingError(
                f"archive listing exceeded {_ARCHIVE_MAX_PAGES} pages at offset={offset} — "
                f"the endpoint never returned an empty page"
            )
    finally:
        session.close()

    return all_posts


def _is_paywalled(post: dict) -> bool:
    """A paid post whose archive-list body is a preview, not the full text."""
    return post.get("audience") == "only_paid"


class SubstackFetchError(RuntimeError):
    """The per-post fetch could not be COMPLETED — a Cloudflare challenge, a transport failure,
    or a non-JSON body on a JSON endpoint.

    Distinct, on purpose, from "this post has no body" — both used to arrive as `None`, which let
    a caller count a Cloudflare block as `no_body`. See doc for the incident. The caller must SKIP
    the post without concluding anything about it (no atom, no `seen` mark, retried next run)."""


def _fetch_full_post(base: str, slug: str, cookies: dict) -> dict | None:
    """Fetch one post's FULL body. Substack serves it at `/api/v1/posts/<slug>` on the
    publication's OWN host; a subscriber session (`cookies`) is what turns a paid post's preview
    into the whole text. `cookies={}` is legitimate and returns full text for `audience`
    `everyone` — see `saved_source` for who passes what.

    Returns the post dict (with full body_html), or None when there is no slug to ask about.
    RAISES `SubstackFetchError` when the request itself failed — see that class for why the
    two are no longer both `None`.

    Retried, and the retry is why: measured 2026-09-07 over a three-post saved list, this
    endpoint 403s intermittently and the SAME request succeeds seconds later. Without a retry
    here that transient block becomes a `body_state='pending'` stub, which every later rail
    pass re-fetches forever — a permanent hourly loop bought by one flaky response. Bounded
    retry inside one pass is the cheap place to absorb it; the list walk already had one."""
    if not slug:
        return None
    try:
        return _authed_get_json_retry(f"{base.rstrip('/')}/api/v1/posts/{slug}", {}, cookies,
                                      referer=base.rstrip("/") + "/")
    except Exception as e:
        log(f"    [warn] full-body fetch failed for {slug!r}: {e}")
        raise SubstackFetchError(f"full-body fetch failed for {slug!r}: {e}") from e


def _publication_url(pub: dict) -> str | None:
    """Canonical base URL for a publication object: its custom domain if it has one,
    else the {subdomain}.substack.com host."""
    if not isinstance(pub, dict):
        return None
    if pub.get("custom_domain"):
        return f"https://{pub['custom_domain']}"
    if pub.get("subdomain"):
        return f"https://{pub['subdomain']}.substack.com"
    return None


def own_user_id(cookies: dict) -> int | None:
    """The logged-in user's own numeric id — needed to key the subscriber-lists call.

    Substack embeds it in the reader page's `window._preloads` JSON (there's no clean
    'me' JSON endpoint that survives the account API's bot protection). One HTML GET.

    THREE outcomes, and collapsing the first two is the defect this signature exists to prevent:

      RAISES `SubstackListingError`  the request was REFUSED — a non-200, or a transport failure.
      returns None                   the page came back and carries no signed-in user: signed out,
                                     or Substack changed the preload shape.
      returns an int                 the id.

    Measured 2026-09-08: `substack.com` served a 5,987-byte "Error - Substack" page with a 403 for
    every reader route for at least half an hour. That page has no `_preloads`, so returning None
    for it made a Cloudflare window indistinguishable from a signed-out session — and one level up
    that became "you follow nobody", recorded as a SUCCESSFUL collector run. On a fresh install
    `onboard` then told the user their sign-in had gone stale and to reconnect, which was false and
    which reconnecting could not fix.
    """
    from curl_cffi import requests as cffi_requests
    try:
        resp = cffi_requests.get(
            "https://substack.com/inbox",
            headers={"User-Agent": _UA, "Cookie": build_cookie_header(cookies)},
            impersonate="chrome120", timeout=30,
        )
    except Exception as e:
        raise SubstackListingError(f"reader page unreachable: {e}") from e
    if resp.status_code != 200:
        raise SubstackListingError(f"reader page refused: HTTP {resp.status_code}")
    html = resp.text
    i = html.find("_preloads")
    q = html.find('"', html.find("JSON.parse(", i))
    if q < 0:
        log("  [warn] reader page carries no _preloads — signed out, or the shape changed")
        return None
    try:
        import json as _json
        data = _json.loads(_json.JSONDecoder().raw_decode(html, q)[0])
    except Exception as e:
        log(f"  [warn] could not decode the reader page preloads: {e}")
        return None
    return (data.get("user") or {}).get("id")


def parse_subscriber_lists(data) -> list[dict]:
    """`subscriber-lists` JSON → [{name, url}]. The ONE reader of that payload's shape.

    The shape is subscriberLists[].groups[].users[], each user carrying a
    `primary_publication` — a person you follow who has no publication is skipped. Local and
    hosted differ only in how the bytes are fetched, so they must not each grow a parser.
    """
    pubs = []
    for lst in (data.get("subscriberLists") or []) if isinstance(data, dict) else []:
        for group in lst.get("groups") or []:
            for user in group.get("users") or []:
                purl = _publication_url(user.get("primary_publication") or {})
                if not purl:
                    continue
                name = (user.get("primary_publication") or {}).get("name") or user.get("name") or ""
                pubs.append({"name": name, "url": purl})
    return pubs


def fetch_follows(cookies: dict, user_id: int | None = None) -> list[dict]:
    """The people the user FOLLOWS on Substack, over the LOCAL cookie transport.

    Hits the reader account API `/api/v1/user/{id}/subscriber-lists?lists=following`.
    `user_id` defaults to the logged-in user (resolved via own_user_id).

    NOT the subscription list, and the endpoint's name is the trap: `lists=following` returns the
    Notes-era SOCIAL graph — people — while a subscription is a person→publication relationship
    served by `/api/v1/subscriptions/page_v2` and read by `fetch_subscription_list` below.
    Verified 2026-09-08 against Substack's own frontend: the one call site for this endpoint
    renders a profile's Following/Followers tabs, under the string `"${r} isn't following anyone
    yet"`. The two graphs overlapped by ONE publication out of 36 on the measured account, so
    neither read substitutes for the other. This function was called `fetch_subscriptions` until
    2026-09-08 and its signal was stamped `subscribe`, which is the defect the rename ended — see
    the `retired-substack-follow-graph-as-subscriptions` guard.

    Returns [{name, url}] — an EMPTY list means the account follows nobody, and it must keep
    meaning only that. A REFUSED read raises `SubstackListingError` instead, the same rule
    `fetch_saved_posts` follows: this endpoint is Cloudflare-protected and hostile to bursty
    automation, so "we were blocked" is a routine outcome and it is not evidence about the
    user's graph."""
    user_id = user_id or own_user_id(cookies)
    if not user_id:
        return []
    url = f"https://substack.com/api/v1/user/{user_id}/subscriber-lists"
    try:
        data = _authed_get_json(url, {"lists": "following"}, cookies)
    except Exception as e:
        raise SubstackListingError(f"follow-list fetch failed: {e}") from e
    return parse_subscriber_lists(data)


# ── The user's own SUBSCRIPTIONS — a different graph, a different endpoint ─────

def parse_subscription_page(data) -> list[dict]:
    """One `subscriptions/page_v2` page → [{name, url, membership_state, is_favorite}]. The ONE
    reader of that payload's shape, for the reason `parse_subscriber_lists` states.

    The page carries two parallel arrays and the join is `subscriptions[].publication_id` →
    `publications[].id` — measured 2026-09-08, 36 of 36 matched. A subscription whose publication
    is absent from the join is SKIPPED: without it there is no URL, and the URL is the entity key.

    The sibling arrays `publicationUsers` and `publicationsWithPledges` came back EMPTY on the
    measured account and nothing here reads them.

    `is_founding` is deliberately not carried. It refines `membership_state == 'subscribed'` —
    the user pays MORE — and no reader distinguishes paid from founding-paid. `is_favorite` is
    carried because it is an independent act with no other representation in the payload."""
    if not isinstance(data, dict):
        return []
    pubs = {p["id"]: p for p in (data.get("publications") or [])
            if isinstance(p, dict) and p.get("id") is not None}
    out = []
    for sub in data.get("subscriptions") or []:
        if not isinstance(sub, dict):
            continue
        pub = pubs.get(sub.get("publication_id"))
        purl = _publication_url(pub or {})
        if not purl:
            continue
        out.append({
            "id": sub.get("id"),
            "name": pub.get("name") or "",
            "url": purl,
            "membership_state": sub.get("membership_state") or "",
            "is_favorite": bool(sub.get("is_favorite")),
        })
    return out


def fetch_subscription_list(source, *, max_pages: int = _SUBS_MAX_PAGES) -> list[dict]:
    """The publications the user SUBSCRIBES to — the newsletter relationship, carrying whether
    they pay for it. The read that makes a Substack reader with no Notes follows visible at all.

    `source` comes from `subscription_list_source()` and supplies pages of
    `/api/v1/subscriptions/page_v2`. The WALK lives here rather than in either transport for the
    reason `fetch_saved_posts` states: pagination is payload shape, not bytes, and two copies of
    it would drift.

    PAGINATION IS UNMEASURED. The one measured response held 36 subscriptions and carried no
    `nextCursor` key at all — Substack's own pager reads `lastPage?.nextCursor` and stops when it
    is absent, so a single-page response simply omits it. The cursor is therefore walked on
    inference, including the `cursor=` parameter NAME, which the sibling routes use. That
    inference is safe to be wrong about in one direction only, and the dedupe below is what makes
    it so: a wrong parameter name means the server ignores it and re-serves page 1, every record
    is a duplicate, `new_this_page` is 0, and the walk stops LOUD instead of looping forever.

    RAISES `SubstackListingError` when page 1 never arrived — an empty subscription list and a
    refused one are different facts, and returning [] for both makes a Cloudflare 403 arrive at
    the caller as "you subscribe to nothing". A LATER page failing returns what arrived, logged
    LOUD: that under-imports this pass and self-corrects on the next one, because nothing removes
    a signal for being absent from a walk."""
    records: list[dict] = []
    seen: set = set()
    cursor: str | None = None
    pages = 0
    while pages < max_pages:
        data = source.page(cursor)
        if data is None:
            if pages == 0:
                raise SubstackListingError("subscription listing refused on page 1")
            log(f"  [warn] subscription page {pages + 1} failed — returning {len(records)} "
                f"subscription(s) from {pages} page(s); LIST MAY BE INCOMPLETE")
            return records
        new_this_page = 0
        for rec in parse_subscription_page(data):
            # Keyed on the subscription row id, which every measured record carried. A payload
            # that ever drops it falls back to the publication URL — also unique per page.
            key = rec.get("id") or rec["url"]
            if key in seen:
                continue
            seen.add(key)
            records.append(rec)
            new_this_page += 1
        pages += 1
        cursor = data.get("nextCursor") if isinstance(data, dict) else None
        if not cursor:
            break  # clean end of the list — and the ONLY case measured so far
        if new_this_page == 0:
            log("  [warn] subscription pagination returned a cursor but no new rows — stopping "
                "to avoid a loop; LIST MAY BE INCOMPLETE")
            break
        time.sleep(1)  # pace the reader endpoint (Cloudflare is hostile to bursts)
    else:
        log(f"  [warn] subscription walk hit the {max_pages}-page cap — LIST MAY BE INCOMPLETE")

    log(f"  Fetched {len(records)} subscription(s) across {pages} page(s).")
    return records


# ── The one transport seam for the user's own Substack account ────────────────

class _LocalFollows:
    """The established cookie transport, behind the same call shape as hosted Chrome."""

    def __init__(self, profile: str | None = None) -> None:
        self.profile = profile

    def follows(self) -> list[dict]:
        return fetch_follows(read_substack_cookies(profile=self.profile))


class _HostedFollows:
    """Chrome makes the request inside the signed-in page; only the JSON body comes back.

    Both `None`s below are REFUSALS, not empty graphs, and they raise for the same reason the
    local transport does. The genuine no-session case never reaches here: `readable()` gates a
    hosted collector on `has_connection()` before it runs, so a None arriving at this point means
    the read failed, not that the user signed out.
    """

    def follows(self) -> list[dict]:
        from pipeline.ingestion import hosted_substack
        user_id = hosted_substack.own_user_id()
        if not user_id:
            raise SubstackListingError("hosted Chrome could not resolve the account id")
        body = hosted_substack.subscriber_lists(user_id)
        if body is None:
            raise SubstackListingError("hosted follow-list request was refused")
        return parse_subscriber_lists(body)


def follow_source(profile: str | None = None):
    """The one transport a caller may use to read who the user FOLLOWS on Substack.

    Selection lives here, not in the collector, so a rail cannot reach the plaintext cookie
    path from a hosted home by accident — the same rule `x_graphql_core.x_session` enforces
    for X. `profile` names a local browser profile and means nothing hosted.

    Called `subscription_source` until 2026-09-08. The name is retired rather than reused: it
    now reads as the OTHER account read, and a stale caller reaching it would silently get the
    wrong graph. `subscription_list_source` below is that other read.
    """
    from pipeline.ingestion import hosted_browser
    if hosted_browser.enabled():
        return _HostedFollows()
    return _LocalFollows(profile)


class _LocalSubscriptionList:
    """The cookie transport for one page of the subscription list.

    One request per page and no account-id lookup: `page_v2` is keyed on the SESSION and takes
    no user id, which is what makes this read cheaper than the follow read's two requests.
    Cookies are read on first use, matching `_LocalSaved`.
    """

    def __init__(self, profile: str | None = None) -> None:
        self.profile = profile
        self._cookies: dict | None = None

    @property
    def cookies(self) -> dict:
        if self._cookies is None:
            self._cookies = read_substack_cookies(profile=self.profile)
        return self._cookies

    def page(self, cursor: str | None) -> dict | None:
        params = {"cursor": cursor} if cursor else {}
        try:
            return _authed_get_json_retry(_SUBS_ENDPOINT, params, self.cookies,
                                          referer="https://substack.com/inbox")
        except Exception as e:
            log(f"  [warn] subscription page fetch failed: {e}")
            return None


class _HostedSubscriptionList:
    """Chrome reads the subscription list inside the signed-in substack.com page.

    A refusal returns None, which `fetch_subscription_list` turns into a raise on page 1 and a
    LOUD partial list afterwards — the same split `_HostedSaved` relies on.
    """

    def page(self, cursor: str | None) -> dict | None:
        from pipeline.ingestion import hosted_substack
        return hosted_substack.subscription_page(cursor)


def subscription_list_source(profile: str | None = None):
    """The one transport a caller may use to read the publications the user SUBSCRIBES to.

    A second function beside `follow_source` rather than a `kind=` argument on one, following the
    rule `set_signal`/`add_signal` states: there is no argument a caller can get wrong. The two
    read different endpoints and return different shapes, so a single selector would have to be
    told which — and being told wrong is exactly the defect the 2026-09-08 rename fixed.
    """
    from pipeline.ingestion import hosted_browser
    if hosted_browser.enabled():
        return _HostedSubscriptionList()
    return _LocalSubscriptionList(profile)


# ── An Oracle's recommendations (PUBLIC — no session, and it must not spend one) ─

class SubstackPublicReadError(RuntimeError):
    """A public Substack read was refused or came back in a shape this module cannot use.

    Separate from `SubstackListingError`, which means an ACCOUNT read failed. The two callers
    respond differently: an account failure is usually a dead session the user must reconnect,
    while this one is a Cloudflare refusal or a publication that moved, and the caller's answer
    is to skip that publication and try again next pass.
    """


def _public_get_json(url: str, params: dict, *, referer: str | None = None):
    """GET a PUBLIC Substack JSON route with no cookie at all.

    Deliberately not `_authed_get_json`: both new reads below answered 200 anonymously (measured
    2026-09-08), and a read that does not need the session must not spend it. Every authenticated
    request carries the same `substack.sid` through the same Cloudflare front door, so an
    anonymous read that borrows it converts a free request into one more chance to trip the 403
    window that rate-limits the reads which genuinely cannot run without it.
    """
    from curl_cffi import requests as cffi_requests
    headers = {"Accept": "application/json", "User-Agent": _UA}
    if referer:
        headers["Referer"] = referer
    resp = cffi_requests.get(url, params=params, headers=headers,
                             impersonate="chrome120", timeout=30)
    if resp.status_code != 200 or not resp.text.strip():
        raise SubstackPublicReadError(f"HTTP {resp.status_code} from {url}")
    return resp.json()


def fetch_publication_id(publication_url: str) -> int:
    """A publication's numeric id, read from the newest post in its PUBLIC archive.

    The recommendations route below is keyed on this id and takes nothing else — a subdomain
    string returns `400 Invalid value` (measured 2026-09-08), so the resolution is unavoidable.
    The archive is the cheapest surface that carries it: one anonymous request with `limit=1`,
    and it works identically for a custom domain and a `{subdomain}.substack.com` host, which is
    what a `blog:`-shaped Substack needs.

    Not cached. The id is immutable, so a cache would never need invalidating, but the caller is
    TTL-gated to a handful of passes a week over a roster the user confirms by hand — so the cache
    would save a few requests a week and add a column, a read path and a staleness question that
    the measured volume does not pay for. Revisit if the Substack roster reaches a size where a
    pass is dominated by resolution rather than by the recommendation reads themselves.

    RAISES `SubstackPublicReadError` when the archive is refused or holds no post. An empty
    archive is a real publication with nothing published, and it is indistinguishable HERE from a
    refusal — both mean this pass cannot read this Oracle, which is the only decision the caller
    makes from it.
    """
    data = _public_get_json(f"{publication_url.rstrip('/')}/api/v1/archive",
                            {"sort": "new", "limit": 1})
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise SubstackPublicReadError(f"archive at {publication_url} carried no post record")
    pub_id = data[0].get("publication_id")
    if not isinstance(pub_id, int):
        raise SubstackPublicReadError(f"archive at {publication_url} carried no publication_id")
    return pub_id


def parse_recommendations(data) -> list[dict]:
    """One `recommendations/from/{id}` response → [{url, name, bio, payments_state}]. The ONE
    reader of that payload's shape, for the reason `parse_subscriber_lists` states.

    The response is a bare JSON ARRAY — no envelope, no cursor field of any kind — and each item
    nests the whole recommended publication under `recommendedPublication`. An item whose nested
    publication yields no URL is SKIPPED: the URL is the entity key, and there is no second way
    to derive one here.

    The nested publication ALSO carries an `author` object with `handle`, `name` and `bio` — which
    the account-level reads do not. The handle is deliberately not carried: the entity key stays
    the publication URL so that a publication the user already subscribes to lands on the SAME
    entity and corroborates, instead of minting a second, handle-keyed id for Stage-3 to merge
    back. `bio` is carried because `screen._classify_prompt` reads it and no other Substack read
    supplies one.
    """
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        pub = item.get("recommendedPublication")
        if not isinstance(pub, dict):
            continue
        purl = _publication_url(pub)
        if not purl:
            continue
        author = pub.get("author") if isinstance(pub.get("author"), dict) else {}
        out.append({
            "url": purl,
            "name": pub.get("name") or "",
            "bio": (author or {}).get("bio") or "",
            "payments_state": pub.get("payments_state") or "",
        })
    return out


def fetch_recommendations(publication_id: int) -> list[dict]:
    """The publications one publication recommends — a public, session-free read.

    `limit` is the only bound the route offers and it has no pagination at all: the response is a
    bare array, so there is no cursor to walk and a truncated list is indistinguishable from a
    complete one except by its length. `_RECOMMENDATIONS_LIMIT` is therefore the measured MAXIMUM
    the route accepts (100 is rejected as `Invalid value`, 99 is not), and a response that comes
    back at exactly that length is logged LOUD as possibly truncated rather than silently trusted.

    That warning exists because the investigation's `limit=25` probe returned exactly 25 and the
    map recorded it as a possible count. Re-read at 99, the same publication returns 6 — the 25
    was the limit binding, not the list ending.

    RAISES `SubstackPublicReadError` on a refusal, for the same reason the account reads raise:
    an empty recommendation list and a refused read are different facts, and a publication that
    recommends nobody is common (measured: one of the three confirmed Substack Oracles).
    """
    data = _public_get_json(
        f"{_RECOMMENDATIONS_ENDPOINT}/{publication_id}",
        {"limit": _RECOMMENDATIONS_LIMIT},
        referer="https://substack.com/",
    )
    recs = parse_recommendations(data)
    if isinstance(data, list) and len(data) >= _RECOMMENDATIONS_LIMIT:
        log(f"  [warn] publication {publication_id} returned {len(data)} recommendations at the "
            f"{_RECOMMENDATIONS_LIMIT}-item route maximum — LIST MAY BE TRUNCATED")
    return recs


# ── Saved posts (post-level curation — the X-bookmarks analog) ────────────────

def _saved_item_to_record(item: dict) -> dict | None:
    """Map one saved-list item to the trimmed field record we keep (the payload is ~95%
    noise — pricing tables, palettes, i18n — all dropped). Returns None for non-posts
    (notes/comments) or an item with no stable id, so the caller just skips it.

    `publication` sits at the ITEM level (not under `post`); it's `post.wordcount` (no
    underscore)."""
    if item.get("type") != "post":
        return None
    post = item.get("post") or {}
    pub = item.get("publication") or {}
    post_id = post.get("id")
    if not post_id:
        return None
    inbox = post.get("inboxItem") if isinstance(post.get("inboxItem"), dict) else {}
    return {
        "id": post_id,
        "url": post.get("canonical_url") or "",
        "title": post.get("title") or "Untitled",
        "subtitle": post.get("subtitle") or "",
        "description": post.get("description") or "",
        "preview": post.get("truncated_body_text") or "",
        "post_date": post.get("post_date") or "",
        "saved_at": post.get("saved_at") or (inbox or {}).get("saved_at") or "",
        "wordcount": post.get("wordcount") or 0,
        "audience": post.get("audience") or "",
        "slug": post.get("slug") or "",
        "publication_name": pub.get("name") or "",
        "publication_url": pub.get("base_url") or _publication_url(pub) or "",
        "author_name": pub.get("author_name") or "",
        "author_handle": pub.get("author_handle") or "",
    }


class _LocalSaved:
    """The user's own browser session, used for BOTH the saved list and each post's body.

    Cookies are read on first use, not at construction: `saved_source()` is called before the
    caller knows it will fetch anything, and reading the jar has a cost the phase probe's
    docstring spells out.
    """

    def authenticated_body(self, base: str) -> bool:
        """Did a body fetched from `base` carry a session the host ACCEPTED?

        Was a class constant `authenticated_bodies = True` until 2026-09-08, and that constant was
        a lie for every custom-domain publication. The body fetch does carry the user's session,
        but only `*.substack.com` honors it — so on a custom domain the fetch is anonymous, a paid
        post comes back as a teaser, and `sync_substack_saved` stored it `body_state='complete'`.
        That is the exact false claim `_HostedSaved`'s `partial` branch exists to prevent, and
        `export` ships `body_state` to shared KBs.

        A per-publication QUESTION, not a per-transport constant: the transport can only answer
        "did I send a cookie", and the honest answer needs the host too.
        """
        return _session_cookie_applies(base)

    def __init__(self, profile: str | None = None) -> None:
        self.profile = profile
        self._cookies: dict | None = None

    @property
    def cookies(self) -> dict:
        if self._cookies is None:
            self._cookies = read_substack_cookies(profile=self.profile)
        return self._cookies

    def page(self, cursor: str | None) -> dict | None:
        params = {"filter": "all"}
        if cursor:
            params["cursor"] = cursor
        try:
            return _authed_get_json_retry(_SAVED_ENDPOINT, params, self.cookies,
                                          referer="https://substack.com/saved")
        except Exception as e:
            log(f"  [warn] saved-posts page fetch failed: {e}")
            return None

    def full_post(self, base: str, slug: str) -> dict | None:
        return _fetch_full_post(base, slug, self.cookies)


class _HostedSaved:
    """Chrome reads the saved list inside the signed-in substack.com page; bodies come from each
    publication's own host with NO session.

    Bodies are unauthenticated on purpose. `ChromeRequestRunner` keeps one page per site because
    a credentialed cross-origin fetch is subject to the target's CORS policy, so an authenticated
    body fetch would mean one Chrome tab per publication — hundreds of megabytes on a box sized
    for ~100 MB children. A publication's own `/api/v1/posts/{slug}` answers cookie-less with the
    full text for `audience: everyone`, which is most of a saved list; a paid post comes back as
    a preview, and `sync_substack_saved` records that as `partial` rather than claiming a body it
    did not get.
    """

    def authenticated_body(self, base: str) -> bool:
        """Never. This transport sends no cookie to any host, so `base` cannot change the answer
        — and it takes one anyway so the two transports present one interface."""
        return False

    def page(self, cursor: str | None) -> dict | None:
        from pipeline.ingestion import hosted_substack
        return hosted_substack.saved_page(cursor)

    def full_post(self, base: str, slug: str) -> dict | None:
        return _fetch_full_post(base, slug, {})


def saved_source(profile: str | None = None):
    """The one transport a caller may use to read the user's own saved posts.

    Selection lives here, not in the collector — the same rule `subscription_source` follows, so
    a rail cannot reach the plaintext cookie path from a hosted home by accident. `profile` names
    a local browser profile and means nothing hosted.
    """
    from pipeline.ingestion import hosted_browser
    if hosted_browser.enabled():
        return _HostedSaved()
    return _LocalSaved(profile)


class SavedPosts(NamedTuple):
    """What the saved-list walk saw, and whether that is ALL of it.

    `complete` exists because THREE of this walk's four exits are truncations that return
    normally — a later page refused, a cursor that stops advancing, and the page cap — and until
    2026-09-13 every one of them announced itself with a log line and nothing else. A caller could
    not tell a full list from a partial one, so "the user saved N posts by this person" and "we saw
    N of an unknown number" arrived as the same value.

    That distinction is load-bearing for exactly one reader today, `ingest_curation
    .sync_substack_saved_signals`: it writes the `save` count with `set_signal` (which REPLACES)
    on a complete walk and `ensure_signal` (presence only) otherwise, because a partial aggregate
    written as a total would lower a correct count — the one way that collector could destroy
    information. A named field rather than a log line is what makes that decidable in code.

    A TUPLE, unpacked at every call site, rather than a `.complete` attribute on a list subclass:
    the flag is impossible to forget when the caller has to name it, and a test double that hands
    back a bare list fails loudly instead of defaulting to "complete".
    """
    records: list[dict]
    complete: bool


def fetch_saved_posts(source, *, max_pages: int = _SAVED_MAX_PAGES) -> SavedPosts:
    """The user's OWN Substack "Saved posts" — post-level curation, a stronger-intent
    signal than the subscription list (that's *who* you follow; this is *what specific
    posts* you deliberately saved). The true analog of reading X bookmarks.

    `source` comes from `saved_source()` and supplies pages of /api/v1/reader/saved?filter=all.
    The WALK is here rather than in either transport because pagination is payload shape, not
    bytes: the next page's token comes back as `result.nextCursor` and is re-sent as the `cursor`
    query param until nextCursor is null. Two copies of that would drift the way two copies of
    `parse_subscriber_lists` would. Filters to `type=="post"` client-side (server-side
    `filter=post` 400s — dropping notes/comments). Returns [{…trimmed field map…}].

    RAISES `SubstackListingError` when page 1 never arrived, which is the same rule
    `_fetch_all_posts` follows and for the same reason: an empty list and a refused one are
    different facts, and returning [] for both makes a Cloudflare 403 arrive at the caller as
    "you have saved nothing". Measured 2026-09-07, live: this endpoint refused four consecutive
    attempts, and the rail reported `status: ok, added: 0` — a rail that believes it succeeded
    is a rail nobody investigates.

    A LATER page failing still returns what arrived, logged LOUD — and returns it as
    `complete=False`, which is the half a log line could not give a caller. That case
    under-imports this pass and self-corrects on the next one, because nothing here removes an
    atom for being absent from a walk.

    No silent truncation (the invariant): dedupes across pages by post.id, and if a page
    hands back a non-null cursor but yields ZERO new posts, it stops and logs LOUD — a
    cursor that doesn't advance would otherwise loop forever or silently under-fetch. All three
    truncations ride back on `SavedPosts.complete`; see that class."""
    records: list[dict] = []
    seen: set = set()
    cursor: str | None = None
    pages = 0
    complete = True
    while pages < max_pages:
        data = source.page(cursor)
        if data is None:
            if pages == 0:
                raise SubstackListingError("saved-posts listing refused on page 1")
            log(f"  [warn] saved-posts fetch failed on page {pages + 1}")
            log(f"  [warn] returning {len(records)} saved post(s) from {pages} page(s) "
                f"— LIST MAY BE INCOMPLETE (later pages unreachable)")
            return SavedPosts(records, False)
        items = (data.get("items") or []) if isinstance(data, dict) else []
        new_this_page = 0
        for it in items:
            rec = _saved_item_to_record(it)
            if rec is None or rec["id"] in seen:
                continue
            seen.add(rec["id"])
            records.append(rec)
            new_this_page += 1
        pages += 1
        cursor = data.get("nextCursor") if isinstance(data, dict) else None
        if not cursor:
            break  # clean end of the list
        if new_this_page == 0:
            log("  [warn] saved-posts pagination returned a cursor but no new posts — "
                "stopping to avoid a loop; LIST MAY BE INCOMPLETE")
            complete = False
            break
        time.sleep(1)  # pace the reader endpoint (Cloudflare is hostile to bursts)
    else:
        log(f"  [warn] saved-posts hit the {max_pages}-page cap — LIST MAY BE INCOMPLETE")
        complete = False

    log(f"  Fetched {len(records)} saved post(s) across {pages} page(s).")
    return SavedPosts(records, complete)


# ── Markdown ─────────────────────────────────────────────────────────────────

def _post_to_markdown(post: dict, author: str, author_name: str) -> str:
    """Convert a Substack archive post to markdown with frontmatter."""
    title = post.get("title", "Untitled")
    subtitle = post.get("subtitle", "")
    body_html = post.get("body_html", "")
    post_date = post.get("post_date", "")
    url = post.get("canonical_url", "")
    word_count = post.get("word_count", 0)

    # Parse date
    date_str = ""
    if post_date:
        try:
            dt = datetime.fromisoformat(post_date.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_str = post_date[:10] if len(post_date) >= 10 else ""

    # Convert HTML to markdown
    body_md = ""
    if body_html:
        body_md = _h2t.handle(body_html).strip()

    # Frontmatter
    fm = (
        f"---\n"
        f"source: substack\n"
        f'author: "{author}"\n'
        f'author_name: "{author_name}"\n'
        f"url: {url}\n"
        f"date: {date_str}\n"
        f"type: article\n"
        f"tags: []\n"
        f"---\n\n"
    )

    # Body
    body = f"# {title}\n\n"
    if subtitle:
        body += f"*{subtitle}*\n\n"
    if body_md:
        body += f"{body_md}\n\n"

    body += f"---\n*Substack · [Original post]({url})*\n"

    return fm + body
