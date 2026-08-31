"""
pipeline/ingestion/sources/substack.py
Substack cookie auth, archive/post/subscription/saved-post fetching, and post→markdown rendering.

RSS feeds only return 15-20 posts. The ``/api/v1/archive`` endpoint returns ALL posts with full
``body_html``, paginated — which is what ``_fetch_all_posts`` walks.

Layer 1 only — see ``pipeline/ingestion/sources/__init__.py``. The Layer-2 note-writing walks
that used to sit above this (``ingest_substack.sync_substack`` / ``sync_saved_posts``) were
DELETED 2026-08-14 with the ``raw/`` rail. This module was deliberately NOT deleted with them:
``pipeline/kb/ingest_curation.py`` imports ``fetch_subscriptions``, ``own_user_id``,
``_fetch_full_post`` and ``SubstackFetchError`` from here to land ATOMS. That layer split is
exactly what made the Layer-2 deletion safe — see the ``retired-sync-tool`` guard.
"""

import time
from datetime import datetime

import html2text
import requests

from pipeline.ingestion.browser_cookies import build_cookie_header, read_cookies
from pipeline.ingestion.utils import log

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

# The user's OWN "Saved posts" list — post-level curation (the true X-bookmarks analog). Subject
# to an intermittent Cloudflare 403 that a retry clears (hence _authed_get_json_retry).
_SAVED_ENDPOINT = "https://substack.com/api/v1/reader/saved"
# Runaway backstop for the cursor loop — 200 pages far exceeds any real saved list.
_SAVED_MAX_PAGES = 200


# ── Cookie auth ────────────────────────────────────────────────────────────────

def read_substack_cookies(profile: str | None = None) -> dict:
    """Read the logged-in Substack session cookies from the user's browser (auto-detected
    across Chrome/Brave/Edge/Firefox/Safari). Raises SyncAuthError
    (via the shared reader) if not logged in / the profile is ambiguous — the caller
    falls back to the public archive rather than crashing."""
    return read_cookies(
        "substack.com", _SUBSTACK_AUTH_COOKIE,
        profile=profile, env_var="SUBSTACK_CHROME_PROFILE", source="substack",
    )


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


def _fetch_all_posts(
    substack_url: str, since: datetime | None = None, cookies: dict | None = None,
) -> list[dict]:
    """Paginate through Substack's archive API to get ALL posts.

    Unauthenticated (cookies=None) this reads the PUBLIC archive — paywalled posts
    come back preview-only. With `cookies` the request carries the subscriber session,
    so the list is fetched AS the user (full bodies come from _fetch_full_post, since
    even the authed archive list can return preview body_html for paid posts).

    Runs the whole walk on ONE `requests.Session` so Cloudflare's `__cf_bm` bot-management cookie
    persists across pages — applying rate limits per session rather than treating every page as a
    fresh first-contact client from the same IP.

    RAISES `SubstackListingError` when a page cannot be fetched after `_ARCHIVE_RETRIES` — see
    that class for why a partial list must never be returned as a complete one."""
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
                    batch = resp.json()
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

            all_posts.extend(batch)
            offset += len(batch)     # advance by what ARRIVED: the API under-fills pages

            # Stop if we've passed the since date
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
    """Fetch one post's FULL body as an authenticated subscriber.

    SPIKE — confirm this endpoint against a real subscribed publication before
    trusting the happy path. Substack serves a single post's full body_html at
    `/api/v1/posts/<slug>` when the request carries a subscriber session; the archive
    list returns only a preview for paid posts.

    Returns the post dict (with full body_html), or None when there is no slug to ask about.
    RAISES `SubstackFetchError` when the request itself failed — see that class for why the
    two are no longer both `None`."""
    if not slug:
        return None
    try:
        return _authed_get_json(f"{base.rstrip('/')}/api/v1/posts/{slug}", {}, cookies)
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
    'me' JSON endpoint that survives the account API's bot protection). One HTML GET;
    returns None on any failure so callers degrade gracefully."""
    from curl_cffi import requests as cffi_requests
    try:
        html = cffi_requests.get(
            "https://substack.com/inbox",
            headers={"User-Agent": _UA, "Cookie": build_cookie_header(cookies)},
            impersonate="chrome120", timeout=30,
        ).text
        i = html.find("_preloads")
        q = html.find('"', html.find("JSON.parse(", i))
        if q < 0:
            return None
        import json as _json
        data = _json.loads(_json.JSONDecoder().raw_decode(html, q)[0])
        return (data.get("user") or {}).get("id")
    except Exception as e:
        log(f"  [warn] could not resolve own user id: {e}")
        return None


def fetch_subscriptions(cookies: dict, user_id: int | None = None) -> list[dict]:
    """The user's OWN Substack "Following" list — the Substack-primary analog of
    reading X bookmarks (auto-discovers their curated source set, no handles typed).

    Hits the reader account API `/api/v1/user/{id}/subscriber-lists?lists=following`.
    `user_id` defaults to the logged-in user (resolved via own_user_id). The shape is
    subscriberLists[].groups[].users[], each user carrying a
    `primary_publication` — a person you follow who has no publication is skipped.
    Returns [{name, url}]; [] on any failure (fail-safe — never crash discovery).

    NOTE: this account API is Cloudflare-protected and hostile to bursty automation —
    call it sparingly (once per sync) and paced, or it will 403 the caller."""
    user_id = user_id or own_user_id(cookies)
    if not user_id:
        return []
    url = f"https://substack.com/api/v1/user/{user_id}/subscriber-lists"
    try:
        data = _authed_get_json(url, {"lists": "following"}, cookies)
    except Exception as e:
        log(f"  [warn] subscription-list fetch failed: {e}")
        return []

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


# ── Saved posts (post-level curation — the X-bookmarks analog) ────────────────

def _authed_get_json_retry(url: str, params: dict, cookies: dict, *,
                           retries: int = 4, backoff: float = 2.0):
    """Authed GET with retry for Substack's intermittent Cloudflare 403 on the reader
    endpoints. Raises the last error if every attempt fails — the caller decides the fail-safe."""
    from curl_cffi import requests as cffi_requests
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = cffi_requests.get(
                url, params=params,
                headers={
                    "Accept": "application/json",
                    "User-Agent": _UA,
                    "Referer": "https://substack.com/saved",
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
    raise last or RuntimeError("saved fetch failed after retries")


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


def fetch_saved_posts(cookies: dict, *, max_pages: int = _SAVED_MAX_PAGES) -> list[dict]:
    """The user's OWN Substack "Saved posts" — post-level curation, a stronger-intent
    signal than the subscription list (that's *who* you follow; this is *what specific
    posts* you deliberately saved). The true analog of reading X bookmarks.

    Hits /api/v1/reader/saved?filter=all and follows cursor pagination — the next page's
    token comes back as `result.nextCursor` and is re-sent as the `cursor` query param, until
    nextCursor is null. Filters to `type=="post"` client-side (server-side `filter=post` 400s —
    dropping notes/comments). Returns [{…trimmed field map…}]; [] on total failure (fail-safe).

    No silent truncation (the invariant): dedupes across pages by post.id, and if a page
    hands back a non-null cursor but yields ZERO new posts, it stops and logs LOUD — a
    cursor that doesn't advance would otherwise loop forever or silently under-fetch."""
    records: list[dict] = []
    seen: set = set()
    cursor: str | None = None
    pages = 0
    while pages < max_pages:
        params = {"filter": "all"}
        if cursor:
            params["cursor"] = cursor
        try:
            data = _authed_get_json_retry(_SAVED_ENDPOINT, params, cookies)
        except Exception as e:
            log(f"  [warn] saved-posts fetch failed on page {pages + 1}: {e}")
            if pages == 0:
                return []  # never got page 1 → fail-safe empty, no partial claim
            log(f"  [warn] returning {len(records)} saved post(s) from {pages} page(s) "
                f"— LIST MAY BE INCOMPLETE (later pages unreachable)")
            return records
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
            break
        time.sleep(1)  # pace the reader endpoint (Cloudflare is hostile to bursts)
    else:
        log(f"  [warn] saved-posts hit the {max_pages}-page cap — LIST MAY BE INCOMPLETE")

    log(f"  Fetched {len(records)} saved post(s) across {pages} page(s).")
    return records


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
