"""A Substack publication's public reads: its archive, one post's full body, its numeric id and
the publications it recommends.

RSS feeds only return 15-20 posts. The ``/api/v1/archive`` endpoint returns all posts with full
``body_html``, paginated — which is what ``fetch_all_posts`` walks.
"""

import time
from datetime import datetime
from typing import Literal, TypeAlias

import requests

from pipeline.ingestion.utils import log
from pipeline.platforms.browser.cookies import build_cookie_header
from pipeline.platforms.substack import http as substack_http
from pipeline.platforms.substack.contracts import (
    SubstackFetchError,
    SubstackListingError,
    SubstackPublicReadError,
)

ArchiveAction: TypeAlias = Literal["finish", "continue", "limit"]


def archive_progress(page: list[dict], last_date: object, since: datetime | None,
                     pages_read: int, page_limit: int) -> ArchiveAction:
    """Decide whether a validated archive page ends, continues, or exhausts the bounded walk."""
    if not page:
        return "finish"
    if since and last_date:
        try:
            parsed = datetime.fromisoformat(last_date.replace("Z", "+00:00"))
            if parsed < since:
                return "finish"
        except (ValueError, TypeError):
            pass
    if pages_read >= page_limit:
        return "limit"
    return "continue"


FETCH_DELAY = 2  # seconds between API pages

# Archive page size. Substack does not return what you ask for — pages come back short of the
# requested limit but are still contiguous, so a short page is not the end of the archive. Advance
# the offset by what arrived, and treat only an empty page as the end.
ARCHIVE_PAGE_SIZE = 50
_ARCHIVE_RETRIES = 3
# Runaway backstop for the offset loop — at 50/page this is 25k
# posts, far past any real publication.
_ARCHIVE_MAX_PAGES = 500

# One publication's outbound recommendations — someone else's curation, read with no session.
# Keyed on the publication's numeric id only, so `fetch_publication_id` resolves one first.
_RECOMMENDATIONS_ENDPOINT = "https://substack.com/api/v1/recommendations/from"
# The route's maximum, not a page size (`limit=100` is rejected, and there is no cursor).
_RECOMMENDATIONS_LIMIT = 99


def archive_posts(payload) -> list | None:
    """The archive endpoint's posts, in either shape it has answered with.

    It used to answer with a bare list and now answers with an envelope — `{"posts": [...],
    "nextCursor": ...}` — so the posts are read from whichever shape arrived. None when the body
    is neither shape.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("posts"), list):
        return payload["posts"]
    return None


def older_than(post: dict, since: datetime) -> bool:
    """Does this archive record definitely predate `since`? Only a parsed date can say so: a
    missing, unparseable or naive `post_date` returns False, so the record is retained."""
    raw = post.get("post_date") or ""
    if not raw:
        return False
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")) < since
    except (ValueError, TypeError):
        return False


def fetch_all_posts(
    substack_url: str, since: datetime | None = None, cookies: dict | None = None,
) -> list[dict]:
    """Paginate through Substack's archive API to get all posts.

    Unauthenticated (cookies=None) this reads the public archive — paywalled posts come back
    preview-only. With `cookies` the list is fetched as the user (full bodies still come from
    fetch_full_post).

    `since` filters as well as bounds: records that definitely predate it are dropped
    (`older_than`), while the raw page decides when to stop. The walk runs on one
    `requests.Session` so Cloudflare's `__cf_bm` cookie persists across pages.

    Raises `SubstackListingError` when a page cannot be fetched, or is not a list of post records,
    after `_ARCHIVE_RETRIES`."""
    base = substack_url.rstrip("/")
    all_posts = []
    offset = 0
    headers = {"User-Agent": substack_http.USER_AGENT}
    if cookies and substack_http.session_cookie_applies(base):
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
                        params={"sort": "new", "offset": offset, "limit": ARCHIVE_PAGE_SIZE},
                        headers=headers,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    # A wrong-shaped 200 is treated like a failed page: retried, then a listing
                    # error.
                    posts = archive_posts(payload)
                    if posts is None or not all(isinstance(post, dict) for post in posts):
                        raise ValueError(f"archive page was not a list of posts "
                                         f"(got {type(payload).__name__})")
                    batch = posts
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

            last_date = batch[-1].get("post_date", "") if batch else None
            # An empty page answers "finish", the only end-of-archive signal: a short page is not.
            progress = archive_progress(batch, last_date, since,
                                        _page + 1, _ARCHIVE_MAX_PAGES)

            # `since` means on/after, so the boundary page that straddles the date is filtered.
            all_posts.extend(post for post in batch
                             if since is None or not older_than(post, since))
            offset += len(batch)     # advance by what arrived (unfiltered): the API under-fills

            # Stop from the raw page's oldest post, independently of the filtered records kept.
            if progress == "finish":
                break
            if progress == "limit":
                raise SubstackListingError(
                    f"archive listing exceeded {_ARCHIVE_MAX_PAGES} pages at offset={offset} — "
                    f"the endpoint never returned an empty page"
                )
            time.sleep(FETCH_DELAY)
    finally:
        session.close()

    return all_posts


def is_paywalled(post: dict) -> bool:
    """A paid post whose archive-list body is a preview, not the full text."""
    return post.get("audience") == "only_paid"


def fetch_full_post(base: str, slug: str, cookies: dict) -> dict | None:
    """Fetch one post's full body from `/api/v1/posts/<slug>` on the publication's own host. A
    subscriber session (`cookies`) turns a paid post's preview into the whole text; `cookies={}`
    returns full text for `audience` `everyone`.

    Returns the post dict, or None when there is no slug. Raises `SubstackFetchError` when the
    request failed. Retried, because this endpoint 403s intermittently."""
    if not slug:
        return None
    # The session goes only to a host that honors it (`session_cookie_applies`): a custom domain
    # ignores it, so sending it there would hand the subscriber's session to a third party.
    if not substack_http.session_cookie_applies(base):
        cookies = {}
    try:
        return substack_http.authed_get_json_retry(f"{base.rstrip('/')}/api/v1/posts/{slug}", {},
                                                   cookies, referer=base.rstrip("/") + "/")
    except Exception as e:
        log(f"    [warn] full-body fetch failed for {slug!r}: {e}")
        raise SubstackFetchError(f"full-body fetch failed for {slug!r}: {e}") from e


def publication_url(pub: dict) -> str | None:
    """Canonical base URL for a publication object: its custom domain if it has one,
    else the {subdomain}.substack.com host."""
    if not isinstance(pub, dict):
        return None
    if pub.get("custom_domain"):
        return f"https://{pub['custom_domain']}"
    if pub.get("subdomain"):
        return f"https://{pub['subdomain']}.substack.com"
    return None


# ── An Oracle's recommendations (public — no session, and it must not spend one) ─

def fetch_publication_id(publication_url: str) -> int:
    """A publication's numeric id, read from the newest post in its public archive (one anonymous
    request, works for custom domains too). Not cached: the caller's volume does not pay for it.

    Raises `SubstackPublicReadError` when the archive is refused or holds no post; both mean this
    pass cannot read this Oracle.
    """
    data = substack_http.public_get_json(f"{publication_url.rstrip('/')}/api/v1/archive",
                            {"sort": "new", "limit": 1})
    posts = archive_posts(data)
    if not posts or not isinstance(posts[0], dict):
        raise SubstackPublicReadError(f"archive at {publication_url} carried no post record")
    pub_id = posts[0].get("publication_id")
    if not isinstance(pub_id, int):
        raise SubstackPublicReadError(f"archive at {publication_url} carried no publication_id")
    return pub_id


def parse_recommendations(data) -> list[dict]:
    """One `recommendations/from/{id}` response → [{url, name, bio, payments_state}], the one
    reader of that payload's shape.

    The response is a bare JSON array; each item nests the publication under
    `recommendedPublication`. An item with no URL is skipped: the URL is the entity key. The
    author handle is not carried, so a publication lands on the same URL-keyed entity; `bio` is,
    because `screen._classify_prompt` reads it.
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
        purl = publication_url(pub)
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

    The route has no pagination, so a response at exactly `_RECOMMENDATIONS_LIMIT` is logged as
    possibly truncated. Raises `SubstackPublicReadError` on a refusal or a non-array body: an
    empty recommendation list and a refused read are different facts.
    """
    data = substack_http.public_get_json(
        f"{_RECOMMENDATIONS_ENDPOINT}/{publication_id}",
        {"limit": _RECOMMENDATIONS_LIMIT},
        referer="https://substack.com/",
    )
    if not isinstance(data, list):
        raise SubstackPublicReadError(
            f"recommendations for publication {publication_id} did not come back as a list")
    recs = parse_recommendations(data)
    if len(data) >= _RECOMMENDATIONS_LIMIT:
        log(f"  [warn] publication {publication_id} returned {len(data)} recommendations at the "
            f"{_RECOMMENDATIONS_LIMIT}-item route maximum — LIST MAY BE TRUNCATED")
    return recs
