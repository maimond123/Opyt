"""Search Substack's posts across every publication, signed out: the site's own post search.

`includePlatformResults=true` is what makes the route answer signed out (without it, nothing), and
`dateRange` windows it on the post's date. The query matches word by word: quoting it gives exact
phrases but drops the date window (results from 2017), so a caller passes it unquoted and checks
the name itself. Results carry no body, so a post whose title does not name the query is read on
its own (`post_text`).
"""

import html
import re

from pipeline.platforms.substack import http as substack_http
from pipeline.platforms.substack.contracts import SubstackFetchError, SubstackPublicReadError
from pipeline.platforms.substack.publication import fetch_full_post, is_paywalled

SEARCH_URL = "https://substack.com/api/v1/post/search"

_POST_URL = re.compile(r"^(https?://[^/]+)/p/([^/?#]+)", re.I)
_TAG = re.compile(r"<[^>]+>")


def range_for(days: float) -> str:
    """The narrowest of the search's windows (`day`, `week`, `month`) that holds the last `days`
    days."""
    return "day" if days <= 1 else "week" if days <= 7 else "month"


def search_posts(query: str, *, date_range: str, page: int = 0) -> tuple[list[dict], bool]:
    """One page (20 at most) of the posts matching `query` within `date_range`: `(posts, more)`.
    Raises `SubstackPublicReadError` when the request is refused."""
    data = substack_http.public_get_json(
        SEARCH_URL, {"query": query, "page": page, "includePlatformResults": "true",
                     "dateRange": date_range})
    return parse_page(data)


def parse_page(data) -> tuple[list[dict], bool]:
    """A search page → `(posts, more)`, each post `{id, title, subtitle, description, url,
    published, paid, publication, author}`. A page without its results list raises rather than
    reading as a quiet week; a result with no id or address is not a post anyone can open."""
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise SubstackPublicReadError("post search answered without a results list")
    publications = {p.get("id"): p for p in data.get("publications") or [] if isinstance(p, dict)}
    posts = []
    for r in data["results"]:
        if not isinstance(r, dict) or not r.get("id") or not r.get("canonical_url"):
            continue
        bylines = [b for b in r.get("publishedBylines") or [] if isinstance(b, dict)]
        posts.append({
            "id": str(r["id"]),
            "title": r.get("title") or "",
            "subtitle": r.get("subtitle") or "",
            "description": r.get("description") or "",
            "url": r["canonical_url"],
            "published": r.get("post_date") or "",
            "paid": is_paywalled(r),
            "publication": (publications.get(r.get("publication_id")) or {}).get("name") or "",
            "author": (bylines[0].get("name") or "") if bylines else "",
        })
    return posts, bool(data.get("more"))


def post_text(url: str) -> str:
    """A post's body as plain text, read anonymously by the one post read
    (`publication.fetch_full_post`, which retries Substack's intermittent 403; a paid post's body
    is its free preview). Raises `SubstackFetchError` when every try is refused."""
    found = _POST_URL.match(url or "")
    if found is None:
        raise SubstackFetchError(f"not a post address: {url!r}")
    post = fetch_full_post(*found.groups(), {})
    body = post.get("body_html") if isinstance(post, dict) else None
    return " ".join(html.unescape(_TAG.sub(" ", body or "")).split())
