"""Search X's posts the way its search box does (SearchTimeline), on the user's own session.

One request returns one page of posts for a raw query in X's own search syntax (quoted phrases,
`lang:`, `since:`, `-filter:replies`); what to ask is the caller's. SearchTimeline is a POST, sent
on the local session, so search runs only when this install is signed into X. Measured: its own
bucket, ~187 requests per 15 minutes.
"""

from urllib.parse import quote

import pipeline.platforms.x.records as x_records
from pipeline.platforms.collection_page import entries, timeline
from pipeline.platforms.x import local as x_local
from pipeline.platforms.x.contracts import HOME_URL
from pipeline.platforms.x.request import XClient

SEARCH_OP = "SearchTimeline"
# Empty: discovery finds it in the main bundle (`irezafrnzav4Ulf1vDkEBg` at one time). A baked
# id would survive X's rotation as a permanent 404.
DEFAULT_SEARCH_QID = ""
PAGE_SIZE = 20


def available() -> bool:
    """Whether this install can search X now: a managed session signed into X."""
    return x_local.has_session()


def referer(raw_query: str) -> str:
    """The page a browser is on when it sends this search: the honest referer for its headers."""
    return f"https://x.com/search?q={quote(raw_query)}&src=typed_query"


def search_posts(client: XClient, raw_query: str, *, product: str = "Top",
                 count: int = PAGE_SIZE) -> list[dict]:
    """One page of posts for `raw_query`, normalized (`records.normalize_tweet`), in X's order.
    `product` is the search tab: "Top" (X's ranking) or "Latest" (newest first)."""
    qid = client.resolve_query_id(SEARCH_OP, default_seed=DEFAULT_SEARCH_QID,
                                  env_var="X_SEARCHTIMELINE_QUERY_ID", page_url=HOME_URL)
    data = client.graphql_post(SEARCH_OP, qid,
                               {"rawQuery": raw_query, "count": count,
                                "querySource": "typed_query", "product": product},
                               x_records.TIMELINE_FEATURES, tolerate_errors=True)
    return parse_posts(data)


def parse_posts(data: dict) -> list[dict]:
    """The posts on one search page. A response without the search timeline raises
    (`collection_page.Incomplete`) rather than reading as a quiet day: an empty page moves a
    watch's window past posts it never saw.

    Each post is an item entry holding `tweet_results`; the page's other entries (its cursors, a
    "relevance prompt") hold none."""
    page = timeline(data, "data", "search_by_raw_query", "search_timeline", "timeline")
    out = []
    for entry in entries(page):
        item = entry["content"].get("itemContent") or {}
        result = (item.get("tweet_results") or {}).get("result")
        tweet = x_records.normalize_tweet(result) if result else None
        if tweet and tweet["id"]:
            out.append(tweet)
    return out
