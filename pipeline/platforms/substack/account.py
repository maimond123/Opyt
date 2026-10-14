"""What Substack's account reads mean: the follow, subscription and saved-list payloads, parsed
once for both transports, and one page of each as `collection_page.Page`."""

import hashlib
import json

import pipeline.platforms.collection_page as collection_page
from pipeline.platforms.substack import publication as substack_publication
from pipeline.platforms.substack.contracts import (
    FollowSource,
    SavedSource,
    SubscriptionSource,
    SubstackListingError,
)


def preload_user(decoded_shape) -> object | None:
    """The user id from the reader page's decoded preloads; None when no user is signed in.
    Raises `SubstackListingError` when the preloads, or the user in them, are not JSON objects:
    a page in an unknown shape is unavailable, not signed out."""
    if not isinstance(decoded_shape, dict):
        raise SubstackListingError("reader page preloads are not a JSON object")
    user = decoded_shape.get("user")
    if not user:
        return None
    if not isinstance(user, dict):
        raise SubstackListingError("reader page preloads carry a user that is not a JSON object")
    return user.get("id")


def parse_subscriber_lists(data) -> list[dict]:
    """`subscriber-lists` JSON → [{name, url}], the one reader of that payload's shape for both
    transports.

    The shape is subscriberLists[].groups[].users[], each user carrying a `primary_publication`;
    a person you follow who has no publication is skipped.
    """
    pubs = []
    for lst in (data.get("subscriberLists") or []) if isinstance(data, dict) else []:
        for group in lst.get("groups") or []:
            for user in group.get("users") or []:
                purl = substack_publication.publication_url(user.get("primary_publication") or {})
                if not purl:
                    continue
                name = (user.get("primary_publication") or {}).get("name") or user.get("name") or ""
                pubs.append({"name": name, "url": purl})
    return pubs


# ── The user's own subscriptions — a different graph, a different endpoint ─────

def parse_subscription_page(data) -> list[dict]:
    """One `subscriptions/page_v2` page → [{name, url, membership_state, is_favorite}], the one
    reader of that payload's shape.

    The join is `subscriptions[].publication_id` → `publications[].id`. A subscription whose
    publication is absent from the join is skipped: the URL is the entity key."""
    if not isinstance(data, dict):
        return []
    pubs = {p["id"]: p for p in (data.get("publications") or [])
            if isinstance(p, dict) and p.get("id") is not None}
    out = []
    for sub in data.get("subscriptions") or []:
        if not isinstance(sub, dict):
            continue
        pub = pubs.get(sub.get("publication_id"))
        purl = substack_publication.publication_url(pub or {})
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


# ── One page of an account list, for a durable walk ────────────────────────────
#
# A durable walk (`pipeline/curation/walks.py`) reads one page per unit, and these say what each
# page holds and whether it ends the list, as `collection_page.Page`. A refused page raises
# `SubstackListingError`, never an empty page, for the reason every account read raises. A
# response without the list it should carry is `Incomplete`: it cannot end the walk.

def _listing(data, key: str) -> list:
    if not isinstance(data, dict) or not isinstance(data.get(key), list):
        raise collection_page.Incomplete(f"Substack listing response lacks {key}")
    return data[key]


def _next(data: dict) -> str | None:
    cursor = data.get("nextCursor")
    if cursor is not None and (not isinstance(cursor, str) or not cursor):
        raise collection_page.Incomplete("invalid Substack listing cursor")
    return cursor


def _saved_item_identity(item) -> str:
    """Every saved item occupies a place in the list, post or not. A post is its id; a note or
    comment, whose identity has never been measured, is its own content."""
    post = item.get("post") if isinstance(item, dict) else None
    if isinstance(post, dict) and post.get("id"):
        return str(post["id"])
    body = json.dumps(item, sort_keys=True, default=str)
    kind = item.get("type") if isinstance(item, dict) else None
    return f"{kind or 'item'}:{hashlib.sha256(body.encode()).hexdigest()[:16]}"


def saved_page(source: SavedSource, cursor: str | None = None) -> collection_page.Page:
    """One page of the user's saved posts (`source` from `saved_source()`). Notes and comments
    are observed but not records. No `nextCursor` ends the list."""
    data = source.page(cursor)
    if data is None:
        raise SubstackListingError("saved-posts page refused")
    items = _listing(data, "items")
    records = tuple((str(rec["id"]), rec) for rec in
                    (saved_item_to_record(it) for it in items if isinstance(it, dict))
                    if rec is not None)
    following = _next(data)
    return collection_page.Page(records, following,
                                tuple(_saved_item_identity(it) for it in items),
                                ended=following is None)


def subscription_page(source: SubscriptionSource,
                      cursor: str | None = None) -> collection_page.Page:
    """One page of the publications the user subscribes to (`source` from
    `subscription_list_source()`). A subscription whose publication is missing from the join is
    observed but not a record. No `nextCursor` ends the list: the only end ever measured."""
    data = source.page(cursor)
    if data is None:
        raise SubstackListingError("subscription page refused")
    rows = _listing(data, "subscriptions")
    records = tuple((str(rec.get("id") or rec["url"]), rec)
                    for rec in parse_subscription_page(data))
    observed = tuple(str(row.get("id") or row.get("publication_id") or "")
                     for row in rows if isinstance(row, dict))
    following = _next(data)
    return collection_page.Page(records, following, observed, ended=following is None)


def follows_page(source: FollowSource, user_id: int) -> collection_page.Page:
    """Everyone the account follows (`source` from `follow_source()`): one read, so one page
    that ends the list."""
    people = source.follows(user_id)
    records = tuple((rec["url"], rec) for rec in people)
    return collection_page.Page(records, None, tuple(key for key, _ in records), ended=True)


# ── Saved posts (post-level curation — the X-bookmarks analog) ────────────────

def saved_item_to_record(item: dict) -> dict | None:
    """Map one saved-list item to the trimmed field record we keep (the payload is ~95%
    noise — pricing tables, palettes, i18n — all dropped). Returns None for non-posts
    (notes/comments) or an item with no stable id, so the caller just skips it.

    `publication` sits at the item level (not under `post`); it's `post.wordcount` (no
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
        "publication_url": pub.get("base_url") or substack_publication.publication_url(pub) or "",
        "author_name": pub.get("author_name") or "",
        "author_handle": pub.get("author_handle") or "",
    }
