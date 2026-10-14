"""Read one page of an account's Originals timeline (UserOriginalsTimeline), addressed by
position, and the user-timeline parsing it rests on."""

import base64
import struct
from datetime import datetime
from typing import Literal, NamedTuple

from pipeline.ingestion import x_render as xt
import pipeline.platforms.x.records as x_records
from pipeline.platforms.x.request import XClient

# ── User timelines: shared parsing (free cookie-scrape) ──────────────────────
# The page parser and the unavailable-account rule. The one timeline OPYT reads is Originals.


class XUserUnavailable(RuntimeError):
    """The account exists as an id but its timeline can't be read — suspended, deactivated,
    or protected. Raised rather than returned as `[]`, since those are opposite facts
    about a candidate (unreadable vs. "posts nothing")."""


def _user_timeline_root(data: dict) -> tuple[dict, str | None]:
    """(the timeline object, an unavailability reason). X ships this under both
    `timeline_v2.timeline` and `timeline.timeline`; both are read, since a rotation
    between them would otherwise look like an empty (not unreadable) timeline. The
    reason is non-None when the user node itself reports `UserUnavailable`."""
    user = ((data.get("data") or {}).get("user") or {}).get("result") or {}
    if user.get("__typename") == "UserUnavailable":
        return {}, (user.get("reason") or user.get("message") or "UserUnavailable")
    tl = (user.get("timeline_v2") or user.get("timeline") or {})
    return (tl.get("timeline") or {}), None


def _parse_user_timeline(timeline: dict) -> tuple[list[dict], str | None, str | None]:
    """One user-timeline page -> (raw tweet `result` nodes in order, bottom cursor, pinned id).

    Three entry shapes, all needed or real posts get dropped:
      - `tweet-...` — a standalone post.
      - `profile-conversation-...` / `profile-originals-conversation-...` — a self-thread,
        delivered as a module of items.
      - `TimelinePinEntry` — the pinned post, arrives as a single `entry`, not in
        an `entries` list. Kept even though it may be older than the rest of the page.

    The pinned tweet's id is returned beside the results because its date is arbitrary: X
    re-delivers it on every page, so a walk measuring how far back it reached must not count it.
    None when the page carries no pin.
    """
    results: list[dict] = []
    next_cursor: str | None = None
    pinned_id: str | None = None

    def _take(entry: dict) -> None:
        nonlocal next_cursor
        eid = entry.get("entryId", "") or ""
        content = entry.get("content", {}) or {}
        if content.get("cursorType") == "Bottom":
            next_cursor = content.get("value")
            return
        if eid.startswith("tweet-") or eid.startswith("profile-tweet-"):
            res = ((content.get("itemContent") or {}).get("tweet_results") or {}).get("result")
            if res:
                results.append(res)
            return
        if eid.startswith("profile-conversation-") or content.get("items"):
            for it in content.get("items") or []:
                ic = ((it.get("item") or {}).get("itemContent") or {})
                res = (ic.get("tweet_results") or {}).get("result")
                if res:
                    results.append(res)

    for ins in timeline.get("instructions", []) or []:
        for e in ins.get("entries", []) or []:
            _take(e)
        pinned = ins.get("entry")            # TimelinePinEntry carries one entry, not a list
        if isinstance(pinned, dict):
            n = len(results)
            _take(pinned)
            if len(results) > n:
                pinned_id = (x_records.normalize_tweet(results[-1]) or {}).get("id") or None
    return results, next_cursor, pinned_id


# ── Originals (UserOriginalsTimeline; free cookie-scrape) ──────────────────────
# One account's own posts and self-threads — no retweets, no replies to other people — addressable
# by position: a cursor built below a post starts the read there (`originals_cursor_below`). Its
# own bucket: 250 requests per 15 minutes.
ORIGINALS_OP = "UserOriginalsTimeline"
DEFAULT_ORIGINALS_QID = ""            # empty → discovery, like every other op (it is in an entry bundle)
_ORIGINALS_PAGE_SIZE = 20
_X_EPOCH_MS = 1288834974657           # X ids are (ms since this instant) << 22


def _timeline_cursor(sort_key: int, tweet_id: str) -> str:
    """A user-timeline bottom cursor: the Thrift struct X ships, URL-safe base64, unpadded.

    In a cursor X hands back, `tweet_id` is the page's oldest tweet id minus one (where the next
    page starts) and `sort_key` is the timeline's own entry sort index — the last one the page
    used, an id-shaped number that is not a tweet time. The trailing i32 is 2 for a bottom cursor
    (1 for a top one). Pinned byte for byte against two real cursors in the tests."""
    tid = str(tweet_id).encode()
    raw = (b"\x0c\x00\x07"                                           # struct, field 7
           b"\x0a\x00\x01" + struct.pack(">q", int(sort_key)) +      #   i64 sort key
           b"\x0b\x00\x02" + struct.pack(">i", len(tid)) + tid +     #   string tweet id
           b"\x08\x00\x03" + struct.pack(">i", 2) +                  #   i32 2 (bottom)
           b"\x00\x00")                                              # end inner, end outer
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def id_at(at: datetime) -> int:
    """The smallest post id X could mint at instant `at`: ids are (ms − epoch) << 22, so ids
    order by time. `at` must be timezone-aware."""
    return (int(at.timestamp() * 1000) - _X_EPOCH_MS) << 22


def originals_cursor_below(tweet_id: str) -> str:
    """A cursor that starts the Originals timeline just below one post: the next page holds only
    posts older than `tweet_id`. Built the way X builds its own bottom cursors (the oldest id read,
    minus one, in both id-shaped fields), so a walk resumes exactly after the last post it read.

    Resuming by id rather than by date is what a backward walk needs: a date cursor at the oldest
    post's second, plus a safety overlap, re-reads the posts inside that overlap, and an account
    that posts 20 or more times inside it never gets past them. `id_at` builds the same position
    for an instant: X ids are time-ordered, so `originals_cursor_below(str(id_at(at) + 1))` starts
    the timeline at `at`."""
    tid = int(tweet_id) - 1
    return _timeline_cursor(tid, str(tid))


class OriginalsWalk(NamedTuple):
    """What one `fetch_user_originals` call got.

    `tweets`   — every tweet the page held, normalized, this account's only, in the order X sent
                 them. The pinned tweet is in here once.
    `finished` — the timeline has nothing older: the page brought nothing but the pin, every
                 dated post on it is older than `after`, or X handed back no cursor.
    `pinned`   — the pinned tweet, normalized, when the account has one. Named because Originals
                 ships it only in its `TimelinePinEntry`, never in its chronological slot, so a
                 thread rooted at the pin never shows the self-reply that would mark it as a
                 thread (`ingest_x_footprint.threads_owed`).
    """
    tweets: list[dict]
    finished: bool
    pinned: dict | None = None


class OriginalsPage(NamedTuple):
    """Pure classification and accepted records from one raw Originals response."""
    action: Literal["accepted", "unavailable", "unrecognized"]
    new: list[dict]
    cursor: str | None
    pinned_id: str | None
    pinned: dict | None
    refusal: str | None


def originals_page(data: dict, user_id: str) -> OriginalsPage:
    """Classify one response and derive its own-author records and pin evidence."""
    root, unavailable = _user_timeline_root(data)
    if unavailable:
        return OriginalsPage("unavailable", [], None, None, None,
                             f"x:user:{user_id} timeline unreadable: {unavailable}")
    if not isinstance(root.get("instructions"), list):
        return OriginalsPage(
            "unrecognized", [], None, None, None,
            f"x:user:{user_id}: {ORIGINALS_OP} answered with no timeline "
            f"(errors: {str((data or {}).get('errors'))[:300]})",
        )

    raw_records, cursor, pinned_id = _parse_user_timeline(root)
    seen: set[str] = set()
    new: list[dict] = []
    next_pinned = None
    for result in raw_records:
        record = x_records.normalize_tweet(result)
        if not record or not record.get("id"):
            continue
        if record["id"] == pinned_id and next_pinned is None:
            next_pinned = record
        if record["id"] in seen:
            continue
        # Originals measured single-author; retain the filter as a guard against a changed shape.
        if str((record.get("author") or {}).get("id") or "") != str(user_id):
            continue
        seen.add(record["id"])
        new.append(record)
    return OriginalsPage("accepted", new, cursor, pinned_id, next_pinned, None)


def originals_finished(fresh_dates: list[float | None], floor: float,
                       cursor: str | None) -> bool:
    """Whether one Originals page shows the timeline has nothing older to read."""
    if not fresh_dates:
        return True
    if all(stamp is not None and stamp < floor for stamp in fresh_dates):
        return True
    return not cursor


def fetch_user_originals(client: XClient, user_id: str, *, after: datetime,
                         below_id: str | None = None) -> OriginalsWalk:
    """One page of an account's originals: from the top, or just below the post `below_id`
    (`originals_cursor_below`). Pages are time-bounded, so a page whose posts are all older than
    `after` is the end of what was asked for.

    The pinned tweet is re-delivered on every page; it is excluded from the `finished` decision,
    since its date says nothing about where the page sits, and kept as content.

    Raises `XUserUnavailable` for a suspended/deactivated/protected account, and
    `XUnrecognizedPage` for a 200 that carries no timeline."""
    qid = client.resolve_query_id(ORIGINALS_OP,
                                  default_seed=DEFAULT_ORIGINALS_QID,
                                  env_var="X_ORIGINALS_QUERY_ID",
                                  page_url="https://x.com/home")
    variables = {"userId": str(user_id), "count": _ORIGINALS_PAGE_SIZE,
                 "includePromotedContent": False, "withVoice": True,
                 "withV2Timeline": True, "withQuickPromoteEligibilityTweetFields": True}
    if below_id is not None:
        variables["cursor"] = originals_cursor_below(below_id)
    # `withArticleRichContentState` is required: without it an Article is a teaser and the body
    # goes missing silently. `withArticlePlainText` is a shape `x_render._article_shape` does not
    # read.
    data = client.graphql_get(ORIGINALS_OP, qid, variables, x_records.TIMELINE_FEATURES,
                              field_toggles={"withArticleRichContentState": True,
                                      "withArticlePlainText": False},
                       tolerate_errors=True)
    page = originals_page(data, user_id)
    if page.action == "unavailable":
        raise XUserUnavailable(page.refusal or "X user timeline unavailable")
    if page.action == "unrecognized":
        raise x_records.XUnrecognizedPage(
            page.refusal or f"x:user:{user_id}: {ORIGINALS_OP} answered with no timeline")

    def _stamp(t: dict) -> float | None:
        d = xt.parse_twitter_date(t.get("createdAt", ""))
        return d.timestamp() if d else None

    fresh = [t for t in page.new if t["id"] != page.pinned_id]
    finished = originals_finished([_stamp(t) for t in fresh], after.timestamp(), page.cursor)
    return OriginalsWalk(page.new, finished, page.pinned)
