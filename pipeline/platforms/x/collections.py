"""X's resumable account collections: Bookmarks, Following, Likes and the user's owned Lists.

One request per page, each returned as a `collection_page.Page`; the caller's walk (curation)
decides continuation. Every read takes the one `XClient` (`pipeline.platforms.sessions.x_session`).

Lists are owned-only: a list you created groups accounts by your own topic — the strongest "who to
fully track" signal short of subscribing. Lists you merely follow or X's suggested lists are
someone else's curation, so they're skipped. `ListsManagementPageTimeline` lists them (keeping only
lists whose owner rest_id is the viewer; the `ListToFollow` suggested module is ignored), and
`ListMembers(listId)` reads one list's accounts.

Likes is a plain tweet timeline (like Bookmarks): each unique liked tweet's author is the signal.
Response root `data.user.result.timeline.timeline` (not `timeline_v2`), tweet entries at
`content.itemContent.tweet_results.result`.
"""

import json
import os

import pipeline.platforms.collection_page as collection_page
import pipeline.platforms.x.records as x_records
from pipeline.platforms.x.contracts import HOME_URL
from pipeline.platforms.x.request import XClient

DEFAULT_PAGE_SIZE = 100
_DISCOVER_PAGE = HOME_URL   # any authed page harvests the shared JS bundles


# ── Bookmarks ─────────────────────────────────────────────────────────────────

BOOKMARKS_OP = "Bookmarks"

# Feature switches the Bookmarks query declares, lifted verbatim from X's bundle. X 400s
# if a required key is missing, so mirror the declared set exactly. Override via
# $X_BOOKMARKS_FEATURES / $X_BOOKMARKS_FIELD_TOGGLES (JSON) if X rotates them.
BOOKMARKS_FEATURES = {
    "rweb_video_screen_enabled": True,
    "rweb_cashtags_enabled": True,
    "profile_label_improvements_pcf_label_in_post_enabled": True,
    "responsive_web_profile_redirect_enabled": True,
    "rweb_tipjar_consumption_enabled": True,
    "verified_phone_label_enabled": True,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": True,
    "premium_content_api_read_enabled": True,
    "communities_web_enable_tweet_community_results_fetch": True,
    "c9s_tweet_anatomy_moderator_badge_enabled": True,
    "responsive_web_grok_analyze_button_fetch_trends_enabled": True,
    "responsive_web_grok_analyze_post_followups_enabled": True,
    "rweb_cashtags_composer_attachment_enabled": True,
    "responsive_web_jetfuel_frame": True,
    "responsive_web_grok_share_attachment_enabled": True,
    "responsive_web_grok_annotations_enabled": True,
    "articles_preview_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "rweb_conversational_replies_downvote_enabled": True,
    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
    "responsive_web_twitter_article_tweet_consumption_enabled": True,
    "content_disclosure_indicator_enabled": True,
    "content_disclosure_ai_generated_indicator_enabled": True,
    "responsive_web_grok_show_grok_translated_post": True,
    "responsive_web_grok_analysis_button_from_backend": True,
    "post_ctas_fetch_enabled": True,
    "freedom_of_speech_not_reach_fetch_enabled": True,
    "standardized_nudges_misinfo": True,
    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
    "longform_notetweets_rich_text_read_enabled": True,
    "longform_notetweets_inline_media_enabled": True,
    "responsive_web_grok_image_annotation_enabled": True,
    "responsive_web_grok_imagine_annotation_enabled": True,
    "responsive_web_grok_community_note_auto_translation_is_enabled": True,
    "responsive_web_enhance_cards_enabled": True,
}

# Article toggles are on: this makes the same free Bookmarks call return full X-Article
# bodies instead of a teaser stub.
BOOKMARKS_FIELD_TOGGLES = {
    "withPayments": False,
    "withAuxiliaryUserLabels": False,
    "withArticleRichContentState": True,
    "withArticlePlainText": True,
    "withArticleSummaryText": True,
    "withArticleVoiceOver": False,
    "withGrokAnalyze": False,
    "withDisallowedReplyControls": False,
}


def bookmark_page(client: XClient, cursor: str | None = None,
                  page_size: int = DEFAULT_PAGE_SIZE) -> collection_page.Page:
    """Read one Bookmarks page through the managed X client."""
    query_id = client.resolve_query_id(BOOKMARKS_OP, env_var="X_BOOKMARKS_QUERY_ID",
                                       page_url="https://x.com/i/bookmarks")
    variables = {"count": page_size, "includePromotedContent": False}
    if cursor:
        variables["cursor"] = cursor
    features = os.getenv("X_BOOKMARKS_FEATURES")
    features = json.loads(features) if features else BOOKMARKS_FEATURES
    toggles = os.getenv("X_BOOKMARKS_FIELD_TOGGLES")
    toggles = json.loads(toggles) if toggles else BOOKMARKS_FIELD_TOGGLES
    data = client.graphql_get(BOOKMARKS_OP, query_id, variables, features,
                              field_toggles=toggles)
    timeline = collection_page.timeline(data, "data", "bookmark_timeline_v2", "timeline")
    records = []
    for entry in collection_page.entries(timeline):
        if not entry.get("entryId", "").startswith("tweet-"):
            continue
        result = ((entry["content"].get("itemContent") or {})
                  .get("tweet_results") or {}).get("result")
        norm = x_records.normalize_tweet(result)
        if norm and norm.get("id"):
            records.append((norm["id"], norm))
    return collection_page.from_timeline(timeline, tuple(records))


# ── Following timeline (person-endorsement signal; free cookie-scrape) ─────────
# Sibling of the Likes read below: a user-scoped timeline whose entries are user
# cards, reusing x_records.normalize_user + parse_members.
FOLLOWING_OP = "Following"
# Empty on purpose: a baked seed 404s forever once X rotates it. Empty routes to
# env -> cache -> live JS-bundle discovery, which finds + caches the real id.
DEFAULT_FOLLOWING_QID = ""


def following_page(client: XClient, viewer_id: str,
                   cursor: str | None = None) -> collection_page.Page:
    """One Following request, including evidence from unavailable user cards."""
    qid = client.resolve_query_id(FOLLOWING_OP, default_seed=DEFAULT_FOLLOWING_QID,
                                  env_var="X_FOLLOWING_QUERY_ID",
                                  page_url=_DISCOVER_PAGE)
    variables = {"userId": viewer_id, "count": DEFAULT_PAGE_SIZE,
                 "includePromotedContent": False}
    if cursor:
        variables["cursor"] = cursor
    data = client.graphql_get(FOLLOWING_OP, qid, variables, x_records.TIMELINE_FEATURES,
                              tolerate_errors=True)
    timeline = collection_page.timeline(data, "data", "user", "result", "timeline", "timeline")
    members = x_records.parse_members(timeline)
    return collection_page.from_timeline(
        timeline, tuple((m["user_id"], m) for m in members), repeat_prefix="user-")


# ── Likes ─────────────────────────────────────────────────────────────────────

LIKES_OP = "Likes"

# Empty on purpose: resolve_query_id returns a non-empty seed without validating it, so a
# guessed seed would 404 forever on rotation; empty routes to live JS-bundle discovery instead.
DEFAULT_LIKES_QID = ""

# Likes shares the web client's timeline feature bundle with Lists; override via
# $X_LIKES_FEATURES (JSON) if a live Likes capture shows a different set.
LIKES_FEATURES = x_records.TIMELINE_FEATURES


def _features() -> dict:
    override = os.getenv("X_LIKES_FEATURES")
    return json.loads(override) if override else LIKES_FEATURES


def _parse_timeline(timeline: dict) -> list[dict]:
    """Pull tweet `result` objects out of a validated Likes timeline (root
    `data.user.result.timeline.timeline`). Tweet entries carry
    `entryId` starting `tweet-` with the tweet at
    `content.itemContent.tweet_results.result`. The paging cursor is `collection_page`'s."""
    tweets: list[dict] = []
    for e in collection_page.entries(timeline):
        entry_id = e.get("entryId", "")
        content = e.get("content", {}) or {}
        if entry_id.startswith("tweet-"):
            result = ((content.get("itemContent") or {})
                      .get("tweet_results") or {}).get("result")
            if result:
                tweets.append(result)
    return tweets


def _unwrap(result: dict) -> dict:
    """TweetWithVisibilityResults nests the real tweet one level down under `tweet`."""
    if result.get("__typename") == "TweetWithVisibilityResults":
        return result.get("tweet") or {}
    return result


def _tweet_id(result: dict) -> str:
    """The dedup key for the timeline unit — the tweet, not its author (two liked tweets
    by one author must count as two likes)."""
    r = _unwrap(result)
    return (r.get("legacy") or {}).get("id_str") or r.get("rest_id") or ""


def _tweet_author(result: dict) -> dict | None:
    """The candidate: the tweet's author at `result.core.user_results.result`, mapped
    through the shared candidate-signal user shape. None if the author can't be reached
    (deleted/suspended) — the caller still counts the tweet, just contributes no author."""
    r = _unwrap(result)
    user = (((r.get("core") or {}).get("user_results") or {}).get("result"))
    return x_records.normalize_user(user)


def likes_page(viewer_id: str, client: XClient,
               cursor: str | None = None) -> collection_page.Page:
    """One Likes page, keyed by liked tweet so distinct likes of one author still count."""
    qid = client.resolve_query_id(LIKES_OP, default_seed=DEFAULT_LIKES_QID,
                                  env_var="X_LIKES_QUERY_ID", page_url=_DISCOVER_PAGE)
    variables = {
        "userId": viewer_id, "count": DEFAULT_PAGE_SIZE,
        "includePromotedContent": False, "withClientEventToken": False,
        "withBirdwatchNotes": False, "withVoice": True, "withV2Timeline": True,
    }
    if cursor:
        variables["cursor"] = cursor
    data = client.graphql_get(LIKES_OP, qid, variables, _features(), tolerate_errors=True)
    timeline = collection_page.timeline(data, "data", "user", "result", "timeline", "timeline")
    tweets = _parse_timeline(timeline)
    records = tuple((tid, author) for tweet in tweets
                    if (tid := _tweet_id(tweet)) and (author := _tweet_author(tweet)))
    return collection_page.from_timeline(timeline, records, repeat_prefix="tweet-")


# ── Lists ─────────────────────────────────────────────────────────────────────

LISTS_MGMT_OP = "ListsManagementPageTimeline"
LIST_MEMBERS_OP = "ListMembers"

# No built-in query ids: a seed outranks the cache, so a rotated one could never be replaced by
# the re-discovery a 400/404 triggers. The env overrides stay authoritative.

# ── Parse: owned lists out of the management timeline ─────────────────────────

def _parse_owned_lists(timeline: dict, viewer_id: str) -> tuple[list[dict], list[str]]:
    """The viewer's owned lists, and the page's observed keys, from one walk of the timeline.

    Owned = the `OwnedSubscribedList` module's lists whose owner rest_id == the viewer; a
    subscribed list (owner != viewer) is dropped. The `ListToFollow` module is X's endless
    suggestions, which begin after owned/subscribed lists end: skipped. A module's entryId
    repeats across pages even when its lists change, so its observed keys are its list IDs,
    subscribed lists included, rather than that module ID."""
    lists: list[dict] = []
    observed: list[str] = []
    for e in collection_page.entries(timeline):
        content = e["content"]
        if content.get("cursorType"):
            continue
        if content.get("__typename") != "TimelineTimelineModule":
            observed.append(e.get("entryId", ""))
            continue
        inj = (((content.get("clientEventInfo") or {}).get("details") or {})
               .get("timelinesDetails") or {}).get("injectionType")
        if inj == "ListToFollow":
            continue
        if inj != "OwnedSubscribedList":
            raise collection_page.Incomplete("unrecognized list module")
        for it in content.get("items", []):
            lst = ((it.get("item") or {}).get("itemContent") or {}).get("list") or {}
            lid = lst.get("id_str")
            if not lid:
                raise collection_page.Incomplete("list module item lacks a list identity")
            observed.append("list-" + str(lid))
            owner = (((lst.get("user_results") or {}).get("result") or {})
                     .get("rest_id"))
            if owner == viewer_id:     # owned only
                lists.append({
                    "id": lid,
                    "name": lst.get("name", ""),
                    "mode": lst.get("mode", ""),
                    "member_count": lst.get("member_count", 0),
                })
    return lists, observed


def owned_lists_page(client: XClient, viewer_id: str,
                     cursor: str | None = None) -> collection_page.Page:
    """One list enumeration request; member requests have their own operation and quota."""
    qid = client.resolve_query_id(LISTS_MGMT_OP, env_var="X_LISTS_MGMT_QUERY_ID",
                                  page_url=_DISCOVER_PAGE)
    variables = {"count": DEFAULT_PAGE_SIZE}
    if cursor:
        variables["cursor"] = cursor
    data = client.graphql_get(LISTS_MGMT_OP, qid, variables, x_records.TIMELINE_FEATURES,
                              tolerate_errors=True)
    timeline = collection_page.timeline(data, "data", "viewer", "list_management_timeline", "timeline")
    lists, observed = _parse_owned_lists(timeline, viewer_id)
    return collection_page.from_timeline(
        timeline, tuple((lst["id"], lst) for lst in lists),
        observed=tuple(observed), repeat_prefix="list-", ended=not observed)


def list_members_page(list_id: str, client: XClient,
                      cursor: str | None = None) -> collection_page.Page:
    """One page of one list's members; normalization does not decide pagination."""
    qid = client.resolve_query_id(LIST_MEMBERS_OP, env_var="X_LIST_MEMBERS_QUERY_ID",
                                  page_url=_DISCOVER_PAGE)
    variables = {"listId": list_id, "count": DEFAULT_PAGE_SIZE}
    if cursor:
        variables["cursor"] = cursor
    data = client.graphql_get(LIST_MEMBERS_OP, qid, variables, x_records.TIMELINE_FEATURES,
                              tolerate_errors=True)
    timeline = collection_page.timeline(data, "data", "list", "members_timeline", "timeline")
    members = x_records.parse_members(timeline)
    return collection_page.from_timeline(
        timeline, tuple((m["user_id"], m) for m in members), repeat_prefix="user-")
