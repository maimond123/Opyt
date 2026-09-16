"""
pipeline/ingestion/x_graphql.py
Bookmarks-shaped adapter over the shared local-session X GraphQL transport. It owns
Bookmarks features/toggles and timeline parsing; the core owns session reads, headers,
query-id discovery, GraphQL GET/rate handling, and `normalize` — the tweet shape is shared
by every X operation, so it is not this adapter's to own.

X uses one OPYT-managed browser session. The core resolves the Bookmarks queryId, with
$X_BOOKMARKS_QUERY_ID as an override.
"""

import json
import os

from pipeline.ingestion.utils import log

BOOKMARKS_OP = "Bookmarks"
DEFAULT_PAGE_SIZE = 100

# Managed-session presence uses the shared cookie reader; the core owns session reads.
from pipeline.ingestion import browser_cookies as _bc  # noqa: E402

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

# Article toggles are ON: this makes the same free Bookmarks call return full X-Article
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


def has_managed_x_session() -> bool:
    """Whether an OPYT-managed browser session is currently logged into X."""
    from pipeline.ingestion import hosted_browser, hosted_x
    if hosted_browser.enabled():
        return hosted_x.has_connection()
    candidates, _ = _bc.list_opyt_logged_in(["x.com", "twitter.com"], "auth_token")
    return bool(candidates)


# ── Bookmarks timeline parse ──

def _parse_timeline(data: dict) -> tuple[list[dict], str | None]:
    timeline = (data.get("data", {})
                    .get("bookmark_timeline_v2", {})
                    .get("timeline", {}))
    entries: list[dict] = []
    for ins in timeline.get("instructions", []):
        if ins.get("type") == "TimelineAddEntries" or "entries" in ins:
            entries.extend(ins.get("entries", []))

    tweets: list[dict] = []
    next_cursor: str | None = None
    for e in entries:
        entry_id = e.get("entryId", "")
        content = e.get("content", {}) or {}
        if entry_id.startswith("tweet-"):
            result = ((content.get("itemContent") or {})
                      .get("tweet_results", {}) or {}).get("result")
            if result:
                tweets.append(result)
        elif content.get("cursorType") == "Bottom":
            next_cursor = content.get("value")
    return tweets, next_cursor


# ── Public iterator ────────────────────────────────────────────────────────────

def iterate_bookmarks(limit: int = 0, page_size: int = DEFAULT_PAGE_SIZE):
    """Yield the user's bookmarks newest-first as normalized tweet dicts. Raises
    SyncAuthError if not logged in / session dead.

    The caller then reports a broken source rather than a silent '0 bookmarks' — but only via a
    specific chain, and this sentence claimed the outcome without it from 2026-07-24 to
    2026-09-06. `run_concurrent` catches this raise to drain the already-fetched window, records
    it as `source_error`, `sync_bookmarks` turns that into `error` (plus `undetermined` for a
    rate limit), and `bookmark_catchup` maps `classify_run`'s verdict onto its own status. Break
    any link and a dead session is again indistinguishable from a quiet week."""
    from pipeline.ingestion import x_graphql_core as core

    session = core.x_session("https://x.com/i/bookmarks")
    query_id = core.resolve_query_id(
        BOOKMARKS_OP, session,
        env_var="X_BOOKMARKS_QUERY_ID", page_url="https://x.com/i/bookmarks",
    )
    log(f"[x-graphql] using Bookmarks queryId={query_id}")

    cursor: str | None = None
    seen: set[str] = set()
    yielded = 0
    page = 0
    while True:
        page += 1
        variables = {"count": page_size, "includePromotedContent": False}
        if cursor:
            variables["cursor"] = cursor
        features = os.getenv("X_BOOKMARKS_FEATURES")
        features = json.loads(features) if features else BOOKMARKS_FEATURES
        toggles = os.getenv("X_BOOKMARKS_FIELD_TOGGLES")
        toggles = json.loads(toggles) if toggles else BOOKMARKS_FIELD_TOGGLES
        data = core.graphql_get(BOOKMARKS_OP, query_id, variables, features, session,
                                field_toggles=toggles)
        results, next_cursor = _parse_timeline(data)
        log(f"[x-graphql] page {page}: {len(results)} tweets")
        if not results:
            break
        for result in results:
            norm = core.normalize(result)
            if norm and norm.get("id"):
                yield norm
                yielded += 1
                if limit and yielded >= limit:
                    return
        if not next_cursor or next_cursor in seen:
            break
        seen.add(next_cursor)
        cursor = next_cursor
