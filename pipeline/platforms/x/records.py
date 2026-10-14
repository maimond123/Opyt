"""X record normalization (tweets and user cards), the unrecognized-page refusal, and the timeline
feature bundle; no transport dependencies."""

from pipeline.platforms.collection_page import entries


class XUnrecognizedPage(RuntimeError):
    """A 200 that carries no timeline its parser recognizes: an errors-only body, or X renaming
    the keys the parser reads. Raised rather than read as an empty page, because an
    empty page ends a walk and a finished walk claims coverage nothing revisits. A real end of a
    timeline still carries its `instructions`."""


TIMELINE_FEATURES = {
    "rweb_video_screen_enabled": False,
    "rweb_cashtags_enabled": True,
    "profile_label_improvements_pcf_label_in_post_enabled": True,
    "responsive_web_profile_redirect_enabled": False,
    "rweb_tipjar_consumption_enabled": False,
    "verified_phone_label_enabled": False,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "premium_content_api_read_enabled": False,
    "communities_web_enable_tweet_community_results_fetch": True,
    "c9s_tweet_anatomy_moderator_badge_enabled": True,
    "responsive_web_grok_analyze_button_fetch_trends_enabled": False,
    "responsive_web_grok_analyze_post_followups_enabled": True,
    "rweb_cashtags_composer_attachment_enabled": True,
    "responsive_web_jetfuel_frame": True,
    "responsive_web_grok_share_attachment_enabled": True,
    "responsive_web_grok_annotations_enabled": True,
    "articles_preview_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "rweb_conversational_replies_downvote_enabled": False,
    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
    "responsive_web_twitter_article_tweet_consumption_enabled": True,
    "content_disclosure_indicator_enabled": True,
    "content_disclosure_ai_generated_indicator_enabled": True,
    "responsive_web_grok_show_grok_translated_post": True,
    "responsive_web_grok_analysis_button_from_backend": True,
    "post_ctas_fetch_enabled": False,
    "freedom_of_speech_not_reach_fetch_enabled": True,
    "standardized_nudges_misinfo": True,
    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
    "longform_notetweets_rich_text_read_enabled": True,
    # Gates the media embedded inside a longform note tweet; on, to match BOOKMARKS_FEATURES.
    "longform_notetweets_inline_media_enabled": True,
    "responsive_web_grok_image_annotation_enabled": True,
    "responsive_web_grok_imagine_annotation_enabled": True,
    "responsive_web_grok_community_note_auto_translation_is_enabled": True,
    "responsive_web_enhance_cards_enabled": False,
}


def normalize_user(result: dict) -> dict | None:
    """Map an X User `result` → the candidate-signal fields. `site` (the bio's outbound
    URL) is the attested-link seed the cross-platform entity-resolution step needs."""
    if not result or result.get("__typename") != "User":
        return None
    rest_id = result.get("rest_id") or ""
    if not rest_id:
        return None
    core_ = result.get("core") or {}
    legacy = result.get("legacy") or {}
    handle = core_.get("screen_name") or legacy.get("screen_name") or ""
    name = core_.get("name") or legacy.get("name") or handle
    bio = ((result.get("profile_bio") or {}).get("description")
           or legacy.get("description") or "")
    # X split the flat `legacy` blob into typed sub-objects; read the modern path first and
    # keep `legacy` only as fallback, or `site`/followers_count silently go ""/0.
    urls = ((((result.get("profile_bio") or {}).get("entities") or {}).get("url") or {})
            .get("urls") or (((legacy.get("entities") or {}).get("url") or {}).get("urls"))
            or [])
    site = urls[0].get("expanded_url", "") if urls else ""
    followers = ((result.get("relationship_counts") or {}).get("followers")
                 if (result.get("relationship_counts") or {}).get("followers") is not None
                 else legacy.get("followers_count", 0))
    following = (result.get("relationship_perspectives") or {}).get("following", False)
    verified = bool(result.get("is_blue_verified")
                    or (result.get("verification") or {}).get("verified"))
    return {
        "user_id": rest_id,
        "handle": handle,
        "display_name": name,
        "bio": bio,
        "site": site,
        "followers_count": followers or 0,
        "verified": verified,
        "i_follow": bool(following),
    }


def parse_members(timeline: dict) -> list[dict]:
    """The user cards on one page; the page's cursor is `collection_page`'s."""
    out: list[dict] = []
    for e in entries(timeline):
        entry_id = e.get("entryId", "")
        content = e.get("content", {}) or {}
        if entry_id.startswith("user-"):
            result = (((content.get("itemContent") or {}).get("user_results") or {})
                      .get("result"))
            norm = normalize_user(result)
            if norm:
                out.append(norm)
    return out


def normalize_tweet(result: dict) -> dict | None:
    """Map an X GraphQL tweet `result` to the normalized tweet dict every renderer and
    consumer in this repo reads. Media/quote paths are best-effort."""
    if not result:
        return None
    if result.get("__typename") == "TweetWithVisibilityResults":
        result = result.get("tweet", {}) or {}
    legacy = result.get("legacy") or {}
    if not legacy:
        return None

    user_result = (((result.get("core") or {}).get("user_results") or {})
                   .get("result") or {})
    u_legacy = user_result.get("legacy") or {}
    u_core = user_result.get("core") or {}
    # screen_name/name/id_str live in `core`/`rest_id`, not `u_legacy`. Read that one
    # path; if X moves it again, `username` goes visibly "unknown" rather than silently
    # falling through a dead fallback.
    username = u_core.get("screen_name") or "unknown"
    name = u_core.get("name") or username
    user_id = user_result.get("rest_id") or ""
    # Author's bio outbound URL — the cross-platform identity seed Stage-3 joins on.
    # `legacy` is the only home of it on this surface; no fallback to `profile_bio.entities`
    # (that path doesn't exist here — see the Lists/Following surface for contrast).
    # Absent → "".
    u_urls = (((u_legacy.get("entities") or {}).get("url") or {}).get("urls") or [])
    u_site = u_urls[0].get("expanded_url", "") if u_urls else ""

    tweet_id = legacy.get("id_str") or result.get("rest_id") or ""
    note = (((result.get("note_tweet") or {}).get("note_tweet_results") or {})
            .get("result") or {})
    text = note.get("text") or legacy.get("full_text", "")
    # A long post's full text carries its own `t.co` codes and link list
    # (`note_tweet.entity_set.urls`), different from the 280-character version's, so use that list
    # with the full text. Media stays the short version's: the full text has no media list. Its
    # mentions come from the same list, since a name past the 280th character is only there.
    entities = legacy.get("entities", {}) or {}
    note_set = (note.get("entity_set") or {}) if note.get("text") else {}
    for key in ("urls", "user_mentions"):
        if note_set.get(key) is not None:
            entities = {**entities, key: note_set[key]}

    norm = {
        "id": tweet_id,
        "author": {"userName": username, "name": name, "id": user_id, "site": u_site},
        "text": text,
        "createdAt": legacy.get("created_at", ""),
        "likeCount": legacy.get("favorite_count", 0),
        "entities": entities,
        # Must stay camelCase: _render_media reads `extendedEntities`.
        "extendedEntities": legacy.get("extended_entities", {}),
        # Link card passthrough: X's result.card.legacy.binding_values already matches
        # the {key, value:{string_value}} shape _parse_card reads. Absent card → no key
        # → _render_link_cards falls back to the bare entities.urls link.
        "card": {"binding_values": (((result.get("card") or {}).get("legacy") or {})
                                    .get("binding_values") or [])},
        "url": f"https://x.com/{username}/status/{tweet_id}",
        "isQuote": legacy.get("is_quote_status", False),
        "conversationId": legacy.get("conversation_id_str", tweet_id),
        "isReply": bool(legacy.get("in_reply_to_status_id_str")),
        "replyCount": legacy.get("reply_count", 0),
        # The two fields ingest_x_footprint._filter_and_stitch decides on (drop RTs +
        # replies-to-others, keep originals + self-threads). `isRetweet` reads
        # `legacy.retweeted_status_result` (not the forgeable "RT @" text prefix);
        # `inReplyToUserId` reads the numeric `legacy.in_reply_to_user_id_str`, not the
        # handle field (unreliable).
        "isRetweet": bool(legacy.get("retweeted_status_result")),
        "inReplyToUserId": legacy.get("in_reply_to_user_id_str") or "",
        # The parent tweet. Read by the resumable thread walk, to tell the end of an author's
        # chain (a self-reply nothing of theirs replies to) from its middle.
        "inReplyToStatusId": legacy.get("in_reply_to_status_id_str") or "",
    }
    # X-Article body: with the article toggles on, X nests the full body at
    # result.article.article_results.result.content_state.blocks. Carry the whole
    # `article` node; `_render_article` digs out title + blocks. Absent → normal post.
    article = result.get("article")
    if article:
        norm["article"] = article

    quoted = (result.get("quoted_status_result") or {}).get("result")
    if quoted:
        norm["quoted_tweet"] = normalize_tweet(quoted)
    return norm
