"""
pipeline/ingestion/x_lists.py
Local-session X Lists reader — the user's OWNED lists and their MEMBERS, emitted as a
person-endorsement candidate signal for oracle onboarding.

Owned-only: a list you CREATED groups accounts by your own topic — the strongest
"who to fully track" signal short of subscribing. Lists you merely FOLLOW or X's
SUGGESTED lists are someone else's curation, so they're skipped.

Two GraphQL ops (queryIds are X-wide, self-heal on drift):
  1. ListsManagementPageTimeline → your lists. Keeps only lists whose owner rest_id ==
     the viewer (`twid` cookie); the `ListToFollow` (suggested) module is ignored.
  2. ListMembers(listId) → the accounts in one list, paginated by a bottom cursor.

Output: state/candidate_signals_x_lists.json — members deduped by rest_id, with the
union of the list names/ids they appear in (cross-list membership = stronger signal).
"""

import argparse
import json
import time
from pathlib import Path

from pipeline.ingestion import x_graphql_core as core
from pipeline.ingestion.utils import log, SyncAuthError

LISTS_MGMT_OP = "ListsManagementPageTimeline"
LIST_MEMBERS_OP = "ListMembers"

# Baked X-wide seeds from the live capture (2026-07-09). Env overrides + drift-triggered
# re-discovery cover rotation.
DEFAULT_LISTS_MGMT_QID = "wgVgVkLURZzQ6flLmOyprA"
DEFAULT_LIST_MEMBERS_QID = "kcsJubZ1BIwpdKrYfiNRtg"
_DISCOVER_PAGE = "https://x.com/home"   # any authed page harvests the shared bundles
_REFERER = "https://x.com/i/lists"
DEFAULT_PAGE_SIZE = 100

# X timelines return a FRESH bottom cursor on every page forever (the infinite
# "discover" scroll), so the real terminator is "this page added nothing new", not the
# cursor. MAX_PAGES is a runaway backstop only — if we ever hit it we log LOUD and stop
# (never silently under-fetch, never hammer X into a 429).
MAX_PAGES = 100

# Feature switches lifted verbatim from the captured request URLs (both ops share them).
# Neither op sends fieldToggles. Override via $X_LISTS_FEATURES (JSON) if X rotates.
LISTS_FEATURES = {
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
    # ON to match BOOKMARKS_FEATURES, which is the bundle whose full-article capture was actually
    # measured. This was the ONE flag differing between the two bundles, and it gates the media
    # embedded INSIDE a longform note tweet — so every timeline consumer of this bundle (Lists,
    # Likes, Following, both user timelines) was dropping it while the bookmark path kept it.
    # Verified live 2026-08-30: both values return 200, so this is a fidelity fix, not a risk.
    "longform_notetweets_inline_media_enabled": True,
    "responsive_web_grok_image_annotation_enabled": True,
    "responsive_web_grok_imagine_annotation_enabled": True,
    "responsive_web_grok_community_note_auto_translation_is_enabled": True,
    "responsive_web_enhance_cards_enabled": False,
}


def _features() -> dict:
    import os
    override = os.getenv("X_LISTS_FEATURES")
    return json.loads(override) if override else LISTS_FEATURES


# ── Parse: owned lists out of the management timeline ─────────────────────────

def _parse_owned_lists(timeline: dict, viewer_id: str) -> tuple[list[dict], str | None]:
    """Owned lists only: the `OwnedSubscribedList` module, entries whose list owner
    rest_id == the viewer. Skips the `ListToFollow` (suggested) module and any
    subscribed list (owner != viewer). Returns (lists, bottom_cursor)."""
    out: list[dict] = []
    next_cursor: str | None = None
    for ins in timeline.get("instructions", []):
        for e in ins.get("entries", []):
            content = e.get("content", {}) or {}
            if content.get("cursorType") == "Bottom":
                next_cursor = content.get("value")
                continue
            if content.get("__typename") != "TimelineTimelineModule":
                continue
            inj = (((content.get("clientEventInfo") or {}).get("details") or {})
                   .get("timelinesDetails") or {}).get("injectionType")
            if inj != "OwnedSubscribedList":     # ignore ListToFollow (suggested)
                continue
            for it in content.get("items", []):
                lst = ((it.get("item") or {}).get("itemContent") or {}).get("list") or {}
                lid = lst.get("id_str")
                owner = (((lst.get("user_results") or {}).get("result") or {})
                         .get("rest_id"))
                if not lid or owner != viewer_id:     # owned only
                    continue
                out.append({
                    "id": lid,
                    "name": lst.get("name", ""),
                    "mode": lst.get("mode", ""),
                    "member_count": lst.get("member_count", 0),
                })
    return out, next_cursor


def fetch_owned_lists(cookies: dict, headers: dict, viewer_id: str) -> list[dict]:
    qid = core.resolve_query_id(LISTS_MGMT_OP, cookies,
                                default_seed=DEFAULT_LISTS_MGMT_QID,
                                env_var="X_LISTS_MGMT_QUERY_ID", page_url=_DISCOVER_PAGE)
    lists: list[dict] = []
    seen_ids: set[str] = set()
    cursor: str | None = None
    seen_cursor: set[str] = set()
    for page_no in range(1, MAX_PAGES + 1):
        variables = {"count": DEFAULT_PAGE_SIZE}
        if cursor:
            variables["cursor"] = cursor
        data = core.graphql_get(LISTS_MGMT_OP, qid, variables, _features(), headers,
                                tolerate_errors=True)
        timeline = (((data.get("data") or {}).get("viewer") or {})
                    .get("list_management_timeline") or {}).get("timeline", {})
        page, next_cursor = _parse_owned_lists(timeline, viewer_id)
        new = 0
        for l in page:
            if l["id"] not in seen_ids:
                seen_ids.add(l["id"])
                lists.append(l)
                new += 1
        # No NEW owned lists this page → we've caught them all (they group on the first
        # page; later pages are the endless suggested-list scroll). Stop regardless of
        # the ever-advancing cursor.
        if new == 0:
            break
        if not next_cursor or next_cursor in seen_cursor:
            break
        seen_cursor.add(next_cursor)
        cursor = next_cursor
    else:
        log(f"[x-lists] hit MAX_PAGES={MAX_PAGES} enumerating owned lists — stopping "
            f"(partial: {len(lists)} so far).")
    return lists


# ── Parse: members out of a list's members_timeline ──────────────────────────

def _normalize_user(result: dict) -> dict | None:
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


def _parse_members(timeline: dict) -> tuple[list[dict], str | None]:
    out: list[dict] = []
    next_cursor: str | None = None
    for ins in timeline.get("instructions", []):
        for e in ins.get("entries", []):
            entry_id = e.get("entryId", "")
            content = e.get("content", {}) or {}
            if content.get("cursorType") == "Bottom":
                next_cursor = content.get("value")
                continue
            if entry_id.startswith("user-"):
                result = (((content.get("itemContent") or {}).get("user_results") or {})
                          .get("result"))
                norm = _normalize_user(result)
                if norm:
                    out.append(norm)
    return out, next_cursor


def fetch_list_members(list_id: str, cookies: dict, headers: dict) -> list[dict]:
    qid = core.resolve_query_id(LIST_MEMBERS_OP, cookies,
                                default_seed=DEFAULT_LIST_MEMBERS_QID,
                                env_var="X_LIST_MEMBERS_QUERY_ID", page_url=_DISCOVER_PAGE)
    members: list[dict] = []
    seen_ids: set[str] = set()
    cursor: str | None = None
    seen_cursor: set[str] = set()
    for page_no in range(1, MAX_PAGES + 1):
        variables = {"listId": list_id, "count": DEFAULT_PAGE_SIZE}
        if cursor:
            variables["cursor"] = cursor
        data = core.graphql_get(LIST_MEMBERS_OP, qid, variables, _features(), headers,
                                tolerate_errors=True)
        timeline = (((data.get("data") or {}).get("list") or {})
                    .get("members_timeline") or {}).get("timeline", {})
        page, next_cursor = _parse_members(timeline)
        new = 0
        for m in page:
            if m["user_id"] not in seen_ids:     # dedup across pages
                seen_ids.add(m["user_id"])
                members.append(m)
                new += 1
        # No NEW users this page → end of the list (the cursor keeps advancing forever).
        if new == 0:
            break
        if not next_cursor or next_cursor in seen_cursor:
            break
        seen_cursor.add(next_cursor)
        cursor = next_cursor
    else:
        log(f"[x-lists] hit MAX_PAGES={MAX_PAGES} paging members of list {list_id} — "
            f"stopping (partial: {len(members)} so far).")
    return members


# ── Orchestrate + write the signal ───────────────────────────────────────────

def _signal_path(config=None) -> Path:
    from pipeline.config import state_paths
    return (config or state_paths()).state_file("candidate_signals_x_lists")


def aggregate_members(owned: list[dict], members_by_list: dict[str, list[dict]],
                      viewer_id: str) -> list[dict]:
    """Fold per-list member lists into deduped candidates. Dedup by user_id, UNION the
    list names/ids (cross-list membership is a stronger signal, not a duplicate), drop
    the viewer themselves. Sorted by breadth of membership then handle. Pure — no IO,
    so it is directly unit-testable against captured fixtures."""
    by_user: dict[str, dict] = {}
    for lst in owned:
        for m in members_by_list.get(lst["id"], []):
            if m["user_id"] == viewer_id:         # never treat yourself as a candidate
                continue
            rec = by_user.get(m["user_id"])
            if rec is None:
                rec = {**m, "list_names": [], "list_ids": []}
                by_user[m["user_id"]] = rec
            if lst["name"] not in rec["list_names"]:
                rec["list_names"].append(lst["name"])
                rec["list_ids"].append(lst["id"])
    return sorted(by_user.values(),
                  key=lambda c: (-len(c["list_names"]), c["handle"].lower()))


def sync_lists(profile: str | None = None, dry_run: bool = False,
               config=None) -> dict:
    """Pull the viewer's owned lists + members → deduped candidate signal. Raises
    SyncAuthError if the session is dead (caller records a broken source, never a
    silent 0). Fail-safe: no twid → skip (can't tell owned from subscribed → no noise)."""
    cookies = core.read_x_cookies(profile=profile)
    vid = core.viewer_id(cookies)
    if not vid:
        log("[x-lists] twid cookie missing — cannot distinguish owned vs subscribed "
            "lists; skipping (fail-safe, no noise).")
        return {"lists": 0, "candidates": 0, "skipped": "no_viewer_id"}

    headers = core.auth_headers(cookies, referer=_REFERER)
    owned = fetch_owned_lists(cookies, headers, vid)
    log(f"[x-lists] {len(owned)} owned list(s): {[l['name'] for l in owned]}")

    members_by_list: dict[str, list[dict]] = {}
    for lst in owned:
        members = fetch_list_members(lst["id"], cookies, headers)
        log(f"[x-lists]   '{lst['name']}' (declares {lst['member_count']}) "
            f"→ {len(members)} member(s)")
        members_by_list[lst["id"]] = members

    candidates = aggregate_members(owned, members_by_list, vid)
    result = {"lists": len(owned), "candidates": len(candidates)}

    if dry_run:
        log(f"[x-lists] DRY RUN — {len(candidates)} candidate(s); not writing.")
        for c in candidates[:15]:
            log(f"    @{c['handle']:<20} {c['list_names']}  ({c['followers_count']} followers)")
        result["preview"] = candidates[:15]
        return result

    payload = {
        "signal": "x_list_member",
        "viewer_id": vid,
        "lists": owned,
        "captured_at": int(time.time()),
        "candidates": candidates,
    }
    path = _signal_path(config)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log(f"[x-lists] wrote {len(candidates)} candidate(s) from {len(owned)} list(s) → {path}")
    result["written"] = len(candidates)
    result["path"] = str(path)
    return result


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Ingest the user's OWNED X lists' members "
                                             "as a candidate signal.")
    ap.add_argument("--profile", help="Chrome profile dir / browser key (else auto-pick)")
    ap.add_argument("--dry-run", action="store_true", help="Print candidates, do not write")
    args = ap.parse_args()
    try:
        out = sync_lists(profile=args.profile, dry_run=args.dry_run)
        log(f"[x-lists] done: {json.dumps({k: v for k, v in out.items() if k != 'preview'})}")
    except SyncAuthError as e:
        log(f"[x-lists] NOT LOGGED IN / session dead: {e}")
        raise SystemExit(2)


if __name__ == "__main__":
    _cli()
