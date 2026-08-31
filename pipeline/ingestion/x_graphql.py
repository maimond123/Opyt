"""
pipeline/ingestion/x_graphql.py
Local-session X GraphQL client — reads the logged-in user's own bookmarks through
x.com's internal GraphQL API using cookies lifted from the browser (no official API,
no third party, no per-read billing).

Onboarding is hands-off: profiles are auto-scanned/picked (list_x_logged_in_profiles),
and the Bookmarks queryId self-discovers and caches (~/.opyt/x_query_ids.json), with
$X_BOOKMARKS_QUERY_ID as an override.
"""

import json
import os
import re
from pathlib import Path

from pipeline.ingestion.utils import log, SyncAuthError

GRAPHQL_HOST = "https://x.com/i/api/graphql"
BOOKMARKS_OP = "Bookmarks"
DEFAULT_PAGE_SIZE = 100

# X-wide queryId for the Bookmarks operation — public, same for every user (names the
# query, not the account). Empty = not seeded → fall through to cache → discovery → env.
DEFAULT_BOOKMARKS_QUERY_ID = "tUVliYsHyxrQIT4HXUWNdA"  # from bundle.Bookmarks, 2026-06-29
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Browser/profile discovery lives in the shared cookie module; X layers account-id
# decoration + ct0 warning on top of it.
from pipeline.ingestion import browser_cookies as _bc  # noqa: E402

# Public web-app bearer — not a secret; the constant the x.com web client ships.
FALLBACK_BEARER = ("AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D"
                   "1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA")

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


# ── Browser sessions ──────────────────────────────────────────────────────────
# browser_cookies.list_logged_in scans every browser/profile logged into X; `profile`
# is the selector CLI/env (X_CHROME_PROFILE) key on.

def list_x_logged_in_profiles() -> list[dict]:
    """Every browser/profile currently logged into X, as {profile, label}. The data layer
    behind the picker — no I/O prompts, no silent choice.

    Identity (the `twid`-derived account_id) and the cookies themselves used to ride along
    here. They cannot any more, and should not: detecting a Chromium session is now a
    row-presence check that never decrypts, so producing either would mean launching a
    browser PER CANDIDATE to answer a question the picker does not ask. `read_x_cookies`
    reads the one profile that gets chosen, and `viewer_id` decodes identity from that."""
    candidates, failures = _bc.list_logged_in(["x.com", "twitter.com"], "auth_token")
    out: list[dict] = []
    for c in candidates:
        out.append({"profile": c["profile"] or c["browser"], "label": c["label"]})
    # No login found but a read was blocked: log the actionable reason (Keychain/FDA)
    # instead of a bare "not logged in". Log-only; read_x_cookies raises the typed error.
    if not out and failures:
        f = _bc._worst_failure(failures)
        log(f"[x-graphql] no X login found; a read was blocked — "
            f"{_bc.remediation(f['kind'], _bc.backend_for(f['browser']), 'X')}")
    return out


def read_x_cookies(profile: str | None = None) -> dict:
    """Return the cookie dict for the chosen X session.

    Delegates selection to the shared multi-browser reader: explicit `profile` arg or
    $X_CHROME_PROFILE wins, else a lone logged-in profile auto-picks, else raises a
    SyncAuthError listing the options. The interactive picker lives in the CLI layer.
    """
    cookies = _bc.read_cookies(
        ["x.com", "twitter.com"], "auth_token",
        profile=profile, env_var="X_CHROME_PROFILE", source="X")
    if not cookies.get("ct0"):
        log("[x-graphql] warning: ct0 (csrf) cookie missing — request may 403")
    return cookies


def auth_headers(cookies: dict) -> dict:
    bearer = os.getenv("X_WEB_BEARER") or FALLBACK_BEARER
    return {
        "authorization": f"Bearer {bearer}",
        "x-csrf-token": cookies.get("ct0", ""),       # MUST equal the ct0 cookie
        "x-twitter-active-user": "yes",
        "x-twitter-auth-type": "OAuth2Session",
        "x-twitter-client-language": "en",
        "content-type": "application/json",
        "User-Agent": _UA,
        "Referer": "https://x.com/i/bookmarks",
        "Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items()),
    }


# ── queryId: env → cache → discover (self-healing) ────────────────────────────

def _qid_cache_path() -> Path:
    from opyt_core.paths import opyt_home
    return opyt_home() / "x_query_ids.json"


def _qid_cache_get(op: str) -> str | None:
    try:
        return json.loads(_qid_cache_path().read_text()).get(op)
    except Exception:
        return None


def _qid_cache_put(op: str, qid: str) -> None:
    p = _qid_cache_path()
    try:
        data = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        data = {}
    data[op] = qid
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data))
    except Exception as e:
        log(f"[x-graphql] could not cache queryId: {e}")


def _qid_cache_clear(op: str) -> None:
    p = _qid_cache_path()
    try:
        data = json.loads(p.read_text())
        data.pop(op, None)
        p.write_text(json.dumps(data))
    except Exception:
        pass


def resolve_query_id(cookies: dict) -> str:
    pinned = os.getenv("X_BOOKMARKS_QUERY_ID")
    if pinned:
        return pinned
    if DEFAULT_BOOKMARKS_QUERY_ID:        # baked seed — the Path A value, once set
        return DEFAULT_BOOKMARKS_QUERY_ID
    cached = _qid_cache_get(BOOKMARKS_OP)
    if cached:
        return cached
    discovered = discover_query_id(cookies)
    if discovered:
        _qid_cache_put(BOOKMARKS_OP, discovered)
        return discovered
    raise RuntimeError(
        "Could not resolve the Bookmarks GraphQL queryId from the JS bundles. This is "
        "a one-time developer seed (X-wide, not per-user): grab it from one request at "
        "x.com/i/bookmarks (DevTools → Network → the 'Bookmarks' request URL) and set "
        "$X_BOOKMARKS_QUERY_ID, or paste the URL and have it pinned in code.")


def discover_query_id(cookies: dict) -> str | None:
    """Best-effort: enumerate the client-web JS chunks and scan for the Bookmarks
    operation's queryId, bookmark-named chunks first. Returns None on miss; caller
    falls back to env/dev-seed.
    """
    try:
        from curl_cffi import requests as cffi
    except ImportError:
        return None

    pat = re.compile(r'queryId:"([^"]+)",operationName:"' + BOOKMARKS_OP + '"')
    base = "https://abs.twimg.com/responsive-web/client-web/"
    cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())

    def _get(url: str) -> str:
        return cffi.get(url, headers={"User-Agent": _UA}, impersonate="chrome120",
                        timeout=20).text

    try:
        html = cffi.get("https://x.com/i/bookmarks",
                        headers={"User-Agent": _UA, "Cookie": cookie_header},
                        impersonate="chrome120", timeout=30).text
    except Exception as e:
        log(f"[x-graphql] queryId discovery: page fetch failed: {e}")
        return None

    entry = set(re.findall(
        r'https://abs\.twimg\.com/responsive-web/client-web[^"\']+?\.js', html))
    chunks: set[str] = set(entry)
    # Harvest chunk filenames from the entry bundles' webpack manifest.
    for url in list(entry)[:6]:
        try:
            js = _get(url)
        except Exception:
            continue
        m = pat.search(js)
        if m:
            log(f"[x-graphql] discovered Bookmarks queryId in {url.split('/')[-1]}")
            return m.group(1)
        for frag in re.findall(r'"((?:bundle|api|endpoints|[\w.\-]+)\.[0-9a-f]{6,}a?\.js)"', js):
            chunks.add(base + frag.split("/")[-1])

    # Scan harvested chunks, bookmark-named first, bounded.
    ordered = sorted(chunks, key=lambda u: ("bookmark" not in u.lower(), u))
    scanned = 0
    for url in ordered[:150]:
        try:
            js = _get(url)
        except Exception:
            continue
        scanned += 1
        m = pat.search(js)
        if m:
            log(f"[x-graphql] discovered Bookmarks queryId in {url.split('/')[-1]} "
                f"(after {scanned} chunks)")
            return m.group(1)
    log(f"[x-graphql] queryId not found after scanning {scanned} chunk(s)")
    return None


# ── The request ───────────────────────────────────────────────────────────────

def _graphql_get(headers: dict, query_id: str, variables: dict) -> dict:
    from curl_cffi import requests as cffi

    feats = os.getenv("X_BOOKMARKS_FEATURES")
    feats = json.loads(feats) if feats else BOOKMARKS_FEATURES
    toggles = os.getenv("X_BOOKMARKS_FIELD_TOGGLES")
    toggles = json.loads(toggles) if toggles else BOOKMARKS_FIELD_TOGGLES
    params = {
        "variables": json.dumps(variables, separators=(",", ":")),
        "features": json.dumps(feats, separators=(",", ":")),
        "fieldToggles": json.dumps(toggles, separators=(",", ":")),
    }
    url = f"{GRAPHQL_HOST}/{query_id}/{BOOKMARKS_OP}"
    resp = cffi.get(url, params=params, headers=headers, impersonate="chrome120", timeout=30)

    if resp.status_code in (401, 403):
        raise SyncAuthError(
            f"x.com rejected the session ({resp.status_code}) — cookie expired or ct0/csrf "
            f"mismatch. Re-log into x.com in Chrome. Body: {resp.text[:300]}")
    if resp.status_code in (400, 404):
        # queryId or features drifted — drop the cache so the next run re-discovers.
        _qid_cache_clear(BOOKMARKS_OP)
        raise RuntimeError(
            f"Bookmarks GraphQL {resp.status_code} — queryId or `features` drifted "
            f"(cache cleared; next run re-discovers). Body: {resp.text[:500]}")
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise RuntimeError(f"Bookmarks GraphQL returned errors: {data['errors']}")
    return data


# ── Timeline parse + normalize to the shape tweet_to_markdown wants ──

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


def _normalize(result: dict) -> dict | None:
    """Map an X GraphQL tweet `result` to the normalized tweet dict every renderer and
    consumer in this repo reads. Media/quote paths are best-effort.

    The shape was originally twitterapi.io's, because that provider was the only producer when
    this was written. It is now the ONLY shape — the provider was removed on 2026-08-30 and this
    function is the sole producer — so it is just "the normalized tweet", not a translation into
    somebody else's vocabulary."""
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

    norm = {
        "id": tweet_id,
        "author": {"userName": username, "name": name, "id": user_id, "site": u_site},
        "text": text,
        "createdAt": legacy.get("created_at", ""),
        "likeCount": legacy.get("favorite_count", 0),
        "entities": legacy.get("entities", {}),
        # MUST stay camelCase: _render_media reads `extendedEntities`. Emitting X's
        # snake_case `extended_entities` here silently drops every photo/video/GIF.
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
        # handle field (unreliable). One path each, no `or` fallback — see
        "isRetweet": bool(legacy.get("retweeted_status_result")),
        "inReplyToUserId": legacy.get("in_reply_to_user_id_str") or "",
    }
    # X-Article body: with the article toggles ON, X nests the full body at
    # result.article.article_results.result.content_state.blocks. Carry the whole
    # `article` node; `_render_article` digs out title + blocks. Absent → normal post.
    article = result.get("article")
    if article:
        norm["article"] = article

    quoted = (result.get("quoted_status_result") or {}).get("result")
    if quoted:
        norm["quoted_tweet"] = _normalize(quoted)
    return norm


# ── Public iterator ────────────────────────────────────────────────────────────

def iterate_bookmarks(limit: int = 0, page_size: int = DEFAULT_PAGE_SIZE,
                      profile: str | None = None):
    """Yield the user's bookmarks newest-first as normalized tweet dicts. Raises
    SyncAuthError if not logged in / session dead (so the caller records a broken
    source, never a silent '0 bookmarks')."""
    cookies = read_x_cookies(profile=profile)
    headers = auth_headers(cookies)
    query_id = resolve_query_id(cookies)
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
        data = _graphql_get(headers, query_id, variables)
        results, next_cursor = _parse_timeline(data)
        log(f"[x-graphql] page {page}: {len(results)} tweets")
        if not results:
            break
        for result in results:
            norm = _normalize(result)
            if norm and norm.get("id"):
                yield norm
                yielded += 1
                if limit and yielded >= limit:
                    return
        if not next_cursor or next_cursor in seen:
            break
        seen.add(next_cursor)
        cursor = next_cursor
