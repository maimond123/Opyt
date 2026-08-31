"""
pipeline/ingestion/x_graphql_core.py
Shared primitives for local-session X GraphQL reads (Lists / following / likes / …):
cookie read, auth headers, the self-healing queryId resolve->discover->cache, and the
GraphQL GET with drift handling. Requests run as the logged-in user, from their own
session cookies — no official API, no per-read billing. Bookmarks (x_graphql.py) stays
on its own copy for now rather than migrating onto this core; see the companion doc.
"""

import json
import os
import re
from pathlib import Path

from pipeline.ingestion.utils import log, SyncAuthError
from pipeline.ingestion import browser_cookies as _bc

GRAPHQL_HOST = "https://x.com/i/api/graphql"

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Public web-app bearer — not a secret; the constant the x.com web client ships.
FALLBACK_BEARER = ("AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D"
                   "1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA")


# ── Browser session ─────────────────────────────────────────────────────────

def read_x_cookies(profile: str | None = None) -> dict:
    """Cookie dict for the chosen X session (delegates to the shared multi-browser
    reader — explicit `profile`/$X_CHROME_PROFILE wins, else a lone logged-in
    browser auto-picks, else a SyncAuthError listing the options). Raises rather than
    returning empty, so a caller records a broken source instead of a silent '0'."""
    cookies = _bc.read_cookies(
        ["x.com", "twitter.com"], "auth_token",
        profile=profile, env_var="X_CHROME_PROFILE", source="X")
    if not cookies.get("ct0"):
        log("[x-graphql] warning: ct0 (csrf) cookie missing — request may 403")
    return cookies


def viewer_id(cookies: dict) -> str | None:
    """The logged-in user's own numeric id (rest_id), decoded from the `twid` cookie
    (`u=<id>`). This is the identity we match list OWNERSHIP against — handles rename,
    ids don't. None if twid is absent (caller should fail safe, not guess)."""
    twid = (cookies.get("twid") or "").replace("u%3D", "").replace("u=", "")
    return twid or None


def auth_headers(cookies: dict, referer: str) -> dict:
    """The header set every GraphQL read needs. `referer` is per-surface (x.com/i/lists,
    x.com/i/bookmarks, …); X is lax about it but we send the honest one."""
    bearer = os.getenv("X_WEB_BEARER") or FALLBACK_BEARER
    return {
        "authorization": f"Bearer {bearer}",
        "x-csrf-token": cookies.get("ct0", ""),       # MUST equal the ct0 cookie
        "x-twitter-active-user": "yes",
        "x-twitter-auth-type": "OAuth2Session",
        "x-twitter-client-language": "en",
        "content-type": "application/json",
        "User-Agent": _UA,
        "Referer": referer,
        "Cookie": "; ".join(f"{k}={v}" for k, v in cookies.items()),
    }


# ── queryId: env → seed → cache → discover (self-healing) ─────────────────────
# queryId names the OPERATION, not the account — X-wide, identical for every user.
# Seeded once from a real request, cached per-op in ~/.opyt/x_query_ids.json, and
# re-discovered from the JS bundles on drift.

def _qid_cache_path() -> Path:
    from opyt_core.paths import opyt_home
    return opyt_home() / "x_query_ids.json"


def qid_cache_get(op: str) -> str | None:
    try:
        return json.loads(_qid_cache_path().read_text()).get(op)
    except Exception:
        return None


def qid_cache_put(op: str, qid: str) -> None:
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


def qid_cache_clear(op: str) -> None:
    p = _qid_cache_path()
    try:
        data = json.loads(p.read_text())
        data.pop(op, None)
        p.write_text(json.dumps(data))
    except Exception:
        pass


def resolve_query_id(op: str, cookies: dict, *, default_seed: str = "",
                     env_var: str | None = None,
                     page_url: str = "https://x.com/home") -> str:
    """env override → baked seed → cache → live discovery → raise. Mirrors the
    Bookmarks resolver, parameterized by operation."""
    if env_var and os.getenv(env_var):
        return os.getenv(env_var)
    if default_seed:
        return default_seed
    cached = qid_cache_get(op)
    if cached:
        return cached
    discovered = discover_query_id(op, page_url, cookies)
    if discovered:
        qid_cache_put(op, discovered)
        return discovered
    raise RuntimeError(
        f"Could not resolve the {op} GraphQL queryId from the JS bundles. This is a "
        f"one-time developer seed (X-wide, not per-user): grab it from one real "
        f"{op} request URL (DevTools → Network) and pass it as default_seed / the env "
        f"override.")


def discover_query_id(op: str, page_url: str, cookies: dict) -> str | None:
    """Best-effort: fetch an authenticated x.com page, harvest the client-web JS chunk
    filenames it references (webpack manifest), and scan them for
    `queryId:"…",operationName:"<op>"`. Op-named chunks are scanned first. None on miss
    → caller falls back to env/seed. (Generalized from the Bookmarks discovery.)"""
    try:
        from curl_cffi import requests as cffi
    except ImportError:
        return None

    pat = re.compile(r'queryId:"([^"]+)",operationName:"' + re.escape(op) + '"')
    base = "https://abs.twimg.com/responsive-web/client-web/"
    cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())

    def _get(url: str) -> str:
        return cffi.get(url, headers={"User-Agent": _UA}, impersonate="chrome120",
                        timeout=20).text

    try:
        html = cffi.get(page_url,
                        headers={"User-Agent": _UA, "Cookie": cookie_header},
                        impersonate="chrome120", timeout=30).text
    except Exception as e:
        log(f"[x-graphql] {op} queryId discovery: page fetch failed: {e}")
        return None

    entry = set(re.findall(
        r'https://abs\.twimg\.com/responsive-web/client-web[^"\']+?\.js', html))
    chunks: set[str] = set(entry)
    for url in list(entry)[:6]:
        try:
            js = _get(url)
        except Exception:
            continue
        m = pat.search(js)
        if m:
            log(f"[x-graphql] discovered {op} queryId in {url.split('/')[-1]}")
            return m.group(1)
        for frag in re.findall(r'"((?:bundle|api|endpoints|[\w.\-]+)\.[0-9a-f]{6,}a?\.js)"', js):
            chunks.add(base + frag.split("/")[-1])

    op_key = op.lower()
    ordered = sorted(chunks, key=lambda u: (op_key not in u.lower(), u))
    scanned = 0
    for url in ordered[:150]:
        try:
            js = _get(url)
        except Exception:
            continue
        scanned += 1
        m = pat.search(js)
        if m:
            log(f"[x-graphql] discovered {op} queryId in {url.split('/')[-1]} "
                f"(after {scanned} chunks)")
            return m.group(1)
    log(f"[x-graphql] {op} queryId not found after scanning {scanned} chunk(s)")
    return None


# ── The request ─────────────────────────────────────────────────────────────

class XRateLimited(RuntimeError):
    """x.com returned 429 — the shared session's request budget for this window is spent.
    Distinct from a per-account failure: a multi-account loop must STOP entirely here,
    not skip the one account and keep going."""


def graphql_get(op: str, query_id: str, variables: dict, features: dict,
                headers: dict, *, field_toggles: dict | None = None,
                tolerate_errors: bool = False) -> dict:
    """One GraphQL GET. On 401/403 -> SyncAuthError (dead session). On 400/404 -> clear
    the queryId cache (queryId/features drifted) and raise so the next run re-discovers.
    `tolerate_errors`: X often returns a partial `errors` array alongside valid `data`;
    when True, keep `data` and just log the error count instead of hard-failing."""
    from curl_cffi import requests as cffi

    params = {
        "variables": json.dumps(variables, separators=(",", ":")),
        "features": json.dumps(features, separators=(",", ":")),
    }
    if field_toggles is not None:
        params["fieldToggles"] = json.dumps(field_toggles, separators=(",", ":"))
    url = f"{GRAPHQL_HOST}/{query_id}/{op}"
    resp = cffi.get(url, params=params, headers=headers, impersonate="chrome120", timeout=30)

    if resp.status_code == 429:
        raise XRateLimited(
            f"{op} rate-limited (429) by x.com — too many requests in a short window. "
            f"Wait a few minutes before retrying. (If this fired mid-pagination it means "
            f"a loop failed to terminate — that is a bug, not just throttling.)")
    if resp.status_code in (401, 403):
        raise SyncAuthError(
            f"x.com rejected the session ({resp.status_code}) — cookie expired or "
            f"ct0/csrf mismatch. Re-log into x.com in Chrome. Body: {resp.text[:300]}")
    if resp.status_code in (400, 404):
        qid_cache_clear(op)
        raise RuntimeError(
            f"{op} GraphQL {resp.status_code} — queryId or `features` drifted "
            f"(cache cleared; next run re-discovers). Body: {resp.text[:500]}")
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        if not tolerate_errors:
            raise RuntimeError(f"{op} GraphQL returned errors: {data['errors']}")
        log(f"[x-graphql] {op}: {len(data['errors'])} partial field error(s) tolerated "
            f"(data present)")
    return data


# ── Following timeline (person-endorsement signal; FREE cookie-scrape) ─────────
# Sibling of x_likes.fetch_liked_authors: a user-scoped timeline whose entries are USER
# cards, reusing x_lists._normalize_user + _parse_members. Reads the viewer's own
# Following via their session cookies. (The paid XFollowingScout path this replaced was deleted
# with its provider on 2026-08-30; every X read is a cookie-scrape now.)
FOLLOWING_OP = "Following"
# Empty on purpose: a baked seed 404s forever once X rotates it. Empty routes to
# env -> cache -> live JS-bundle discovery, which finds + caches the real id.
DEFAULT_FOLLOWING_QID = ""
_FOLLOWING_MAX_PAGES = 100
_FOLLOWING_PAGE_SIZE = 100


def _following_features() -> dict:
    """The Following op shares the web client's timeline feature bundle with Lists/Likes.
    Reference it rather than duplicate 40 drift-prone lines; override via
    $X_FOLLOWING_FEATURES (JSON) if a live capture ever shows a different set."""
    from pipeline.ingestion import x_lists as xlists   # lazy: x_lists imports THIS module
    override = os.getenv("X_FOLLOWING_FEATURES")
    return json.loads(override) if override else xlists.LISTS_FEATURES


def fetch_following(cookies: dict, headers: dict, viewer_id: str) -> list[dict]:
    """Walk the viewer's Following timeline → one normalized user per followed account
    (the `x_lists._normalize_user` shape, `site` included as the identity seed). Dedups by
    rest_id and stops the instant a page adds no NEW user — the only terminator that holds
    against X's forever-advancing infinite-scroll cursor (same guard as likes/lists)."""
    from pipeline.ingestion import x_lists as xlists   # lazy: avoids the import cycle
    qid = resolve_query_id(FOLLOWING_OP, cookies, default_seed=DEFAULT_FOLLOWING_QID,
                           env_var="X_FOLLOWING_QUERY_ID", page_url="https://x.com/home")
    features = _following_features()
    out: list[dict] = []
    seen_ids: set[str] = set()
    cursor: str | None = None
    seen_cursor: set[str] = set()
    for _page in range(1, _FOLLOWING_MAX_PAGES + 1):
        variables = {"userId": viewer_id, "count": _FOLLOWING_PAGE_SIZE,
                     "includePromotedContent": False}
        if cursor:
            variables["cursor"] = cursor
        data = graphql_get(FOLLOWING_OP, qid, variables, features, headers,
                           tolerate_errors=True)
        timeline = ((((data.get("data") or {}).get("user") or {}).get("result") or {})
                    .get("timeline") or {}).get("timeline", {})
        page, next_cursor = xlists._parse_members(timeline)   # 'user-' entries → normalized
        new = 0
        for m in page:
            if m["user_id"] not in seen_ids:
                seen_ids.add(m["user_id"])
                out.append(m)
                new += 1
        if new == 0:              # no NEW follow this page → caught them all
            break
        if not next_cursor or next_cursor in seen_cursor:
            break
        seen_cursor.add(next_cursor)
        cursor = next_cursor
    else:
        log(f"[x-following] hit MAX_PAGES={_FOLLOWING_MAX_PAGES} paging following — "
            f"stopping (partial: {len(out)} so far).")
    return out


# ── Conversation / thread context (TweetDetail; FREE cookie-scrape) ────────────
# A bookmarked reply needs the chain it replies to. TweetDetail returns it: ancestor
# tweets as top-level `tweet-` entries before the focal, in order; the author's own
# self-thread continuation in `conversationthread-` modules below.
TWEETDETAIL_OP = "TweetDetail"
DEFAULT_TWEETDETAIL_QID = ""      # empty → discovery (a guessed seed would 404 on rotation)
_CONVO_MAX_ANCESTORS = 20         # bound a pathologically deep reply chain (keep the nearest)
_CONVO_MAX_CONTINUATION = 25      # bound a long self-thread


def _tweetdetail_features() -> dict:
    from pipeline.ingestion import x_lists as xlists   # lazy: shared timeline bundle
    override = os.getenv("X_TWEETDETAIL_FEATURES")
    return json.loads(override) if override else xlists.LISTS_FEATURES


def _convo_tweets(data: dict) -> tuple[list[dict], list[dict]]:
    """(top-level tweets in order, conversationthread module tweets in order), normalized to the
    x_graphql._normalize shape. Both live under threaded_conversation_with_injections_v2."""
    from pipeline.ingestion import x_graphql as xg
    root = ((data.get("data") or {})
            .get("threaded_conversation_with_injections_v2") or {})
    top: list[dict] = []
    modules: list[dict] = []
    for ins in root.get("instructions", []):
        for e in ins.get("entries", []):
            eid = e.get("entryId", "")
            c = e.get("content", {}) or {}
            if eid.startswith("tweet-"):
                res = ((c.get("itemContent") or {}).get("tweet_results") or {}).get("result")
                n = xg._normalize(res) if res else None
                if n and n.get("id"):
                    top.append(n)
            elif eid.startswith("conversationthread-"):
                for it in c.get("items", []) or []:
                    res = ((((it.get("item") or {}).get("itemContent") or {})
                            .get("tweet_results") or {}).get("result"))
                    n = xg._normalize(res) if res else None
                    if n and n.get("id"):
                        modules.append(n)
    return top, modules


def _author_key(t: dict) -> str:
    """Same-author match key: `author.userName`, lowercased; "" if absent.

    Written to be shape-agnostic because two backends fed it — `x_graphql._normalize` and
    twitterapi.io's thread_context, which both populated that field. Only the first survives
    (2026-08-30), so this is no longer a compatibility choice, just the right field."""
    return ((t.get("author") or {}).get("userName") or "").lower()


def reconstruct_chain(tweets: list[dict], focal_id: str, *,
                      max_ancestors: int = _CONVO_MAX_ANCESTORS,
                      max_continuation: int = _CONVO_MAX_CONTINUATION) -> list[dict]:
    """From a flat, conversation-ordered tweet list → the chain the focal sits in:
    `[ancestors…, focal, same-author self-continuation…]`. Ancestors = tweets before the focal
    (the DEBATE context); continuation = same-author tweets after it (the self-thread). Shape-
    agnostic (matches on author.userName), so the cookie AND twitterapi.io backends share it.
    Returns [] when there's no real context (len ≤ 1). Deduped, order-preserving."""
    focal_id = str(focal_id)
    idx = next((i for i, t in enumerate(tweets) if str(t.get("id")) == focal_id), None)
    if idx is None:
        return []                                    # focal not in the conversation → render solo
    focal = tweets[idx]
    fa = _author_key(focal)
    ancestors = tweets[max(0, idx - max_ancestors):idx]
    continuation: list[dict] = []
    for t in tweets[idx + 1:]:
        if _author_key(t) == fa and str(t.get("id")) != focal_id:
            continuation.append(t)
        if len(continuation) >= max_continuation:
            break
    chain, seen = [], set()
    for t in ancestors + [focal] + continuation:
        tid = str(t.get("id"))
        if tid and tid not in seen:
            seen.add(tid)
            chain.append(t)
    return chain if len(chain) > 1 else []


def fetch_conversation(focal_id: str, cookies: dict, headers: dict) -> list[dict]:
    """The ordered chain a tweet sits in — `[ancestors…, focal, same-author self-continuation]` —
    via the FREE cookie-scrape TweetDetail (the keyless fallback; the twitterapi.io thread_context
    path is primary). Returns [] on no-context or any failure (fail-safe: caller renders solo).
    Subject to X's 150/15-min TweetDetail limit — the caller paces + degrades on that."""
    qid = resolve_query_id(TWEETDETAIL_OP, cookies, default_seed=DEFAULT_TWEETDETAIL_QID,
                           env_var="X_TWEETDETAIL_QUERY_ID", page_url="https://x.com/home")
    variables = {
        "focalTweetId": str(focal_id), "with_rux_injections": False,
        "includePromotedContent": False, "withCommunity": True,
        "withQuickPromoteEligibilityTweetFields": True, "withBirdwatchNotes": True,
        "withVoice": True, "withV2Timeline": True,
    }
    # No article fieldToggles, and that is correct rather than an oversight. This returns CONTEXT —
    # the ancestors a focal tweet replies to and the author's own continuation — and a caller
    # always holds its own rich copy of the FOCAL tweet from the path that fetched it
    # (`iterate_bookmarks` and `fetch_tweets_by_ids` both set `withArticleRichContentState`;
    # `ingest_x._fetch_one_tweet` splices that copy back over the chain's). An ancestor renders as
    # its text, never as an article body, so asking for 16 KB of `content_state` per ancestor would
    # buy bytes nothing reads.
    data = graphql_get(TWEETDETAIL_OP, qid, variables, _tweetdetail_features(), headers,
                       tolerate_errors=True)
    top, modules = _convo_tweets(data)
    # Flatten: ancestors + focal live in `top` (in order), replies/continuation in `modules` after.
    # reconstruct_chain then picks ancestors-before-focal + same-author-after uniformly.
    return reconstruct_chain(top + modules, focal_id)


# ── User timeline (UserTweets; FREE cookie-scrape) ────────────────────────────
# The candidate-list Proposer's light content pull: one shallow, recent page of a
# candidate's own posts (docs/plans/2026-08-11-proposer-candidate-loop.md). Shares the
# Lists/Likes/Following feature bundle and pagination terminator. FREE; rate is the
# only ceiling, and every GraphQL consumer here shares one session budget, so this
# function deliberately does not sleep — pacing is left to the caller.
USERTWEETS_OP = "UserTweets"
# Empty on purpose: a baked seed survives rotation as a permanent 404 since
# resolve_query_id never re-checks a non-empty seed. Empty routes to
# env -> cache -> live JS-bundle discovery, which self-heals.
DEFAULT_USERTWEETS_QID = ""

# The two user-scoped timelines X exposes. They differ ONLY in the operation, the page its
# queryId is discovered from, and one variable — parsing, normalization, termination and the
# unavailable-account rule are identical — so this is one walk with a selector rather than two
# functions whose names differ by an adjective.
#
# "posts" is the x.com/<user> Posts tab: originals, quotes, retweets and self-threads. It OMITS
# standalone replies to other accounts, and that omission is a real population, not an edge case:
# measured against twitterapi.io over one 108-day window, 200 of the 214 tweets only twitterapi.io
# returned were replies. "replies" (x.com/<user>/with_replies) is what carries them.
#
# `UserTweetsAndReplies` is NOT the reply operation. X still ships its name and queryId in the JS
# bundle, but every call 404s (verified 2026-08-30) — which is exactly why no queryId here is
# baked in as a seed: `resolve_query_id` returns a non-empty seed WITHOUT validating it, so a
# stale id becomes a permanent 404 the cache is never consulted to fix.
#
# ⚠️ THE REPLIES TIMELINE RETURNS OTHER PEOPLE'S TWEETS. Every entry it ships is a
# `profile-conversation-` module holding the whole exchange — the tweet being replied TO as well as
# the reply — so a raw walk of @hypersoren's replies came back 321 tweets from 92 DISTINCT AUTHORS,
# 145 of them (45%) not his (measured 2026-08-30). The Posts tab has no such shape: the same walk
# returned 80 tweets from exactly 1 author. That is why this function filters by `user_id` below
# and does not leave it to callers: `fetch_user_tweets(user_id)` means one account's own timeline,
# and every downstream consumer is built on that. `ingest_x_footprint._majority_author` states the
# invariant outright — "the author of a from:handle pull is uniform" — and a 45%-contaminated list
# would not merely dilute its vote, it would slip other people's ORIGINAL posts past
# `_filter_and_stitch` entirely, which drops retweets and replies-to-others but has no reason to
# suspect a plain original of belonging to somebody else.
_TIMELINES = {
    "posts":   (USERTWEETS_OP, "X_USERTWEETS_QUERY_ID",
                {"withQuickPromoteEligibilityTweetFields": True}),
    "replies": ("UserRepliesTimeline", "X_USERREPLIES_QUERY_ID",
                {"withCommunity": True}),
}
_USERTWEETS_PAGE_SIZE = 20        # what one request returns (~20-41 tweets observed)
# Hard cap — the cursor NEVER exhausts (see the loop). Raised 20 -> 60 on 2026-08-30, when this
# path took over the deep oracle backfill from twitterapi.io's sharded search. At 20 it silently
# TRUNCATED a prolific account: @hypersoren needs 26 requests to cover 183 days (515 tweets, 20 s),
# so the ceiling, not the date window, was deciding how much history an oracle got.
#
# This stays a runaway guard, not the terminator. The date-window `after_page` stop is what
# actually ends a real pull — measured 2 requests for @waterloo_intern (340 days) and 2 for
# @AMelhede (216 days), because the window closed long before any page count mattered.
_USERTWEETS_MAX_PAGES = 60


class XUserUnavailable(RuntimeError):
    """The account exists as an id but its timeline can't be read — suspended, deactivated,
    or protected. Raised rather than returned as `[]`, since those are opposite facts
    about a candidate (unreadable vs. "posts nothing")."""


def _usertweets_features() -> dict:
    """UserTweets shares the web client's timeline feature bundle with Lists/Likes/Following.
    Reference it rather than duplicate 40 drift-prone lines; override via $X_USERTWEETS_FEATURES."""
    from pipeline.ingestion import x_lists as xlists   # lazy: x_lists imports THIS module
    override = os.getenv("X_USERTWEETS_FEATURES")
    return json.loads(override) if override else xlists.LISTS_FEATURES


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


def _parse_user_timeline(timeline: dict) -> tuple[list[dict], str | None]:
    """One UserTweets page -> (raw tweet `result` nodes in order, bottom cursor).

    Three entry shapes, all needed or real posts get dropped:
      - `tweet-...` — a standalone post.
      - `profile-conversation-...` — a self-thread, delivered as a module of items.
      - `TimelinePinEntry` — the pinned post, arrives as a single `entry`, not in
        an `entries` list. Kept even though it may be older than the rest of the page.
    """
    results: list[dict] = []
    next_cursor: str | None = None

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
        pinned = ins.get("entry")            # TimelinePinEntry carries ONE entry, not a list
        if isinstance(pinned, dict):
            _take(pinned)
    return results, next_cursor


def fetch_user_tweets(cookies: dict, headers: dict, user_id: str, *, pages: int = 1,
                      page_size: int = _USERTWEETS_PAGE_SIZE,
                      after_page=None, timeline: str = "posts") -> list[dict]:
    """Walk `pages` of one account's own timeline -> normalized tweets (the
    `x_graphql._normalize` / twitterapi.io shape).

    `timeline` selects which of X's two user-scoped timelines to walk — "posts" (the default,
    unchanged for every existing caller) or "replies" for the standalone replies the Posts tab
    omits. See `_TIMELINES` for why that is one selector and not two functions.

    `pages=1` is the design default (one request per candidate); higher values are for
    a deliberate deep pull. Raises `XUserUnavailable` for a suspended/deactivated/
    protected account; returns `[]` for an account that has genuinely posted nothing.

    Termination: X hands back a fresh bottom cursor forever, so three independent stops
    apply (matching `fetch_following`): no new tweet id on a page, a repeated cursor,
    and a hard page cap. `after_page(tweets) -> bool` is an optional fourth, caller-
    supplied stop checked after each page, alongside the other three rather than
    instead of them, and it is also the caller's pacing point since this function does
    not sleep itself. See the companion doc for the full rationale."""
    try:
        op, qid_env, extra_vars = _TIMELINES[timeline]
    except KeyError:
        raise ValueError(f"timeline must be one of {sorted(_TIMELINES)}, got {timeline!r}")
    qid = resolve_query_id(op, cookies, default_seed=DEFAULT_USERTWEETS_QID,
                           env_var=qid_env, page_url="https://x.com/home")
    from pipeline.ingestion import x_graphql as xg      # lazy: shared tweet normalizer

    features = _usertweets_features()
    # Clamped, not trusted: a cap a caller can raise past isn't a cap.
    want = max(1, int(pages))
    n_pages = min(want, _USERTWEETS_MAX_PAGES)
    out: list[dict] = []
    seen_ids: set[str] = set()
    seen_cursor: set[str] = set()
    cursor: str | None = None
    for _page in range(1, n_pages + 1):
        variables = {"userId": str(user_id), "count": int(page_size),
                     "includePromotedContent": False,
                     "withVoice": True, "withV2Timeline": True, **extra_vars}
        if cursor:
            variables["cursor"] = cursor
        # tolerate_errors: X often returns partial errors beside valid data; don't
        # throw away a good page over one bad sub-field.
        #
        # `withArticleRichContentState` is what makes an X Article a FIELD READ instead of a second
        # fetch, and it is not optional. Without it the `article` node on a timeline tweet is a
        # 1.1 KB TEASER — cover image, title, `preview_text` — and nothing else; the request still
        # returns 200, `article` is still truthy, and every "does this have an article" check still
        # passes, so the body goes missing silently. With it the same tweet carries the full
        # 107-block `content_state` that `x_render._article_shape` reads (measured 2026-08-30:
        # 16,328 bytes of rendered markdown, byte-identical to what `fetch_tweets_by_ids` returns).
        #
        # `withArticlePlainText` stays False deliberately. It is a THIRD shape — a flat `plain_text`
        # string that `_article_shape` does not read — so turning it on adds ~18 KB per article to
        # every page and changes nothing that renders.
        data = graphql_get(op, qid, variables, features, headers,
                           field_toggles={"withArticleRichContentState": True,
                                          "withArticlePlainText": False},
                           tolerate_errors=True)
        # `root`, not `timeline` — that name is the caller's selector and rebinding it here made
        # the ceiling log below print a response dict instead of "posts" / "replies".
        root, unavailable = _user_timeline_root(data)
        if unavailable:
            raise XUserUnavailable(f"x:user:{user_id} timeline unreadable: {unavailable}")
        page, next_cursor = _parse_user_timeline(root)
        new = 0            # NEW tweets by this account
        others = 0         # NEW tweets by somebody else (a replies page's conversation partners)
        for res in page:
            norm = xg._normalize(res)
            if not norm or not norm.get("id") or norm["id"] in seen_ids:
                continue
            seen_ids.add(norm["id"])
            # Whose tweet is this? On "replies" the answer is often "somebody else's" — see the
            # warning on `_TIMELINES`. Counted separately from the id dedupe so the two reasons a
            # page can add nothing stay distinguishable in the terminator below.
            if str((norm.get("author") or {}).get("id") or "") != str(user_id):
                others += 1
                continue
            out.append(norm)
            new += 1
        # A page of pure conversation partners is NOT an exhausted cursor — on a replies walk that
        # is an ordinary page. Terminating on `new == 0` alone would stop the walk dead the first
        # time every reply on a page sat under someone else's post. `others` keeps it going; the
        # repeated-cursor and page-cap stops below still bound it.
        if new == 0 and others == 0:          # nothing new at all → the cursor is spinning
            break
        # Caller's own "enough" test and pacing point, run only after a page landed.
        if after_page is not None and after_page(out):
            break
        if not next_cursor or next_cursor in seen_cursor:
            break
        seen_cursor.add(next_cursor)
        cursor = next_cursor
    else:
        # Only worth logging when the CAP is what stopped us, not an intentional pages=1.
        if n_pages < want or n_pages == _USERTWEETS_MAX_PAGES:
            log(f"[x-usertweets] x:user:{user_id} ({timeline}): stopped at the {n_pages}-page "
                f"ceiling (asked {want}) — partial: {len(out)} tweets.")
    return out


# ── User profile (UserByScreenName; FREE cookie-scrape) ───────────────────────
# One handle -> the identity fields two callers need: `oracles._fetch_x_identity` (which keys
# `x:user:{id}` entities on `rest_id`, the load-bearing field) and
# `discover_profile._probe_twitter_bio` (which mines the bio for a person's other homes).
USERBYSCREENNAME_OP = "UserByScreenName"
# Empty for the same reason as every other queryId here: `resolve_query_id` returns a non-empty
# seed WITHOUT validating it, so a baked id survives rotation as a permanent 404.
DEFAULT_USERBYSCREENNAME_QID = ""
# Measured 2026-08-30 off `x-rate-limit-limit`: 150 per 15 minutes, its own bucket. Recorded
# because it is the number that decides whether a caller walking many handles needs to pace.
_USERBYSCREENNAME_LIMIT_PER_WINDOW = 150


def _expand_tco(url: str, url_entities: dict) -> str:
    """A t.co short link -> the real destination, via the `urls` list X ships beside it. Returns
    `url` unchanged when nothing matches, so a caller never gets an empty string for a link that
    exists. X hands back the shortened form in `website.url` and the expansion only in the
    entities block, so this is a join, not a fallback."""
    for u in (url_entities.get("urls") or []):
        if u.get("url") == url and u.get("expanded_url"):
            return u["expanded_url"]
    return url


def fetch_user_profile(cookies: dict, headers: dict, screen_name: str) -> dict | None:
    """One X handle -> `{user_id, handle, display_name, bio, website, bio_urls, verified,
    followers}`, or None when the account cannot be read.

    ⚠️ THERE IS NO `legacy` BLOCK. X restructured this response, so the classic
    `legacy.followers_count` / `legacy.description` paths that a first pass written from memory of
    the old shape reaches for yield `None` for EVERY field — silently, since the request still
    returns 200. The live paths are `core.*`, `profile_bio.*`, `relationship_counts.*`,
    `verification.*` and `website.url`; see the companion doc for the full mapping table.

    Returns None rather than raising `XUserUnavailable` — unlike `fetch_user_tweets` directly
    above, which does raise. The difference is that neither caller here distinguishes the cases:
    both report "unresolved" for a suspended account and for a network failure alike, so a second
    exception type would be a distinction nothing acts on.

    `website` and every entry in `bio_urls` are EXPANDED past t.co. Both callers did that
    themselves, against two slightly different copies of the same loop; it belongs here, once,
    beside the entities block it needs. Self-links to x.com are deliberately NOT filtered — that
    is a consumer policy (one caller blanks them, the other skips the source) and the two disagree.
    """
    from pipeline.ingestion import x_lists as xlists   # lazy: x_lists imports THIS module
    qid = resolve_query_id(USERBYSCREENNAME_OP, cookies,
                           default_seed=DEFAULT_USERBYSCREENNAME_QID,
                           env_var="X_USERBYSCREENNAME_QUERY_ID",
                           page_url="https://x.com/home")
    handle = (screen_name or "").strip().lstrip("@")
    if not handle:
        return None
    data = graphql_get(USERBYSCREENNAME_OP, qid,
                       {"screen_name": handle, "withSafetyModeUserFields": True},
                       xlists.LISTS_FEATURES, headers,
                       field_toggles={"withAuxiliaryUserLabels": True}, tolerate_errors=True)
    user = ((data.get("data") or {}).get("user") or {}).get("result") or {}
    # A handle that does not exist comes back as `{"data": {}}` — no `user` key at all — and a
    # suspended one as `__typename: "UserUnavailable"`. Both are "cannot read this account".
    if not user or user.get("__typename") == "UserUnavailable":
        return None
    uid = str(user.get("rest_id") or "").strip()
    if not uid:
        return None

    core_ = user.get("core") or {}
    bio = user.get("profile_bio") or {}
    entities = bio.get("entities") or {}
    counts = user.get("relationship_counts") or {}
    site = (user.get("website") or {}).get("url") or ""
    return {
        "user_id": uid,
        "handle": core_.get("screen_name") or handle,
        "display_name": core_.get("name") or "",
        "bio": bio.get("description") or "",
        "website": _expand_tco(site, entities.get("url") or {}),
        # The OTHER homes a person lists — a Substack, a podcast, a personal site. Expanded and
        # de-t.co'd here; classifying them is the caller's job.
        "bio_urls": [u.get("expanded_url") or u.get("url") or ""
                     for u in ((entities.get("description") or {}).get("urls") or [])
                     if u.get("expanded_url") or u.get("url")],
        # Two separate facts on this response and the consumers only ever wanted one: a blue
        # check. `verification.verified` is the legacy-blue flag (false for nearly everyone now),
        # `is_blue_verified` the subscription. OR-ed, matching what twitterapi.io's
        # `isBlueVerified` meant to the callers that read it.
        "verified": bool(user.get("is_blue_verified")
                         or (user.get("verification") or {}).get("verified")),
        "followers": counts.get("followers"),
    }


# ── Tweets by id (TweetResultsByRestIds; FREE cookie-scrape) ──────────────────
# The one thing the cookie path genuinely could not do before: fetch a SOLO post. `TweetDetail`
# returns a CONVERSATION, and `reconstruct_chain` returns [] for a chain of one, so a standalone
# tweet was invisible to every keyless path. This fetches the post itself.
TWEETSBYIDS_OP = "TweetResultsByRestIds"
DEFAULT_TWEETSBYIDS_QID = ""      # empty -> discovery, like every other op here
# Measured 2026-08-30 off `x-rate-limit-limit`: 500 per 15 minutes, its own bucket — the most
# generous surface on this core, and more generous than the paid endpoint it replaces.
_TWEETSBYIDS_LIMIT_PER_WINDOW = 500
# X's per-request maximum. Deliberately NOT wrapped in a chunking loop: every caller today asks
# for exactly one id, so a loop would be code for a caller that does not exist. Passing more
# raises, because the alternative — letting X truncate the list — loses tweets silently.
_TWEETSBYIDS_MAX_IDS = 100


def fetch_tweets_by_ids(cookies: dict, headers: dict, ids: list) -> list[dict]:
    """Tweets by id -> normalized tweets (the `x_graphql._normalize` shape), misses DROPPED.

    X answers positionally — ask for four ids and get four entries, with a deleted or protected
    post as an empty `{}` in its slot. Those are dropped rather than returned as placeholders, so
    the result is "the tweets that exist" and a caller matches on `id` instead of on position.

    Raises ValueError past `_TWEETSBYIDS_MAX_IDS`; see that constant for why there is no loop."""
    from pipeline.ingestion import x_graphql as xg      # lazy: shared tweet normalizer
    from pipeline.ingestion import x_lists as xlists    # lazy: x_lists imports THIS module

    wanted = [str(i) for i in (ids or []) if str(i).strip()]
    if not wanted:
        return []
    if len(wanted) > _TWEETSBYIDS_MAX_IDS:
        raise ValueError(
            f"{TWEETSBYIDS_OP} takes at most {_TWEETSBYIDS_MAX_IDS} ids per request, got "
            f"{len(wanted)}. Chunk at the call site — no caller has needed it yet, so the loop "
            f"is not written here.")
    qid = resolve_query_id(TWEETSBYIDS_OP, cookies, default_seed=DEFAULT_TWEETSBYIDS_QID,
                           env_var="X_TWEETSBYIDS_QUERY_ID", page_url="https://x.com/home")
    data = graphql_get(
        TWEETSBYIDS_OP, qid,
        # `withVoice` and `withCommunity` are not optional extras: X rejects the request with a
        # 422 GRAPHQL_VALIDATION_FAILED naming them if either is absent.
        {"tweetIds": wanted, "includePromotedContent": False,
         "withVoice": True, "withCommunity": True},
        xlists.LISTS_FEATURES, headers,
        field_toggles={"withArticleRichContentState": True, "withArticlePlainText": False,
                       "withGrokAnalyze": False, "withDisallowedReplyControls": False},
        tolerate_errors=True)
    out: list[dict] = []
    for entry in ((data.get("data") or {}).get("tweetResult") or []):
        res = entry.get("result")
        norm = xg._normalize(res) if res else None
        if norm and norm.get("id"):
            out.append(norm)
    return out
