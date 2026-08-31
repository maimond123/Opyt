"""
pipeline/ingestion/x_likes.py
Local-session X Likes reader — the AUTHORS of the tweets you've liked, emitted as a
content-aggregate candidate signal for oracle onboarding.

Repeated likes of one account signal "worth my attention," weaker than List membership
(your own deliberate grouping) but real — Tier-3 in the acquisition ladder (noisiest and
priciest to gather, pulled last).

Cookie-scrape, and there is no alternative: no paid provider ever exposed a likes-read
endpoint, and X made the
Likes tab private in June 2024 (served only to the owner's own authenticated session), so
reading your own likes requires the same GraphQL cookie-scrape path as x_lists.py.

Likes is a plain tweet timeline (like Bookmarks): walk it, take each unique liked tweet's
AUTHOR, count how many of your likes that author earned (`liked_count`), write the
candidate signal. Output: state/candidate_signals_x_likes.json.

Response shape: op `Likes`, root `data.user.result.timeline.timeline` (not `timeline_v2`),
tweet entries at `content.itemContent.tweet_results.result` — see
"""

import argparse
import json
import time
from pathlib import Path

from pipeline.ingestion import x_graphql_core as core
from pipeline.ingestion import x_lists as xlists   # reuse _normalize_user + the feature bundle
from pipeline.ingestion.utils import log, SyncAuthError

LIKES_OP = "Likes"

# Empty on purpose: resolve_query_id returns a non-empty seed WITHOUT validating it, so a
# guessed seed would 404 forever on rotation; empty routes to live JS-bundle discovery instead.
DEFAULT_LIKES_QID = ""

_DISCOVER_PAGE = "https://x.com/home"   # any authed page harvests the shared JS bundles
_REFERER = "https://x.com/i/likes"
DEFAULT_PAGE_SIZE = 100

# X timelines hand back a FRESH bottom cursor on every page forever (the infinite
# scroll), so the real terminator is "this page added no NEW tweet", not the cursor.
# MAX_PAGES is a runaway backstop only — hitting it logs LOUD and stops (never silently
# under-fetch, never hammer X into a 429). Same guard as x_lists.
MAX_PAGES = 100

# Likes shares the web client's timeline feature bundle with Lists (both captured off the
# same 2026-07-09 client). Reference it rather than duplicate 40 drift-prone lines; if a
# live Likes capture shows a different set, override via $X_LIKES_FEATURES (JSON).
LIKES_FEATURES = xlists.LISTS_FEATURES


def _features() -> dict:
    import os
    override = os.getenv("X_LIKES_FEATURES")
    return json.loads(override) if override else LIKES_FEATURES


# ── Tweet-timeline parse (mirror of x_graphql._parse_timeline, root key swapped) ──────

def _parse_timeline(data: dict) -> tuple[list[dict], str | None]:
    """Pull tweet `result` objects + the bottom cursor out of a Likes response. Root key
    `data.user.result.timeline.timeline` (confirmed live 2026-07-10); tweet entries carry
    `entryId` starting `tweet-` with the tweet at
    `content.itemContent.tweet_results.result`; the paging cursor is the entry whose
    `content.cursorType == "Bottom"`."""
    timeline = ((((data.get("data") or {}).get("user") or {}).get("result") or {})
                .get("timeline") or {}).get("timeline", {})
    tweets: list[dict] = []
    next_cursor: str | None = None
    for ins in timeline.get("instructions", []):
        for e in ins.get("entries", []):
            entry_id = e.get("entryId", "")
            content = e.get("content", {}) or {}
            if content.get("cursorType") == "Bottom":
                next_cursor = content.get("value")
                continue
            if entry_id.startswith("tweet-"):
                result = ((content.get("itemContent") or {})
                          .get("tweet_results") or {}).get("result")
                if result:
                    tweets.append(result)
    return tweets, next_cursor


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
    return xlists._normalize_user(user)


# ── Page loop (copy of x_lists.fetch_list_members' terminator; unit = the tweet) ──────

def fetch_liked_authors(viewer_id: str, cookies: dict, headers: dict) -> list[dict]:
    """Walk the viewer's Likes timeline, returning one normalized author per UNIQUE liked
    tweet (duplicates preserved across DIFFERENT tweets by the same author, so the caller
    can count). Dedups by tweet id (X repeats tweets with an ever-advancing cursor) and
    stops the page loop the instant a page yields no new tweet — the only termination that
    holds against a cursor that never goes empty."""
    qid = core.resolve_query_id(LIKES_OP, cookies, default_seed=DEFAULT_LIKES_QID,
                                env_var="X_LIKES_QUERY_ID", page_url=_DISCOVER_PAGE)
    authors: list[dict] = []
    seen_tweets: set[str] = set()
    cursor: str | None = None
    seen_cursor: set[str] = set()
    for _page_no in range(1, MAX_PAGES + 1):
        variables = {
            "userId": viewer_id, "count": DEFAULT_PAGE_SIZE,
            "includePromotedContent": False, "withClientEventToken": False,
            "withBirdwatchNotes": False, "withVoice": True, "withV2Timeline": True,
        }
        if cursor:
            variables["cursor"] = cursor
        data = core.graphql_get(LIKES_OP, qid, variables, _features(), headers,
                                tolerate_errors=True)
        tweets, next_cursor = _parse_timeline(data)
        new = 0
        for t in tweets:
            tid = _tweet_id(t)
            if not tid or tid in seen_tweets:      # dedup across pages by tweet
                continue
            seen_tweets.add(tid)
            new += 1
            author = _tweet_author(t)              # count the tweet even if authorless
            if author:
                authors.append(author)
        # No NEW tweet this page → end of your likes (the cursor keeps advancing forever).
        if new == 0:
            break
        if not next_cursor or next_cursor in seen_cursor:
            break
        seen_cursor.add(next_cursor)
        cursor = next_cursor
    else:
        log(f"[x-likes] hit MAX_PAGES={MAX_PAGES} paging likes — stopping "
            f"(partial: {len(seen_tweets)} liked tweet(s) so far).")
    return authors


# ── Aggregate + write the signal ──────────────────────────────────────────────────────

def _signal_path(config=None) -> Path:
    from pipeline.config import state_paths
    return (config or state_paths()).state_file("candidate_signals_x_likes")


def aggregate_authors(authors: list[dict], viewer_id: str) -> list[dict]:
    """Fold the per-liked-tweet author list into deduped candidates. Dedup by user_id,
    accumulate a `liked_count` (how many of your likes this author earned — the action
    strength), drop the viewer themselves. Sorted by liked_count then handle. Pure — no
    IO, so it's directly unit-testable against captured fixtures."""
    by_user: dict[str, dict] = {}
    for a in authors:
        if not a or a["user_id"] == viewer_id:     # never treat yourself as a candidate
            continue
        rec = by_user.get(a["user_id"])
        if rec is None:
            rec = {**a, "liked_count": 0}
            by_user[a["user_id"]] = rec
        rec["liked_count"] += 1
    return sorted(by_user.values(),
                  key=lambda c: (-c["liked_count"], c["handle"].lower()))


def sync_likes(profile: str | None = None, dry_run: bool = False,
               config=None) -> dict:
    """Pull the viewer's liked-tweet authors → deduped candidate signal. Raises
    SyncAuthError if the session is dead (caller records a broken source, never a silent
    0). Fail-safe: no twid → skip (likes are viewer-scoped; without the viewer id there's
    nothing to scope to)."""
    cookies = core.read_x_cookies(profile=profile)
    vid = core.viewer_id(cookies)
    if not vid:
        log("[x-likes] twid cookie missing — likes are viewer-scoped and can't be read "
            "without the viewer id; skipping (fail-safe, no crash).")
        return {"liked_tweets": 0, "candidates": 0, "skipped": "no_viewer_id"}

    headers = core.auth_headers(cookies, referer=_REFERER)
    authors = fetch_liked_authors(vid, cookies, headers)
    log(f"[x-likes] {len(authors)} liked tweet(s) with an extractable author")

    candidates = aggregate_authors(authors, vid)
    result = {"liked_tweets": len(authors), "candidates": len(candidates)}

    if dry_run:
        log(f"[x-likes] DRY RUN — {len(candidates)} candidate(s); not writing.")
        for c in candidates[:15]:
            log(f"    @{c['handle']:<20} liked×{c['liked_count']}  "
                f"({c['followers_count']} followers)")
        result["preview"] = candidates[:15]
        return result

    payload = {
        "signal": "x_liked_author",
        "viewer_id": vid,
        "captured_at": int(time.time()),
        "candidates": candidates,
    }
    path = _signal_path(config)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    log(f"[x-likes] wrote {len(candidates)} candidate(s) → {path}")
    result["written"] = len(candidates)
    result["path"] = str(path)
    return result


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Ingest the authors of the user's X likes "
                                             "as a candidate signal.")
    ap.add_argument("--profile", help="Chrome profile dir / browser key (else auto-pick)")
    ap.add_argument("--dry-run", action="store_true", help="Print candidates, do not write")
    args = ap.parse_args()
    try:
        out = sync_likes(profile=args.profile, dry_run=args.dry_run)
        log(f"[x-likes] done: {json.dumps({k: v for k, v in out.items() if k != 'preview'})}")
    except SyncAuthError as e:
        log(f"[x-likes] NOT LOGGED IN / session dead: {e}")
        raise SystemExit(2)


if __name__ == "__main__":
    _cli()
