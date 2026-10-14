"""Read the posts a conversation needs: one TweetDetail page and its chain, and tweets by id."""

import pipeline.platforms.x.records as x_records
from pipeline.platforms.x.request import XClient

# ── Conversation / thread context (TweetDetail; free cookie-scrape) ────────────
# A bookmarked reply needs the chain it replies to. TweetDetail returns it: ancestor
# tweets as top-level `tweet-` entries before the focal, in order; the author's own
# self-thread continuation in `conversationthread-` modules below.
TWEETDETAIL_OP = "TweetDetail"
DEFAULT_TWEETDETAIL_QID = ""      # empty → discovery (a guessed seed would 404 on rotation)
_CONVO_MAX_ANCESTORS = 20         # bound a pathologically deep reply chain (keep the nearest)
_CONVO_MAX_CONTINUATION = 25      # bound a long self-thread


def _convo_tweets(data: dict) -> tuple[list[dict], list[dict]]:
    """(top-level tweets in order, conversationthread module tweets in order), normalized to the
    `normalize_tweet` shape. Both live under threaded_conversation_with_injections_v2."""
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
                n = x_records.normalize_tweet(res) if res else None
                if n and n.get("id"):
                    top.append(n)
            elif eid.startswith("conversationthread-"):
                for it in c.get("items", []) or []:
                    res = ((((it.get("item") or {}).get("itemContent") or {})
                            .get("tweet_results") or {}).get("result"))
                    n = x_records.normalize_tweet(res) if res else None
                    if n and n.get("id"):
                        modules.append(n)
    return top, modules


def reconstruct_chain(tweets: list[dict], focal_id: str, *,
                      max_ancestors: int = _CONVO_MAX_ANCESTORS,
                      max_continuation: int = _CONVO_MAX_CONTINUATION) -> list[dict]:
    """From a flat, conversation-ordered tweet list → the chain the focal sits in:
    `[ancestors…, focal, self-thread…]`. Ancestors = tweets before the focal (the debate
    context). The self-thread = later self-replies by the focal's author to a post already in
    the chain; the author's replies to commenters are not part of it.
    Returns [] when there's no real context (len ≤ 1). Deduped, order-preserving."""
    focal_id = str(focal_id)
    idx = next((i for i, t in enumerate(tweets) if str(t.get("id")) == focal_id), None)
    if idx is None:
        return []                                    # focal not in the conversation → render solo
    focal = tweets[idx]
    author = str((focal.get("author") or {}).get("id") or "")
    ancestors = tweets[max(0, idx - max_ancestors):idx]
    in_chain = {str(t.get("id")) for t in ancestors} | {focal_id}
    continuation: list[dict] = []
    for t in tweets[idx + 1:]:
        if is_self_reply(t, author) and str(t.get("inReplyToStatusId") or "") in in_chain:
            continuation.append(t)
            in_chain.add(str(t.get("id")))
        if len(continuation) >= max_continuation:
            break
    chain, seen = [], set()
    for t in ancestors + [focal] + continuation:
        tid = str(t.get("id"))
        if tid and tid not in seen:
            seen.add(tid)
            chain.append(t)
    return chain if len(chain) > 1 else []


def is_self_reply(t: dict, author_id: str) -> bool:
    """A reply by the author to themself, never a reply to a commenter."""
    return (str((t.get("author") or {}).get("id") or "") == author_id
            and bool(t.get("isReply"))
            and str(t.get("inReplyToUserId") or "") == author_id)


def fetch_thread_page(focal_id: str, client: XClient, *, rich: bool = True) -> list[dict]:
    """One TweetDetail request, with its normalized tweets in conversation order.

    A malformed response raises instead of completing an owed conversation as empty. Rate and
    session refusals propagate to the quota loop. `rich` supplies the focal article when the
    caller does not already have the Bookmarks feed's copy.
    """
    qid = client.resolve_query_id(TWEETDETAIL_OP,
                                  default_seed=DEFAULT_TWEETDETAIL_QID,
                                  env_var="X_TWEETDETAIL_QUERY_ID",
                                  page_url="https://x.com/home")
    variables = {
        "focalTweetId": str(focal_id), "with_rux_injections": False,
        "includePromotedContent": False, "withCommunity": True,
        "withQuickPromoteEligibilityTweetFields": True, "withBirdwatchNotes": True,
        "withVoice": True, "withV2Timeline": True,
    }
    options = {"field_toggles": {"withArticleRichContentState": True,
                                  "withArticlePlainText": False}} if rich else {}
    data = client.graphql_get(TWEETDETAIL_OP, qid, variables, x_records.TIMELINE_FEATURES,
                              tolerate_errors=True, **options)
    timeline = ((data.get("data") or {})
                .get("threaded_conversation_with_injections_v2"))
    if not isinstance(timeline, dict) or not isinstance(timeline.get("instructions"), list):
        raise x_records.XUnrecognizedPage("TweetDetail response has no conversation instructions")
    top, modules = _convo_tweets(data)
    return top + modules


def fetch_conversation(focal_id: str, client: XClient) -> list[dict]:
    """One page's ancestors and self-thread for a foreground saved post.

    The caller holds the focal post's rich copy already. Background thread walks use
    `fetch_thread_page` directly and persist their position after each request.
    """
    return reconstruct_chain(fetch_thread_page(focal_id, client, rich=False), focal_id)


# ── Tweets by id (TweetResultsByRestIds; free cookie-scrape) ──────────────────
# Fetches a solo post, which `TweetDetail` + `reconstruct_chain` cannot (a chain of one is []).
TWEETSBYIDS_OP = "TweetResultsByRestIds"
DEFAULT_TWEETSBYIDS_QID = ""      # empty -> discovery, like every other op here
# X's per-request maximum. Passing more raises rather than letting X truncate the list silently.
TWEETSBYIDS_MAX_IDS = 100


def fetch_tweets_by_ids(client: XClient, ids: list) -> list[dict]:
    """Tweets by id -> normalized tweets (the `normalize_tweet` shape), misses dropped.

    X answers positionally, with a deleted or protected post as `{}` in its slot; those are
    dropped, so callers match on `id`, not position. Raises ValueError past `TWEETSBYIDS_MAX_IDS`."""

    wanted = [str(i) for i in (ids or []) if str(i).strip()]
    if not wanted:
        return []
    if len(wanted) > TWEETSBYIDS_MAX_IDS:
        raise ValueError(
            f"{TWEETSBYIDS_OP} takes at most {TWEETSBYIDS_MAX_IDS} ids per request, got "
            f"{len(wanted)}. Chunk at the call site — no caller has needed it yet, so the loop "
            f"is not written here.")
    qid = client.resolve_query_id(TWEETSBYIDS_OP,
                                  default_seed=DEFAULT_TWEETSBYIDS_QID,
                                  env_var="X_TWEETSBYIDS_QUERY_ID",
                                  page_url="https://x.com/home")
    data = client.graphql_get(
        TWEETSBYIDS_OP, qid,
        # `withVoice` and `withCommunity` are not optional extras: X rejects the request with a
        # 422 GRAPHQL_VALIDATION_FAILED naming them if either is absent.
        {"tweetIds": wanted, "includePromotedContent": False,
         "withVoice": True, "withCommunity": True},
        x_records.TIMELINE_FEATURES,
        field_toggles={"withArticleRichContentState": True, "withArticlePlainText": False,
                       "withGrokAnalyze": False, "withDisallowedReplyControls": False},
        tolerate_errors=True)
    entries = (data.get("data") or {}).get("tweetResult")
    if not isinstance(entries, list):
        raise x_records.XUnrecognizedPage("TweetResultsByRestIds response has no result list")
    out: list[dict] = []
    for entry in entries:
        res = entry.get("result")
        norm = x_records.normalize_tweet(res) if res else None
        if norm and norm.get("id"):
            out.append(norm)
    return out
