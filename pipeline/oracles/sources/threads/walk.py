"""Decide the next request in an author's self-thread from saved tweets and visited focals."""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.platforms.x.conversation import is_self_reply


@dataclass(frozen=True)
class Walk:
    tweets: tuple[dict, ...] = ()
    focused: tuple[str, ...] = ()
    gone: bool = False

    def next_focal(self, root: str) -> str | None:
        if self.gone:
            return None
        if not self.focused:
            return root
        parents = {str(t.get("inReplyToStatusId") or "") for t in self.tweets}
        return next((str(t["id"]) for t in self.tweets
                     if str(t["id"]) != root and str(t["id"]) not in self.focused
                     and str(t["id"]) not in parents and (t.get("replyCount") or 0) > 0), None)

    def saved(self) -> dict:
        return {"tweets": list(self.tweets), "focused": list(self.focused), "gone": self.gone}


def restore(saved: dict | None) -> Walk:
    saved = saved or {}
    return Walk(tuple(saved.get("tweets") or ()), tuple(saved.get("focused") or ()),
                bool(saved.get("gone")))


def advance(prior: Walk, root: str, author: str, focal: str, page: list[dict]) -> Walk:
    """Merge one successful response. Never re-focus a visited tweet; no total request cap.

    Only the root and replies by this author to themself in this conversation belong here.
    A response to another focal may omit the root; only the root's own response can say gone.
    """
    focused = tuple(dict.fromkeys((*prior.focused, focal)))
    if focal == root and not any(str(t.get("id")) == root for t in page):
        return Walk(prior.tweets, focused, gone=True)
    kept = {str(t["id"]): t for t in prior.tweets}
    for tweet in page:
        tid = str(tweet.get("id") or "")
        own = str((tweet.get("author") or {}).get("id") or "") == author
        continuation = (is_self_reply(tweet, author)
                        and str(tweet.get("conversationId") or "") == root)
        if tid and own and (tid == root or continuation):
            kept[tid] = tweet
    return Walk(tuple(kept[tid] for tid in sorted(kept, key=int)), focused)
