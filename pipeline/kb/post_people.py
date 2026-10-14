"""
pipeline/kb/post_people.py — who an Oracle's post engages with: the people it quotes, @mentions
or links to, each with the Oracle's own words.

A decision over one stitched group of normalized tweets (`platforms/x/records.normalize_tweet`).
It runs at ingest, where the structured post is still in hand: the stored snapshot is markdown,
and parsing the Oracle's words back out of it leaks the quoted post's lines. The candidates' mention
arm (`oracles/mentions.py`) reads the result from `payload.engages`.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pipeline.ingestion.x_render import expand_urls, is_quote
from pipeline.kb.link_router import parse_tweet_id

# How strongly each kind says "this Oracle engaged with this person". One entry per person per post
# keeps the strongest kind, so the order matters.
KINDS = ("quote", "link", "mention")

# The Oracle's words per entry, cut on a word boundary. Enough for a host to read the stance.
WORDS_LIMIT = 500


def _words(tweet: dict) -> str:
    """The tweet's own text with links expanded and the quote's trailing link removed, collapsed
    and capped. The quoted post is a separate object, so it never appears here."""
    text = " ".join(expand_urls(tweet.get("text") or "", tweet.get("entities"),
                                quoting=is_quote(tweet)).split())
    if len(text) <= WORDS_LIMIT:
        return text
    cut = text.rfind(" ", 0, WORDS_LIMIT)
    return text[:cut if cut > 0 else WORDS_LIMIT].rstrip() + "…"


def _tweet_url(tweet: dict) -> str:
    handle = (tweet.get("author") or {}).get("userName") or "i"
    return tweet.get("url") or f"https://x.com/{handle}/status/{tweet.get('id')}"


def _named(tweet: dict) -> list[tuple[str, str | None, str]]:
    """Everyone one tweet names → `(kind, x id or None, handle)`, the Oracle included."""
    out: list[tuple[str, str | None, str]] = []
    qt = tweet.get("quoted_tweet")
    quoted_id = None
    if isinstance(qt, dict):
        qa = qt.get("author") or {}
        quoted_id = str(qt.get("id") or "")
        if qa.get("userName"):
            out.append(("quote", str(qa.get("id") or "") or None, qa["userName"]))
    entities = tweet.get("entities") or {}
    for m in entities.get("user_mentions") or []:
        if m.get("screen_name"):
            out.append(("mention", str(m.get("id_str") or "") or None, m["screen_name"]))
    for u in entities.get("urls") or []:
        url = u.get("expanded_url") or ""
        tid = parse_tweet_id(url)
        if not tid or tid == quoted_id:          # the quote's own link is the quote
            continue
        handle = urlparse(url).path.strip("/").split("/")[0]
        if handle and handle.lower() != "i":     # x.com/i/status/… names no one
            out.append(("link", None, handle))
    return out


def post_people(group: list[dict]) -> list[dict]:
    """One entry per person the group names, other than its author:
    `{kind, who, handle, words, url}`. `kind` is the strongest way any tweet of the group named
    them (quote > link > mention), and `words`/`url` are that tweet's. `who` is `x:user:<id>`
    when X gave the id (quotes and mentions), else None (a link carries only a handle)."""
    if not group:
        return []
    author = group[0].get("author") or {}
    self_id, self_handle = str(author.get("id") or ""), (author.get("userName") or "").lower()
    best: dict[str, dict] = {}
    for tweet in group:
        for kind, uid, handle in _named(tweet):
            key = handle.lower()
            if key == self_handle or (uid and uid == self_id):
                continue
            entry = best.get(key)
            if entry is None or KINDS.index(kind) < KINDS.index(entry["kind"]):
                who = f"x:user:{uid}" if uid else (entry or {}).get("who")
                best[key] = {"kind": kind, "who": who, "handle": handle,
                             "words": _words(tweet), "url": _tweet_url(tweet)}
            elif uid and not entry["who"]:
                entry["who"] = f"x:user:{uid}"
    return list(best.values())
