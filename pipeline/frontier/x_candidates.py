"""
pipeline/frontier/x_candidates.py — X as a Frontier source: post search
(`pipeline/platforms/x/search.py`) on the user's own session, mapped to `Candidate`s.

A post is shown by its author (the title, as an X atom's link is labelled once it is admitted) and
its text (the summary).
"""

from __future__ import annotations

import html
from datetime import datetime

from pipeline.frontier.candidates import MAX_SUMMARY_CHARS, Candidate
from pipeline.ingestion.x_render import parse_twitter_date
from pipeline.platforms import sessions
from pipeline.platforms.x import search as x_search


class XAdapter:
    """X's Top tab for the watch's name: English, no replies, since the window's first day, one
    page of 20, which caps a busy name ("x402", ~600 posts a day) at what X ranks first. Context
    words narrow a name that is also an idiom or another thing ("Chai-2" + antibody)."""
    slug = "x"
    host = "x.com"
    min_interval_s = 2.0               # the probes' gap; SearchTimeline's bucket is ~187 / 15 min
    min_lookback_days = 0              # `since:` windows on the post's date, which X indexes at once
    # `since:` takes any date, but the answer is one page of 20 from `Top` however wide the
    # window: over years that is twenty all-time-popular posts, not a survey.
    max_window_days = 30

    def available(self) -> bool:
        """Only on an install signed into X: search runs on the user's own session."""
        return x_search.available()

    def search(self, query: str, *, since: datetime | None, now: datetime, limit: int = 20,
               context=()) -> list[Candidate]:
        del now  # X windows only on the lower date bound.
        raw = raw_query(query, context, since)
        posts = x_search.search_posts(sessions.x_session(x_search.referer(raw)), raw)
        return [from_post(p) for p in posts][:limit]


def _bare(text: str) -> str:
    """A word or phrase with no double quote, which would end X's phrase early."""
    return text.replace('"', "").strip()


def raw_query(name: str, context, since: datetime | None) -> str:
    """X's search syntax for one watch: the name as a phrase, any one of its context words,
    English, no replies, since the window's first day."""
    words = [f'"{w}"' if " " in w else w for w in map(_bare, context or ()) if w]
    parts = [f'"{_bare(name)}"']
    if words:
        parts.append(words[0] if len(words) == 1 else f"({' OR '.join(words)})")
    parts += ["lang:en", "-filter:replies"]
    if since is not None:
        parts.append(f"since:{since:%Y-%m-%d}")
    return " ".join(parts)


def from_post(post: dict) -> Candidate:
    """A normalized post → its `Candidate`, keyed `x:{id}` like the atom it would become."""
    author = post["author"]
    posted = parse_twitter_date(post.get("createdAt") or "")
    return Candidate(candidate_id=f"x:{post['id']}",
                     source="x",
                     kind="x",
                     title=f"{author['name']} (@{author['userName']})",
                     url=post["url"],
                     published=posted.isoformat() if posted else None,
                     # X escapes &, < and > in a post's text.
                     summary=html.unescape(post.get("text") or "")[:MAX_SUMMARY_CHARS],
                     payload={"handle": author["userName"], "likes": post.get("likeCount", 0)})
