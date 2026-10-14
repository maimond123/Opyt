"""
pipeline/frontier/substack_candidates.py — Substack as a Frontier source: post search
(`pipeline/platforms/substack/search.py`), signed out, mapped to `Candidate`s.

A post is shown by its publication (the title, as a Substack atom's link is labelled once it is
admitted) and by what it says: its title and subtitle, and the passage that names the watch.
"""

from __future__ import annotations

from datetime import datetime

from pipeline.frontier import intake
from pipeline.frontier.candidates import MAX_SUMMARY_CHARS, Candidate
from pipeline.platforms.substack import search as substack_search
from pipeline.timeparse import parse_ts

# The search matches word by word and its results carry no body, so a post whose title,
# subtitle and description do not name the watch is read to find where it does: at most this many
# per search. A multi-word name brought 20 results of which 3 named it.
MENTIONS = 10
# Characters each side of the name: the passage follows the post's title and subtitle, and the
# whole fits the 400 characters `latest` and `look` show of a find.
PASSAGE = 150
PAGES = 2                              # 20 posts a page


class SubstackAdapter:
    """Substack's post search for the watch's name, unquoted (quoting drops the date window),
    within the narrowest of its windows that holds the pull's, up to two pages; then only the
    posts published since the window's start."""
    slug = "substack"
    host = "substack.com"
    min_interval_s = 1.5               # the probes' gap
    min_lookback_days = 0              # `dateRange` windows on the post's date
    # `dateRange` is day, week or month and nothing wider, so a longer window would search a
    # month and then drop every older post below — zero finds, silently.
    max_window_days = 30

    def search(self, query: str, *, since: datetime | None, now: datetime, limit: int = 40,
               context=()) -> list[Candidate]:
        del context  # Substack's search has no use for X's context words.
        days = (now - since).total_seconds() / 86400 if since is not None else 30
        window = substack_search.range_for(days)
        found: dict[str, dict] = {}
        for page in range(PAGES):
            posts, more = substack_search.search_posts(query, date_range=window, page=page)
            found.update((p["id"], p) for p in posts)
            if not more:
                break
        recent = [p for p in found.values() if since is None or _since(p, since)][:limit]
        out, reads = [], 0
        for post in recent:
            passage = None
            if not intake.names(query, _head(post)) and reads < MENTIONS:
                reads += 1
                passage = intake.mention(query, substack_search.post_text(post["url"]),
                                         width=PASSAGE)
            out.append(from_post(post, passage))
        return out


def _head(post: dict) -> str:
    return "\n".join((post["title"], post["subtitle"], post["description"]))


def _since(post: dict, since: datetime) -> bool:
    """Published at or after `since`; a post with no readable date is kept, never dropped."""
    published = parse_ts(post["published"])
    return published is None or published >= since


def from_post(post: dict, passage: str | None) -> Candidate:
    """A search result → its `Candidate`, keyed `substack:{post id}` like the atom it would
    become. A paid post says so in its title: the library stores public posts only."""
    lead = " — ".join(dict.fromkeys(t for t in (post["title"], post["subtitle"],
                                                post["description"]) if t))
    by = post["publication"] or post["author"] or "Substack"
    return Candidate(candidate_id=f"substack:{post['id']}",
                     source="substack",
                     kind="substack",
                     title=f"{by} (paid)" if post["paid"] else by,
                     url=post["url"],
                     published=post["published"] or None,
                     summary="\n".join(t for t in (lead, passage) if t)[:MAX_SUMMARY_CHARS],
                     payload={"publication": post["publication"], "author": post["author"],
                              "paid": post["paid"]})
