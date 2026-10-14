"""
pipeline/topics/preview.py — flow: what a topic would have brought in over the last month, shown
before the user agrees to it. Writes nothing.

Each spelling is searched once on each of its sources, over the last 30 days, by the search
`look` also runs (`frontier/lanes.py`: the daily pulls' adapters and name check, one lane per
source, side by side, cut off at `BUDGET_S`), so the counts are what the topic will bring and a
spelling not searched in time says so. How many of the user's people's posts its words and people
match over the same month is counted from the store.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from pipeline.frontier import adapters, lanes
from pipeline.frontier.searching import PER_QUERY_LIMIT

from . import lists, posts

LOOKBACK_DAYS = 30
BUDGET_S = 40.0
EXAMPLES = 3

# The answer's instruction: what to show, and the one judgement the counts support.
PRESENT = (
    "Show the user, in plain words, each watch with how many finds it would have brought last "
    "month and an example or two, each with its `cite` link, then how many of their people's "
    "posts a day the words and people would have matched. A watch that found nothing names "
    "something quiet, which is fine. One that found more than about 30 in a month is busy: say "
    "you will check its finds before showing any, and suggest a narrower name only if this one "
    "names a category rather than a thing; if its examples on X are about something else, give "
    "it a `context` word. A source marked `unavailable` cannot be searched on this install now "
    "(X needs the user's X connected). When they agree, call watch(action=\"create\") with the "
    "same lists.")


def run(conn, *, name: str | None, watches, words, people, now: datetime,
        registry: dict | None = None) -> dict:
    registry = adapters.adapters() if registry is None else registry
    topic, problems = lists.draft(conn, name=name, watches=watches, words=words, people=people,
                                  every_source=list(registry))
    if problems:
        return {"status": "refused", "reason": " ".join(problems)}

    found = lanes.search(topic["rows"], registry, since=now - timedelta(days=LOOKBACK_DAYS),
                         now=now, limit=PER_QUERY_LIMIT, budget_s=BUDGET_S)
    matched = posts.recent(conn, topic, now, LOOKBACK_DAYS)
    return {"status": "preview", "name": topic["name"],
            "watches": _by_watch(topic["rows"], found),
            "words": topic["words"], "people": [p["name"] for p in topic["people"]],
            "posts_last_month": matched, "posts_per_day": round(matched / LOOKBACK_DAYS, 1),
            "next": PRESENT}


def _by_watch(rows: list[dict], found: dict) -> list[dict]:
    """One entry per watch name: its spellings, how many distinct finds last month, what each
    source came to, and a few of the finds."""
    out: dict[str, dict] = {}
    for row in rows:
        entry = out.setdefault(row["name"], {"name": row["name"], "spellings": [],
                                             "by_source": {}, "_finds": {}})
        entry["spellings"].append(row["text"])
        for source in row["sources"]:
            got = found.get((row["text"], source), lanes.UNCHECKED)
            if isinstance(got, str):
                entry["by_source"].setdefault(source, got)
                continue
            entry["_finds"].update({c.candidate_id: c for c in got})
            counted = entry["by_source"].get(source)
            entry["by_source"][source] = (counted if isinstance(counted, int) else 0) + len(got)
    for entry in out.values():
        finds = list(entry.pop("_finds").values())
        entry["found_last_month"] = len(finds)
        entry["examples"] = [lanes.card(c) for c in finds[:EXAMPLES]]
    return list(out.values())
