"""
pipeline/frontier/look.py — flow: search outside the library now for a few named things, on the
watch sources (X, Substack, arXiv, OpenAlex, GitHub), and hand back what each found, with links
and the `hopper` arguments that would save it. Reads nothing of the store and writes nothing.

The search is the one a topic's preview runs (`lanes`), through the daily pulls' adapters and
name check, so `look` and a watch on the same name find the same things.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from pipeline.frontier import adapters, cadence, lanes
from pipeline.frontier import queries as fq

MAX_NAMES = 3
# No ceiling on how far back to ask: a researcher's back catalogue is a real
# window, each source narrows the ask to what it can serve (`lanes.window_days`), and `limit`
# bounds what any window can return.
DAYS = 7
BUDGET_S = 40.0
ASKED = 20          # results asked of each source per spelling
SHOWN = 8           # results shown per source per name, newest first; the rest are counted

_SUBSTACK_HOST = re.compile(r"^https?://([^/]+\.)?substack\.com/", re.I)

# What the host does with the answer.
PRESENT = (
    "These were found just now, outside the user's library, by searching each name over the "
    "last {days} days. None of it comes from the user's own people: never present a result as "
    "theirs or as something their people said. Read every result and show the user only what is "
    "worth their time, each with its `cite` link; say nothing about the rest. A source marked "
    "`unavailable` cannot be searched on this install now (X needs the user's X connected); "
    "`failed` or `unchecked` means it was not searched this time. To save a result, call hopper "
    "with its `save` arguments; a `paid` post cannot be saved, because Opyt keeps public posts "
    "only. After showing them, offer in one line to follow the name with watch, so new finds on "
    "it come to the user every day. If they ask what their own people have said about it, call "
    "search(query=<the name>, entry_mode=[\"oracle-footprint\", \"user-saved\"]).")

# Appended when a source could not be asked as far back as the user wanted.
SHORTER = (
    "\n\nThese sources cannot be searched that far back and were searched over less: {sources}. "
    "So do not tell the user there is nothing on one of them over the whole window — you did not "
    "look. Say what you searched where, in your own plain words, only if it changes what they "
    "should make of the answer.")


def run(*, names, sources=None, days=None, now: datetime, registry: dict | None = None) -> dict:
    registry = adapters.adapters() if registry is None else registry
    rows, problems = asks(names, sources, list(registry))
    days, more = window(days)
    if problems := problems + more:
        return {"status": "refused", "reason": " ".join(problems)}
    found = lanes.search(rows, registry, since=now - timedelta(days=days), now=now, limit=ASKED,
                         budget_s=BUDGET_S)
    short = narrowed(rows, registry, days)
    answer = {"status": "ok", "days": days,
              "names": by_name(rows, found, newest_first=not cadence.landscape(days)),
              "next": PRESENT.format(days=days)}
    if short:
        answer["searched_less"] = short
        answer["next"] += SHORTER.format(
            sources=", ".join(f"{s} over {d} days" for s, d in sorted(short.items())))
    return answer


def asks(raw, sources, every_source) -> tuple[list[dict], list[str]]:
    """`([{name, text, sources, context}], problems)`: one row per spelling of each named thing,
    its name first, searched on `sources` (every source when none is given)."""
    items = _list(raw)
    chosen = _strings(sources) or list(every_source)
    problems = []
    if not 1 <= len(items) <= MAX_NAMES:
        problems.append(f"Name 1 to {MAX_NAMES} things to look for (got {len(items)}).")
    if bad := [s for s in chosen if s not in every_source]:
        problems.append(f"Unknown sources {', '.join(bad)}; look searches "
                        f"{', '.join(every_source)}.")
    rows = []
    for i, item in enumerate(items[:MAX_NAMES]):
        item = {"name": item} if isinstance(item, str) else item if isinstance(item, dict) else {}
        name = item.get("name").strip() if isinstance(item.get("name"), str) else ""
        texts = fq.spellings(name, _strings(item.get("spellings")))
        if not texts or texts[0] != name:
            problems.append(f"names[{i}] needs a name.")
            continue
        rows += [{"name": name, "text": text, "sources": chosen,
                  "context": _strings(item.get("context"))} for text in texts]
    return rows, problems


def window(days) -> tuple[int, list[str]]:
    """`(days, problems)`: how far back to look, `DAYS` unless given."""
    if days is None:
        return DAYS, []
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        return DAYS, [f"`days` is a whole number of 1 or more (got {days!r})."]
    return days, []


def narrowed(rows: list[dict], registry: dict, days: int) -> dict[str, int]:
    """The searched sources that cannot be asked as far back as `days`, each with the window it
    was searched over instead. Empty when every source served the whole ask."""
    asked = {s for row in rows for s in row["sources"]}
    got = {s: int(lanes.window_days(registry[s], days)) for s in asked}
    return {s: d for s, d in got.items() if d < days}


def by_name(rows: list[dict], found: dict, *, newest_first: bool = True) -> list[dict]:
    """Per name: what each source came to (how many distinct finds, or why it has none), and the
    finds themselves, `SHOWN` per source.

    Newest first for a window short enough that every find is news. Over a long window the
    sources' own order is kept instead: asked for a landscape, OpenAlex answers most-cited first
    (`openalex_candidates.sort_for`) and GitHub by stars, and re-sorting that page by date would
    cut the most-cited work out of the `SHOWN` the host ever sees.
    """
    out: dict[str, dict] = {}
    for row in rows:
        entry = out.setdefault(row["name"], {"name": row["name"], "by_source": {}, "_finds": {}})
        for source in row["sources"]:
            got = found.get((row["text"], source), lanes.UNCHECKED)
            if isinstance(got, str):
                entry["by_source"].setdefault(source, got)
            else:
                entry["_finds"].setdefault(source, {}).update({c.candidate_id: c for c in got})
    for entry in out.values():
        entry["results"] = {}
        for source, held in entry.pop("_finds").items():
            shown = list(held.values())
            if newest_first:
                shown.sort(key=lambda c: c.published or "", reverse=True)
            entry["by_source"][source] = len(shown)
            if shown:
                entry["results"][source] = [_result(c) for c in shown[:SHOWN]]
    return list(out.values())


def _result(c) -> dict:
    """A find with what saving it takes. A custom-domain Substack post needs `kind_hint`: its
    address says nothing about Substack."""
    if (c.payload or {}).get("paid"):
        return {**lanes.card(c), "paid": True}
    save = {"reference": c.url, "confirm": True}
    if c.source == "substack" and not _SUBSTACK_HOST.match(c.url or ""):
        save["kind_hint"] = "substack"
    return {**lanes.card(c), "save": save}


def _list(value) -> list:
    return value if isinstance(value, list) else [] if value is None else [value]


def _strings(value) -> list[str]:
    return [v.strip() for v in _list(value) if isinstance(v, str) and v.strip()]
