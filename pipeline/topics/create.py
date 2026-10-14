"""
pipeline/topics/create.py — flows: start following a topic, change it, stop it, list them.

Every check runs before the first write, so a refused call writes nothing. Handles in `people`
resolve against the store only (`lists.persons`), never the network. Creating a topic wakes the
daily pulls, whose first look at each watch reaches back 14 days (`frontier/cadence.py`).
"""
from __future__ import annotations

from pipeline.frontier import adapters, candidates
from pipeline.kb.writes import write_tx
from pipeline.timeparse import utc_iso

from . import lists, store

# Opyt cannot open a session; only the host's own scheduler can, and only it knows whether it has
# one. So the ask rides the answer, the same words on every client.
DAILY = ("Ask the user once, in one line, whether they want a daily update on their topics. If "
         "they say yes and you can create scheduled tasks, create ONE daily task covering every "
         "topic with this exact prompt: \"Call Opyt's latest tool, check what is new on my topics, "
         "and tell me what is worth my time.\" Skip the question if that task already exists. If "
         "you cannot schedule tasks, tell them you will mention new finds when they next talk to "
         "you.")


def create(conn, *, name: str | None, watches, words, people, now: str | None = None) -> dict:
    topic, problems = lists.draft(conn, name=name, watches=watches, words=words, people=people,
                                  every_source=list(adapters.adapters()))
    if problems:
        return _refused(problems)
    stamp = now or utc_iso()
    with write_tx(conn, site="topics.create"):
        topic_id = store.new_topic(conn, name=topic["name"], words=topic["words"],
                                   people=topic["people"], now=stamp)
        store.add_watches(conn, topic_id, topic["rows"], now=stamp)
    return {"status": "ok", "topic_id": topic_id, **_shown(conn, store.get(conn, topic_id)),
            "first": ("The first search of each watch runs in the background and looks back 14 "
                      "days; new finds and posts show up in latest()."),
            "next": DAILY}


def edit(conn, *, topic_id: str | None, name: str | None = None, watches=None,
         drop_watches=None, words=None, people=None, now: str | None = None) -> dict:
    """Add watches, drop watches by name, and replace the name, the words or the people: each
    only when given."""
    topic = store.get(conn, topic_id or "")
    if topic is None or topic["status"] != store.ACTIVE:
        return _refused([f"There is no topic {topic_id!r} being followed; watch(action=\"list\") "
                         "shows them."])
    rows, problems = lists.watch_rows(watches, list(adapters.adapters()))
    held = store.watches(conn, topic_id)
    dropped = set(lists.strings(drop_watches))
    if unknown := dropped - {w["name"] for w in held}:
        problems.append(f"Not a watch of this topic: {', '.join(sorted(unknown))}.")
    phrases, more = lists.phrases(words) if words is not None else (None, [])
    persons, still = lists.persons(conn, people) if people is not None else (None, [])
    problems += more + still
    if problems:
        return _refused(problems)
    stamp = now or utc_iso()
    with write_tx(conn, site="topics.edit"):
        store.update(conn, topic_id, now=stamp, name=(name or "").strip() or None,
                     words=phrases, people=persons)
        store.drop_watches(conn, topic_id, [w["query_id"] for w in held if w["name"] in dropped])
        store.add_watches(conn, topic_id, rows, now=stamp)
    return {"status": "ok", "topic_id": topic_id, **_shown(conn, store.get(conn, topic_id))}


def drop(conn, *, topic_id: str | None, now: str | None = None) -> dict:
    """Stop following the topic. Its watches stop unless another topic owns them; what it already
    brought into the library stays."""
    topic = store.get(conn, topic_id or "")
    if topic is None or topic["status"] != store.ACTIVE:
        return _refused([f"There is no topic {topic_id!r} being followed."])
    with write_tx(conn, site="topics.drop"):
        store.update(conn, topic_id, now=now or utc_iso(), status=store.DROPPED)
    return {"status": "ok", "topic_id": topic_id, "dropped": topic["name"]}


def listing(conn) -> dict:
    """Every topic being followed, with how many finds wait for the check."""
    return {"status": "ok", "topics": [{"topic_id": t["topic_id"], **_shown(conn, t)}
                                       for t in store.topics(conn)]}


def _shown(conn, topic: dict) -> dict:
    """A topic as the host reads it: its watches by name with their spellings, sources and, when
    they have some, context words."""
    held = store.watches(conn, topic["topic_id"])
    by_name: dict[str, dict] = {}
    for w in held:
        entry = by_name.setdefault(w["name"], {"name": w["name"], "spellings": [], "sources": []})
        entry["spellings"].append(w["text"])
        entry["sources"] = list(dict.fromkeys([*entry["sources"], *w["sources"]]))
        if w["context"]:
            entry["context"] = list(dict.fromkeys([*entry.get("context", []), *w["context"]]))
    return {"name": topic["name"], "watches": list(by_name.values()), "words": topic["words"],
            "people": [p["name"] for p in topic["people"]],
            "waiting": len(candidates.waiting(conn, [w["query_id"] for w in held]))}


def _refused(problems: list[str]) -> dict:
    return {"status": "refused", "reason": " ".join(problems)}
