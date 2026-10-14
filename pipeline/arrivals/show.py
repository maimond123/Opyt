"""
pipeline/arrivals/show.py — `latest`'s answer: per topic the user follows, the finds its watches
brought that wait for the host's check and the user's people's new posts on it; then what was newly
published outside every topic, each atom with its link and why it is here; and, counted one line per
Oracle or kind, the older items that arrived with it (`fresh.py`).

Only a look at everything (no `topic_id`) marks the posts seen: a look at one topic must not hide
the others. A find is not marked seen: it waits until the host checks it (`watch(checked)`), so a
look that never records its check shows the same finds again.
"""
from __future__ import annotations

import json
from datetime import datetime

from pipeline.arrivals import fresh, grouping, provenance, window
from pipeline.frontier import candidates
from pipeline.kb.id_batches import rows_for
from pipeline.retrieval.cards import cite
from pipeline.topics import store as topics_store

# Finds handed over per topic in one look, and how much of each summary: enough to judge it by, and
# small enough that a busy topic still fits one reply. The rest wait for the next look.
FINDS = 20
SUMMARY_CHARS = 400

# What the host does with a look that holds finds or topic posts.
CHECK = (
    "Check every find and every topic post against its topic, from its title and summary, and keep "
    "only what is genuinely about the topic and worth the user's time. Show those under their "
    "topic, each with its `cite` link, and never mention what you passed over. Then call "
    "watch(action=\"checked\", keep=[...], passed=[...]) with the candidate_id of every find "
    "listed here (topic posts need nothing). A find titled \"(paid)\" cannot be stored, because "
    "Opyt keeps public posts only: show it with its link if it is worth it, but put it in "
    "`passed`. After showing them, offer once: \"Want to know what "
    "your people have said about these?\" If yes, for each, search(query=<what it is about>, "
    "entry_mode=[\"oracle-footprint\", \"user-saved\"]) and report only what they have said, with "
    "links; if none of them has said anything on it, say so. Never say what they would think.")


def show(conn, *, topic_id: str | None, limit: int, now: datetime) -> dict:
    followed = (topics_store.topics(conn) if topic_id is None
                else [topics_store.get(conn, topic_id)])
    if None in followed:
        return {"status": "error",
                "message": f"There is no topic {topic_id!r}; latest() with no topic_id shows "
                           "every topic that has something new."}
    since = window.start(conn, now)
    arrived = window.arrivals(conn, since)
    new = sorted((r for r in arrived if fresh.is_new(r)),
                 key=lambda r: (str(fresh.published(r) or r["arrived"][:10]), r["arrived"]),
                 reverse=True)                                  # newest published first
    page = grouping.cut(grouping.groups(conn, [r["atom_id"] for r in new], followed,
                                        unfiled=topic_id is None), limit)
    waiting = {t["topic_id"]: candidates.waiting(
        conn, [w["query_id"] for w in topics_store.watches(conn, t["topic_id"])])
        for t in followed}
    older = [] if topic_id is not None else [r for r in arrived if not fresh.is_new(r)]
    lines = fresh.older_lines(older, _oracles(conn, {r["who_id"] for r in older}))
    answer = _answer(since, followed, page, waiting, *_facts(conn, page), older=lines,
                     scoped=topic_id is not None)
    if topic_id is None:
        window.mark_seen(conn, now)
    return answer


def _answer(since, followed, page, waiting, atoms, names, *, older, scoped) -> dict:
    """The look as the host reads it. `new` counts each arrival once, though an atom two topics
    match is listed under both."""
    posts = {g["topic"]["topic_id"]: g for g in page if g["topic"]}
    out = {"since": since, "new": len({a for g in page for a in g["atom_ids"]}),
           "topics": [_topic(t, posts.get(t["topic_id"]), waiting[t["topic_id"]], atoms, names)
                      for t in followed if posts.get(t["topic_id"]) or waiting[t["topic_id"]]]}
    if unfiled := next((g for g in page if g["topic"] is None), None):
        out["unfiled"] = _posts(unfiled, atoms, names)
    if older:
        out["older"] = older
    if out["topics"]:
        out["next"] = CHECK
    elif not page:
        out["note"] = f"Nothing new was published since {since}."
    if any(len(g["listed"]) < len(g["atom_ids"]) for g in page):
        out["more"] = ("Not every post is listed: each `more` counts what it left out. "
                       + ("Call again with a larger `limit` for the rest." if scoped else
                          "This look marked all of it seen, so it will not be listed again; it "
                          "stays in the library, where search finds it."))
    return out


def _topic(topic, group, finds, atoms, names) -> dict:
    out = {"topic_id": topic["topic_id"], "name": topic["name"]}
    if finds:
        out["finds"] = [_find(row) for row in finds[:FINDS]]
        if len(finds) > FINDS:
            out["finds_more"] = (f"{len(finds) - FINDS} more finds wait; check these first, then "
                                 f"latest(topic_id=\"{topic['topic_id']}\") hands over the next.")
    if group:
        out["posts"] = _posts(group, atoms, names)
    return out


def _posts(group, atoms, names) -> dict:
    out = {"new": len(group["atom_ids"]),
           "atoms": [_card(atoms[a], names) for a in group["listed"]]}
    if more := len(group["atom_ids"]) - len(group["listed"]):
        out["more"] = more
    return out


def _find(row) -> dict:
    """A find as the host checks it: what it is, its link and summary, and which watch found it."""
    return {"candidate_id": row["candidate_id"], "kind": row["kind"], "title": row["title"],
            "when": (row["published"] or "")[:10] or None,
            "cite": cite(row["url"], row["title"], row["kind"], row["published"]),
            "summary": (row["summary"] or "")[:SUMMARY_CHARS], "found_by": row["found_by"]}


def _card(atom, names) -> dict:
    name = names.get(atom["who_id"])
    return {"atom_id": atom["atom_id"], "who": name or atom["who_id"],
            "when": (atom["when_ts"] or "")[:10] or None,
            "cite": cite(atom["source_url"], name, atom["source_type"], atom["when_ts"]),
            "description": atom["description"],
            "why": provenance.why(atom, names)}


# ── reads ───────────────────────────────────────────────────────────────────────

def _facts(conn, page) -> tuple[dict, dict]:
    """For the atoms the page lists: their rows, and the names of their authors and of whoever
    linked them."""
    listed = list(dict.fromkeys(a for group in page for a in group["listed"]))
    atoms = {r["atom_id"]: {**dict(r), "payload": json.loads(r["payload"] or "{}")}
             for r in rows_for(conn, "SELECT atom_id, who_id, when_ts, source_url, source_type, "
                                     "description, entry_mode, payload FROM atoms "
                                     "WHERE atom_id IN ({ids})", listed)}
    people = {a["who_id"] for a in atoms.values()} | {
        r["by"] for a in atoms.values() for r in a["payload"].get("referenced_by") or []}
    names = {r[0]: r[1] for r in rows_for(
        conn, "SELECT entity_id, name FROM entities WHERE entity_id IN ({ids}) AND name != ''",
        people)}
    return atoms, names


def _oracles(conn, who_ids) -> dict[str, tuple[str, str]]:
    """`fresh.oracle_names` over every member of the authors' clusters."""
    return fresh.oracle_names(
        (r["entity_id"], r["canonical_id"], r["name"]) for r in rows_for(
            conn, "SELECT DISTINCT m.entity_id, m.canonical_id, m.name FROM entities e "
                  "JOIN entities m ON m.entity_id = e.entity_id OR m.canonical_id = e.canonical_id "
                  "WHERE e.entity_id IN ({ids})", who_ids))
