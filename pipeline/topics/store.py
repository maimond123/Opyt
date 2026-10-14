"""
pipeline/topics/store.py — the topic tables, read and written.

A topic is a subject the user follows in `user_topics`:
its name, and the words and people that pick out the user's own people's posts on it. Its watches
search outside the store. A watch names one thing, and each of its spellings is one standing query
(`frontier_queries`, which the daily pulls search), with the watch's context words for X
(`watch_context`); `topic_watches` says which topic owns each query, under the watch's name. A
query runs while an active topic owns it (`frontier/queries.active_queries`).

Writers here do not commit: the flow that calls them owns the transaction.
"""
from __future__ import annotations

import json
import secrets

from pipeline.frontier import chain
from pipeline.frontier import queries as fq
from pipeline.kb.writes import after_commit

ACTIVE, DROPPED = "active", "dropped"


def _topic(row) -> dict:
    topic = dict(row)
    topic["words"] = json.loads(topic["words"] or "[]")
    topic["people"] = json.loads(topic["people"] or "[]")
    return topic


def new_topic(conn, *, name: str, words: list[str], people: list[str], now: str) -> str:
    """Insert a topic and return its id."""
    topic_id = f"t_{secrets.token_hex(4)}"
    conn.execute(
        "INSERT INTO user_topics (topic_id, name, words, people, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (topic_id, name, json.dumps(words), json.dumps(people), ACTIVE, now, now))
    return topic_id


def get(conn, topic_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM user_topics WHERE topic_id = ?", (topic_id,)).fetchone()
    return _topic(row) if row else None


def topics(conn, *, status: str | None = ACTIVE) -> list[dict]:
    """Topics in `status` (every topic when None), oldest first."""
    sql, params = "SELECT * FROM user_topics", ()
    if status is not None:
        sql, params = sql + " WHERE status = ?", (status,)
    return [_topic(r) for r in conn.execute(sql + " ORDER BY created_at, topic_id", params)]


def update(conn, topic_id: str, *, now: str, name: str | None = None,
           words: list[str] | None = None, people: list[str] | None = None,
           status: str | None = None) -> None:
    """Replace the fields given; the rest stay."""
    fields = {"name": name, "status": status,
              "words": None if words is None else json.dumps(words),
              "people": None if people is None else json.dumps(people)}
    given = {**{k: v for k, v in fields.items() if v is not None}, "updated_at": now}
    sets = ", ".join(f"{k} = ?" for k in given)
    conn.execute(f"UPDATE user_topics SET {sets} WHERE topic_id = ?", (*given.values(), topic_id))


def watches(conn, topic_id: str) -> list[dict]:
    """The topic's standing queries: `{name, query_id, text, sources, context}`, in the order
    added."""
    return [{"name": r["name"], "query_id": r["query_id"], "text": r["text"],
             "sources": json.loads(r["target_sources"] or "[]"),
             "context": json.loads(r["context"] or "[]")}
            for r in conn.execute(
                "SELECT w.name, q.query_id, q.text, q.target_sources, c.words AS context "
                "FROM topic_watches w JOIN frontier_queries q ON q.query_id = w.query_id "
                "LEFT JOIN watch_context c ON c.query_id = w.query_id WHERE w.topic_id = ? "
                "ORDER BY w.added_at, w.rowid", (topic_id,))]


def add_watches(conn, topic_id: str, rows: list[dict], *, now: str) -> None:
    """Write each `{name, text, sources, context, query_id}` as a standing query the topic owns,
    with its context words when it has some, and wake the daily pulls once the caller's
    transaction commits."""
    if not rows:
        return
    for row in rows:
        fq.add(conn, row["text"], row["sources"], now=now)
        if row["context"]:
            fq.set_context(conn, row["query_id"], row["context"])
    conn.executemany("INSERT OR IGNORE INTO topic_watches (topic_id, query_id, name, added_at) "
                     "VALUES (?, ?, ?, ?)",
                     [(topic_id, row["query_id"], row["name"], now) for row in rows])
    after_commit(conn, chain.queries_added)


def drop_watches(conn, topic_id: str, query_ids: list[str]) -> None:
    """Remove the topic's ownership of these queries. A query no active topic owns stops running;
    its row, watermarks and finds stay, so asking for the same name again resumes where it left
    off and never re-shows a find the user already saw."""
    conn.executemany("DELETE FROM topic_watches WHERE topic_id = ? AND query_id = ?",
                     [(topic_id, q) for q in query_ids])
