"""
pipeline/frontier/queries.py — the standing queries the crawler searches: their identity, their
context words, and which of them run.

A standing query is one spelling of a topic's watch: `pipeline/topics/store.py` writes the query
here and the `topic_watches` row naming the topic that owns it. Identity is `normalized`
(lowercase, whitespace-collapsed), hashed into `query_id`, so the same words asked twice land on
one row and the pair watermarks keyed on `query_id` (`pulls.py`) never reset. A query runs while at
least one active topic owns it.

A watch's context words (`watch_context`) narrow its name where a source searches loose text: only
X reads them. Their own table, so a store made before
them needs no re-onboard.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z0-9]+")


def normalize(text: str) -> str:
    """The identity form: case- and whitespace-insensitive."""
    return _WS.sub(" ", (text or "").strip().lower())


def query_id_for(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def spellings(name: str, others: list[str]) -> list[str]:
    """`name` and its other spellings, each searched on its own: once each by identity, and only
    those holding a word to search."""
    out, seen = [], set()
    for text in [name, *others]:
        if _WORD.search(text) and (norm := normalize(text)) not in seen:
            seen.add(norm)
            out.append(text)
    return out


def add(conn: sqlite3.Connection, text: str, target_sources: list[str], *, now: str) -> None:
    """Write the query for `text`, or widen an existing one with the same identity to these
    sources too. It keeps its text and `created_at`: it is the same watch whichever topic asked
    first. Its sources only widen, so a name watched before X and Substack were sources gains
    them when it is asked for again, and each new source starts from its own first look (the
    watermarks are per source, `pulls.py`). Joins the caller's transaction."""
    norm = normalize(text)
    query_id = query_id_for(norm)
    held = conn.execute("SELECT target_sources FROM frontier_queries WHERE query_id = ?",
                        (query_id,)).fetchone()
    if held is None:
        conn.execute("INSERT INTO frontier_queries "
                     "(query_id, text, normalized, target_sources, created_at) "
                     "VALUES (?, ?, ?, ?, ?)",
                     (query_id, text.strip(), norm, json.dumps(target_sources), now))
        return
    sources = json.loads(held[0] or "[]")
    wider = list(dict.fromkeys([*sources, *target_sources]))
    if wider != sources:
        conn.execute("UPDATE frontier_queries SET target_sources = ? WHERE query_id = ?",
                     (json.dumps(wider), query_id))


def set_context(conn: sqlite3.Connection, query_id: str, words: list[str]) -> None:
    """Give the query these context words, replacing any it had: the last watch to name them is
    the one the user just asked for. Joins the caller's transaction."""
    conn.execute("INSERT OR REPLACE INTO watch_context (query_id, words) VALUES (?, ?)",
                 (query_id, json.dumps(words)))


def context(row) -> list[str]:
    """The context words on a query row read by `active_queries`."""
    return json.loads(row["context"] or "[]")


def active_queries(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every query an active topic owns, with its context words: what the crawler searches. A
    watch a topic dropped, or a topic no longer followed, leaves it unless another active topic
    still owns it."""
    # 'active' is `user_topics.status`'s followed state (pipeline/topics/store.ACTIVE).
    return list(conn.execute(
        "SELECT q.*, c.words AS context FROM frontier_queries q "
        "LEFT JOIN watch_context c ON c.query_id = q.query_id WHERE EXISTS ("
        "  SELECT 1 FROM topic_watches w JOIN user_topics t ON t.topic_id = w.topic_id"
        "   WHERE w.query_id = q.query_id AND t.status = 'active') "
        "ORDER BY q.created_at, q.query_id"))
