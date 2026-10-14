"""
pipeline/topics/posts.py — which of the user's own people's posts belong to a topic: those holding
one of its words as a phrase, and every post by one of its people.

Matched when the user looks, never stored, so an edited list applies to what arrives next and
nothing runs in the background. The host checks every match before showing it, so the lists are
written for recall (`lists.RULES`). Phrases are spelled once, in `retrieval/rank.fts_phrase`.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from pipeline.kb.id_batches import rows_for
from pipeline.retrieval.rank import fts_phrase

# How the user's own people's material enters: their Oracles' posts, what they saved, and the
# pages their Oracles linked. A watch's kept finds (`frontier`) are not theirs.
PEOPLES = ("oracle-footprint", "user-saved", "author_referenced")


def fts_query(words) -> str | None:
    """The FTS5 query matching any of `words`, each as an exact phrase, or None when none holds a
    word."""
    parts = [q for q in (fts_phrase(w) for w in words) if q != '""']
    return " OR ".join(parts) or None


def matching(conn, topic: dict, among: set[str]) -> set[str]:
    """The atoms in `among` that hold one of the topic's words or were written by its people."""
    if not among:
        return set()
    found: set[str] = set()
    if query := fts_query(topic["words"]):
        found |= among & {r[0] for r in conn.execute(
            "SELECT b.atom_id FROM bodies_fts JOIN bodies b ON b.body_id = bodies_fts.rowid "
            "WHERE bodies_fts MATCH ?", (query,))}
    authors = {w for person in topic["people"] for w in person["who_ids"]}
    if authors:
        found |= {r["atom_id"] for r in rows_for(
            conn, "SELECT atom_id, who_id FROM atoms WHERE atom_id IN ({ids})", among)
            if r["who_id"] in authors}
    return found


def recent(conn, topic: dict, now: datetime, days: int = 30) -> int:
    """How many of the user's people's posts from the last `days` the topic matches: what its
    check would have seen."""
    since = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    marks = ",".join("?" * len(PEOPLES))
    among = {r[0] for r in conn.execute(
        f"SELECT atom_id FROM atoms WHERE entry_mode IN ({marks}) AND when_ts >= ?",
        (*PEOPLES, since))}
    return len(matching(conn, topic, among))
