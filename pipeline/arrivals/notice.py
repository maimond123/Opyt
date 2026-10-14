"""
pipeline/arrivals/notice.py — the line that rides the user's own searches: how many new items
arrived since they last looked, and which topics have finds waiting for the host's check, pointing
at `latest`. New means recently published (`fresh.is_new`), so an Oracle's archive landing does
not count. Marks nothing seen; only `latest` does.
"""
from __future__ import annotations

from datetime import datetime

from pipeline.arrivals import fresh, window
from pipeline.frontier import candidates
from pipeline.topics import store as topics_store


def notice(conn, now: datetime) -> dict | None:
    """`{arrived, finds, call, message}`, or None when nothing is new."""
    arrived = sum(1 for row in window.arrivals(conn, window.start(conn, now)) if fresh.is_new(row))
    finds = {t["name"]: n for t in topics_store.topics(conn)
             if (n := len(candidates.waiting(
                 conn, [w["query_id"] for w in topics_store.watches(conn, t["topic_id"])])))}
    if not (arrived or finds):
        return None
    said = [f"{arrived} new item{'s' if arrived != 1 else ''} arrived in the user's library since "
            f"they last looked"] if arrived else []
    if finds:
        said.append("new finds wait to be checked on "
                    + ", ".join(f"{name} ({n})" for name, n in finds.items()))
    return {"arrived": arrived, "finds": finds, "call": "latest()",
            "message": "; ".join(said) + ". latest() shows them by topic, with why each is here."}
