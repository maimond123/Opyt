"""
pipeline/oracles/told.py — what the user has been told: the shown ledger, and the "since you
last looked" notice.

`milestones_shown` records which of a pick's milestones the user has already been told, so each
is said once (`unshown`, `mark_shown`); a new ask reopens the ones it makes owed again (`reopen`).
A source discovery added for a pick is told once too, under its own key `found:{review_id}`
(`found_shown`, `mark_found_shown`).
`completion_notice` is how a user who walked away learns what landed: each pick's new milestones,
said once, on whatever they call next. OPYT cannot push — an MCP server speaks only when called.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

from pipeline.ingestion.utils import log
from pipeline.kb import schema
from pipeline.kb import writes
from pipeline.oracles import milestones
from pipeline.oracles.asks import list_asks
from pipeline.timeparse import utc_iso
from pipeline.timeparse import utc_now


_DDL = """
CREATE TABLE IF NOT EXISTS milestones_shown (
  canonical_id TEXT NOT NULL,
  milestone    TEXT NOT NULL,        -- newest_posts | archives | reach | stuck
  shown_at     TEXT NOT NULL,
  PRIMARY KEY (canonical_id, milestone)
);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    """Idempotent DDL, safe on every writable open."""
    conn.executescript(_DDL)
    conn.commit()


def _shown(conn) -> set[tuple[str, str]]:
    return {(r[0], r[1]) for r in conn.execute(
        "SELECT canonical_id, milestone FROM milestones_shown")}


def unshown(conn: sqlite3.Connection, picks: list[dict] | None = None) -> list[dict]:
    """Milestones reached and not yet told, as `{name, canonical_id, milestone, …}`: `reach`
    carries how far back X now goes, `stuck` its cause and step. Asked picks only."""
    try:
        asked = {schema.current_canonical(conn, a.canonical_id) for a in list_asks(conn)}
        shown = _shown(conn)
    except sqlite3.OperationalError:
        return []
    out = []
    for pick in milestones.for_picks(conn) if picks is None else picks:
        cid = pick["canonical_id"]
        if cid not in asked:
            continue
        for m in milestones.reached(pick):
            if (cid, m) in shown:
                continue
            event = {"name": pick["name"], "canonical_id": cid, "milestone": m}
            if m == "reach":
                event["covered_from"] = pick["reach"]["covered_from"]
            if m == "stuck":
                event.update(pick["stuck"])
            out.append(event)
    return out


def mark_shown(conn: sqlite3.Connection, picks: list[dict], *,
               now: datetime | None = None) -> None:
    """These picks' milestones have been told: stamp every one they reached, and forget a
    `stuck` that has cleared, so the next one is told too. Never raises: a stamp that fails costs
    a repeat, never a loss."""
    stamp = utc_iso(now or utc_now())
    try:
        with writes.write_tx(conn, site="milestones_shown"):
            for pick in picks:
                cid = pick["canonical_id"]
                conn.executemany(
                    "INSERT OR IGNORE INTO milestones_shown (canonical_id, milestone, shown_at) "
                    "VALUES (?, ?, ?)", [(cid, m, stamp) for m in milestones.reached(pick)])
                if not pick.get("stuck"):
                    conn.execute("DELETE FROM milestones_shown WHERE canonical_id=? "
                                 "AND milestone='stuck'", (cid,))
    except sqlite3.Error as exc:
        log(f"[scrapers] could not stamp milestones as shown: {type(exc).__name__}: {exc}")


def found_shown(conn: sqlite3.Connection) -> set[tuple[str, int]]:
    """`(canonical_id, review_id)` for every found source the user has been told about."""
    try:
        return {(cid, int(m.split(":", 1)[1])) for cid, m in _shown(conn)
                if m.startswith("found:")}
    except sqlite3.OperationalError:
        return set()


def mark_found_shown(conn: sqlite3.Connection, told_of: list[tuple[str, int]], *,
                     now: datetime | None = None) -> None:
    """These found sources have been told. Never raises: a stamp that fails costs a repeat."""
    if not told_of:
        return
    stamp = utc_iso(now or utc_now())
    try:
        with writes.write_tx(conn, site="milestones_shown"):
            conn.executemany(
                "INSERT OR IGNORE INTO milestones_shown (canonical_id, milestone, shown_at) "
                "VALUES (?, ?, ?)", [(cid, f"found:{rid}", stamp) for cid, rid in told_of])
    except sqlite3.Error as exc:
        log(f"[scrapers] could not stamp found sources as shown: {type(exc).__name__}: {exc}")


def reopen(conn: sqlite3.Connection, canonical_id: str, milestones) -> None:
    """A new ask made these milestones owed again: let them be told again once reached."""
    cid = schema.current_canonical(conn, canonical_id)
    ms = list(milestones)
    with writes.write_tx(conn, site="milestones_shown"):
        conn.execute(f"DELETE FROM milestones_shown WHERE canonical_id IN (?, ?) "
                     f"AND milestone IN ({', '.join('?' for _ in ms)})",
                     (canonical_id, cid, *ms))


def completion_notice(conn, *, kind: str = "ingest") -> dict | None:
    """What reached a milestone while nobody was looking, as one plain notice, or None.

    OPYT cannot push, so what landed while the user was away rides on whatever they call next.
    Read from the milestones, not from a run; each is said once and stamped as it is taken
    (`mark_shown`). `kind` is unused, kept so no caller changes."""

    picks = milestones.for_picks(conn)
    events = unshown(conn, picks)
    if not events:
        return None
    mark_shown(conn, picks)
    return {"milestones": events, "message": _since_you_looked(events)}


def people(names: list[str]) -> str:
    """A list of writers joined the way a person says it — "Andrej", "Andrej and Martin",
    "Andrej, Martin and Soren"."""
    if len(names) <= 1:
        return names[0] if names else "these writers"
    return f"{', '.join(names[:-1])} and {names[-1]}"


_STUCK = {
    "x_session": "X needs reconnecting before more of {who}'s posts can come in.",
    "needs_topics": "Nothing of {who}'s papers is pulled until the user picks which subjects "
                    "to follow.",
    "discovery_failed": "Finding {who}'s other sources keeps failing.",
    "unsaved": "Some of {who}'s posts keep arriving but won't save.",
    "not_running": "Nothing is collecting {who}'s writing right now.",
}


def _since_you_looked(events: list[dict]) -> str:
    """The milestones as one plain message, in the user's terms."""
    def who(milestone: str) -> list[str]:
        return list(dict.fromkeys(e["name"] for e in events if e["milestone"] == milestone))

    said = []
    if names := who("newest_posts"):
        said.append(f"The newest X posts from {people(names)} are in.")
    if names := who("archives"):
        said.append(f"The rest of what {people(names)} publish beyond X is in.")
    for e in events:
        if e["milestone"] == "reach":
            said.append(f"{e['name']}'s X history now goes back to {e['covered_from'][:10]}, as "
                        f"far as was asked." if e.get("covered_from") else
                        f"{e['name']}'s X history is in, as far as was asked.")
    for e in events:
        if e["milestone"] == "stuck":
            cause = e.get("cause") or ""
            line = ("The model provider is refusing requests, so nothing more of {who}'s can be "
                    "saved yet." if cause.startswith("model:") else _STUCK.get(cause, ""))
            if line:
                said.append(line.format(who=e["name"]))
    return ("Since the user last looked: " + " ".join(said)
            + " Mention it once, briefly, beside whatever they asked; don't make it the subject "
              "of the turn. Say what is there and, for anything stuck, the step in `milestones` "
              "they can take. Never explain how it works.")
