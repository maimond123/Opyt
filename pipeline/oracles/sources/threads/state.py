"""Thread progress and retries for the atom that owes the work. The walk so far is kept in
`thread_walks`, off the atom, because search and open return an atom's payload whole. A saved
post's conversation is owed and checked through curation's ledger (`curation/conversations.py`).
An Oracle thread whose photos are still owed waits for them (`footprint_photos.WAITING`), so its
tail's rewrite renders them and the thread is rewritten once for both."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from pipeline.curation import conversations
from pipeline.kb import footprint_photos, schema
from pipeline.kb.writes import write_tx
from pipeline.timeparse import parse_ts
from pipeline.worker.quota_loop import retries
from pipeline.worker.quota_loop.source import Changed, Footprint, Item, Retry, Unit
from pipeline.oracles import roster
from pipeline.oracles.sources.threads import tail_state
from pipeline.oracles.sources.threads.walk import Walk
from pipeline.oracles.sources.threads.walk import restore
from pipeline.worker.quota_loop.source import RetryRows


QUOTA = "x:TweetDetail"


@dataclass(frozen=True)
class Work:
    atom_id: str
    version: int
    raw_hash: str
    who_id: str
    payload: dict
    oracle: str | None
    walk: Walk

    @property
    def root(self) -> str:
        return str(self.payload.get("thread_root") or self.atom_id.split(":", 1)[1])


def read(conn, atom_id: str) -> Work | None:
    row = conn.execute("SELECT a.atom_id, a.version, a.raw_hash, a.who_id, a.payload, "
                       "a.entry_mode, w.walk FROM atoms a "
                       "LEFT JOIN thread_walks w ON w.atom_id=a.atom_id WHERE a.atom_id=?",
                       (atom_id,)).fetchone()
    if row is None:
        return None
    oracle = (schema.current_canonical(conn, row["who_id"])
              if row["entry_mode"] == "oracle-footprint" else None)
    return Work(row["atom_id"], row["version"], row["raw_hash"], row["who_id"],
                json.loads(row["payload"] or "{}"), oracle,
                restore(json.loads(row["walk"]) if row["walk"] else None))


def require_current(conn, work: Work) -> Work:
    current = read(conn, work.atom_id)
    if (current is None or current.version != work.version
            or current.raw_hash != work.raw_hash or current.oracle != work.oracle
            or current.who_id != work.who_id
            or (work.oracle is not None and not roster.on_roster(conn, work.oracle))):
        raise Changed(work.atom_id)
    return current


def owed(conn, kind: str) -> list[Work]:
    if kind == "tails":
        query = ("SELECT atom_id FROM atoms WHERE source_type='x' "
                 "AND entry_mode='oracle-footprint' "
                 "AND json_extract(payload, '$.thread_tail')=? "
                 f"AND NOT {footprint_photos.WAITING} ORDER BY when_ts DESC")
        atom_ids = [row[0] for row in conn.execute(query, (tail_state.TAIL_OWED,))]
    else:
        atom_ids = conversations.owed_bookmarks(conn)
    items = [read(conn, atom_id) for atom_id in atom_ids]
    heads = roster.heads(conn) if kind == "tails" else set()
    return sorted((w for w in items if w is not None
                   and (w.oracle is None or w.oracle in heads)),
                  key=lambda w: len(w.walk.focused))


def save_walk(conn, work: Work, walk: Walk) -> Work:
    with write_tx(conn, site="threads.walk"):
        require_current(conn, work)
        conn.execute("INSERT INTO thread_walks (atom_id, walk) VALUES (?, ?) "
                     "ON CONFLICT(atom_id) DO UPDATE SET walk=excluded.walk",
                     (work.atom_id, json.dumps(walk.saved())))
        return require_current(conn, work)


def move_payload_walks(conn) -> None:
    """Walks saved by an older version sit in `atoms.payload`. Move each to `thread_walks`, so it keeps its progress and leaves the
    payload."""
    held = "json_type(payload, '$.thread_walk') IS NOT NULL"
    with write_tx(conn, site="threads.move_walks"):
        conn.execute("INSERT OR IGNORE INTO thread_walks (atom_id, walk) "
                     f"SELECT atom_id, json_extract(payload, '$.thread_walk') FROM atoms WHERE {held}")
        conn.execute(f"UPDATE atoms SET payload=json_remove(payload, '$.thread_walk') WHERE {held}")


def finish(conn, work: Work, *, gone: bool = False, complete: bool = False) -> None:
    """Complete only an existing, current atom. Called after its successful durable write; a
    saved post's caller keeps its chain on the saved row in the same transaction."""
    with write_tx(conn, site="threads.complete"):
        require_current(conn, work)
        if work.oracle is not None:
            tail_state.set_tail_state(
                conn, work.atom_id, tail_state.TAIL_GONE if gone else tail_state.TAIL_FETCHED,
                body="complete" if complete else None)
        else:
            conversations.mark_checked(conn, work.root)
        conn.execute("DELETE FROM thread_walks WHERE atom_id=?", (work.atom_id,))
        conn.execute("UPDATE atoms SET payload=json_remove(payload, '$.thread_retry', "
                     "'$.bookmark_refresh') WHERE atom_id=?", (work.atom_id,))


class ThreadRows(RetryRows):
    """The loop's retry adapter for the atom's own thread work."""

    def read(self, work: Work) -> Retry:
        return Retry(**(work.payload.get("thread_retry") or {}))

    def write(self, conn, key: tuple, retry: Retry) -> None:
        with write_tx(conn, site="threads.retry"):
            conn.execute("UPDATE atoms SET payload=json_set(payload, '$.thread_retry', json(?)) "
                         "WHERE atom_id=?", (json.dumps(vars(retry)), *key))

    def clear(self, conn, key: tuple) -> None:
        with write_tx(conn, site="threads.retry_clear"):
            conn.execute("UPDATE atoms SET payload=json_remove(payload, '$.thread_retry') "
                         "WHERE atom_id=?", key)


ROWS = ThreadRows()


def next_unit(conn, kind: str, now: datetime) -> Unit | None:
    for work in owed(conn, kind):
        prior = ROWS.read(work)
        if retries.may_run(prior.retry_at, prior.failures, now):
            return Unit(work.atom_id, work, Item(ROWS, (work.atom_id,), prior, "x"),
                        Footprint(work.oracle) if work.oracle is not None else None)
    return None


def next_due(conn, kind: str, now: datetime) -> datetime | None:
    times = [at for work in owed(conn, kind)
             if (at := parse_ts(ROWS.read(work).retry_at)) is not None and at > now]
    return min(times) if times else None
