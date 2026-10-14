"""Saved-content work: which saved posts still owe their content, at which input revision.

A listing page offers its records here in its own transaction; a content source takes them one
at a time. The work outlives the listing that found it: a new walk never deletes owed work, a
unit settling an older revision cannot mark a newer one done, and re-offering an unchanged post
leaves its retry state alone, so a refresh cannot escape a backoff. Retry timing belongs to the
quota loop; this table only holds the columns it charges.

The row also keeps what a later re-render of the item needs beyond its record (`Rendering`), so
the item's atom holds only the finished result.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from pipeline.kb.writes import write_tx
from pipeline.timeparse import utc_iso
from pipeline.worker.quota_loop.source import Changed, Retry

_DDL = """
CREATE TABLE IF NOT EXISTS curation_saved_items (
    platform TEXT NOT NULL, item TEXT NOT NULL, account TEXT NOT NULL,
    revision TEXT NOT NULL, data TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('owed', 'done', 'superseded')),
    offered_at TEXT NOT NULL,
    status TEXT, failures INTEGER, retry_at TEXT, last_error TEXT,
    chain TEXT,
    rendered_from_teaser INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(platform, item)
);
CREATE INDEX IF NOT EXISTS curation_saved_items_owed
    ON curation_saved_items(platform, state, offered_at);
"""

# What changes on a saved X post without changing what is saved of it. `replyCount` is not
# here: a post that gains replies becomes owed to the conversation lane.
_X_VOLATILE = ("likeCount",)


@dataclass(frozen=True)
class Saved:
    platform: str
    item: str
    account: str
    revision: str
    data: dict
    retry: Retry


@dataclass(frozen=True)
class Rendering:
    """What a saved item's row keeps for re-rendering it. `chain`: the conversation the thread
    lane last read, None when none was read. `rendered_from_teaser`: the input last recorded for
    the stored atom carried only an article's teaser, so a full article must re-render it."""
    data: dict
    chain: list | None
    rendered_from_teaser: bool


def init_schema(conn: sqlite3.Connection) -> None:
    # executescript commits a caller's transaction, so initialization is entry-only.
    if conn.in_transaction:
        raise RuntimeError("initialize saved-item tables before starting a transaction")
    conn.executescript(_DDL)


def _without(post: dict, keys: tuple[str, ...]) -> dict:
    return {k: (_without(v, keys) if isinstance(v, dict) else v)
            for k, v in post.items() if k not in keys}


def revision(record: dict, volatile: tuple[str, ...] = ()) -> str:
    """The revision of a saved item's record: a hash of everything its saved content reads,
    leaving out what changes without changing that."""
    body = json.dumps(_without(record, volatile), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def x_revision(post: dict) -> str:
    """The revision of a normalized X post."""
    return revision(post, _X_VOLATILE)


def offer(conn: sqlite3.Connection, platform: str, account: str,
          records: Iterable[tuple[str, str, dict]], now: datetime) -> None:
    """Make each `(item, revision, data)` owed unless the same account already holds that
    revision. Joins the caller's transaction, so a page and its offers land together. A
    re-offer leaves the row's re-render inputs alone: a new revision of a post is still the same
    conversation, and its stored atom was still rendered from what the flag says."""
    rows = [(platform, item, account, revision, json.dumps(data, sort_keys=True), utc_iso(now))
            for item, revision, data in records]
    if any(not item for _, item, *_ in rows):
        raise ValueError("a saved item requires its identity")
    with write_tx(conn, site="saved.offer"):
        conn.executemany(
            "INSERT INTO curation_saved_items(platform,item,account,revision,data,state,offered_at) "
            "VALUES(?,?,?,?,?,'owed',?) ON CONFLICT(platform,item) DO UPDATE SET "
            "account=excluded.account, revision=excluded.revision, data=excluded.data, "
            "state='owed', offered_at=excluded.offered_at, "
            "status=NULL, failures=NULL, retry_at=NULL, last_error=NULL "
            "WHERE curation_saved_items.revision!=excluded.revision "
            "OR curation_saved_items.account!=excluded.account "
            "OR curation_saved_items.state='superseded'", rows)


def supersede(conn: sqlite3.Connection, platform: str, account: str) -> None:
    """Another account connected: what the old one still owes is no longer this home's to save.
    Joins the caller's transaction."""
    with write_tx(conn, site="saved.supersede"):
        conn.execute("UPDATE curation_saved_items SET state='superseded' "
                     "WHERE platform=? AND account!=? AND state='owed'", (platform, account))


def owed(conn: sqlite3.Connection, platform: str) -> list[Saved]:
    """Owed work, oldest offer first. A home that never listed has none."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                        "AND name='curation_saved_items'").fetchone():
        return []
    return [Saved(platform, row[0], row[1], row[2], json.loads(row[3]), Retry(*row[4:]))
            for row in conn.execute(
                "SELECT item,account,revision,data,status,failures,retry_at,last_error "
                "FROM curation_saved_items WHERE platform=? AND state='owed' "
                "ORDER BY offered_at,item", (platform,))]


def require_current(conn: sqlite3.Connection, saved: Saved) -> None:
    """Refuse a write for work that was re-offered, superseded or settled meanwhile."""
    row = conn.execute("SELECT account,revision,state FROM curation_saved_items "
                       "WHERE platform=? AND item=?", (saved.platform, saved.item)).fetchone()
    if row is None or tuple(row) != (saved.account, saved.revision, "owed"):
        raise Changed(f"saved {saved.platform} item {saved.item} changed")


def settle(conn: sqlite3.Connection, saved: Saved) -> None:
    """Mark this revision's content saved. Joins the caller's transaction, so the content and
    its done-state land together."""
    with write_tx(conn, site="saved.settle"):
        require_current(conn, saved)
        conn.execute("UPDATE curation_saved_items SET state='done',status=NULL,failures=NULL,"
                     "retry_at=NULL,last_error=NULL WHERE platform=? AND item=?",
                     (saved.platform, saved.item))


def rendering(conn: sqlite3.Connection, platform: str, item: str) -> Rendering | None:
    """The row's re-render inputs, or None for an item no listing offered (a hand-saved post)."""
    row = conn.execute("SELECT data,chain,rendered_from_teaser FROM curation_saved_items "
                       "WHERE platform=? AND item=?", (platform, item)).fetchone()
    if row is None:
        return None
    return Rendering(json.loads(row[0]), None if row[1] is None else json.loads(row[1]),
                     bool(row[2]))


def record_chain(conn: sqlite3.Connection, platform: str, item: str, chain: list) -> None:
    """Keep the conversation just read for the item. Joins the caller's transaction, so the atom
    rendered from it and the chain land together. A hand-saved post has no row to keep it on."""
    conn.execute("UPDATE curation_saved_items SET chain=? WHERE platform=? AND item=?",
                 (json.dumps(chain), platform, item))


def record_rendered(conn: sqlite3.Connection, platform: str, item: str, *, teaser: bool) -> None:
    """Record whether the input just recorded for the item's atom carried only a teaser. Joins
    the caller's transaction, so it lands with the atom write it describes."""
    conn.execute("UPDATE curation_saved_items SET rendered_from_teaser=? "
                 "WHERE platform=? AND item=?", (int(teaser), platform, item))


class SavedRows:
    """The quota loop owns retry timing and charging; this adapter owns the item's columns."""

    def read(self, saved: Saved) -> Retry:
        return saved.retry

    def write(self, conn: sqlite3.Connection, key: tuple, retry: Retry) -> None:
        platform, item, revision = key
        with write_tx(conn, site="saved.retry"):
            conn.execute(
                "UPDATE curation_saved_items SET status=?,failures=?,retry_at=?,last_error=? "
                "WHERE platform=? AND item=? AND revision=? AND state='owed'",
                (retry.status, retry.failures, retry.retry_at, retry.last_error,
                 platform, item, revision))

    def clear(self, conn: sqlite3.Connection, key: tuple) -> None:
        self.write(conn, key, Retry())


ROWS = SavedRows()
