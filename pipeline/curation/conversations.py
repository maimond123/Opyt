"""The X conversation ledger: which saved posts' conversations are checked, and which are owed.

A saved post that is a reply, or that has replies, owes its conversation to oracles' TweetDetail
lane; a standalone post is checked when it is saved. The ledger is `sync_dedup`'s
`x_convo_checked` namespace, keyed by tweet id. Curation owns it; the thread
lane reads what is owed here and marks what it finished here.
"""

import sqlite3
from pathlib import Path

from pipeline.dedup_store import SyncSet
from pipeline.ingestion.utils import load_state
from pipeline.worker.quota_loop import retries

NAMESPACE = "x_convo_checked"


def checked(home: Path) -> SyncSet:
    """Every checked tweet id."""
    return load_state(home / f"{NAMESPACE}.json")


def mark_checked(conn: sqlite3.Connection, tweet_id: str) -> None:
    """Joins the caller's transaction, so a post and its checked state land together."""
    conn.execute("INSERT OR IGNORE INTO sync_dedup(namespace, item_id, synced_at) "
                 "VALUES (?, ?, datetime('now'))", (NAMESPACE, tweet_id))


# A saved X atom whose conversation is owed: a reply, a post with replies, or one marked for
# refresh, that the ledger has not checked (a refresh is owed regardless).
_OWED = ("source_type='x' AND entry_mode='user-saved' "
         "AND (json_extract(payload, '$.is_reply') "
         "OR COALESCE(json_extract(payload, '$.reply_count'), 0)>0 "
         "OR json_extract(payload, '$.bookmark_refresh')) "
         "AND (json_extract(payload, '$.bookmark_refresh') OR NOT EXISTS "
         "(SELECT 1 FROM sync_dedup d WHERE d.namespace=? "
         "AND d.item_id=substr(a.atom_id, instr(a.atom_id, ':')+1)))")

# Its photos are still being read: an owed `saved_images` row not given up. The thread lane waits
# for it, so the post is rewritten once for both.
_IMAGES_OWED = ("EXISTS (SELECT 1 FROM curation_saved_images i WHERE i.platform='x' "
                "AND i.item=substr(a.atom_id, instr(a.atom_id, ':')+1) AND i.state='owed' "
                "AND COALESCE(i.failures, 0)<?)")


def owed_bookmarks(conn: sqlite3.Connection) -> list[str]:
    """The saved X atoms whose conversation is owed and whose photos are not still being read,
    newest first."""
    waiting, args = "", [NAMESPACE]
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='curation_saved_images'").fetchone():
        waiting, args = f" AND NOT {_IMAGES_OWED}", [NAMESPACE, retries.GIVE_UP_AFTER]
    return [row[0] for row in conn.execute(
        f"SELECT atom_id FROM atoms a WHERE {_OWED}{waiting} ORDER BY when_ts DESC", args)]


def owed_conversations(conn: sqlite3.Connection, items: list[str]) -> list[str]:
    """Of these saved X posts, the ones whose conversation is owed."""
    marks = ",".join("?" * len(items))
    return [row[0].split(":", 1)[1] for row in conn.execute(
        f"SELECT atom_id FROM atoms a WHERE atom_id IN ({marks}) AND {_OWED}",
        [f"x:{i}" for i in items] + [NAMESPACE])]
