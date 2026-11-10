"""
service/pages/store.py — pages of sources: their rows in `service.db`, their comments, and their
PDFs on disk.

The one writer of the `pages` and `page_comments` tables and of `pages/<id>/`, where a page's
papers keep their PDFs. A page's comments are deleted wherever the page is (`_drop`). A
page lasts `LIFETIME` from when it was last opened (`opened`), or from when it was made if nobody
opened it. An expired one is deleted when it is next asked for, and every night by `purge`
(`service/DEPLOY.md`, "Nightly"). Every function that needs the time takes `now` (unix seconds)
from its caller.
"""
from __future__ import annotations

import json
import re
import secrets
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import NamedTuple

from opyt_core.paths import opyt_path
from service import db, uploads

DAYS = 90                    # deleted this long after it was last opened
LIFETIME = DAYS * 24 * 3600
_DIR = "pages"
_ID = re.compile(r"^[A-Za-z0-9_-]{22}$")
# A folder this young may belong to a page still being made, whose row has not landed yet.
_MAKING = 3600


class Page(NamedTuple):
    id: str
    owner: str
    title: str
    note: str | None
    answer: str | None    # markdown: the host's answer, drawn above the sources
    items: list[dict]
    made: float


class Comment(NamedTuple):
    id: str
    thread: str           # the id of the comment that started its thread: its own, for that one
    item: int | None      # which source it is on, counted from 1 (0 is the answer); None on a reply
    quote: str | None     # the passage; None for a whole source, and on a reply
    prefix: str | None    # the source's text just before the passage, and just after it: how
    suffix: str | None    # the page finds the passage again
    name: str
    body: str
    made: float


def new_id() -> str:
    """128 random bits, the link's only secret."""
    return secrets.token_urlsafe(16)


def valid_id(page_id: str) -> bool:
    """Whether `page_id` could be one `new_id` made. Checked before it names a row or a folder."""
    return bool(_ID.match(page_id or ""))


def day(ts: float) -> date:
    """The UTC day of a stored time."""
    return datetime.fromtimestamp(ts, timezone.utc).date()


def path(page_id: str) -> str:
    """Where a page is read, under the service's own address; its PDFs are read under it."""
    return f"/p/{page_id}"


def pdf_path(page_id: str, n: int) -> Path:
    """Where the PDF of a page's `n`th source is kept."""
    return opyt_path(_DIR, page_id, f"{n}.pdf")


def has_room() -> bool:
    """Whether the disk can spare a page's PDFs: the free space uploads keep in reserve."""
    return shutil.disk_usage(opyt_path()).free >= uploads.FREE_BYTES_FLOOR


def save(page: Page) -> None:
    """Store a new page, to last `LIFETIME` from when it was made. Its PDFs are already in its
    folder."""
    conn = db.connect()
    try:
        conn.execute("INSERT INTO pages (id, owner, title, note, answer, items, made, expires) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (page.id, page.owner, page.title, page.note, page.answer,
                      json.dumps(page.items), page.made, page.made + LIFETIME))
        conn.commit()
    finally:
        conn.close()


def read(page_id: str, now: float) -> Page | None:
    """The page, or None when there is none or it has expired. An expired page is deleted here."""
    if not valid_id(page_id):
        return None
    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
        expired = row is not None and row["expires"] <= now
        if expired:
            _drop(conn, "id = ?", (page_id,))
            conn.commit()
    finally:
        conn.close()
    if row is None:
        return None
    if expired:
        _remove_folder(page_id)
        return None
    return Page(row["id"], row["owner"], row["title"], row["note"], row["answer"],
                json.loads(row["items"]), row["made"])


def opened(page_id: str, now: float) -> None:
    """The page was just opened: it now lasts `LIFETIME` from `now`. When is all a visit leaves."""
    conn = db.connect()
    try:
        conn.execute("UPDATE pages SET expires = ? WHERE id = ?", (now + LIFETIME, page_id))
        conn.commit()
    finally:
        conn.close()


def delete(owner: str, page_id: str) -> bool:
    """Take one of `owner`'s pages down now. False when `owner` has no page by that id."""
    if not valid_id(page_id):
        return False
    conn = db.connect()
    try:
        gone = _drop(conn, "id = ? AND owner = ?", (page_id, owner))
        conn.commit()
    finally:
        conn.close()
    if gone:
        _remove_folder(page_id)
    return bool(gone)


def delete_all(owner: str) -> int:
    """Take every page `owner` made down now, with its comments and its folder: part of deleting
    the library (`deletion.py`). Returns how many. A folder that outlives its row is the purge's
    to clear."""
    conn = db.connect()
    try:
        ids = [r[0] for r in conn.execute("SELECT id FROM pages WHERE owner = ?", (owner,))]
        _drop(conn, "owner = ?", (owner,))
        conn.commit()
    finally:
        conn.close()
    for page_id in ids:
        _remove_folder(page_id)
    return len(ids)


def purge(now: float) -> int:
    """Delete every expired page with its folder, and every older folder no page names (a making
    or a deletion that died midway). Returns how many pages expired. The nightly cron calls this."""
    conn = db.connect()
    try:
        expired = {r[0] for r in conn.execute("SELECT id FROM pages WHERE expires <= ?", (now,))}
        _drop(conn, "expires <= ?", (now,))
        conn.commit()
        live = {r[0] for r in conn.execute("SELECT id FROM pages")}
    finally:
        conn.close()
    root = opyt_path(_DIR)
    for folder in root.iterdir() if root.is_dir() else ():
        if folder.name in expired or (folder.name not in live
                                      and folder.stat().st_mtime < now - _MAKING):
            shutil.rmtree(folder, ignore_errors=True)
    return len(expired)


def _drop(conn, where: str, params: tuple) -> int:
    """Delete the pages `where` matches, and their comments with them. Returns how many pages.
    Every deletion of a page goes through here."""
    conn.execute(f"DELETE FROM page_comments WHERE page_id IN (SELECT id FROM pages WHERE {where})",
                 params)
    return conn.execute(f"DELETE FROM pages WHERE {where}", params).rowcount


def _remove_folder(page_id: str) -> None:
    shutil.rmtree(opyt_path(_DIR, page_id), ignore_errors=True)


# ── a page's comments ─────────────────────────────────────────────────────────────

_COMMENT = "id, thread, item, quote, prefix, suffix, name, body, made"


def comments(page_id: str) -> list[Comment]:
    """Every comment on the page, oldest first."""
    conn = db.connect()
    try:
        rows = conn.execute(f"SELECT {_COMMENT} FROM page_comments WHERE page_id = ? "
                            f"ORDER BY made, rowid", (page_id,)).fetchall()
    finally:
        conn.close()
    return [Comment(*row) for row in rows]


def add_comment(page_id: str, comment: Comment, key_sha256: str) -> None:
    """Keep a new comment, with the hash of the key that can delete it."""
    conn = db.connect()
    try:
        conn.execute(f"INSERT INTO page_comments (page_id, key_sha256, {_COMMENT}) "
                     f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (page_id, key_sha256, *comment))
        conn.commit()
    finally:
        conn.close()


def comment_key(page_id: str, comment_id: str) -> str | None:
    """The hash of the key that deletes this comment, or None when the page has no such comment."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT key_sha256 FROM page_comments WHERE page_id = ? AND id = ?",
                           (page_id, comment_id)).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def delete_comment(page_id: str, comment_id: str) -> None:
    """Delete one comment, and the replies to it when it started a thread."""
    conn = db.connect()
    try:
        conn.execute("DELETE FROM page_comments WHERE page_id = ? AND (id = ? OR thread = ?)",
                     (page_id, comment_id, comment_id))
        conn.commit()
    finally:
        conn.close()
