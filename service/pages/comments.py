"""
service/pages/comments.py — the comments on a page of sources: reading them, adding one, taking
one's own down. Anyone holding the page's link does all three, from the page's own script
(`comments.js`); the link is the only lock, as it is for reading.

A comment is on a passage of one source or of the answer above them, on a whole source, or a
reply to either (`store.Comment` says how each is recorded). Nobody signs in, so a new comment
comes back with a key, once. The browser that posted it keeps the key, and that key is all that
deletes the comment before its page goes. Only the key's hash is stored.

Every check that can refuse a comment is `_checked`, which runs before anything is written.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

from service.pages import store

MAX_BODY = 2000        # characters in one comment
MAX_NAME = 40
MAX_QUOTE = 2000       # in the passage a comment is on
MAX_CONTEXT = 64       # kept of the text either side of the passage
MAX_PER_PAGE = 500
# What the page's script needs of the above, to stop a reader before the service refuses them.
LIMITS = {"body": MAX_BODY, "name": MAX_NAME, "quote": MAX_QUOTE}


class Refused(ValueError):
    """The comment was not kept. The message says why, as a sentence for the reader."""


def listed(page_id: str, now: float) -> dict | None:
    """`{comments, limits}` for a live page, oldest comment first; None when no live page has this
    id."""
    if store.read(page_id, now) is None:
        return None
    return {"comments": [c._asdict() for c in store.comments(page_id)], "limits": LIMITS}


def post(page_id: str, fields: dict, now: float) -> dict | None:
    """Keep a new comment: `{comment, key}`, the key being what later deletes it. None when no
    live page has this id. Raises `Refused` before anything is written."""
    page = store.read(page_id, now)
    if page is None:
        return None
    comment = _checked(fields, sources=len(page.items), answered=bool(page.answer),
                       there=store.comments(page_id), comment_id=secrets.token_urlsafe(9), now=now)
    key = secrets.token_urlsafe(16)
    store.add_comment(page_id, comment, _sha(key))
    return {"comment": comment._asdict(), "key": key}


def delete(page_id: str, comment_id: str, key: str) -> bool:
    """Delete a comment, with the replies to it, for whoever holds its key. False when the page
    has no such comment or the key is not its key."""
    held = store.comment_key(page_id, comment_id)
    if held is None or not hmac.compare_digest(held, _sha(key)):
        return False
    store.delete_comment(page_id, comment_id)
    return True


def _checked(fields: dict, *, sources: int, answered: bool, there: list[store.Comment],
             comment_id: str, now: float) -> store.Comment:
    """The comment these fields make on a page of `sources` sources, with an answer above them
    when `answered`, that already holds `there`, or `Refused`."""
    name = " ".join((fields.get("name") or "").split())
    body = (fields.get("body") or "").strip()
    if not name:
        raise Refused("A comment needs your name.")
    if not body:
        raise Refused("A comment needs some words.")
    if len(name) > MAX_NAME:
        raise Refused(f"A name can be at most {MAX_NAME} characters.")
    if len(body) > MAX_BODY:
        raise Refused(f"A comment can be at most {MAX_BODY:,} characters.")
    if len(there) >= MAX_PER_PAGE:
        raise Refused(f"This page already has {MAX_PER_PAGE} comments, which is all it holds.")
    thread = fields.get("thread")
    if thread:
        if not any(c.id == thread and c.thread == c.id for c in there):
            raise Refused("The comment you are replying to is gone.")
        return store.Comment(comment_id, thread, None, None, None, None, name, body, now)
    item = fields.get("item")
    if not isinstance(item, int) or not (0 if answered else 1) <= item <= sources:
        raise Refused("A comment has to be on the answer or one of this page's sources.")
    quote = fields.get("quote") or None
    if quote is None:
        return store.Comment(comment_id, comment_id, item, None, None, None, name, body, now)
    if len(quote) > MAX_QUOTE:
        raise Refused(f"Select at most {MAX_QUOTE:,} characters to comment on.")
    return store.Comment(comment_id, comment_id, item, quote,
                         (fields.get("prefix") or "")[-MAX_CONTEXT:],
                         (fields.get("suffix") or "")[:MAX_CONTEXT], name, body, now)


def _sha(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
