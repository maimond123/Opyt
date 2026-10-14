"""
service/owner_tools.py — the tools this service answers by itself for a library's owner: its
status, a reader view of sources, an invite, and cutting one reader off.

Each is one flow with two thin entries: the connector's tool, for an owner signed in with Google
(`connector.py`), and a `/v1/library/…` route, for the owner's agent holding an agent key
(`app.py`). They write what this service owns (pages, invites, tokens) and never the served copy.
The tools that change the library itself are `library_writes.py`.

`reader` is the name a call's reads are counted under: the hash a reader token's reads are
counted under, whoever the caller is.
"""
from __future__ import annotations

import os
import time
from urllib.parse import urlparse

from pipeline.retrieval import reads
from pipeline.sharing import agent_invite, invite, page_words, readers
from service import grants, library_writes, publications, reading, tokens
from service.pages import make, store

BASE_URL_ENV = "OPYT_SERVICE_BASE_URL"
SETUP_URL = "https://useopyt.com/connector.html"


def base_url() -> str:
    """This service's own address, as its links are written."""
    return os.environ.get(BASE_URL_ENV, "").rstrip("/")


def status(owner: str) -> dict:
    """What `owner`'s served library holds and when it was last published, and what was asked to
    change in it lately. Not counted as a read."""
    total = reads.kb_aggregate(kb=owner).get("total", 0)
    return {"status": "ready", "items": total,
            "last_published_at": publications.publish_demand(owner)["last_upload_at"],
            "search": "keyword", "setup": SETUP_URL,
            "changes": library_writes.recent(owner)}


def reader_view(owner: str, reader: str, atom_ids: list[str] | None, title: str | None,
                note: str | None, answer: str | None, delete: str | None) -> dict:
    """Make a page of `answer` and `atom_ids` from `owner`'s served copy, or take down the one
    `delete` links to. Each item on a page is counted as a read."""
    if delete:
        page_id = urlparse(delete.strip()).path.rstrip("/").rsplit("/", 1)[-1]
        if not store.delete(owner, page_id):
            return {"status": "not_found", "message": "this library made no page with that "
                                                      "link: it may have expired already."}
        return {"status": "deleted",
                "message": "Taken down. The link now says the page has expired."}
    items, missing = _served_items(owner, reader, atom_ids or [])
    if not items:
        return {"status": "not_found", "missing": missing,
                "message": "none of these ids is in the user's library, so there is no page "
                           "to make. Pass the `atom_id` of search hits or open results."}
    try:
        made = make.create(owner, items, title, note, answer, time.time())
    except make.Refused as e:
        return {"status": "refused", "message": str(e)}
    return {"status": "made", "url": base_url() + made["path"],
            **({"missing": missing} if missing else {}),
            "message": page_words.made_message(made["days"], missing)}


def share(owner: str, confirm: bool, for_whom: str | None, agent: bool) -> dict:
    """Preview what an invite to `owner`'s library gives, or mint one: for a person, who reads,
    or (`agent`) for the owner's own agent, which acts as them."""
    if not confirm:
        return _share_preview(owner, agent)
    code = grants.mint_grant(owner, label=(for_whom or "").strip() or None, agent=agent)
    return {"status": "shared", "invite": invite.link(code),
            "message": agent_invite.handover(base_url()) if agent else _shared(owner, for_whom)}


def unshare(owner: str, reader: str, confirm: bool) -> dict:
    """Preview cutting off the one reader or agent `reader` names, or do it. The pick and its
    sentences are `readers`', which the Mac's `unshare` uses too."""
    held = readers.readers_of(tokens.list_tokens(owner))
    got = readers.pick(held, reader)
    if isinstance(got, dict):
        return got
    if not confirm:
        return readers.preview(got.row, held, base_url())
    return readers.revoked(got.row, held, tokens.revoke(owner, got.row["token_sha256"]))


def _share_preview(owner: str, agent: bool) -> dict:
    """What an invite would give, a person or (`agent`) the user's own agent, from the served
    copy. Mints nothing and is not counted as a read."""
    agg = reads.kb_aggregate(kb=owner)
    held = readers.readers_of(tokens.list_tokens(owner))
    total = agg.get("total", 0)
    return {
        "status": "preview",
        "items": total,
        "by_source_type": agg.get("by_source_type"),
        "by_what_kind": agg.get("by_what_kind"),
        "top_entities": agg.get("top_entities"),
        "readers": len(held),
        "last_published_at": publications.publish_demand(owner)["last_upload_at"],
        "consent": list(agent_invite.CONSENT) if agent else [
            f"The person who redeems the link can search all {total} items in this library. "
            f"There is no way to share part of it.",
            "It is standing access, not a one-time copy: they see the library as it updates, "
            "until they are cut off with `unshare`.",
            "The link works once. They redeem it in Opyt on their own computer, or by pasting it "
            "into claude.ai or ChatGPT with the Opyt connector added.",
        ],
    }


def _shared(owner: str, for_whom: str | None) -> str:
    who = f"to {for_whom.strip()}" if for_whom and for_whom.strip() else "to the person it is for"
    unnamed = ("" if tokens.owner_label(owner) else
               " Their Opyt names this library by a code unless they give it a name when they "
               "accept it, so tell them what to call it.")
    return (f"Give the user the invite to send {who}. It works once: in Opyt on their computer, "
            f"or pasted into claude.ai or ChatGPT with the Opyt connector added."
            f"{unnamed}")


def _served_items(owner: str, reader: str, atom_ids: list[str]) -> tuple[list[dict], list[str]]:
    """Each atom from the served copy, once, in the order given, each counted as a read; and the
    ids the copy lacks."""
    items, missing = [], []
    for atom_id in dict.fromkeys(atom_ids):
        got = reading.open_whole(owner, reader, atom_id)
        if "error" in got:
            missing.append(atom_id)
        else:
            items.append(got)
    return items, missing


