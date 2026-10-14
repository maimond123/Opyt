"""
pipeline/sharing/page.py — `reader_view`'s page of sources: the items an answer used, from this
library, on one private link the service serves until nobody has opened it for 90 days
(`service/pages/`); and taking one down sooner.

Opyt cannot see the conversation, so the host names the items. Each is read as the store keeps it,
without `open()`'s paper full-text read: the page shows a paper's PDF, which the service fetches.
An install that never shared registers with no label and holds the token aside, as the Google
connection does (`client.held_token`), so making a page publishes nothing else of the library.

Every failure is a sentence in the return value, never a raise.
"""
from __future__ import annotations

from urllib.parse import urlparse

from pipeline.kb import schema
from pipeline.retrieval.engine import json_obj
from pipeline.sharing import account, client, page_words

_SELECT = ("SELECT atom_id, source_type, source_url, when_ts, description, payload "
           "FROM atoms WHERE atom_id = ?")


def make(atom_ids: list[str], title: str | None, note: str | None,
         answer: str | None) -> dict:
    """A page of the answer and these atoms, in this order: `made` with its link, or the sentence
    why not."""
    items, missing = _items(atom_ids)
    if not items:
        return {"status": "not_found", "missing": missing,
                "message": "none of these ids is in the user's library, so there is no page to "
                           "make. Pass the `atom_id` of search hits or open results."}
    acct = account.load()
    try:
        made = client.make_page(acct._replace(token=client.held_token(acct)), title, note, answer,
                                items)
    except client.Failed as e:
        return {"status": "failed", "message": f"the page was not made: {e}"}
    return {"status": "made", "url": acct.url + made["path"],
            **({"missing": missing} if missing else {}),
            "message": page_words.made_message(made["days"], missing)}


def delete(link: str) -> dict:
    """Take a page this install made down now, by its link."""
    acct = account.load()
    token = acct.token or account.pending_token()
    if token is None:
        return {"status": "not_found", "message": "this install has made no reader views."}
    page_id = urlparse(link.strip()).path.rstrip("/").rsplit("/", 1)[-1]
    try:
        got = client.delete_page(acct._replace(token=token), page_id)
    except client.Failed as e:
        return {"status": "failed", "message": f"the page was not taken down: {e}"}
    if not got["deleted"]:
        return {"status": "not_found",
                "message": "no page of this install's has that link: it may have expired already."}
    return {"status": "deleted", "message": "Taken down. The link now says the page has expired."}


def _items(atom_ids: list[str]) -> tuple[list[dict], list[str]]:
    """Each atom as the page draws it, once, in the order given; and the ids the store lacks."""
    conn = schema.connect(read_only=True)
    try:
        items, missing = [], []
        for atom_id in dict.fromkeys(atom_ids):
            row = conn.execute(_SELECT, (atom_id,)).fetchone()
            if row is None:
                missing.append(atom_id)
                continue
            items.append({"atom_id": row["atom_id"], "source_type": row["source_type"],
                          "source_url": row["source_url"], "when_ts": row["when_ts"],
                          "description": row["description"],
                          "raw": schema.read_body(conn, row["atom_id"]),
                          "payload": json_obj(row["payload"])})
        return items, missing
    finally:
        conn.close()
