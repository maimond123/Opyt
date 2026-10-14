"""
pipeline/sharing/unshare.py — the unshare flow: cut off one reader, disconnect the user's own
claude.ai and ChatGPT, or withdraw the whole knowledge base.

Two-phase, with two scopes under one verb: naming a reader cuts off that one person and leaves
the copy serving, and naming the Google account (`link.NAMES`) disconnects it the same way;
naming nobody deletes everything the service holds for this library (the copy, every reader, the
read counts, the Google link), and this install drops its token, so nothing publishes from here
again. Because the larger act is the default, each preview leads with its
scope. `for_whom` at share time makes a reader nameable here.

Every failure is a sentence in the return value, never a raise.
"""
from __future__ import annotations

from pipeline.sharing import account, client, link, push_gate, readers


def run(confirm: bool, reader: str | None) -> dict:
    """Preview or do one of the two scopes: `reader` names one person, None means everybody."""
    acct = account.load()
    if not acct.token:
        return {"status": "not_shared",
                "message": "this library is not on Opyt's server, so there is nothing to stop "
                           "or delete."}
    if reader is None and confirm:
        return _withdraw(acct)
    try:
        listed = client.tokens(acct)
    except client.Failed as e:
        return {"status": "unreachable", "message": str(e)}
    held, google = readers.readers_of(listed["tokens"]), link.connected(listed)
    if reader is None:
        return _preview_everyone(acct, held, google)
    # A reader the user labelled "google" is still theirs to cut off by that name.
    if google and link.names(reader) and not readers.match(held, reader):
        return link.disconnect(acct, confirm)
    return _with_google(_cut_off(acct, held, reader, confirm), google)


# Says what the server no longer holds rather than what this call deleted, so it is true of a
# library that was never linked to Google or never sent a share link.
_DELETED = (
    "SAY this as written: \"Deleted for good. Opyt's server no longer holds anything of your "
    "library: no copy, no link to your Google account, no share links and no count of its reads. "
    "claude.ai and ChatGPT can no longer read it, and every share link you sent has stopped "
    "working. Your library on this computer is untouched, and nothing is uploaded again unless "
    "you share it or connect Opyt to Google again.\"")


def _withdraw(acct: account.Account) -> dict:
    """Everything the service holds for this library deleted, then this install's token dropped,
    so nothing publishes from here again. A failed call changes nothing here."""
    try:
        res = client.unpublish(acct)
    except client.Failed as e:
        return {"status": "failed", "message": str(e)}
    push_gate.forget_publication()
    account.drop_token()
    return {"status": "unshared",
            "readers_revoked": res["readers_revoked"],
            "export_deleted": res["export_deleted"],
            "message": _DELETED}


def _cut_off(acct: account.Account, held: list[dict], wanted: str, confirm: bool) -> dict:
    """Cut off one reader. The served copy stays and everybody else keeps reading. The pick and
    its sentences are `readers`', which the connector's `unshare` uses too."""
    got = readers.pick(held, wanted)
    if isinstance(got, dict):
        return got
    if not confirm:
        return readers.preview(got.row, held, acct.url)
    try:
        res = client.revoke(acct, got.row["token_sha256"])
    except client.Failed as e:
        return {"status": "failed", "message": str(e)}
    return readers.revoked(got.row, held, bool(res.get("revoked")))


def _with_google(answer: dict, google: str | None) -> dict:
    """The answer that lists who holds access, with the Google connection beside the readers:
    the one list of everything that can reach the library."""
    if not google or answer.get("status") != "no_such_reader":
        return answer
    return {**answer, "readers": [*answer["readers"], link.roster_row(google)],
            "message": answer["message"] + " The user's own claude.ai and ChatGPT are connected "
                                           "too: `reader='google'` disconnects them."}


def _preview_everyone(acct: account.Account, held: list[dict], google: str | None) -> dict:
    return {"status": "preview",
            "scope": "everyone",
            "readers": len(held),
            "reader_labels": [t["label"] for t in held if t["label"]],
            # Everything that can reach the library: each reader and agent, then the user's own
            # claude.ai and ChatGPT when they are connected.
            "who_has_access": [*readers.roster(held),
                               *([link.roster_row(google)] if google else [])],
            "service": acct.url,
            "consent": [
                f"This takes the whole knowledge base down, not one reader, and cannot be "
                f"undone. All {len(held)} of them lose access on their next request.",
                "Everything Opyt's server holds for it is deleted: its copy, every invite and "
                "share link, the count of its reads, and its link to a Google account if it has "
                "one, so claude.ai and ChatGPT can no longer read it.",
                "The knowledge base on this computer stays as it is. Sharing it or connecting "
                "Opyt to Google again starts over, with new invites for everyone.",
                "To cut off one person instead, pass their name as `reader`; "
                "`reader='google'` disconnects only the user's own claude.ai and ChatGPT.",
            ]}
