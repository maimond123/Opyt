"""
pipeline/sharing/accept.py — the accept flow: turns an invite into a knowledge base this install
can read.

Single-phase: a preview cannot validate a one-time code without burning it, and pasting the
invite is itself the yes. The peer row it writes is `pipeline/retrieval/peers.py`'s; this is the
only surface that writes a reader's row.

Every failure is a sentence in the return value, never a raise.
"""
from __future__ import annotations

from pipeline.retrieval import peers
from pipeline.sharing import account, client, invite


def run(pasted: str, name: str | None) -> dict:
    """Redeem the invite inside `pasted` and register the knowledge base it opens, under `name`
    or the owner's own."""
    if peers.reserved(name):
        # Decided before the redeem, which spends the one-time code.
        return {"status": "reserved_name",
                "message": f"'{peers.LOCAL_KB}' is how this install names its own knowledge "
                           f"base, so a shared one cannot be called that. Pass another `name`, "
                           f"or leave it out to use the owner's."}
    found = invite.parse(pasted, account.service_url())
    if found is None:
        return {"status": "not_an_invite",
                "message": f"that does not contain an Opyt invite code. An invite looks like "
                           f"{invite.link('<code>')}, where the code is {invite.CODE_LEN} letters "
                           f"and digits. Ask them to send the link again."}
    try:
        res = client.redeem(found.url, found.code, account.install_id())
    except client.Failed:
        return {"status": "code_unavailable",
                "message": ("that invite code has already been used or is not one this "
                            "service issued — a code works exactly once. Ask them to send a "
                            "new one.")}
    asked = peers.local_name(name, res.get("suggested_name"), res["owner"])
    kb = peers.add(asked, client.kb_location(found.url, res["owner"]),
                   res.get("suggested_name"), token=res["token"])
    return _accepted(kb, asked, res)


def _accepted(kb: str, asked: str, res: dict) -> dict:
    """The answer: the name to pass as `kb=`, and why it differs from the one asked for."""
    out = {"status": "accepted", "kb": kb, "owner": res["owner"],
           "message": f"Search it with kb='{kb}'."}
    if kb != asked:
        out["renamed_from"] = asked
        out["message"] += (f" '{asked}' was already another knowledge base on this install, "
                           f"so this one is '{kb}' — tell the user that.")
    if res.get("notice"):
        out["service_notice"] = res["notice"]
    return out
