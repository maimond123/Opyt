"""
service/account_peers.py — the libraries shared with a Google account, read through the connector.

Someone who was sent an invite and uses claude.ai or ChatGPT has no install to redeem it in, so
the connector redeems it for them (`accept`): the code buys a reader token as it does anywhere
(`grants.redeem_grant`), and a row here ties that token to the Google account that pasted it.
The owner sees and cuts off this reader like any other. The token itself is not kept: the
connector reads in-process, and a row whose token is gone has been cut off.

A name is what the caller passes as `kb=`: the owner's own label, with a number after it when
this account already reads another library by that name.
"""
from __future__ import annotations

from typing import NamedTuple

from pipeline.retrieval import peers
from pipeline.sharing import invite
from service import db, grants, tokens


class Peer(NamedTuple):
    """One library shared with an account. `reader` is the hash its reads are counted under;
    `live` is false once the owner has cut this reader off."""
    name: str
    owner: str
    reader: str
    live: bool


def accept(sub: str, pasted: str) -> dict:
    """Redeem the invite inside `pasted` for the Google account `sub`: `accepted` with the name
    to read it by, or the answer that says why not. An agent's invite is refused unspent."""
    found = invite.parse(pasted, "")
    if found is None:
        return {"status": "not_an_invite",
                "message": f"that does not contain an Opyt invite code. An invite looks like "
                           f"{invite.link('<code>')}. Ask the person who sent it to send the "
                           f"link again."}
    if grants.is_agent_invite(found.code):
        return {"status": "agent_invite",
                "message": "that link is for an agent the user runs, one that calls web "
                           "addresses, and it would be used up here. Nothing was used: give it "
                           "to the agent. To read a library in this app, the person sharing it "
                           "sends an ordinary invite."}
    try:
        owner, token, _role = grants.redeem_grant(found.code, install_id=None)
    except grants.GrantUnavailable:
        return {"status": "code_unavailable",
                "message": "that invite has already been used, or is not one Opyt issued: a "
                           "link works exactly once. Ask them to send a new one."}
    label = tokens.owner_label(owner)
    asked = peers.local_name(None, label, owner)
    name = _keep(sub, asked, owner, tokens.token_hash(token))
    out = {"status": "accepted", "kb": name, "from": label}
    if name != asked:
        out["renamed_from"] = asked
    return out


def _keep(sub: str, name: str, owner: str, reader: str) -> str:
    """Store the row and return the name it landed under. An account that already reads `owner`
    keeps that name with the new token; a name another library holds gets the next number."""
    conn = db.connect()
    try:
        with conn:
            held = conn.execute("SELECT name FROM account_peers WHERE google_sub = ? "
                                "AND owner = ?", (sub, owner)).fetchone()
            if held is not None:
                conn.execute("UPDATE account_peers SET token_sha256 = ? WHERE google_sub = ? "
                             "AND owner = ?", (reader, sub, owner))
                return held["name"]
            taken = {r["name"].lower() for r in conn.execute(
                "SELECT name FROM account_peers WHERE google_sub = ?", (sub,))}
            candidate, n = name, 1
            while candidate.lower() in taken:
                n += 1
                candidate = f"{name}-{n}"
            conn.execute("INSERT INTO account_peers (google_sub, name, owner, token_sha256) "
                         "VALUES (?, ?, ?, ?)", (sub, candidate, owner, reader))
            return candidate
    finally:
        conn.close()


def held(sub: str) -> list[Peer]:
    """Every library shared with `sub`, oldest first, each with whether its token still stands."""
    conn = db.connect()
    try:
        return [Peer(r["name"], r["owner"], r["token_sha256"], bool(r["live"]))
                for r in conn.execute(
                    "SELECT p.name, p.owner, p.token_sha256, EXISTS (SELECT 1 FROM tokens t "
                    "WHERE t.token_sha256 = p.token_sha256 AND t.owner = p.owner) AS live "
                    "FROM account_peers p WHERE p.google_sub = ? ORDER BY p.added_at, p.name",
                    (sub,))]
    finally:
        conn.close()


def forget(sub: str, peer: Peer) -> None:
    """Drop a row whose owner cut this account off, once its reader has been told: nothing is
    kept about a share that has ended."""
    conn = db.connect()
    try:
        with conn:
            conn.execute("DELETE FROM account_peers WHERE google_sub = ? AND name = ? "
                         "AND token_sha256 = ?", (sub, peer.name, peer.reader))
    finally:
        conn.close()


def named(peers: list[Peer], name: str) -> Peer | None:
    """The peer `name` picks, whatever its case."""
    return next((p for p in peers if p.name.lower() == name.strip().lower()), None)
