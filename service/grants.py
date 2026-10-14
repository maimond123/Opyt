"""service/grants.py — one-time invite codes, `grant_codes`, and which of them are agent invites,
`agent_grants`.

A code buys exactly one token: a reader's, or an agent's when the code has an `agent_grants` row.
Only its hash is stored (`tokens.token_hash`), and its alphabet is the invite format's
(`pipeline/sharing/invite.py`).
"""
from __future__ import annotations

import secrets

from pipeline.sharing import invite
from service import db, tokens


def mint_grant(owner: str, *, label: str | None = None, agent: bool = False) -> str:
    """Mint a one-time code and return it once; only the hash is stored. `agent` makes it an
    agent invite, in the same transaction."""
    code = "".join(secrets.choice(invite.CODE_ALPHABET) for _ in range(invite.CODE_LEN))
    h = tokens.token_hash(code)
    conn = db.connect()
    try:
        conn.execute("INSERT INTO grant_codes (code_sha256, owner, label) VALUES (?,?,?)",
                     (h, owner, label))
        if agent:
            conn.execute("INSERT INTO agent_grants (code_sha256, owner) VALUES (?,?)", (h, owner))
        conn.commit()
    finally:
        conn.close()
    return code


class GrantUnavailable(RuntimeError):
    """A code that is not a code, or one already spent; one type so a stranger cannot tell which."""


def redeem_grant(code: str, install_id: str | None = None) -> tuple[str, str, str]:
    """Exchange a code for the token it buys, exactly once. Returns `(owner, token, role)`, the
    role being `agent` for an agent invite and `reader` for any other.

    The claim is one conditional UPDATE, so of two simultaneous redeems the loser updates zero rows.
    """
    h = tokens.token_hash(code)
    conn = db.connect()
    try:
        cur = conn.execute(
            "UPDATE grant_codes SET redeemed_at = datetime('now') "
            "WHERE code_sha256 = ? AND redeemed_at IS NULL", (h,))
        if cur.rowcount != 1:
            conn.commit()
            raise GrantUnavailable("that code is not valid, or has already been used.")
        row = conn.execute(
            "SELECT owner, label, EXISTS (SELECT 1 FROM agent_grants a "
            "WHERE a.code_sha256 = g.code_sha256) AS agent "
            "FROM grant_codes g WHERE g.code_sha256 = ?", (h,)).fetchone()
        owner, label, role = row["owner"], row["label"], "agent" if row["agent"] else "reader"
        conn.commit()
    finally:
        conn.close()
    # After the claim commits: a crash leaves the code spent with no token, the safe direction.
    return owner, tokens.mint_token(owner, role, label=label, install_id=install_id), role


def is_agent_invite(code: str) -> bool:
    """Whether `code` is an agent invite, read without spending it."""
    conn = db.connect()
    try:
        return conn.execute("SELECT 1 FROM agent_grants WHERE code_sha256 = ?",
                            (tokens.token_hash(code),)).fetchone() is not None
    finally:
        conn.close()


def agent_invite_open(owner: str) -> bool:
    """Whether `owner` has an agent invite nobody has redeemed yet."""
    conn = db.connect()
    try:
        return conn.execute(
            "SELECT 1 FROM agent_grants a JOIN grant_codes g ON g.code_sha256 = a.code_sha256 "
            "WHERE a.owner = ? AND g.redeemed_at IS NULL LIMIT 1", (owner,)).fetchone() is not None
    finally:
        conn.close()
