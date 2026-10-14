"""service/tokens.py — who may call this service: bearer tokens, with the routing-key claim each
owner token holds.

`tokens` holds a credential's hash, never the credential; revocation is a row delete, effective on
the next request. `owner_claims` holds one row per published routing key, since the key is the
served file and the path every reader's peer row points at. A claim outlives its tokens, and is
released only when the whole library is deleted (`deletion.py`), with every token that could
reach it.

Three roles, each a kind of key for one library: `owner` (the install that publishes it),
`reader` (someone the owner invited, who only reads) and `agent` (the owner's own agent, which
reads and acts as the owner on the web tools). What each may reach is `access.py`'s.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3

from service import db

# A URL-safe 256-bit secret; long enough that auth attempts need no rate limit.
_TOKEN_BYTES = 32

ROLES = ("owner", "reader", "agent")


def token_hash(token: str) -> str:
    """The one hashing rule; every read and write of a credential goes through it."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class NameClaimed(RuntimeError):
    """An owner name already published under, or a reclaim past a live owner token."""


def _claim_name(conn: sqlite3.Connection, owner: str, token_sha256: str, *, reclaim: bool) -> None:
    """Take or repoint the name's claim, inside the mint's own transaction.

    The primary key decides a simultaneous-claim race. A reclaim requires no live owner token, so
    rotating a leaked token is revoke, then `reclaim=True`."""
    if not reclaim:
        try:
            conn.execute("INSERT INTO owner_claims (owner, token_sha256) VALUES (?, ?)",
                         (owner, token_sha256))
        except sqlite3.IntegrityError:
            raise NameClaimed(
                f"'{owner}' is already claimed, and a name is never released — it is the address "
                f"in every reader's peer row. To rotate this owner's token: revoke it, then mint "
                f"again with reclaim=True.") from None
        return
    live = conn.execute("SELECT 1 FROM tokens WHERE owner = ? AND role = 'owner' LIMIT 1",
                        (owner,)).fetchone()
    if live is not None:
        raise NameClaimed(f"'{owner}' still has a live owner token — revoke it before reclaiming.")
    conn.execute("INSERT INTO owner_claims (owner, token_sha256) VALUES (?, ?) "
                 "ON CONFLICT(owner) DO UPDATE SET token_sha256 = excluded.token_sha256",
                 (owner, token_sha256))


def mint_token(owner: str, role: str, *, label: str | None = None,
               install_id: str | None = None, reclaim: bool = False) -> str:
    """Create a token for `owner` and return it once, in the clear. Only its hash is stored.

    An owner mint also takes the name's claim, refusing (`NameClaimed`) when it is held, even by a
    revoked token. `reclaim=True` moves the claim once no live owner token remains."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    conn = db.connect()
    try:
        if role == "owner":
            _claim_name(conn, owner, token_hash(token), reclaim=reclaim)
        conn.execute(
            "INSERT INTO tokens (token_sha256, owner, role, label, install_id) VALUES (?,?,?,?,?)",
            (token_hash(token), owner, role, label, install_id),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def register_owner(label: str | None = None) -> tuple[str, str]:
    """A brand-new knowledge base: `(routing_key, owner_token)`. The token is returned once.

    The routing key is an assigned opaque address (`secrets.token_hex(6)`, which satisfies
    `_OWNER` and is never `"me"`). No rate limit or identity check, by design. `NameClaimed`
    here is a key collision, so the loop retries with a fresh key."""
    for _ in range(5):
        key = secrets.token_hex(6)
        try:
            return key, mint_token(key, "owner", label=label)
        except NameClaimed:
            continue
    raise NameClaimed("could not assign a free knowledge-base key")


def claim_holder(owner: str) -> str | None:
    """The hash of the one token allowed to publish under this name, or None if none was minted."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT token_sha256 FROM owner_claims WHERE owner = ?",
                           (owner,)).fetchone()
    finally:
        conn.close()
    return row["token_sha256"] if row is not None else None


def owner_label(owner: str) -> str | None:
    """The label on this knowledge base's owner token, or None; what `redeem` hands a new reader
    as `suggested_name`. A suggestion only, never a routing fact or unique.

    Read from the owner token because the registry row may not exist yet when a reader redeems."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT label FROM tokens WHERE owner = ? AND role = 'owner' "
            "ORDER BY created_at LIMIT 1", (owner,)).fetchone()
    finally:
        conn.close()
    return row["label"] if row is not None else None


def resolve_token(token: str) -> dict | None:
    """The row a bearer token names, as a plain dict (the connection closes first), or None.
    The `owner` column is the scope."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT token_sha256, owner, role, label, install_id, created_at FROM tokens "
            "WHERE token_sha256 = ?", (token_hash(token),)).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def list_tokens(owner: str) -> list[dict]:
    """Every token issued for this knowledge base, oldest first, with the hash `revoke` takes."""
    conn = db.connect()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT token_sha256, role, label, install_id, created_at FROM tokens "
            "WHERE owner = ? ORDER BY created_at, token_sha256", (owner,))]
    finally:
        conn.close()


def revoke(owner: str, token_sha256: str) -> bool:
    """Delete one token. Returns whether a row was there.

    Scoped in the WHERE clause, so another owner's hash deletes nothing and reveals nothing."""
    conn = db.connect()
    try:
        n = conn.execute("DELETE FROM tokens WHERE owner = ? AND token_sha256 = ?",
                         (owner, token_sha256)).rowcount
        conn.commit()
    finally:
        conn.close()
    return n > 0


