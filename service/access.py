"""service/access.py — what the calling token may reach.

FastAPI dependencies, one per question a route asks about its caller. Each returns the token's
row or raises the HTTP refusal, so a route never runs for a caller it would refuse.
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from service import tokens


def bearer(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    return authorization[7:].strip()


def token_row(token: str = Depends(bearer)) -> dict:
    """The row a bearer token names; 401 otherwise, including a revoked token (a deleted row).

    No connection crosses the dependency boundary: FastAPI runs dependencies and handlers on
    different threads, and a `sqlite3` connection is bound to its thread."""
    row = tokens.resolve_token(token)
    if row is None:
        raise HTTPException(status_code=401, detail="unknown or revoked token")
    return row


def owner_token(auth: dict = Depends(token_row)) -> dict:
    """An owner token, acting on its own `owner`, never a path segment."""
    if auth["role"] != "owner":
        raise HTTPException(status_code=403, detail="this action needs an owner token")
    return auth


def agent_token(auth: dict = Depends(token_row)) -> dict:
    """An agent token, acting as the owner on the web tools for its own `owner`, never a path
    segment. A reader token never acts, and the owner's own token is for its install's routes."""
    if auth["role"] != "agent":
        raise HTTPException(status_code=403, detail="this action needs an agent token")
    return auth


def reader_of(owner: str, auth: dict = Depends(token_row)) -> dict:
    """A reader or agent token for the knowledge base named in the path. An owner token does not
    read here."""
    if auth["role"] not in ("reader", "agent") or auth["owner"] != owner:
        raise HTTPException(status_code=403, detail="this token cannot read that knowledge base")
    return auth


def publisher(owner: str, auth: dict = Depends(owner_token)) -> dict:
    """An owner token allowed to replace the export served under `owner`: the path must agree
    with the token, and the token must hold the name's claim (`tokens.claim_holder`)."""
    if auth["owner"] != owner:
        raise HTTPException(status_code=403, detail="this token cannot upload that knowledge base")
    if tokens.claim_holder(owner) != auth["token_sha256"]:
        # A second owner token for the same name, which an older database can hold.
        raise HTTPException(status_code=403,
                            detail="this name is published under a different owner token")
    return auth
