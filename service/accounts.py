"""
service/accounts.py — which library a Google account reads, and the sign-in that ties them.

The connector knows a caller by their Google account (`sub`). The library is known by its routing
key. `link` ties them. A builder holding the library's owner token calls it, so the row proves
both halves: the token proves the library, and Google's answer proves the account.

One route drives the whole sign-in, as `opyt_core.github_device` does on the client: a sign-in in
flight is asked about once, otherwise a fresh code is requested. `account_links` holds the one in
flight per library.
"""
from __future__ import annotations

from service import db, google_device


def owner_of(sub: str) -> str | None:
    """The library this Google account reads, or None when no builder has linked one."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT owner FROM accounts WHERE google_sub = ?", (sub,)).fetchone()
    finally:
        conn.close()
    return row["owner"] if row is not None else None


def linked(owner: str) -> bool:
    """Whether any Google account reads `owner` through the connector."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT 1 FROM accounts WHERE owner = ? LIMIT 1", (owner,)).fetchone()
    finally:
        conn.close()
    return row is not None


def linked_at(owner: str) -> str | None:
    """When a Google account was last linked to `owner`, or None when none is."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT MAX(linked_at) AS at FROM accounts WHERE owner = ?",
                           (owner,)).fetchone()
    finally:
        conn.close()
    return row["at"]


def unlink(owner: str) -> bool:
    """Cut every Google account off from `owner`, with any sign-in in flight. Returns whether
    one was linked. The served copy and every reader stay."""
    conn = db.connect()
    try:
        with conn:
            n = conn.execute("DELETE FROM accounts WHERE owner = ?", (owner,)).rowcount
            conn.execute("DELETE FROM account_links WHERE owner = ?", (owner,))
    finally:
        conn.close()
    return n > 0


def link(owner: str, now: float) -> dict:
    """One step of tying `owner` to a Google account:

      `{status: "waiting", user_code, verification_url}`  the user has a code to enter
      `{status: "linked"}`                                 approved; the account reads `owner`
      `{status: "failed", message}`                        denied or expired; the next call
                                                           starts a new sign-in

    Raises `google_device.Unavailable` when Google cannot be asked; nothing is written then, so
    the same code is asked about again on the next call."""
    pending = _in_flight(owner, now)
    if pending is None:
        return _start(owner, now)
    answer = google_device.poll(pending["device_code"])
    if answer.status == "waiting":
        return _waiting(pending)
    conn = db.connect()
    try:
        if answer.status == "approved":
            conn.execute("INSERT INTO accounts (google_sub, owner) VALUES (?, ?) "
                         "ON CONFLICT(google_sub) DO UPDATE SET owner = excluded.owner, "
                         "linked_at = datetime('now')", (answer.sub, owner))
        conn.execute("DELETE FROM account_links WHERE owner = ?", (owner,))
        conn.commit()
    finally:
        conn.close()
    if answer.status == "approved":
        return {"status": "linked"}
    return {"status": "failed", "message": answer.message}


def _in_flight(owner: str, now: float) -> dict | None:
    """The sign-in in flight for `owner`, or None when there is none or Google's code expired."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM account_links WHERE owner = ?", (owner,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None and row["expires_at"] > now else None


def _start(owner: str, now: float) -> dict:
    """Ask Google for a code and keep it as this library's sign-in in flight, replacing any
    expired one."""
    code = google_device.request_code()
    conn = db.connect()
    try:
        conn.execute("INSERT INTO account_links "
                     "(owner, device_code, user_code, verification_url, expires_at) "
                     "VALUES (?, ?, ?, ?, ?) ON CONFLICT(owner) DO UPDATE SET "
                     "device_code = excluded.device_code, user_code = excluded.user_code, "
                     "verification_url = excluded.verification_url, "
                     "expires_at = excluded.expires_at",
                     (owner, code["device_code"], code["user_code"], code["verification_url"],
                      now + code["expires_in"]))
        conn.commit()
    finally:
        conn.close()
    return _waiting(code)


def _waiting(link_row: dict) -> dict:
    return {"status": "waiting", "user_code": link_row["user_code"],
            "verification_url": link_row["verification_url"]}
