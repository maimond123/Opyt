"""
pipeline/retrieval/peers.py — which knowledge bases this install can read, and how to open one.

A peer is somebody else's knowledge base, projected into an export file by
`pipeline/sharing/export.py` (or served over HTTP) and registered here under a short name that
`search`/`open`/`aggregate` take as `kb=`.

  • Read-only, mechanically: `open_peer` opens with `read_only=True`, so SQLite itself refuses a
    write to somebody else's KB (invariant I3).
  • One failure type at the boundary: unregistered and unreadable both raise `PeerUnavailable`.
  • A `location` is an absolute path to an export on this disk or an `https://` base URL;
    `is_remote()` tells them apart. This module opens only files; `served.py` routes remote peers.

The `accept` tool (`pipeline/sharing/accept.py`) writes a reader's row;
`service/uploads.Receiver.commit` writes the service's own row. A file peer takes a Python prompt:

    python -c "from pipeline.retrieval import peers; peers.add('alice', '/path/export.db', \"Alice's KB\")"
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pipeline.kb import schema
from pipeline.kb.writes import write_tx

# The one name that can never be a peer: `kb="me"` means the caller's own store.
LOCAL_KB = "me"


def reserved(name: str | None) -> bool:
    """Whether `name` is `LOCAL_KB`, the one name a peer can never have. The one place the rule is
    checked, by `add`, `accept` and the sharing service."""
    return name == LOCAL_KB


def local_name(name: str | None, suggested: str | None, owner: str) -> str:
    """What a reader calls a knowledge base shared with them: the name asked for, else the
    owner's own label, else the routing key. An owner who labelled themselves with the reserved
    local name falls back to the routing key, which is never that name
    (`service/tokens.register_owner`). One rule for an install's `accept` and the connector's."""
    if name:
        return name
    if suggested and not reserved(suggested):
        return suggested
    return owner


class PeerUnavailable(RuntimeError):
    """A `kb=` name this install cannot read: not registered, or registered and unreadable.

    The tool layer catches this and returns an empty envelope with a notice; it must never escape
    a tool call."""


def is_remote(location: str) -> bool:
    """Whether this peer is served over HTTP rather than being a file on this disk.

    Read by `add` (do not resolve a URL as a path), `open_peer` (refuse) and `reads._served_row`
    (route over the network)."""
    return location.startswith(("http://", "https://"))


def add(name: str, location: str | Path, label: str | None = None, *,
        token: str | None = None, rename_on_collision: bool = True) -> str:
    """Register a knowledge base under `name`. Returns the name it actually landed under.

    `peers.token` is the only copy of a reader bearer token, so a row is never overwritten unless
    it is the same location. Tried at `name`, then `name-2`, `name-3`, …:

      • nothing registered there → insert
      • registered at the same location → update label and token (a re-redeem)
      • registered somewhere else → try the next suffix

    `rename_on_collision=False` makes the name authoritative and overwrites (used by
    `service/uploads.Receiver.commit`, where name and location are tied by `export_path(owner)`).

    A path is resolved and `~` expanded here; a URL is stored verbatim minus a trailing slash.
    Nothing is contacted; failures surface at read time."""
    if reserved(name):
        raise ValueError(f"'{LOCAL_KB}' names your own store and cannot be a peer")
    if not name:
        raise ValueError("a peer needs a name")
    location = str(location)
    path = location.rstrip("/") if is_remote(location) else str(
        Path(location).expanduser().resolve())
    conn = schema.connect()
    try:
        with write_tx(conn, site="peers.add"):
            candidate, n = name, 1
            while True:
                row = conn.execute("SELECT location FROM peers WHERE name = ?",
                                   (candidate,)).fetchone()
                if row is None:
                    conn.execute("INSERT INTO peers (name, location, label, token) "
                                 "VALUES (?, ?, ?, ?)", (candidate, path, label, token))
                    break
                if row["location"] == path or not rename_on_collision:
                    conn.execute("UPDATE peers SET location = ?, label = ?, token = ? "
                                 "WHERE name = ?", (path, label, token, candidate))
                    break
                n += 1
                candidate = f"{name}-{n}"
    finally:
        conn.close()
    return candidate


def remove(name: str) -> bool:
    """Deregister a knowledge base. Returns whether a row was there.

    Does not delete the export file: a file peer's `location` is data this module never wrote."""
    conn = schema.connect()
    try:
        with write_tx(conn, site="peers.remove"):
            n = conn.execute("DELETE FROM peers WHERE name = ?", (name,)).rowcount
    finally:
        conn.close()
    return n > 0


def _read_registry(sql: str, params=()) -> list:
    """Run one SELECT against the registry, read-only, and hand back plain rows.

    Read-only so concurrent lookups take no write lock and skip the DDL. An `OperationalError`
    means nothing is registered, so it returns no rows."""
    try:
        conn = schema.connect(read_only=True)
    except sqlite3.OperationalError:
        return []
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def list_peers() -> list[dict]:
    """Every registered peer, oldest first."""
    return [dict(r) for r in _read_registry(
        "SELECT name, location, label, added_at FROM peers ORDER BY added_at, name")]


def get(name: str) -> dict | None:
    """One peer's row, or None: `location` picks the transport, `token` is the remote credential."""
    rows = _read_registry("SELECT name, location, label, token FROM peers WHERE name = ?", (name,))
    return dict(rows[0]) if rows else None


def open_peer(name: str) -> tuple[sqlite3.Connection, str | None]:
    """`(conn, label)` for a registered peer, opened read-only (invariant I3). Raises
    `PeerUnavailable`.

    Only the connect is caught, so a store that is corrupt past opening still raises loudly. A
    copy with no `bodies_fts` was made by an Opyt older than whole-item search, which this
    search cannot read; it is answered as unavailable until its owner's next push replaces it."""
    rows = _read_registry("SELECT location, label FROM peers WHERE name = ?", (name,))
    if not rows:
        raise PeerUnavailable(f"no knowledge base is registered as '{name}'.")
    row = rows[0]
    location = row["location"]
    if is_remote(location):
        # An entry point skipped the remote branch; say so rather than report a file failure.
        raise PeerUnavailable(
            f"'{name}' is served over HTTP; this path opens local files — the kb entry points "
            f"route remote peers before reaching here.")
    try:
        conn = schema.connect(location, read_only=True)
    except sqlite3.OperationalError as e:
        raise PeerUnavailable(
            f"'{name}' is registered at {location}, but that file could not be opened ({e})."
        ) from e
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'bodies_fts'").fetchone() is None:
        conn.close()
        raise PeerUnavailable(
            f"'{name}' is a copy made by an older Opyt; it refreshes on its owner's next push.")
    return conn, row["label"]
