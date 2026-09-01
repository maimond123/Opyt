"""
pipeline/kb/peers.py — which knowledge bases this install can read, and how to open one.

One sentence, one module. A `peer` is somebody else's knowledge base, projected into an export
file by `pipeline/kb/export.py` and registered here under a short name. `search`/`open`/`aggregate`
take that name as `kb=` and read the peer's store instead of the local one; nothing else about
retrieval changes, because an export is the same substrate the local tools already run against.
Design record: docs/plans/2026-08-26-foreign-kb-export-builder-phase1.md (Part 2).

Three properties this module holds:

  • READ-ONLY, MECHANICALLY. `open_peer` returns `schema.connect(..., read_only=True)`. A reader
    cannot write to somebody else's KB because SQLite refuses, not because every caller remembers
    to. That is invariant I3, enforced at the one place a peer store is ever opened.

  • ONE FAILURE TYPE AT THE BOUNDARY. `kb=` is a string a host model typed. Unregistered and
    registered-but-unreadable are two causes of one fact — *this name cannot be read right now* —
    and the tool layer answers both the same way: an empty result plus a notice that says why
    (P3, fail-safe). So both raise `PeerUnavailable`, with the cause in the message.

  • THE LOCATION IS RESOLVED ON THE WAY IN, when it is a path. `add()` stores an absolute one; a
    relative path would resolve against whatever directory the process happened to start in, so
    the same registry row would name a different file per invocation. A URL is stored VERBATIM —
    putting one through `Path.resolve()` mangles it into a filesystem path, silently.

A `location` is one of two things: a path to an export on this disk, or an `https://` base URL
for a knowledge base served by `service/app.py`. `is_remote()` is the one place they are told
apart. THIS MODULE OPENS ONLY THE FIRST KIND — a remote peer is routed over HTTP by
`opyt_core/kb_remote.py`, before an entry point reaches `open_peer`, which is the whole reason
the seam was drawn at the store-opening boundary rather than inside `retrieve.py`.

WHO WRITES A ROW HERE. `mcp_server/share_tools.accept` is the surface a person reaches — it
redeems an invite and registers the peer in one call, because a reader who has to open a shell is
a reader the design does not get. `opyt-redeem` does the same thing from a terminal and stays as
the operator rail. `service/uploads.Receiver.commit` writes the SERVICE's own row for the export
it just received, which is a different job under the same schema. A file peer on this disk still
takes a Python prompt, and has no other caller:

    python -c "from pipeline.kb import peers; peers.add('david', '/path/export.db', \"David's KB\")"
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import schema

# The one name that can never be a peer: `kb="me"` is how a caller says "my own store", so a row
# under that name would be permanently unreachable — registered, listed, and never openable.
LOCAL_KB = "me"


class PeerUnavailable(RuntimeError):
    """A `kb=` name this install cannot read: not registered, or registered and unreadable.

    One type for both because a caller does the same thing with either — there is no result, and
    the notice says which it was. The tool layer catches this and returns an empty envelope; it
    must never escape a tool call."""


def is_remote(location: str) -> bool:
    """Whether this peer is served over HTTP rather than being a file on this disk.

    One predicate, read by `add` (do not resolve a URL as a path), by `open_peer` (refuse), and by
    the three `opyt_core/kb.py` entry points (route over the network). A second table for remote
    peers would have made every one of those a two-lookup question instead."""
    return location.startswith(("http://", "https://"))


def add(name: str, location: str | Path, label: str | None = None, *,
        token: str | None = None, rename_on_collision: bool = True) -> str:
    """Register a knowledge base under `name`. Returns the name it ACTUALLY landed under.

    ⚠️`peers.token` is the only copy of a reader bearer token in existence — the service stores
    only its sha256 and hands the clear text over exactly once, at redemption — so a row this
    function overwrote is an UNRECOVERABLE credential, restorable only by a fresh grant code from
    that owner. That is why it never replaces a row it did not recognise.

    ONE rule, applied to `name`, then `name-2`, `name-3`, … until it settles:

      • nothing registered there → insert, and that is the name
      • registered at the SAME location → the same knowledge base, registered again: update the
        label and token, which is the genuine re-redeem (a revoked token being replaced)
      • registered somewhere else → a different knowledge base wearing this name → try the next

    Applying it at every candidate rather than only the first is what makes re-redeeming an
    already-suffixed peer land back on `alex-2` instead of minting `alex-3` on every attempt.
    Auto-suffixing rather than prompting is the frictionless constraint: the caller reads the
    returned name and says which one it got.

    `rename_on_collision=False` turns the suffix off and makes the name authoritative —
    overwrite whatever is there. Its one caller is `service/uploads.Receiver.commit`, where name
    and location are welded by `export_path(owner)`, so a differing location does not mean two
    knowledge bases; it means the exports directory moved and the row is stale. Suffixing there
    would leave `open_peer(owner)` resolving the OLD file and serving a stale export silently.

    A PATH is resolved and `~` expanded here rather than at open time, so the row means the same
    file no matter which directory a later process runs from. A URL is stored verbatim: putting
    one through `Path.resolve()` turns `https://host/v1/kb/x` into a path under the working
    directory, which fails much later and says nothing useful when it does. Only the trailing
    slash goes, so the transport can join paths without checking for one.

    Neither kind is contacted here — a peer can be registered before its export lands or while
    its service is down, and `open_peer` (or the transport) reports the failure at read time.

    `token` is the reader bearer token for a remote peer. It lives in the registry beside the peer
    it opens rather than in `.env`, because a reader can hold tokens for several knowledge bases
    at once and one environment variable has nowhere to put the second."""
    if name == LOCAL_KB:
        raise ValueError(f"'{LOCAL_KB}' names your own store and cannot be a peer")
    if not name:
        raise ValueError("a peer needs a name")
    location = str(location)
    path = location.rstrip("/") if is_remote(location) else str(
        Path(location).expanduser().resolve())
    conn = schema.connect()
    try:
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
        conn.commit()
    finally:
        conn.close()
    return candidate


def remove(name: str) -> bool:
    """Deregister a knowledge base. Returns whether a row was there.

    The mirror of `add`. This row is what `open_peer` resolves, so deleting it is what makes the
    name stop answering — which is why an owner unpublishing calls this on the SERVICE's registry
    before deleting the export file.

    Deleting that file is deliberately NOT this function's job: a file peer's `location` is a
    path on somebody else's disk, and a module that removed it would be deleting data it never
    wrote."""
    conn = schema.connect()
    try:
        n = conn.execute("DELETE FROM peers WHERE name = ?", (name,)).rowcount
        conn.commit()
    finally:
        conn.close()
    return n > 0


def _read_registry(sql: str, params=()) -> list:
    """Run one SELECT against the registry, READ-ONLY, and hand back plain rows.

    Read-only is load-bearing here and was not always: `schema.connect()` opens WRITABLE and runs
    the whole idempotent DDL — `_ensure_column`, the backfills, the entry-mode rename. Finding
    out WHERE a peer's file lives is a single SELECT and needs none of it. On one laptop the
    difference is invisible; on a server it means every query runs the full migration against
    `opyt.db` before it can even locate the export, and several at once take a write lock on the
    same file. MEASURED 2026-08-26: 8 concurrent reads → three `database is locked` 500s.

    An `OperationalError` here means the store or its `peers` table is not there, which is one
    fact — NOTHING IS REGISTERED — so it degrades to no rows rather than raising. That is also
    why the read-only open is safe on a never-ingested install: a writable open would have
    CREATED the table as a side effect of asking whether it had anything in it."""
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
    """Every registered peer, oldest first. Used by the tool layer to tell a caller who guessed a
    name what the real ones are."""
    return [dict(r) for r in _read_registry(
        "SELECT name, location, label, added_at FROM peers ORDER BY added_at, name")]


def get(name: str) -> dict | None:
    """One peer's row, or None. The lookup the `kb=` entry points do BEFORE opening anything:
    `location` says which transport, and `token` is the credential the remote one needs.

    Deliberately not folded into `list_peers` + a filter — that reads the whole registry to
    answer a keyed question, on every single search."""
    rows = _read_registry("SELECT name, location, label, token FROM peers WHERE name = ?", (name,))
    return dict(rows[0]) if rows else None


def open_peer(name: str) -> tuple[sqlite3.Connection, str | None]:
    """`(conn, label)` for a registered peer, opened READ-ONLY. Raises `PeerUnavailable`.

    Read-only is how invariant I3 is enforced: this is the only place a peer store is opened, so
    "a reader never writes to somebody else's KB" is a property of SQLite rather than of every
    caller's discipline. It also skips the idempotent DDL, which is what makes reading a peer
    leave its file byte-identical.

    The `OperationalError` catch is scoped to the connect itself — the failure it names is a file
    that moved, was deleted, or is not a database. A query failing later is a different problem and
    still raises, so a genuinely corrupt store is loud rather than silently empty.

    Both opens here are read-only: the registry lookup via `_read_registry` (see its docstring for
    why that matters under concurrency) and the export itself. So resolving and opening a peer
    takes no write lock anywhere, which is what lets several readers be served at once."""
    rows = _read_registry("SELECT location, label FROM peers WHERE name = ?", (name,))
    if not rows:
        raise PeerUnavailable(f"no knowledge base is registered as '{name}'.")
    row = rows[0]
    location = row["location"]
    if is_remote(location):
        # The producer is an entry point that skipped the remote branch, so the message names
        # that rather than the file-shaped failure `schema.connect` would report about a string
        # that was never a path.
        raise PeerUnavailable(
            f"'{name}' is served over HTTP; this path opens local files — the kb entry points "
            f"route remote peers before reaching here.")
    try:
        return schema.connect(location, read_only=True), row["label"]
    except sqlite3.OperationalError as e:
        raise PeerUnavailable(
            f"'{name}' is registered at {location}, but that file could not be opened ({e})."
        ) from e
