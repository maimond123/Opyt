"""service/uploads.py — an owner replaces the export this service serves for them.

One sentence, one module. Receiving bytes and making them the served file is all this does; who
was allowed to send them is `service/store.py`'s question and `service/app.py` asks it first.

FULL REPLACE, NEVER A PATCH. An export is a projection of a store, not a log of changes to one
(I11), so "the newest upload wins" is the whole update model. An upload path that merged the
incoming file into the served one would be exactly the synchronization machinery the design
forbids — two writable homes for the same atoms, plus a reconciler that can only ever drift.

ATOMIC, so a partial upload is never served (P3). Bytes land in `<owner>.db.uploading` and become
`<owner>.db` with one `os.replace` — the same discipline `pipeline/kb/export.py` already uses
when it builds one. A reader holding the old file during the swap keeps reading the old file; a
killed upload leaves the old one untouched and still queryable.

Registration goes through `pipeline/kb/peers.add`, which is idempotent and writes the SAME row a
local reader's `kb=` resolves through. That is what makes the query path the identical function
call whether the export sits on the reader's disk or on this server, and it is why nothing here
records when an upload happened: the file's own mtime and hash answer that, and a second record
of who is served is a second thing that can be wrong.

Design record: docs/plans/2026-08-26-foreign-kb-service-phase3.md §3.3.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from opyt_core.paths import opyt_path
from pipeline.kb import peers

# `owner` reaches this module from a URL path segment and becomes a FILENAME, so it is validated
# rather than trusted. An allow-list is the only version of this check that cannot be argued
# with: `..`, `/` and a leading dot are all outside it without being enumerated.
_OWNER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class BadOwner(ValueError):
    """A `{owner}` path segment that is not a name this service will turn into a filename."""


def valid_owner(owner: str) -> str:
    """The one place an owner name is checked, at the boundary it arrives through.

    Also rejects `peers.LOCAL_KB` ("me"): that name is how a reader says "my own store", so a
    knowledge base served under it would be permanently unreachable — registered, listed, and
    never openable. `peers.add` raises on it too; catching it here means a doomed upload never
    writes the file first."""
    if not _OWNER.match(owner or "") or owner == peers.LOCAL_KB:
        raise BadOwner(
            f"{owner!r} is not a usable knowledge-base name: lowercase letters, digits, "
            f"'-' and '_', starting with a letter or digit, and not '{peers.LOCAL_KB}'.")
    return owner


def exports_dir() -> Path:
    return opyt_path("exports")


def export_path(owner: str) -> Path:
    """Where this owner's served export lives — the WRITER's answer. A reader asks `peers`, which
    `Receiver.commit` registers, so the two cannot disagree by construction."""
    return exports_dir() / f"{valid_owner(owner)}.db"


class Receiver:
    """One upload in progress: `write` the bytes as they arrive, then `commit` or `abort`.

    A class rather than a function taking an iterable because the bytes arrive from an ASYNC HTTP
    stream and this module must not learn what HTTP is. The caller drives the loop; the
    tmp-file/fsync/rename/register discipline stays here, where "how an export becomes the served
    file" is the one thing this module owns.

    Memory is one chunk, whatever size the caller reads in — a 115 MB export is never held whole.
    The sha256 is computed on the way past rather than by re-reading the finished file, because it
    is what the owner compares against their local copy to know the upload arrived intact, and a
    hash of a re-read file would agree with itself even if the write had gone wrong.
    """

    def __init__(self, owner: str):
        self.owner = valid_owner(owner)
        self.dest = export_path(self.owner)
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.dest.with_name(self.dest.name + ".uploading")
        self.tmp.unlink(missing_ok=True)
        self._fh = open(self.tmp, "wb")
        self._digest = hashlib.sha256()
        self.bytes = 0

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._fh.write(chunk)
        self._digest.update(chunk)
        self.bytes += len(chunk)

    def commit(self, *, label: str | None = None) -> dict:
        """Make what was written the served export. `{owner, bytes, sha256, path}`.

        The `os.replace` is the commit: everything before it is invisible to a reader and
        everything after it is already durable, so there is no half-served state to design for.
        `peers.add` follows rather than precedes it — a registry row pointing at a file that does
        not exist yet is exactly the window this ordering removes."""
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        os.replace(self.tmp, self.dest)
        peers.add(self.owner, self.dest, label)
        return {"owner": self.owner, "bytes": self.bytes,
                "sha256": self._digest.hexdigest(), "path": str(self.dest)}

    def abort(self) -> None:
        """Drop a partial upload. Fail-safe: the PREVIOUS export is still in place and still
        queryable, because `commit` is the only thing that ever touches it. Idempotent, so a
        caller can abort from an exception handler without knowing how far it got."""
        if not self._fh.closed:
            self._fh.close()
        self.tmp.unlink(missing_ok=True)
