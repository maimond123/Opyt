"""service/uploads.py — an owner's bytes onto this service's disk, capped, and made the served copy.

A whole export replaces what is served: bytes land in `<owner>.db.uploading` and become
`<owner>.db` with one `os.replace`, so a partial upload is never served. A delta lands beside it
in `<owner>.db.delta` and is never served itself; `serving.receive_delta` patches the copy with
it. Registration writes the same peers row a local reader's `kb=` resolves through.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from opyt_core.paths import opyt_path
from pipeline.retrieval import peers

# A latency and disk limit: every served copy and its search index live on this box.
MAX_EXPORT_BYTES = 500 * 1_000_000

# Checked before an upload starts, so a full disk cannot short-write a truncated export.
# At least 4x the cap, with headroom for `service.db`'s backups on the same volume.
FREE_BYTES_FLOOR = 2 * 1024 ** 3

# `owner` arrives from a URL path and becomes a filename, so it is checked against an allow-list.
_OWNER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class BadOwner(ValueError):
    """A `{owner}` path segment that is not a name this service will turn into a filename."""


class TooLarge(ValueError):
    """The body exceeded `MAX_EXPORT_BYTES`, counted as it arrives (not from `Content-Length`)."""


class NoSpace(RuntimeError):
    """Not enough free disk to accept an upload safely."""


def valid_owner(owner: str) -> str:
    """The one place an owner name is checked. Also rejects `peers.LOCAL_KB` ("me"), which a
    reader uses for their own store."""
    if not _OWNER.match(owner or "") or peers.reserved(owner):
        raise BadOwner(
            f"{owner!r} is not a usable knowledge-base name: lowercase letters, digits, "
            f"'-' and '_', starting with a letter or digit, and not '{peers.LOCAL_KB}'.")
    return owner


def exports_dir() -> Path:
    return opyt_path("exports")


def export_path(owner: str) -> Path:
    """Where this owner's served export lives (the writer's answer; readers ask `peers`)."""
    return exports_dir() / f"{valid_owner(owner)}.db"


class Receiver:
    """One upload in progress: `write` the bytes as they arrive, then `commit` (a whole export),
    `close` (a delta, to be applied), or `abort`.

    Holds one chunk at a time. The sha256 is computed on the way in, not by re-reading the file.
    """

    def __init__(self, owner: str, suffix: str = ".uploading"):
        self.owner = valid_owner(owner)
        self.dest = export_path(self.owner)
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        # Before the first byte; after the mkdir, since `disk_usage` needs an existing path.
        if shutil.disk_usage(self.dest.parent).free < FREE_BYTES_FLOOR:
            raise NoSpace(
                "this service is too low on disk to accept an upload right now. Nothing was "
                "changed and the knowledge base it was already serving is still being served.")
        self.tmp = self.dest.with_name(self.dest.name + suffix)
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
        if self.bytes > MAX_EXPORT_BYTES:
            raise TooLarge(
                f"this knowledge base is larger than the {MAX_EXPORT_BYTES // 1_000_000} MB "
                f"this service serves per knowledge base, so one oversized export cannot crowd out "
                f"every other reader's.")

    def close(self) -> Path:
        """Finish the file without serving it, and return where it is. `abort` removes it."""
        self._fh.flush()
        self._fh.close()
        return self.tmp

    def commit(self, *, label: str | None = None) -> dict:
        """Make what was written the served export. `{owner, bytes, sha256, path}`.

        `os.replace` is the commit; `peers.add` follows it, so no row names a missing file.
        `rename_on_collision=False`: here the name derives the path, so an existing row is stale
        and must be overwritten, not suffixed."""
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        # A patch that died mid-commit leaves the old file's rollback journal; beside the new
        # file, SQLite would play the old pages into it on the next writable open.
        self.dest.with_name(self.dest.name + "-journal").unlink(missing_ok=True)
        os.replace(self.tmp, self.dest)
        peers.add(self.owner, self.dest, label, rename_on_collision=False)
        return {"owner": self.owner, "bytes": self.bytes,
                "sha256": self._digest.hexdigest(), "path": str(self.dest)}

    def abort(self) -> None:
        """Drop a partial upload; the previous export is untouched. Idempotent."""
        if not self._fh.closed:
            self._fh.close()
        self.tmp.unlink(missing_ok=True)


def remove(owner: str) -> bool:
    """Stop serving this owner: deregister the export, then delete the file, so no row names a
    missing file. Returns whether a file was there. Idempotent.

    A partial upload goes too: a restart mid-upload leaves `<owner>.db.uploading` behind, and an
    upload still running then fails at its commit instead of serving again. So do a delta that was
    landing (`.delta`) and the journal of a patch that died mid-commit (`-journal`): both hold
    rows of the library."""
    dest = export_path(owner)
    peers.remove(owner)
    existed = dest.exists()
    dest.unlink(missing_ok=True)
    for suffix in (".uploading", ".delta", "-journal"):
        dest.with_name(dest.name + suffix).unlink(missing_ok=True)
    return existed
