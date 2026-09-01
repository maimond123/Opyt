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

TWO LIMITS, AND THEY ARGUE FROM DIFFERENT THINGS.

`MAX_EXPORT_BYTES` is about LATENCY, not disk. Every search brute-force scans the export's vector
column — measured at 58% of a 117 MB file — on the one shared core, so a 2 GB export spends
seconds per query and degrades EVERY other owner's readers long before any disk fills. The harm
is real at ONE honest publisher and needs no abuser. Enforced here rather than in the client
because this service is the trust boundary; a client-side check would be advice.

`FREE_BYTES_FLOOR` is a fail-safe fix, not abuse defence. A disk that fills mid-upload
short-writes, and `commit` then `os.replace`s a TRUNCATED SQLite file into place and serves it —
partial state, served, which P3 forbids. Checked before the first byte is written, so a full disk
refuses the upload and leaves the previous export answering.

Registration goes through `pipeline/kb/peers.add` and writes the SAME row a local reader's `kb=`
resolves through. That is what makes the query path the identical function call whether the
export sits on the reader's disk or on this server.

Nothing HERE records that an upload happened — `service/app.py` calls `store.record_upload` after
this module's `commit` returns, so `service.db` and this module stay the two separate sentences
they were. What that table holds is not a mirror of the file: `first_published_at` is a fact the
filesystem never had, and it cannot be recovered once the first upload is over.

Design record: docs/plans/2026-08-26-foreign-kb-service-phase3.md §3.3.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from opyt_core.paths import opyt_path
from pipeline.kb import peers

# ~500 MB. Sized against the measured real export (117 MB, 2,876 atoms) with room for a store
# several times larger, and set here rather than in the deploy command because it is a property
# of what this service can serve at an acceptable latency, not of one box's disk.
MAX_EXPORT_BYTES = 500 * 1_000_000

# 2 GiB, at least 4x the cap so a permitted upload can never be the thing that crosses it, with
# headroom for `service.db`'s rotating backups, which share the volume.
FREE_BYTES_FLOOR = 2 * 1024 ** 3

# `owner` reaches this module from a URL path segment and becomes a FILENAME, so it is validated
# rather than trusted. An allow-list is the only version of this check that cannot be argued
# with: `..`, `/` and a leading dot are all outside it without being enumerated.
_OWNER = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class BadOwner(ValueError):
    """A `{owner}` path segment that is not a name this service will turn into a filename."""


class TooLarge(ValueError):
    """The body exceeded `MAX_EXPORT_BYTES`. Raised mid-stream, from `write`, because that is the
    first moment it is knowable: `Content-Length` is the client's claim about the body and this
    service counts what actually arrived."""


class NoSpace(RuntimeError):
    """Not enough free disk to accept an upload safely. Distinct from `TooLarge` because they
    answer differently: the owner can make their export smaller, and can do nothing at all about
    this box being full."""


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
        # BEFORE the first byte, and before the tmp file exists. See the module docstring: the
        # failure this prevents is a short write that `commit` would then serve as a truncated
        # database. The mkdir has to precede it — `disk_usage` needs a path that exists.
        if shutil.disk_usage(self.dest.parent).free < FREE_BYTES_FLOOR:
            raise NoSpace(
                "this service is too low on disk to accept an upload right now. Nothing was "
                "changed and the knowledge base it was already serving is still being served.")
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
        if self.bytes > MAX_EXPORT_BYTES:
            raise TooLarge(
                f"this knowledge base is larger than the {MAX_EXPORT_BYTES // 1_000_000} MB "
                f"this service serves per knowledge base. Every search scans the whole vector "
                f"column, so one oversized export slows down every other reader.")

    def commit(self, *, label: str | None = None) -> dict:
        """Make what was written the served export. `{owner, bytes, sha256, path}`.

        The `os.replace` is the commit: everything before it is invisible to a reader and
        everything after it is already durable, so there is no half-served state to design for.
        `peers.add` follows rather than precedes it — a registry row pointing at a file that does
        not exist yet is exactly the window this ordering removes.

        `rename_on_collision=False` because the name is authoritative HERE and nowhere else: a
        reader's registry can hold two people called `alex`, so `add` auto-suffixes for them,
        but on this service `export_path(owner)` derives the location FROM the name, so a row
        under `owner` naming a different file is a stale row (the exports directory moved), not
        a second knowledge base. Suffixing it would register the new export as `owner-2` while
        `app.py`'s `open_peer(owner)` kept resolving the old file — a stale export served
        silently, forever."""
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        os.replace(self.tmp, self.dest)
        peers.add(self.owner, self.dest, label, rename_on_collision=False)
        return {"owner": self.owner, "bytes": self.bytes,
                "sha256": self._digest.hexdigest(), "path": str(self.dest)}

    def abort(self) -> None:
        """Drop a partial upload. Fail-safe: the PREVIOUS export is still in place and still
        queryable, because `commit` is the only thing that ever touches it. Idempotent, so a
        caller can abort from an exception handler without knowing how far it got."""
        if not self._fh.closed:
            self._fh.close()
        self.tmp.unlink(missing_ok=True)


def remove(owner: str) -> bool:
    """Stop serving this owner: deregister the export, then delete the file. Returns whether a
    file was there.

    The mirror of `Receiver.commit`, undoing its two steps in the REVERSE order for the same
    reason `commit` chose its own. `commit` renames before it registers, so no registry row ever
    names a file that does not exist yet; `remove` deregisters before it unlinks, so no registry
    row ever names a file that no longer exists. The window this closes is the one where
    `open_peer` resolves a row and then fails on a missing file — a confusing error for a state
    that is not confusing at all.

    Idempotent. Unpublishing a knowledge base that is not served is not an error, because the
    caller wants the same end state either way."""
    dest = export_path(owner)
    peers.remove(owner)
    existed = dest.exists()
    dest.unlink(missing_ok=True)
    return existed
