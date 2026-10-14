"""
pipeline/sharing/push_gate.py — whether the served copy is owed a push.

Push when the local store has changed since the last push, whether or not anybody has read since
(a reader sees the newest copy, not the one before their read). The change is
this store's position against the watermark the last successful push left, so the gate asks
nothing of the service. `decide` is the gate itself, plain values in and out; the rest of this
module gathers those values.

A share, a Google connection and `share(refresh=True)` owe a push (`owe`), which makes the next
pass due now. A share and a Google connection also start the fast hour (`start_fast_hour`). A pass that leaves the served copy current pays the debt (`record`), and
`forget_publication` clears it, with the watermark, when the copy is deleted.

Beside the watermark, this install keeps what the service holds row by row (`read_held`), which
`publish` cuts the next delta against.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from opyt_core.paths import opyt_path
from pipeline.kb import schema
from pipeline.sharing import fingerprint
from pipeline.timeparse import parse_ts, utc_iso


def decide(position: dict | None, watermark: dict | None, owed: bool = False) -> dict | None:
    """None to push, or the status that says why not.

      • nothing served, none owed → `not_published`. No push has succeeded since this install
                                    shared, or `unshare` withdrew the copy; only a new share
                                    or connection publishes again. A first push that died
                                    stays owed, so it is retried.
      • store has not moved       → `no_change`. An unreadable store counts as moved: the
                                    safe direction is a redundant push, never a missed one.
    """
    if watermark is None and not owed:
        return {"status": "not_published",
                "message": "nothing from this library is being served: sharing was stopped, "
                           "or never finished. A new `share` publishes it again."}
    if position is not None and position == watermark:
        return {"status": "no_change",
                "message": "the shared copy already matches this library."}
    return None


def _owed_path() -> Path:
    return opyt_path("push_owed")


def owe() -> None:
    """A push is wanted now: the next pass is due at once. An unwritable marker is ignored, and
    the push waits for the next regular pass."""
    try:
        p = _owed_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except OSError:
        pass


def owed_at() -> datetime | None:
    """When a push was last owed, or None when none is."""
    try:
        return datetime.fromtimestamp(_owed_path().stat().st_mtime, timezone.utc)
    except OSError:
        return None


def start_fast_hour() -> None:
    """A copy was just handed over (a share, the Google connection): passes run every
    `publishing.FAST` for `publishing.FAST_FOR` from now. An unwritable marker is ignored, and
    passes stay hourly."""
    try:
        p = _fast_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    except OSError:
        pass


def fast_since() -> datetime | None:
    """When the last fast hour started, or None when none has."""
    try:
        return datetime.fromtimestamp(_fast_path().stat().st_mtime, timezone.utc)
    except OSError:
        return None


def _fast_path() -> Path:
    return opyt_path("push_fast_since")


def record(status: str, position: dict | None, at: datetime) -> None:
    """What a push pass that started `at` leaves behind on this install, from the status it ended
    with.

    Every pass leaves its time (`passed_at`), which the next pass is due `publishing.EVERY`
    after. A push
    that succeeded moves the watermark — only then: see `write_watermark`. A push that succeeded,
    and a `no_change`, both leave the served copy current, so either pays a push owed by a share.
    Every other status changes nothing, so it is retried."""
    _write(_passed_path(), utc_iso(at))
    if status == "ok" and position is not None:
        write_watermark(position)
    if status in ("ok", "no_change"):
        _owed_path().unlink(missing_ok=True)


def _passed_path() -> Path:
    return opyt_path("push_passed")


def passed_at() -> datetime | None:
    """When the last push pass started, or None if none has been recorded."""
    try:
        return parse_ts(_passed_path().read_text().strip())
    except OSError:
        return None


def _write(p: Path, text: str) -> None:
    """Fail-safe: an unwritable marker costs one redundant pass, never a missed one."""
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    except OSError:
        pass


def forget_publication() -> None:
    """The served copy is gone. Nothing is published any more, so the watermark goes, and so does
    any push still owed: a debt left behind would re-publish what the owner just withdrew. What
    the service held goes too, so the next share sends the whole export."""
    _watermark_path().unlink(missing_ok=True)
    _owed_path().unlink(missing_ok=True)
    _held_path().unlink(missing_ok=True)


def _watermark_path() -> Path:
    """Resolved at call time so it honors `$OPYT_HOME` (Distributable: derive paths at runtime).
    A marker bound at import resolves to the real home under a sandboxed one."""
    return opyt_path("push_watermark")


def store_position() -> dict | None:
    """The store's `{atoms_at, atom_count, atom_writes}`, or None if it cannot be read: new,
    changed or re-observed atoms move the first (`upsert_atom` stamps `ingested_at`), removals the
    second, and every atom write the third (`upsert_atom` bumps `version`), so a rewrite in the
    same second as the last push still counts. The atom and its body are written together
    (`ingest_common.store_atom`), so a changed body moves it too."""
    try:
        conn = schema.connect(read_only=True)
    except Exception:
        return None
    try:
        atoms_at, atom_count, atom_writes = conn.execute(
            "SELECT MAX(ingested_at), COUNT(*), TOTAL(version) FROM atoms").fetchone()
    except sqlite3.OperationalError:
        return None            # a store that has never ingested — nothing to publish either
    finally:
        conn.close()
    return {"atoms_at": atoms_at, "atom_count": atom_count, "atom_writes": int(atom_writes)}


def read_watermark() -> dict | None:
    """The store position at the last successful push, or None if nothing is served from here:
    no push has succeeded, or `forget_publication` withdrew the copy."""
    try:
        return json.loads(_watermark_path().read_text())
    except Exception:
        return None


def write_watermark(position: dict) -> None:
    """Record what was just published. Called only after `publish` reports success, so a failed
    push never looks completed."""
    _write(_watermark_path(), json.dumps(position))


def _held_path() -> Path:
    return opyt_path("push_held.json")


def read_held() -> fingerprint.Fingerprint | None:
    """What the service holds, row by row, as of the last push that landed; None when no push has
    landed since sharing began (or the file is unreadable), and the next push then goes whole."""
    return fingerprint.load(_held_path())


def write_held(held: fingerprint.Fingerprint) -> None:
    """Record what the service now holds. Called only after the service accepted a push. An
    unwritable file leaves an older one, whose digest the service refuses as a base, so it costs
    one whole upload, never a wrong copy."""
    try:
        fingerprint.save(held, _held_path())
    except OSError:
        pass
