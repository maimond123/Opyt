"""
pipeline/kb/raw_store.py — the stored raw snapshot behind every atom's pointer.

An atom is a THIN card; on a LOCAL store the actual content lives here as a markdown
file, written UNDER `opyt_home()/kb_raw/{source}/`. (An EXPORT — pipeline/kb/export.py
— inlines those same bodies as a `kb_raw` TABLE instead, because it travels with no
filesystem beside it. `read_body` is the one reader that knows both shapes; everything
above it just asks for an atom's text.) Two reasons the files are NOT the vault `raw/`:
  1. The old indexer/processing pipeline walks the vault `raw/` and would
     double-represent every atom as a legacy note (Risk #1 in the plan).
  2. `kb_raw/` is the KB's own private store — its lifecycle is the atom's, not the
     vault's.

`raw_ref` is stored in the DB relative to `opyt_home()` (e.g.
`kb_raw/x/x_123_1a2b3c4d5e6f7081.md`), never absolute — so the store is distributable
(no hardcoded home path; Risk: the Distributable invariant). `resolve_ref()` rehydrates
it against the CURRENT home, so a copied `~/.opyt` still finds its snapshots.

The ref stored on an atom is authoritative, and `write_snapshot` alone derives filenames,
so a change to `_safe_name` needs no reader fallback and no rekey pass: existing atoms keep
reading the files they already name, and each one moves to the new name the next time its
source actually changes, leaving its old file orphaned under `kb_raw/`. Those orphans are
inert — nothing lists the directory to find an atom's body.

`raw_hash = sha256(snapshot)` is the idempotency + change key: identical raw → same
hash → the ingester skips (no re-embed, no re-write). A changed source → new hash →
re-snapshot + bump version.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from opyt_core.paths import opyt_home

_KB_RAW_DIRNAME = "kb_raw"


def snapshot_hash(markdown: str) -> str:
    """sha256 of the snapshot text — the change-detector. Computed on the STRING so a
    caller can hash-skip BEFORE touching disk or paying to embed."""
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


# How much of the readable prefix survives into the basename. 96 + `_` + 16 hex + `.md` stays far
# under the 255-BYTE filename limit even when every kept character is multi-byte UTF-8.
_NAME_PREFIX_CHARS = 96


def _safe_name(key: str) -> str:
    """`atom_id` → a filesystem-safe basename that is UNIQUE per atom_id and stable across runs.

    A readable prefix plus a digest of the WHOLE key: `x:123` → `x_123_a2c1…`,
    `github:owner/name` → `github_owner_name_9f04…`. Both halves are load-bearing.

    The digest is what makes the map injective, and injectivity is the correctness property, not a
    nicety: `raw_ref` is the only pointer from an atom to its real text, so two atom_ids sharing a
    filename means the second write silently replaces the body the first atom still points at —
    while its `raw_hash`, computed on the string before the write, keeps describing the body that
    is now gone. The prefix-only version had exactly that shape: every separator collapsed to `_`,
    and blog atom_ids preserve the URL path, so `blog:example.com/a/b` and `blog:example.com/a:b`
    both landed on `blog_example.com_a_b.md`.

    The prefix is what keeps `kb_raw/` greppable by hand, and the length cap is why it is a digest
    rather than a percent-encoding of the key: a blog atom_id is `blog:{host}{path}[?query]`, whose
    encoded form has no bound and can exceed the filesystem's 255-byte limit on one name.
    """
    prefix = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_"
                     for ch in key[:_NAME_PREFIX_CHARS])
    return f"{prefix}_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"


def kb_raw_dir(source: str) -> Path:
    """`opyt_home()/kb_raw/{source}` — created lazily by write_snapshot, resolved live."""
    return opyt_home() / _KB_RAW_DIRNAME / source


def write_snapshot(source: str, key: str, markdown: str) -> tuple[str, str]:
    """Write the snapshot and return `(raw_ref, raw_hash)`.

    `raw_ref` is RELATIVE to `opyt_home()` (portable). Overwriting with identical bytes
    is harmless (same path, same content) — callers still hash-skip upstream to avoid the
    expensive re-embed, this write is only the cheap tail.
    """
    d = kb_raw_dir(source)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{_safe_name(key)}.md"
    path.write_text(markdown, encoding="utf-8")
    raw_ref = str(path.relative_to(opyt_home()))
    return raw_ref, snapshot_hash(markdown)


def resolve_ref(raw_ref: str) -> Path:
    """A stored `raw_ref` → an absolute path against the CURRENT `opyt_home()`. Tolerates
    an already-absolute ref (older rows) by returning it as-is."""
    p = Path(raw_ref)
    return p if p.is_absolute() else (opyt_home() / p)


def read_snapshot(raw_ref: str) -> str | None:
    """The REAL raw text behind an atom — what `open()` injects to satisfy the trust
    invariant. Missing file → None (fail-safe: a deleted snapshot degrades, never crashes)."""
    p = resolve_ref(raw_ref)
    try:
        return p.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None


def _has_inlined_bodies(conn) -> bool:
    """Does this store carry its bodies in a table rather than beside it on disk?

    A `sqlite_master` lookup rather than a caught `OperationalError`: catching "no such table"
    by string would also swallow a genuinely corrupt database, and a corrupt store must fail
    loudly instead of degrading into "this atom has no body"."""
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='kb_raw'").fetchone() is not None


def read_body(conn, atom_id: str, raw_ref: str | None) -> str | None:
    """The REAL raw text behind an atom, from wherever THIS store keeps it.

    Two live shapes, not a fallback for a dead one: the local store keeps bodies as files under
    `opyt_home()/kb_raw/` and carries no `kb_raw` table, while an EXPORT (pipeline/kb/export.py)
    inlines them as that table and travels with no filesystem beside it. Both exist; the store
    itself says which it is.

    Fail-safe either way: a missing row or a missing file returns None, and `kb_open` reports
    that as `raw_available: False` rather than crashing."""
    if _has_inlined_bodies(conn):
        row = conn.execute("SELECT text FROM kb_raw WHERE atom_id = ?", (atom_id,)).fetchone()
        if row is not None:
            return row[0]
        return None
    return read_snapshot(raw_ref) if raw_ref else None
