"""
pipeline/dedup_store.py

SQLite-backed dedup state. Each add is a single-row ``INSERT OR IGNORE`` keyed by
``(namespace, item_id)``, so concurrent writers never clobber each other's IDs.
``namespace`` is a source's state name (``synced_ids``, ``blog_synced``, ...).
"""

import json
import sqlite3
from pipeline.timeparse import utc_now
from pathlib import Path

from pipeline.kb.writes import write_tx
from pipeline.sqlite_db import default_db_path

# Never reuse a retired table name for this ledger, or a stale reader silently joins the wrong data.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_dedup (
    namespace TEXT NOT NULL,
    item_id   TEXT NOT NULL,
    synced_at TEXT NOT NULL,
    PRIMARY KEY (namespace, item_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS sync_dedup_seeded (
    namespace TEXT PRIMARY KEY,
    seeded_at TEXT NOT NULL
);
"""


def _now() -> str:
    # Full precision, unlike `utc_iso()`: stored rows are already at microsecond width, and
    # narrowing would make new rows sort against old ones on a shared prefix.
    return utc_now().isoformat()


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the ledger on a caller's connection, before its first transaction."""
    conn.executescript(_SCHEMA)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")     # readers don't block the writer
    conn.execute("PRAGMA busy_timeout=5000")    # wait up to 5s for a concurrent writer, don't error
    init_schema(conn)
    conn.commit()
    return conn


class SyncSet:
    """A set-like dedup store backed by one ``namespace`` of ``sync_dedup``.

    Quacks like a ``set`` (``in``, ``add``, ``len``, iteration); every ``.add()`` is a durable
    single-row insert. Add-only: dropping a synced ID would re-pull (and re-bill) that item.
    """

    def __init__(
        self,
        namespace: str,
        legacy_json: Path | None = None,
        db_path: Path | None = None,
    ):
        self.namespace = namespace
        self._db_path = Path(db_path) if db_path else default_db_path()
        self._conn = _connect(self._db_path)
        self._maybe_seed(legacy_json)
        # Pull this namespace's IDs into RAM once so __contains__ stays O(1) and
        # the hot ingest loop doesn't round-trip to SQLite per membership test.
        self._cache: set[str] = {
            row[0]
            for row in self._conn.execute(
                "SELECT item_id FROM sync_dedup WHERE namespace=?", (self.namespace,)
            )
        }

    def _maybe_seed(self, legacy_json: Path | None) -> None:
        """One-time import of the legacy JSON blob, guarded by the ``sync_dedup_seeded`` marker
        so it runs once per namespace; skipping it would re-pull the whole archive."""
        seeded = self._conn.execute(
            "SELECT 1 FROM sync_dedup_seeded WHERE namespace=?", (self.namespace,)
        ).fetchone()
        if seeded:
            return
        ids = []
        if legacy_json and legacy_json.exists():
            try:
                ids = json.loads(legacy_json.read_text())
            except (json.JSONDecodeError, OSError):
                ids = []
        now = _now()
        with write_tx(self._conn, site="dedup.seed"):
            self._conn.executemany(
                "INSERT OR IGNORE INTO sync_dedup(namespace, item_id, synced_at) VALUES (?,?,?)",
                [(self.namespace, str(i), now) for i in ids],
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO sync_dedup_seeded(namespace, seeded_at) VALUES (?,?)",
                (self.namespace, now),
            )

    def __contains__(self, item_id) -> bool:
        return str(item_id) in self._cache

    def __iter__(self):
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def add(self, item_id) -> None:
        sid = str(item_id)
        if sid in self._cache:
            return
        with write_tx(self._conn, site="dedup.add"):   # durable per add: a crash loses nothing
            self._conn.execute(
                "INSERT OR IGNORE INTO sync_dedup(namespace, item_id, synced_at) VALUES (?,?,?)",
                (self.namespace, sid, _now()),
            )
        self._cache.add(sid)

    def update(self, ids) -> None:
        new = [str(i) for i in ids if str(i) not in self._cache]
        if not new:
            return
        now = _now()
        with write_tx(self._conn, site="dedup.update"):
            self._conn.executemany(
                "INSERT OR IGNORE INTO sync_dedup(namespace, item_id, synced_at) VALUES (?,?,?)",
                [(self.namespace, i, now) for i in new],
            )
        self._cache.update(new)

    def close(self) -> None:
        self._conn.close()
