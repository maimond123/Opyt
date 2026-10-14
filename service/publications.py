"""service/publications.py — each owner's served copy: its size, its dates, its push demand.

`owner_uploads`, one row per knowledge base, written from the first upload because neither
`first_published_at` nor `bytes` can be backfilled. `reads_at_upload` is a watermark on the read
meter (`service/usage.py`), so push demand is two counters compared exactly.
"""
from __future__ import annotations

from service import db, usage


def record_upload(owner: str, size: int) -> None:
    """This owner now stores `size` bytes. Called after the upload commits.

    `bytes` is replaced, not summed; `first_published_at` is never updated. Moving
    `reads_at_upload` here consumes the push demand."""
    conn = db.connect()
    try:
        reads = usage.total_reads(conn, owner)
        conn.execute(
            "INSERT INTO owner_uploads (owner, bytes, reads_at_upload) VALUES (?, ?, ?) "
            "ON CONFLICT(owner) DO UPDATE SET bytes = excluded.bytes, "
            "reads_at_upload = excluded.reads_at_upload, "
            "last_published_at = datetime('now')", (owner, size, reads))
        conn.commit()
    finally:
        conn.close()


def publish_demand(owner: str) -> dict:
    """`{last_upload_at, reads_since_last_upload}`: the demand half of the push gate.

    `last_upload_at` is NULL for a knowledge base never published, which the rail treats as
    "publish". Demand is total reads minus reads at the last upload, not reads since a date
    (`usage_daily` is day-granular)."""
    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT last_published_at, reads_at_upload FROM owner_uploads WHERE owner = ?",
            (owner,)).fetchone()
        last = row["last_published_at"] if row is not None and row["last_published_at"] else None
        reads = 0
        if last is not None:
            total = usage.total_reads(conn, owner)
            reads = max(0, total - row["reads_at_upload"])
    finally:
        conn.close()
    return {"last_upload_at": last, "reads_since_last_upload": reads}


def stored_bytes() -> list[dict]:
    """`{owner, bytes, owner_since, last_published_at}` per knowledge base, largest first. Keyed by
    the opaque routing key, with no label."""
    conn = db.connect()
    try:
        return [{"owner": r["owner"], "bytes": r["bytes"], "owner_since": r["first_published_at"],
                 "last_published_at": r["last_published_at"]}
                for r in conn.execute(
                    "SELECT owner, bytes, first_published_at, last_published_at "
                    "FROM owner_uploads ORDER BY bytes DESC, owner")]
    finally:
        conn.close()
