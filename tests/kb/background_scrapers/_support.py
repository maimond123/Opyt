"""Builders the background-scraper tests share: a page, a pair, a scouting sample."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pipeline.oracles import roster
from pipeline.oracles import registry
from pipeline.oracles.sources.timeline import pages

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def days(n: float) -> datetime:
    return NOW - timedelta(days=n)


def page(oldest_id, oldest_at, *, read_at=None, ended=False, complete=True, added=1):
    """What `walk_page` returns for one page: nothing for `write_page` to write."""
    return pages.Page(read_at=read_at or NOW, fetched=20, oldest_id=oldest_id, oldest_at=oldest_at,
                   ended=ended, complete=complete, added=added, submitted=added)


def sample(oldest_id, oldest_at, *, ended=False, fetched=20, written=2):
    """What `candidate_probe.sample_page` returns for one page."""
    return {"fetched": fetched, "fresh": fetched, "oldest_id": oldest_id, "oldest_at": oldest_at,
            "ended": ended, "written": written}


def pair(conn, handle="carol", cid=None, **cols):
    """A confirmed Oracle with one X pair, its `oracle_sources` columns set from `cols`."""
    cid = cid or f"x:user:{handle}"
    roster.upsert_oracle(conn, cid, name=handle)
    registry.upsert_source(conn, registry.SourceRow(cid, "x", handle))
    if cols:
        sets = ", ".join(f"{k}=?" for k in cols)
        conn.execute(f"UPDATE oracle_sources SET {sets} WHERE canonical_id=? AND source_key=?",
                     (*cols.values(), cid, handle))
    conn.commit()
    return cid


def stored(conn, handle="carol") -> registry.SourceRow:
    return next(r for r in registry.list_sources(conn) if r.source_key == handle)
