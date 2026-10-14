"""
pipeline/oracles/freshness.py — how far behind each Oracle is, and the notice that says so.

`status_summary` is a read-only snapshot, per Oracle and per source, worst first, derived from
what the scrapers write (`oracle_sources` and `found_sources`). `stale_notice` is the sentence a
search carries when refreshing has fallen past twice its target. Never raises: a freshness read
degrades, it never fails a tool call.
"""

from __future__ import annotations

from pipeline.kb import schema
from pipeline.oracles import found
from pipeline.oracles import registry
from pipeline.worker.quota_loop import retries
from pipeline.timeparse import utc_now
from pipeline.oracles import windows

# Fraction of tracked pairs overdue above which the freshness notice fires. A ratio rather than
# an absolute count so it means the same thing at 8 Oracles or 500; 0.5 means the refresh cycle
# has stretched past twice its target TTL.
STALE_FRACTION_ATTENTION = 0.5


def status_summary(conn) -> dict:
    """A read-only freshness snapshot — per Oracle, per source, worst first, derived from
    `oracle_sources` and `found_sources`. Never raises: a missing table degrades to an empty
    snapshot."""
    try:
        rows = registry.list_sources(conn)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tracked_pairs": 0, "oracles": []}

    # A found source whose first pull keeps failing has no pair yet, so `rows` is blind to it too.
    # Only the failing ones: a first pull merely owed is the archives milestone's to report.
    try:
        failing = [f for f in found.list_found(conn) if f.pulled_at is None and f.failures]
        names = {r["head"]: r["name"] for r in conn.execute(
            schema.ROSTER_HEADS_SQL + " ORDER BY o.confirmed_at DESC")}
    except Exception:
        failing, names = [], {}

    now = utc_now()
    by_oracle: dict = {}

    def entry_for(canonical_id: str, name: str | None) -> dict:
        return by_oracle.setdefault(canonical_id,
                                    {"canonical_id": canonical_id, "name": name,
                                     "sources": [], "stale_sources": 0})

    overdue = 0
    for r in rows:
        lag = windows.staleness_hours(r, now)
        stale = windows.is_stale(r, now)
        overdue += 1 if stale else 0
        entry = entry_for(r.canonical_id, r.name)
        entry["sources"].append({
            "source_type": r.source_type, "source_key": r.source_key,
            "last_pulled_at": r.last_pulled_at,
            # How far back this source goes; NULL means no pull has reported a lower bound.
            "covered_from": r.covered_from,
            # The effective ttl (base × this pair's stable jitter), not the base — otherwise the
            # report and the gate disagree, and `hours_overdue` reads off by up to 10%.
            "ttl_hours": round(windows.pair_ttl_hours(r), 1),
            "hours_overdue": None if lag == float("inf") else round(lag, 1),
            "never_refreshed": r.last_pulled_at is None,
            "last_status": r.last_status, "stale": stale,
            **_failure(r.failures, r.retry_at, r.last_error),
        })
        entry["stale_sources"] += 1 if stale else 0
    for f in failing:
        entry_for(f.canonical_id, names.get(schema.current_canonical(conn, f.canonical_id)))["sources"].append({
            "source_type": f.source_type, "source_key": f.url, "never_refreshed": True,
            "last_status": f.last_status, **_failure(f.failures, f.retry_at, f.last_error),
        })

    out = {
        "tracked_pairs": len(rows),
        "stale_pairs": overdue,
        "last_refreshed_at": max((r.last_pulled_at for r in rows if r.last_pulled_at),
                                 default=None),
        "ttl_hours": dict(windows.FLAT_TTL_HOURS),
        "oracles": sorted(by_oracle.values(), key=lambda e: -e["stale_sources"]),
    }
    out["stale_fraction"] = round(overdue / len(rows), 3) if rows else 0.0

    # The verdict lives here, not at call sites, so two surfaces cannot disagree. Silent on a
    # store with no Oracles yet.
    out["needs_attention"] = bool(rows and out["stale_fraction"] > STALE_FRACTION_ATTENTION)
    if out["needs_attention"]:
        out["note"] = (f"{overdue} of {len(rows)} Oracle sources are overdue — refreshing has "
                       f"fallen past twice its target. The background scrapers take the most "
                       f"overdue first and catch up on their own. To pull one person now, call "
                       f"oracle(action='ingest', canonical_ids=[...], x_lookback='since_last').")
    return out


def stale_notice(conn) -> dict | None:
    """The sentence `search` carries when Oracles have fallen behind, or None.

    Gated on `needs_attention` (unlike `oracle(action='screen')`, which always prints freshness):
    `search` is high-frequency, and an unconditional block trains the reader to skip it. The
    server's rider flow asks for it only on a search of the reader's own, non-empty store."""
    fresh = status_summary(conn)
    if not fresh.get("needs_attention"):
        return None
    return {"code": "oracles_stale", "freshness": fresh,
            "message": "Some of your Oracles have not been refreshed in a while, so "
                       "these results may miss what they published recently."}


def _failure(failures: int | None, retry_at: str | None, last_error: str | None) -> dict:
    """One source's failure record, as the scrapers' retry policy left it. Empty when it has none:
    a success clears the count. `given_up` means it is no longer tried until the user asks for
    that Oracle again; otherwise `retry_at` is when it next may run."""
    if not failures:
        return {}
    return {"failures": failures, "given_up": retries.given_up(failures),
            "retry_at": retry_at, "last_error": last_error}
