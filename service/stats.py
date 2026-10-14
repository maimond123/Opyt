"""service/stats.py — the public transparency roll-up, served as JSON (`/v1/stats`) or as a page
(`/stats`).

The collection ruling's transparency obligation: every number `TELEMETRY.md` says is collected,
as totals. No owner label, no reader and no query appears in it.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pipeline.retrieval import peers
from service import db, publications


def rollup() -> dict:
    """Every published number, from `service.db` and the peers registry."""
    return {**_service_db_counts(),
            "kbs_published": len(peers.list_peers()),
            "stored_bytes_by_kb": publications.stored_bytes(),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _service_db_counts() -> dict:
    """The `service.db` counts, in one open. Totals only: no owner, reader or label.

    `zero_result_rate` is None, not 0.0, when nothing has been searched."""
    conn = db.connect()
    try:
        total = conn.execute("SELECT COALESCE(SUM(n), 0) FROM usage_daily").fetchone()[0]
        recent = conn.execute("SELECT COALESCE(SUM(n), 0) FROM usage_daily "
                              "WHERE day >= date('now', '-30 days')").fetchone()[0]
        by_tool = {r["tool"]: r["n"] for r in conn.execute(
            "SELECT tool, SUM(n) AS n FROM usage_daily GROUP BY tool")}
        readers_30d = conn.execute("SELECT COUNT(DISTINCT reader) FROM usage_daily "
                                   "WHERE day >= date('now', '-30 days')").fetchone()[0]
        readers = conn.execute("SELECT COUNT(*) FROM tokens WHERE role = 'reader'").fetchone()[0]
        minted = conn.execute("SELECT COUNT(*) FROM grant_codes").fetchone()[0]
        redeemed = conn.execute(
            "SELECT COUNT(redeemed_at) FROM grant_codes").fetchone()[0]
        searches, zeroes = conn.execute(
            "SELECT COALESCE(SUM(n), 0), COALESCE(SUM(zero_results), 0) FROM usage_daily "
            "WHERE tool = 'search'").fetchone()
        stored = conn.execute("SELECT COALESCE(SUM(bytes), 0) FROM owner_uploads").fetchone()[0]
    finally:
        conn.close()
    return {
        "stored_bytes_total": stored,
        "reads_total": total,
        "reads_30d": recent,
        "reads_by_tool": by_tool,
        "active_readers_30d": readers_30d,
        "readers_total": readers,
        "codes_minted": minted,
        "codes_redeemed": redeemed,
        "zero_result_rate": (zeroes / searches) if searches else None,
    }


# Self-contained by requirement: fetching a font or script would leak the visitor list.
_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>OPYT sharing service — what it has served</title>
<style>
 body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem; }}
 dt {{ font-weight: 600; margin-top: 1.2rem; }}
 dd {{ margin: 0.1rem 0 0; color: #444; }}
 .n {{ font-size: 1.6rem; font-variant-numeric: tabular-nums; }}
 footer {{ margin-top: 3rem; color: #666; font-size: 0.9rem; }}
</style>
<h1>What this service has served</h1>
<p>Every number on this page is a total. No reader and no query appears here or in the database
behind it. Knowledge bases do appear, by their routing key — the address of a file, which
<code>register</code> now assigns at random, though a key claimed before it did so may still
read like a name; <code>/v1/stats</code> lists how much disk each one uses. See TELEMETRY.md in
the Opyt repository for the full schema.</p>
<dl>
 <dt class="n">{kbs_published}</dt><dd>knowledge bases published to this service.</dd>
 <dt class="n">{stored_mb}</dt><dd>of exports stored, across all of them.</dd>
 <dt class="n">{readers_total}</dt><dd>reader tokens currently held. Revoked ones leave this count
   and stay in the read totals below.</dd>
 <dt class="n">{codes_redeemed} / {codes_minted}</dt>
   <dd>grant codes redeemed, of those minted. The gap is invitations nobody used.</dd>
 <dt class="n">{reads_total}</dt><dd>reads served in total.</dd>
 <dt class="n">{reads_30d}</dt><dd>reads served in the last 30 days.</dd>
 <dt class="n">{active_readers_30d}</dt><dd>readers who read something in the last 30 days.</dd>
 <dt class="n">{by_tool}</dt><dd>reads by tool. Opens against searches is how often a result
   was worth reading in full.</dd>
 <dt class="n">{zero_rate}</dt><dd>of searches returned nothing.</dd>
</dl>
<footer>Generated {generated_at}.</footer>
"""


def page() -> str:
    """The same numbers as `/v1/stats`, filled from the same roll-up."""
    d = rollup()
    rate = d["zero_result_rate"]
    return _PAGE.format(
        by_tool=", ".join(f"{v} {k}" for k, v in sorted(d["reads_by_tool"].items())) or "none yet",
        zero_rate="not yet known" if rate is None else f"{rate:.0%}",
        stored_mb=f"{d['stored_bytes_total'] / 1_000_000:,.0f} MB",
        **d)
