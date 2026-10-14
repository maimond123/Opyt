"""
pipeline/oracles/sources/first_pull.py — route a found or reviewed source to its first pull: the
archive walk that pulls it, begun under the caller's unit (`begin`).

One source → one route:

  blog / substack / github → begin → resolve
  scholar / orcid          → not built (needs a profile→publication resolver)

Every source arriving here is one discovery was sure of or the user picked, so
nothing judges authorship: a source wrongly added is removed after the fact with
`forget(source=…)`.

Fail-safe: a source that raises is caught per-source (reported as "error"), never aborting the
other sources.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Callable

from pipeline.ingestion.utils import log
from pipeline.kb import resolve

_WALKED = ("blog", "substack", "github")


def onboard_footprint(conn: sqlite3.Connection, oracle_id: str, sources: list[dict], *,
                      begin: Callable[..., dict | None], author_name: str | None = None,
                      since: datetime | None = None, isolate: bool = True) -> dict:
    """Route a confirmed Oracle's found `sources` to their first pull.

    `sources` are found-source dicts ({source_type, url, metadata, …}), each one discovery was sure
    of or the user picked: there is no trust or authorship check here.
    `begin(conn, url, *, author_name, since, handle)` begins one source's archive walk and returns
    `{"archive_started": True, …}`, or None when the url names nothing to walk.

    Each source ends as one of `ingested` (its walk began) / `error` / `not-built` / `skipped` /
    `unsupported`. `isolate=False` lets a source's exception out instead of recording it as
    `error`, for a caller routing one source.
    """
    oracle_id = (oracle_id or "").strip()
    if not oracle_id:
        return {"error": "onboard_footprint needs an oracle_id (e.g. x:user:123) to attribute to.",
                "results": []}

    results: list[dict] = []

    def _record(url, stype, action, detail="", started=False):
        rec = {"url": url, "type": stype, "action": action, "detail": detail}
        if started:
            rec["stats"] = {"archive_started": True}
        results.append(rec)

    for src in sources or []:
        stype = src.get("source_type") or ""
        url = src.get("url") or ""
        meta = src.get("metadata") or {}
        if not url:
            continue

        # 1. Scholar / ORCID → not built. `not-built`, not `deferred`: nothing resumes this one.
        if stype in ("scholar", "orcid"):
            _record(url, stype, "not-built", "scholar/orcid url→author-id resolver not built yet")
            continue
        if stype not in _WALKED:
            _record(url, stype, "unsupported", f"no atom-KB footprint adapter for {stype!r}")
            continue

        # 2. Blog / Substack / GitHub → its walk.
        try:
            kw = {"handle": meta.get("handle")} if stype != "github" else {}
            started = begin(conn, url, author_name=author_name, since=since, **kw)
            if started is None:
                _record(url, stype, "skipped", "no walkable site, publication or repo in url")
                continue
            # The walk's start upserts the source's entity: resolve folds it in now.
            resolve.resolve_entities(conn)
            _record(url, stype, "ingested", started=True)
        except Exception as e:                              # one bad source never aborts the rest
            if not isolate:
                raise
            log(f"[onboard] {stype} first pull failed for {url}: {type(e).__name__}: {e}")
            _record(url, stype, "error", f"{type(e).__name__}: {e}")

    # What the MCP `add_oracle`/`oracle` tool hands the model, i.e. what the user is told.
    return {
        "oracle_id": oracle_id,
        "ingested": sum(1 for r in results if r["action"] == "ingested"),
        "errors": sum(1 for r in results if r["action"] == "error"),
        "not_built": sum(1 for r in results if r["action"] == "not-built"),
        "results": results,
    }

