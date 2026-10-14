#!/usr/bin/env python3
"""
scripts/backfill_paper_authors.py — put an author list on paper atoms that have none.

One-time: papers are immutable, so no later run adds `payload.authors` to a paper atom written
before `atomize_paper` recorded authors. Writes `payload.authors` only; nothing is re-fetched or
re-embedded. One free OpenAlex request per atom, paced. Idempotent: atoms that already have an
author list are skipped, so a re-run after an interrupted pass costs only what is left.
`papers._papers_missing_an_author_list` counts what is due, and the papers pass logs it.

    python scripts/backfill_paper_authors.py --dry-run     # what would change, no writes
    python scripts/backfill_paper_authors.py               # do it
"""
from __future__ import annotations

import argparse
import json
import sys
import time

# Repo root on the path, so this runs as `python scripts/…` from a clean checkout.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from pipeline.kb import ingest_papers, schema                      # noqa: E402
from pipeline.platforms.papers.registry import get                 # noqa: E402
from pipeline.platforms.papers.openalex import openalex_authors           # noqa: E402

_API = "https://api.openalex.org/works"
_SELECT = "id,title,authorships"


def _lookup_key(atom_id: str) -> str | None:
    """A paper atom id → the OpenAlex `/works` filter that finds it, or None.

    Atom ids are `paper:arXiv:{id}` or `paper:DOI:{doi}` (`derive.canonical_paper_id`).
    An arXiv id resolves through its registered DOI, `10.48550/arXiv.{id}`, which is what
    OpenAlex indexes it under.
    """
    body = atom_id.removeprefix("paper:")
    if body.lower().startswith("arxiv:"):
        return f"doi:10.48550/arXiv.{body.split(':', 1)[1]}"
    if body.lower().startswith("doi:"):
        return f"doi:{body.split(':', 1)[1]}"
    return None


def _fetch_authors(key: str) -> list[dict] | None:
    """The work's authors in OPYT's stored shape, or None when OpenAlex does not have it."""
    import urllib.parse

    url = f"{_API}?{urllib.parse.urlencode({'filter': key, 'select': _SELECT})}"
    body = get(url)
    if body is None:
        return None
    try:
        results = (json.loads(body) or {}).get("results") or []
    except ValueError:
        return None
    if not results:
        return None
    # The adapter's extractor and `atomize_paper`'s normalizer, so a backfilled row is
    # byte-identical to one written fresh.
    oa = openalex_authors(results[0])
    return ingest_papers.atom_authors({
        "authors": [{"name": a["name"],
                     **({"openalexId": oid} if (oid := a.get("openalex_id")) else {}),
                     **({"orcid": orc} if (orc := a.get("orcid")) else {}),
                     **({"position": pos} if (pos := a.get("position")) else {})}
                    for a in oa]})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--pace-seconds", type=float, default=1.0,
                    help="gap between requests (default 1.0 — OpenAlex documents 10/s)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N atoms (0 = all)")
    args = ap.parse_args(argv)

    conn = schema.connect()
    try:
        rows = conn.execute(
            "SELECT atom_id, entry_mode, payload FROM atoms WHERE source_type='paper' "
            "ORDER BY entry_mode='user-saved' DESC, atom_id").fetchall()
        todo = []
        for atom_id, entry_mode, payload in rows:
            try:
                have = (json.loads(payload or "{}") or {}).get("authors") or []
            except (TypeError, ValueError):
                have = []
            if not have:
                todo.append((atom_id, entry_mode, payload))
        if args.limit:
            todo = todo[:args.limit]

        print(f"[backfill] {len(rows)} paper atom(s); {len(todo)} without an author list")
        if args.dry_run:
            for atom_id, entry_mode, _ in todo[:40]:
                print(f"   {entry_mode:18} {atom_id}  key={_lookup_key(atom_id)}")
            return 0

        filled = missing = unkeyable = 0
        for i, (atom_id, entry_mode, payload) in enumerate(todo):
            key = _lookup_key(atom_id)
            if not key:
                unkeyable += 1
                continue
            if i:
                time.sleep(args.pace_seconds)
            authors = _fetch_authors(key)
            if not authors:
                missing += 1
                print(f"   -- {atom_id}: OpenAlex has no author list")
                continue
            merged = {**(json.loads(payload or "{}") or {}), "authors": authors}
            # Payload only: `raw_hash` and the body stay, so nothing is re-indexed.
            conn.execute("UPDATE atoms SET payload=? WHERE atom_id=?",
                         (json.dumps(merged), atom_id))
            conn.commit()
            filled += 1
            print(f"   ok {atom_id} ({entry_mode}): {len(authors)} author(s)")

        print(f"[backfill] filled={filled} no_openalex_record={missing} unkeyable={unkeyable}")
        print("[backfill] next: the paper-author signals are recomputed by the scrapers' "
              "hourly papers pass, or now with "
              "`python -c \"from pipeline.curation import papers; from pipeline.kb import schema; "
              "print(papers.sync_paper_author_signals(schema.connect()))\"`")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
