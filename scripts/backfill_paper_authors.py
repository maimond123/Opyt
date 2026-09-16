#!/usr/bin/env python3
"""
scripts/backfill_paper_authors.py — put an author list on paper atoms written before 2026-09-08.

ONE-TIME, and it exists because papers are immutable under Policy B. `atomize_paper` began
recording every author's registry ids on 2026-09-08 (stage 1 of
docs/plans/2026-09-08-researchers-as-oracles.md); every paper atom written before that carries
none, and no later run revisits a paper. So without this, `paper_authors` produces nothing at all
for a store's existing saves — on the live store that was 82 paper atoms, 11 of them `user-saved`,
which is the entire population the feature was built for.

WHAT IT WRITES: `payload.authors` only. It does NOT touch `raw_ref`, `raw_hash`, the snapshot, or
any chunk — nothing is re-fetched and nothing is re-embedded, so Policy B's actual subject (the
paper's BODY, which is immutable and expensive) is untouched. Adding a metadata key to a row is
the same class of write `promote_atom` already makes.

WHAT IT COSTS: one free OpenAlex request per atom, paced. No embed, no LLM, no money.

IDEMPOTENT: an atom that already has a non-empty `payload.authors` is skipped, so a re-run after a
partial pass costs only the atoms still missing one. Safe to interrupt.

IT HAS A DETECTOR, which is the bar the 2026-09-08 hand-run-entry-point audit set for a hand-run
pass that survives: `paper_authors._papers_missing_an_author_list` counts what is due and every
`curation_pull` report carries the number plus this command. `rechunk.py` was deleted that same
day for lacking one — "a maintenance pass whose trigger nothing can detect is a pass nobody runs"
— and `scripts/restrip_embed_surface.py` survives on exactly this basis.

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
from pipeline.kb.frontier_sources import _get, _openalex_authors   # noqa: E402

_API = "https://api.openalex.org/works"
_SELECT = "id,title,authorships"


def _lookup_key(atom_id: str) -> str | None:
    """A paper atom id → the OpenAlex `/works` filter that finds it, or None.

    Atom ids are `paper:arXiv:{id}` or `paper:DOI:{doi}` (`ingest_papers._canonical_paper_id`).
    An arXiv id resolves through its registered DOI — `10.48550/arXiv.{id}` — which is what
    OpenAlex indexes it under. Verified 2026-09-08 against 10.48550/arXiv.2404.07344.
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
    body = _get(url)
    if body is None:
        return None
    try:
        results = (json.loads(body) or {}).get("results") or []
    except ValueError:
        return None
    if not results:
        return None
    # Through the SAME extractor the adapter uses, then the same normalizer `atomize_paper` uses,
    # so a backfilled row is byte-identical to one written fresh. Two spellings of one shape is
    # how a backfill silently produces rows its own readers mis-parse.
    oa = _openalex_authors(results[0])
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
            # Payload only. `raw_hash`, `raw_ref` and every chunk are untouched, so the paper's
            # BODY — Policy B's actual subject — is not re-fetched, re-rendered or re-embedded.
            conn.execute("UPDATE atoms SET payload=? WHERE atom_id=?",
                         (json.dumps(merged), atom_id))
            conn.commit()
            filled += 1
            print(f"   ok {atom_id} ({entry_mode}): {len(authors)} author(s)")

        print(f"[backfill] filled={filled} no_openalex_record={missing} unkeyable={unkeyable}")
        print("[backfill] next: the paper-author signals are recomputed on the next "
              "`curation_catchup` pass, or now with "
              "`python -c \"from pipeline.kb import paper_authors, schema; "
              "print(paper_authors.sync_paper_author_signals(schema.connect()))\"`")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
